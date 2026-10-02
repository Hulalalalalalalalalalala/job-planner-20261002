import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from job_planner import execute_job, local_path, preview_plan, preview_retry, run_plan

ROOT = Path(__file__).resolve().parent


class JobPlannerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "notes.txt").write_text("one\ntwo\n", encoding="utf-8")
        (self.root / "sales.csv").write_text("item,count\nbook,2\npen,4\n", encoding="utf-8")

    def test_operations_are_deterministic(self):
        self.assertEqual(execute_job(self.root, {"operation": "count-lines", "input": "notes.txt"}), {"lines": 2})
        checksum = {"operation": "sha256", "input": "notes.txt"}
        self.assertEqual(execute_job(self.root, checksum), execute_job(self.root, checksum))
        self.assertEqual(execute_job(self.root, {"operation": "csv-summary", "input": "sales.csv"}), {"columns": ["item", "count"], "rows": 2})

    def test_path_boundaries_and_input_preservation(self):
        for relative in ("../outside.txt", str(ROOT / "README.md")):
            with self.assertRaises(ValueError):
                local_path(self.root, relative)
        jobs = [{"name": "notes", "operation": "sha256", "input": "notes.txt"}]
        with self.assertRaises(ValueError):
            run_plan(self.root, jobs, "notes.txt")
        self.assertEqual((self.root / "notes.txt").read_text(), "one\ntwo\n")

    def test_symlink_escape_rejected(self):
        (self.root / "outside").symlink_to(ROOT / "README.md")
        with self.assertRaises(ValueError):
            execute_job(self.root, {"operation": "sha256", "input": "outside"})

    def test_failure_recorded_without_losing_other_job(self):
        jobs = [{"name": "bad", "operation": "shell", "input": "notes.txt"}, {"name": "good", "operation": "count-lines", "input": "notes.txt"}]
        result = run_plan(self.root, jobs, "results/report.json")
        self.assertEqual([row["status"] for row in result], ["failed", "completed"])
        self.assertEqual(json.loads((self.root / "results/report.json").read_text())["results"], result)

    def test_cli_plan_and_duplicate_names(self):
        jobs = [{"name": "notes", "operation": "count-lines", "input": "notes.txt"}]
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json", "--root", str(self.root)]
        run = subprocess.run(prefix, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"completed": 1, "failed": 0})
        plan.write_text(json.dumps({"jobs": jobs * 2}))
        self.assertEqual(subprocess.run(prefix, capture_output=True).returncode, 2)

    def test_forward_references_run_in_deterministic_order(self):
        jobs = [
            {"name": "final", "operation": "sha256", "input": "notes.txt", "depends_on": ["mid", "indirect"]},
            {"name": "mid", "operation": "count-lines", "input": "notes.txt", "depends_on": ["base"]},
            {"name": "indirect", "operation": "count-lines", "input": "sales.csv", "depends_on": ["base"]},
            {"name": "base", "operation": "sha256", "input": "notes.txt"},
        ]
        result = run_plan(self.root, jobs, "results/report.json")
        self.assertEqual([row["name"] for row in result], ["base", "mid", "indirect", "final"])
        self.assertTrue(all(row["status"] == "completed" for row in result))
        report = json.loads((self.root / "results/report.json").read_text())["results"]
        self.assertEqual([row["name"] for row in report], ["base", "mid", "indirect", "final"])
        # Rerunning the same plan produces the same report bytes.
        self.assertEqual((self.root / "results/report.json").read_text(),
                         json.dumps({"results": run_plan(self.root, jobs, "results/report.json")},
                                    ensure_ascii=False, indent=2) + "\n")

    def test_failed_dependency_blocks_downstream_while_other_branch_runs(self):
        jobs = [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "downstream", "operation": "count-lines", "input": "notes.txt", "depends_on": ["broken"]},
            {"name": "grandchild", "operation": "sha256", "input": "notes.txt", "depends_on": ["downstream"]},
            {"name": "joiner", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["broken", "healthy"]},
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
        ]
        result = run_plan(self.root, jobs, "results/report.json")
        # Each round picks the ready task earliest in plan order: after
        # broken, downstream (index 1) precedes healthy (index 4), and its
        # own dependent grandchild (index 2) precedes healthy as well.
        self.assertEqual([row["name"] for row in result],
                         ["broken", "downstream", "grandchild", "healthy", "joiner"])
        by_name = {row["name"]: row for row in result}
        self.assertEqual(by_name["broken"]["status"], "failed")
        self.assertIn("error", by_name["broken"])
        self.assertEqual(by_name["healthy"]["status"], "completed")
        self.assertEqual(by_name["downstream"],
                         {"name": "downstream", "status": "blocked", "blocked_by": ["broken"]})
        self.assertEqual(by_name["joiner"],
                         {"name": "joiner", "status": "blocked", "blocked_by": ["broken"]})
        self.assertEqual(by_name["grandchild"],
                         {"name": "grandchild", "status": "blocked", "blocked_by": ["downstream"]})
        # Each task produces exactly one record.
        self.assertEqual(len(result), len({row["name"] for row in result}))

    def test_blocked_job_does_not_read_input_but_still_validates_path(self):
        jobs = [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "stray", "operation": "count-lines", "input": "missing.txt", "depends_on": ["broken"]},
        ]
        result = run_plan(self.root, jobs, "results/report.json")
        self.assertEqual(result[1],
                         {"name": "stray", "status": "blocked", "blocked_by": ["broken"]})
        # The pre-run root boundary check applies to blocked jobs as well.
        escaping = [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "stray", "operation": "count-lines", "input": "../outside.txt", "depends_on": ["broken"]},
        ]
        with self.assertRaises(ValueError):
            run_plan(self.root, escaping, "results/report.json")

    def test_empty_dependency_list_is_equivalent_to_no_field(self):
        jobs = [{"name": "notes", "operation": "count-lines", "input": "notes.txt", "depends_on": []}]
        result = run_plan(self.root, jobs, "results/report.json")
        self.assertEqual([row["status"] for row in result], ["completed"])

    def test_invalid_dependencies_raise_value_error(self):
        good = {"name": "good", "operation": "count-lines", "input": "notes.txt"}
        cases = [
            [{"name": "bad", "operation": "count-lines", "input": "notes.txt", "depends_on": "good"}],
            [{"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["a"]}],
            [{"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["good", "good"]}, good],
            [{"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["  "]}, good],
            [{"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": [1]}, good],
            [{"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": "good"}, good],
            [{"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["b"]},
             {"name": "b", "operation": "count-lines", "input": "notes.txt", "depends_on": ["a"]}],
        ]
        for jobs in cases:
            with self.subTest(jobs=jobs):
                with self.assertRaises(ValueError):
                    run_plan(self.root, jobs, "results/report.json")

    def test_invalid_plan_preserves_existing_report(self):
        report = self.root / "results/report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        marker = '{"results": [{"name": "kept"}]}'
        report.write_text(marker, encoding="utf-8")
        jobs = [{"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["ghost"]}]
        with self.assertRaises(ValueError):
            run_plan(self.root, jobs, "results/report.json")
        self.assertEqual(report.read_text(), marker)

    def test_cli_blocked_summary_and_exit_code(self):
        jobs = [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "downstream", "operation": "count-lines", "input": "notes.txt", "depends_on": ["broken"]},
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
        ]
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json", "--root", str(self.root)]
        run = subprocess.run(prefix, capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"completed": 1, "failed": 1, "blocked": 1})
        # An invalid dependency plan returns 2, emits the error JSON and never writes the report.
        bad_jobs = [{"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["x"]}]
        plan.write_text(json.dumps({"jobs": bad_jobs}))
        run = subprocess.run(prefix + ["--output", "results/other.json"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertIn("error", json.loads(run.stdout))
        self.assertFalse((self.root / "results/other.json").exists())

    def test_targets_run_only_selection_and_prerequisites(self):
        jobs = [
            {"name": "final", "operation": "sha256", "input": "notes.txt", "depends_on": ["mid", "indirect"]},
            {"name": "mid", "operation": "count-lines", "input": "notes.txt", "depends_on": ["base"]},
            {"name": "indirect", "operation": "count-lines", "input": "sales.csv", "depends_on": ["base"]},
            {"name": "base", "operation": "sha256", "input": "notes.txt"},
            {"name": "unrelated", "operation": "shell", "input": "missing.txt"},
        ]
        result = run_plan(self.root, jobs, "results/report.json", targets=["final"])
        self.assertEqual([row["name"] for row in result], ["base", "mid", "indirect", "final"])
        self.assertTrue(all(row["status"] == "completed" for row in result))
        report = json.loads((self.root / "results/report.json").read_text())["results"]
        self.assertEqual(report, result)
        # None keeps the full-plan behavior, including the unrelated failure.
        full = run_plan(self.root, jobs, "results/report.json")
        self.assertEqual([row["name"] for row in full],
                         ["base", "mid", "indirect", "final", "unrelated"])
        self.assertEqual(full[-1]["status"], "failed")

    def test_targets_share_prerequisites_and_ignore_declaration_order(self):
        jobs = [
            {"name": "left", "operation": "count-lines", "input": "notes.txt", "depends_on": ["base"]},
            {"name": "right", "operation": "sha256", "input": "notes.txt", "depends_on": ["base"]},
            {"name": "base", "operation": "sha256", "input": "sales.csv"},
            {"name": "other", "operation": "count-lines", "input": "notes.txt"},
        ]
        for targets in (["left", "right"], ["right", "left"]):
            result = run_plan(self.root, jobs, "results/report.json", targets=targets)
            self.assertEqual([row["name"] for row in result], ["base", "left", "right"])

    def test_failed_shared_prerequisite_blocks_one_target_only(self):
        jobs = [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "needy", "operation": "count-lines", "input": "notes.txt", "depends_on": ["broken"]},
            {"name": "free", "operation": "count-lines", "input": "notes.txt"},
            {"name": "unrelated", "operation": "count-lines", "input": "notes.txt"},
        ]
        result = run_plan(self.root, jobs, "results/report.json", targets=["needy", "free"])
        self.assertEqual([row["name"] for row in result], ["broken", "needy", "free"])
        self.assertEqual([row["status"] for row in result], ["failed", "blocked", "completed"])
        self.assertEqual(result[1], {"name": "needy", "status": "blocked", "blocked_by": ["broken"]})

    def test_invalid_targets_raise_value_error_and_keep_report(self):
        jobs = [{"name": "notes", "operation": "count-lines", "input": "notes.txt"}]
        report = self.root / "results/report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        marker = '{"results": [{"name": "kept"}]}'
        report.write_text(marker, encoding="utf-8")
        for targets in ([], "notes", [1], ["  "], ["notes", "notes"], ["ghost"], ["notes", "ghost"]):
            with self.subTest(targets=targets):
                with self.assertRaises(ValueError):
                    run_plan(self.root, jobs, "results/report.json", targets=targets)
        self.assertEqual(report.read_text(), marker)

    def test_unselected_branch_still_validated(self):
        jobs = [
            {"name": "wanted", "operation": "count-lines", "input": "notes.txt"},
            {"name": "stray", "operation": "count-lines", "input": "../outside.txt"},
        ]
        with self.assertRaises(ValueError):
            run_plan(self.root, jobs, "results/report.json", targets=["wanted"])
        bad_deps = [
            {"name": "wanted", "operation": "count-lines", "input": "notes.txt"},
            {"name": "stray", "operation": "count-lines", "input": "notes.txt", "depends_on": ["ghost"]},
        ]
        with self.assertRaises(ValueError):
            run_plan(self.root, bad_deps, "results/report.json", targets=["wanted"])
        self.assertFalse((self.root / "results/report.json").exists())

    def test_cli_only_flag(self):
        jobs = [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "downstream", "operation": "count-lines", "input": "notes.txt", "depends_on": ["broken"]},
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
        ]
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json", "--root", str(self.root)]
        # Selecting only the healthy job skips the failing branch entirely.
        run = subprocess.run(prefix + ["--only", "healthy"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"completed": 1, "failed": 0})
        report = json.loads((self.root / ".results/latest.json").read_text())["results"]
        self.assertEqual([row["name"] for row in report], ["healthy"])
        # Selecting the dependent pulls in its failing prerequisite and reports blocked.
        run = subprocess.run(prefix + ["--only", "downstream"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"completed": 0, "failed": 1, "blocked": 1})
        # Repeated --only selects several targets.
        run = subprocess.run(prefix + ["--only", "downstream", "--only", "healthy"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"completed": 1, "failed": 1, "blocked": 1})
        # An unknown or repeated target returns 2 with error JSON and keeps the old report.
        before = (self.root / ".results/latest.json").read_text()
        for extra in (["--only", "ghost"], ["--only", "healthy", "--only", "healthy"], ["--only", " "]):
            run = subprocess.run(prefix + extra, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)
            self.assertIn("error", json.loads(run.stdout))
        self.assertEqual((self.root / ".results/latest.json").read_text(), before)

    def test_cli_summary_without_dependencies_keeps_fields(self):
        jobs = [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
        ]
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json", "--root", str(self.root)]
        run = subprocess.run(prefix, capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"completed": 1, "failed": 1})

    def test_preview_full_plan(self):
        jobs = [
            {"name": "final", "operation": "sha256", "input": "notes.txt", "depends_on": ["mid"]},
            {"name": "mid", "operation": "count-lines", "input": "notes.txt", "depends_on": ["base"]},
            {"name": "base", "operation": "sha256", "input": "notes.txt"},
            {"name": "alone", "operation": "shell", "input": "missing.txt", "depends_on": []},
        ]
        preview = preview_plan(self.root, jobs, "results/report.json")
        self.assertEqual(preview, {"jobs": [
            {"name": "base", "depends_on": [], "reason": "all", "required_by": []},
            {"name": "mid", "depends_on": ["base"], "reason": "all", "required_by": []},
            {"name": "final", "depends_on": ["mid"], "reason": "all", "required_by": []},
            {"name": "alone", "depends_on": [], "reason": "all", "required_by": []},
        ]})

    def test_preview_targets_order_reasons_and_required_by(self):
        jobs = [
            {"name": "final", "operation": "sha256", "input": "notes.txt", "depends_on": ["mid", "indirect"]},
            {"name": "mid", "operation": "count-lines", "input": "notes.txt", "depends_on": ["base"]},
            {"name": "indirect", "operation": "count-lines", "input": "sales.csv", "depends_on": ["base"]},
            {"name": "before", "operation": "count-lines", "input": "notes.txt", "depends_on": ["base"]},
            {"name": "base", "operation": "sha256", "input": "notes.txt"},
            {"name": "unrelated", "operation": "shell", "input": "missing.txt"},
        ]
        expected = {"jobs": [
            {"name": "base", "depends_on": [], "reason": "prerequisite", "required_by": ["final", "before"]},
            {"name": "mid", "depends_on": ["base"], "reason": "prerequisite", "required_by": ["final"]},
            {"name": "indirect", "depends_on": ["base"], "reason": "prerequisite", "required_by": ["final"]},
            {"name": "final", "depends_on": ["mid", "indirect"], "reason": "target", "required_by": ["final"]},
            {"name": "before", "depends_on": ["base"], "reason": "target", "required_by": ["before"]},
        ]}
        # Swapping target argument order must not change the preview.
        self.assertEqual(preview_plan(self.root, jobs, "results/report.json", targets=["final", "before"]),
                         expected)
        self.assertEqual(preview_plan(self.root, jobs, "results/report.json", targets=["before", "final"]),
                         expected)

    def test_preview_does_not_execute_read_or_write(self):
        jobs = [
            {"name": "missing", "operation": "count-lines", "input": "nope.txt"},
            {"name": "bad-op", "operation": "shell", "input": "notes.txt"},
            {"name": "bad-csv", "operation": "csv-summary", "input": "notes.txt"},
            {"name": "ok", "operation": "sha256", "input": "notes.txt", "depends_on": ["missing"]},
        ]
        before = {p.name for p in self.root.iterdir()}
        preview = preview_plan(self.root, jobs, "results/new/deep/report.json")
        self.assertEqual([row["name"] for row in preview["jobs"]],
                         ["missing", "bad-op", "bad-csv", "ok"])
        self.assertFalse({row.get("status") for row in preview["jobs"]} - {None})
        self.assertFalse((self.root / "results").exists())
        self.assertEqual({p.name for p in self.root.iterdir()}, before)

    def test_preview_validates_whole_plan_invalid_targets_raise(self):
        jobs = [
            {"name": "wanted", "operation": "count-lines", "input": "notes.txt"},
            {"name": "stray", "operation": "count-lines", "input": "../outside.txt"},
        ]
        with self.assertRaises(ValueError):
            preview_plan(self.root, jobs, "results/report.json", targets=["wanted"])
        bad_deps = [
            {"name": "wanted", "operation": "count-lines", "input": "notes.txt"},
            {"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["b"]},
            {"name": "b", "operation": "count-lines", "input": "notes.txt", "depends_on": ["a"]},
        ]
        with self.assertRaises(ValueError):
            preview_plan(self.root, bad_deps, "results/report.json")
        good = [{"name": "notes", "operation": "count-lines", "input": "notes.txt"}]
        with self.assertRaises(ValueError):
            preview_plan(self.root, good, "notes.txt")
        for targets in ([], "notes", [1], ["  "], ["notes", "notes"], ["ghost"]):
            with self.subTest(targets=targets):
                with self.assertRaises(ValueError):
                    preview_plan(self.root, good, "results/report.json", targets=targets)

    def test_preview_matches_run_order_for_selected_target(self):
        jobs = [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "downstream", "operation": "count-lines", "input": "notes.txt", "depends_on": ["broken"]},
            {"name": "grandchild", "operation": "sha256", "input": "notes.txt", "depends_on": ["downstream"]},
            {"name": "joiner", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["broken", "healthy"]},
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
        ]
        preview = preview_plan(self.root, jobs, "results/report.json", targets=["joiner"])
        self.assertEqual([row["name"] for row in preview["jobs"]],
                         ["broken", "healthy", "joiner"])
        self.assertEqual([row["reason"] for row in preview["jobs"]],
                         ["prerequisite", "prerequisite", "target"])
        self.assertTrue(all(row["required_by"] == ["joiner"] for row in preview["jobs"]))
        result = run_plan(self.root, jobs, "results/report.json", targets=["joiner"])
        self.assertEqual([row["name"] for row in result],
                         [row["name"] for row in preview["jobs"]])

    def test_cli_preview_flag(self):
        jobs = [
            {"name": "broken", "operation": "shell", "input": "missing.txt"},
            {"name": "downstream", "operation": "count-lines", "input": "notes.txt", "depends_on": ["broken"]},
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
        ]
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json", "--root", str(self.root)]
        run = subprocess.run(prefix + ["--preview"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        payload = json.loads(run.stdout)
        self.assertEqual([row["name"] for row in payload["jobs"]],
                         ["broken", "downstream", "healthy"])
        self.assertNotIn("completed", run.stdout)
        self.assertFalse((self.root / ".results").exists())
        # With --only the preview shares the same selection semantics.
        run = subprocess.run(prefix + ["--preview", "--only", "downstream"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        payload = json.loads(run.stdout)
        self.assertEqual([row["name"] for row in payload["jobs"]], ["broken", "downstream"])
        self.assertEqual(payload["jobs"][0]["required_by"], ["downstream"])
        self.assertEqual(payload["jobs"][1]["reason"], "target")
        self.assertFalse((self.root / ".results").exists())
        # Validation failures and illegal targets still return 2 error JSON
        # and leave existing reports untouched.
        report = self.root / ".results/latest.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("KEEP", encoding="utf-8")
        bad_jobs = [{"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["x"]}]
        plan.write_text(json.dumps({"jobs": bad_jobs}))
        for extra in (["--preview"], ["--preview", "--only", "ghost"]):
            run = subprocess.run(prefix + extra, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)
            self.assertIn("error", json.loads(run.stdout))
        self.assertEqual(report.read_text(), "KEEP")
        # The plan file itself is protected in preview mode as well.
        plan.write_text(json.dumps({"jobs": jobs}))
        run = subprocess.run(prefix + ["--preview", "--output", "plan.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertIn("error", json.loads(run.stdout))


class PreviewRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "notes.txt").write_text("one\ntwo\n", encoding="utf-8")
        (self.root / "sales.csv").write_text("item,count\nbook,2\npen,4\n", encoding="utf-8")
        self.jobs = [
            {"name": "final", "operation": "sha256", "input": "notes.txt", "depends_on": ["mid", "indirect"]},
            {"name": "mid", "operation": "count-lines", "input": "notes.txt", "depends_on": ["base"]},
            {"name": "indirect", "operation": "count-lines", "input": "sales.csv", "depends_on": ["base"]},
            {"name": "base", "operation": "sha256", "input": "notes.txt"},
            {"name": "unrelated", "operation": "shell", "input": "missing.txt"},
        ]

    def write_report(self, rows, rel="results/report.json"):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"results": rows}), encoding="utf-8")
        return rel

    def test_targets_in_plan_order_with_closure_regardless_of_report_order(self):
        # Report order deliberately scrambled relative to the plan.
        rel = self.write_report([
            {"name": "unrelated", "status": "failed", "error": "x"},
            {"name": "base", "status": "completed", "result": {"sha256": "old", "bytes": 0}},
            {"name": "mid", "status": "failed", "error": "y"},
            {"name": "final", "status": "blocked", "blocked_by": ["mid"]},
            {"name": "indirect", "status": "completed", "result": {"lines": 1}},
        ])
        preview = preview_retry(self.root, self.jobs, "results/report.json", rel)
        # Targets follow current plan order, not record order; unrelated (no
        # prerequisites) sorts after the dependency chain members.
        self.assertEqual(preview["targets"], ["final", "mid", "unrelated"])
        expected_jobs = preview_plan(self.root, self.jobs, "results/report.json",
                                     targets=["final", "mid", "unrelated"])["jobs"]
        self.assertEqual(preview["jobs"], expected_jobs)
        names = [row["name"] for row in preview["jobs"]]
        self.assertEqual(names, ["base", "mid", "indirect", "final", "unrelated"])
        reasons = {row["name"]: row["reason"] for row in preview["jobs"]}
        self.assertEqual(reasons, {"base": "prerequisite", "mid": "target",
                                   "indirect": "prerequisite", "final": "target",
                                   "unrelated": "target"})
        self.assertEqual({n: row["required_by"] for n, row in
                          ((j["name"], j) for j in preview["jobs"])}["base"],
                         ["final", "mid"])

    def test_completed_prerequisites_still_included(self):
        rel = self.write_report([
            {"name": "base", "status": "completed", "result": {"sha256": "old", "bytes": 8}},
            {"name": "mid", "status": "completed", "result": {"lines": 2}},
            {"name": "final", "status": "failed", "error": "boom"},
        ])
        preview = preview_retry(self.root, self.jobs, "results/out.json", rel)
        self.assertEqual(preview["targets"], ["final"])
        self.assertEqual([row["name"] for row in preview["jobs"]],
                         ["base", "mid", "indirect", "final"])
        self.assertTrue(all(row["reason"] in ("target", "prerequisite") for row in preview["jobs"]))

    def test_partial_report_includes_only_required_unrecorded_jobs(self):
        # Last run covered just the mid branch; base was recorded completed
        # while final/indirect/unrelated have no records at all.
        rel = self.write_report([
            {"name": "base", "status": "completed"},
            {"name": "mid", "status": "failed"},
        ])
        preview = preview_retry(self.root, self.jobs, "results/out.json", rel)
        self.assertEqual(preview["targets"], ["mid"])
        self.assertEqual([row["name"] for row in preview["jobs"]], ["base", "mid"])

    def test_blocked_records_are_targets_and_shared_prerequisite_once(self):
        jobs = [
            {"name": "left", "operation": "count-lines", "input": "notes.txt", "depends_on": ["base"]},
            {"name": "right", "operation": "sha256", "input": "notes.txt", "depends_on": ["base"]},
            {"name": "base", "operation": "sha256", "input": "notes.txt"},
            {"name": "lonely", "operation": "count-lines", "input": "notes.txt"},
        ]
        rel = self.write_report([
            {"name": "base", "status": "failed"},
            {"name": "left", "status": "blocked", "blocked_by": ["base"]},
            {"name": "right", "status": "blocked", "blocked_by": ["base"]},
        ])
        preview = preview_retry(self.root, jobs, "results/out.json", rel)
        self.assertEqual(preview["targets"], ["left", "right", "base"])
        self.assertEqual([row["name"] for row in preview["jobs"]],
                         ["base", "left", "right"])

    def test_no_failed_or_blocked_records_gives_empty_preview(self):
        for rows in ([], [{"name": "base", "status": "completed"}],
                     [{"name": "base", "status": "completed", "extra": "ignored"},
                      {"name": "mid", "status": "completed"}]):
            with self.subTest(rows=rows):
                rel = self.write_report(rows)
                self.assertEqual(preview_retry(self.root, self.jobs, "results/out.json", rel),
                                 {"targets": [], "jobs": []})

    def test_does_not_execute_read_inputs_create_or_write(self):
        rows = [
            {"name": "unrelated", "status": "failed"},
            {"name": "final", "status": "failed"},
        ]
        rel = self.write_report(rows)
        before = {p.name for p in self.root.iterdir()}
        before_report = (self.root / rel).read_text()
        # Bad operation, missing input, nonexistent deep output dir: none of
        # these block a retry preview and nothing is created.
        preview = preview_retry(self.root, self.jobs, "results/new/deep/out.json", rel)
        self.assertEqual(preview["targets"], ["final", "unrelated"])
        self.assertFalse((self.root / "results" / "new").exists())
        self.assertEqual({p.name for p in self.root.iterdir()}, before)
        # The source report stays byte-identical.
        self.assertEqual((self.root / rel).read_text(), before_report)

    def test_report_may_equal_output(self):
        rel = self.write_report([{"name": "mid", "status": "failed"}])
        preview = preview_retry(self.root, self.jobs, rel, rel)
        self.assertEqual(preview["targets"], ["mid"])
        self.assertEqual([row["name"] for row in preview["jobs"]], ["base", "mid"])

    def test_invalid_reports_raise_value_error(self):
        valid_rows = [{"name": "mid", "status": "failed"}]
        good = json.dumps({"results": valid_rows})

        def use(text, name="results/bad.json", raw=False):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if raw:
                path.write_bytes(text)
            else:
                path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                preview_retry(self.root, self.jobs, "results/out.json", name)

        use("not json at all")
        use(json.dumps([1, 2, 3]))                       # not an object
        use(json.dumps({"results": {}}))                # results not a list
        use(json.dumps({"results": "nope"}))
        use(json.dumps({}))                             # results missing
        use(json.dumps({"results": [42]}))             # row not an object
        use(json.dumps({"results": [{"status": "failed"}]}))  # name missing
        use(json.dumps({"results": [{"name": "ghost", "status": "failed"}]}))
        use(json.dumps({"results": [{"name": 7, "status": "failed"}]}))
        use(json.dumps({"results": [{"name": "mid", "status": "nope"}]}))
        use(json.dumps({"results": [{"name": "mid"}]}))  # status missing
        use(json.dumps({"results": [{"name": "mid", "status": "FAILED"}]}))
        use(json.dumps({"results": valid_rows * 2}))    # duplicate name
        use(b'{"results": []}\xff', raw=True)           # invalid UTF-8
        # An empty results list is legal.
        (self.root / "results/empty.json").write_text(
            json.dumps({"results": []}), encoding="utf-8")
        self.assertEqual(preview_retry(self.root, self.jobs, "results/out.json",
                                       "results/empty.json"),
                         {"targets": [], "jobs": []})

    def test_missing_and_unreadable_report_raises(self):
        with self.assertRaises(ValueError):
            preview_retry(self.root, self.jobs, "results/out.json", "results/missing.json")
        (self.root / "dir").mkdir()
        with self.assertRaises(ValueError):
            preview_retry(self.root, self.jobs, "results/out.json", "dir")

    def test_report_path_must_stay_inside_root(self):
        self.write_report([{"name": "mid", "status": "failed"}], "inside.json")
        for rel in ("../outside.json", str(ROOT / "README.md"), ".."):
            with self.subTest(rel=rel):
                with self.assertRaises(ValueError):
                    preview_retry(self.root, self.jobs, "results/out.json", rel)
        (self.root / "link").symlink_to(ROOT / "README.md")
        with self.assertRaises(ValueError):
            preview_retry(self.root, self.jobs, "results/out.json", "link")

    def test_whole_plan_validated_including_empty_retry_scope(self):
        # Unselected branch with an escaping input still rejects, even
        # though no retry targets exist.
        rel = self.write_report([{"name": "final", "status": "completed"}])
        bad_path = self.jobs + [{"name": "stray", "operation": "count-lines",
                                 "input": "../outside.txt"}]
        with self.assertRaises(ValueError):
            preview_retry(self.root, bad_path, "results/out.json", rel)
        bad_deps = [
            {"name": "final", "operation": "count-lines", "input": "notes.txt"},
            {"name": "stray", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["ghost"]},
        ]
        with self.assertRaises(ValueError):
            preview_retry(self.root, bad_deps, "results/out.json", rel)
        # Output colliding with an input is rejected even for an empty scope.
        with self.assertRaises(ValueError):
            preview_retry(self.root, self.jobs, "notes.txt", rel)

    def test_cli_retry_preview_flag(self):
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": self.jobs}))
        report = self.root / "history.json"
        report.write_text(json.dumps({"results": [
            {"name": "mid", "status": "failed", "error": "x"},
            {"name": "base", "status": "completed"},
        ]}), encoding="utf-8")
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json",
                  "--root", str(self.root)]
        run = subprocess.run(prefix + ["--retry-preview", "history.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        payload = json.loads(run.stdout)
        self.assertEqual(payload["targets"], ["mid"])
        self.assertEqual([row["name"] for row in payload["jobs"]], ["base", "mid"])
        # No report or output directories created.
        self.assertFalse((self.root / ".results").exists())

        # report == output works on the CLI too.
        run = subprocess.run(prefix + ["--output", "history.json",
                                       "--retry-preview", "history.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["targets"], ["mid"])

        # Empty scope prints the empty object and still returns 0.
        report.write_text(json.dumps({"results": []}), encoding="utf-8")
        run = subprocess.run(prefix + ["--retry-preview", "history.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"targets": [], "jobs": []})

    def test_cli_retry_preview_errors_and_conflicts(self):
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": self.jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json",
                  "--root", str(self.root)]
        marker = '{"results": [{"name": "mid", "status": "failed"}]}'
        report = self.root / "history.json"
        report.write_text(marker, encoding="utf-8")

        def fails(extra):
            run = subprocess.run(prefix + extra, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)
            self.assertEqual(set(json.loads(run.stdout)), {"error"})

        # Conflicting options rejected before reading anything.
        fails(["--retry-preview", "history.json", "--preview"])
        fails(["--retry-preview", "history.json", "--only", "mid"])
        # Missing report, bad JSON, unknown name, escaping report path.
        fails(["--retry-preview", "nope.json"])
        (self.root / "broken.json").write_text("{oops", encoding="utf-8")
        fails(["--retry-preview", "broken.json"])
        (self.root / "unknown.json").write_text(
            json.dumps({"results": [{"name": "ghost", "status": "failed"}]}), encoding="utf-8")
        fails(["--retry-preview", "unknown.json"])
        fails(["--retry-preview", "../outside.json"])
        # Invalid whole plan and output/plan collision reject as well.
        bad = [{"name": "a", "operation": "count-lines", "input": "notes.txt",
                "depends_on": ["x"]}]
        plan.write_text(json.dumps({"jobs": bad}))
        fails(["--retry-preview", "history.json"])
        plan.write_text(json.dumps({"jobs": self.jobs}))
        fails(["--output", "plan.json", "--retry-preview", "history.json"])
        # Existing files are byte-for-byte unchanged by success or failure.
        self.assertEqual(report.read_text(), marker)
        self.assertFalse((self.root / ".results").exists())


if __name__ == "__main__":
    unittest.main()
