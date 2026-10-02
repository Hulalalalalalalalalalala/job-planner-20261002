import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from job_planner import (compare_reports, execute_job, local_path, preview_plan,
                         preview_retry, run_plan, run_retry)

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


    def _write_report(self, relative, payload, raw=None):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if raw is not None:
            path.write_bytes(raw)
        else:
            path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _retry_jobs(self):
        return [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "downstream", "operation": "count-lines", "input": "notes.txt", "depends_on": ["broken"]},
            {"name": "grandchild", "operation": "sha256", "input": "notes.txt", "depends_on": ["downstream"]},
            {"name": "joiner", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["broken", "healthy"]},
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
            {"name": "unrelated", "operation": "csv-summary", "input": "sales.csv"},
        ]

    def test_retry_targets_and_prerequisites_from_real_run(self):
        jobs = self._retry_jobs()
        run_plan(self.root, jobs, "results/report.json")
        preview = preview_retry(self.root, jobs, "never/created.json", "results/report.json")
        self.assertEqual(preview["targets"], ["broken", "downstream", "grandchild", "joiner"])
        self.assertEqual([row["name"] for row in preview["jobs"]],
                         ["broken", "downstream", "grandchild", "healthy", "joiner"])
        by_name = {row["name"]: row for row in preview["jobs"]}
        self.assertEqual(by_name["broken"],
                         {"name": "broken", "depends_on": [], "reason": "target",
                          "required_by": ["broken", "downstream", "grandchild", "joiner"]})
        # A completed shared prerequisite is still scheduled for the retry.
        self.assertEqual(by_name["healthy"],
                         {"name": "healthy", "depends_on": [], "reason": "prerequisite",
                          "required_by": ["joiner"]})
        # The successful, unrelated branch never enters the preview.
        self.assertNotIn("unrelated", by_name)
        # The output path's directory was not created.
        self.assertFalse((self.root / "never").exists())

    def test_retry_record_order_does_not_change_targets_order(self):
        jobs = self._retry_jobs()
        self._write_report("r.json", {"results": [
            {"name": "joiner", "status": "blocked", "blocked_by": ["broken"]},
            {"name": "unrelated", "status": "completed"},
            {"name": "grandchild", "status": "blocked", "blocked_by": ["downstream"]},
            {"name": "healthy", "status": "completed"},
            {"name": "downstream", "status": "blocked", "blocked_by": ["broken"]},
            {"name": "broken", "status": "failed", "error": "bad"},
        ]})
        preview = preview_retry(self.root, jobs, "deep/out.json", "r.json")
        self.assertEqual(preview["targets"], ["broken", "downstream", "grandchild", "joiner"])
        self.assertEqual([row["name"] for row in preview["jobs"]],
                         ["broken", "downstream", "grandchild", "healthy", "joiner"])
        self.assertFalse((self.root / "deep").exists())

    def test_retry_partial_report_includes_only_required_unrecorded(self):
        jobs = self._retry_jobs()
        # A localised run recorded only mid-chain: 'broken' and 'healthy'
        # are unrecorded but become prerequisites; 'unrelated' does not.
        self._write_report("partial.json", {"results": [
            {"name": "grandchild", "status": "blocked"},
            {"name": "downstream", "status": "failed"},
        ]})
        preview = preview_retry(self.root, jobs, "out.json", "partial.json")
        self.assertEqual(preview["targets"], ["downstream", "grandchild"])
        self.assertEqual([row["name"] for row in preview["jobs"]],
                         ["broken", "downstream", "grandchild"])
        # joiner needs healthy too; with joiner recorded failed elsewhere...
        self._write_report("partial2.json", {"results": [
            {"name": "joiner", "status": "blocked"},
            {"name": "downstream", "status": "failed"},
        ]})
        preview = preview_retry(self.root, jobs, "out.json", "partial2.json")
        self.assertEqual(preview["targets"], ["downstream", "joiner"])
        self.assertEqual([row["name"] for row in preview["jobs"]],
                         ["broken", "downstream", "healthy", "joiner"])

    def test_retry_empty_or_all_completed(self):
        jobs = [
            {"name": "notes", "operation": "count-lines", "input": "notes.txt"},
            {"name": "sales", "operation": "csv-summary", "input": "sales.csv"},
        ]
        self._write_report("empty.json", {"results": []})
        self.assertEqual(preview_retry(self.root, jobs, "out.json", "empty.json"),
                         {"targets": [], "jobs": []})
        self._write_report("done.json", {"results": [
            {"name": "sales", "status": "completed", "result": {"columns": ["a"], "rows": 1}},
            {"name": "notes", "status": "completed", "result": {"lines": 2}},
        ]})
        self.assertEqual(preview_retry(self.root, jobs, "out.json", "done.json"),
                         {"targets": [], "jobs": []})
        self.assertFalse((self.root / "out.json").exists())

    def test_retry_ignores_unknown_fields_and_execution_outcomes(self):
        jobs = [
            {"name": "missing", "operation": "count-lines", "input": "nope.txt"},
            {"name": "bad-op", "operation": "shell", "input": "notes.txt"},
            {"name": "ok", "operation": "sha256", "input": "notes.txt", "depends_on": ["missing"]},
        ]
        self._write_report("r.json", {"results": [
            {"name": "missing", "status": "failed", "error": "no file", "severity": 5},
            {"name": "bad-op", "status": "completed", "result": {}, "trace": ["x"]},
        ]})
        preview = preview_retry(self.root, jobs, "out.json", "r.json")
        self.assertEqual(preview["targets"], ["missing"])
        self.assertEqual([row["name"] for row in preview["jobs"]], ["missing"])
        # A blocked target still pulls the closure; the preview promises nothing.
        self._write_report("r2.json", {"results": [{"name": "ok", "status": "blocked"}]})
        preview = preview_retry(self.root, jobs, "out.json", "r2.json")
        self.assertEqual(preview["targets"], ["ok"])
        self.assertEqual([row["name"] for row in preview["jobs"]], ["missing", "ok"])

    def test_retry_invalid_reports_raise_value_error(self):
        jobs = self._retry_jobs()
        run_plan(self.root, jobs, "results/report.json")
        raw_cases = {
            "bad-encoding": b'{"results": []}\xff',
            "bad-json": b"{not json",
            "not-object": b"[1, 2]",
            "no-results": b"{}",
            "results-not-list": b'{"results": {}}',
            "entry-not-object": b'{"results": [1]}',
            "no-name": b'{"results": [{"status": "completed"}]}',
            "unknown-name": b'{"results": [{"name": "ghost", "status": "completed"}]}',
            "duplicate-name": b'{"results": [{"name": "healthy", "status": "completed"},'
                              b' {"name": "healthy", "status": "failed"}]}',
            "bad-status": b'{"results": [{"name": "healthy", "status": "done"}]}',
            "null-status": b'{"results": [{"name": "healthy", "status": null}]}',
        }
        for label, raw in raw_cases.items():
            with self.subTest(label=label):
                self._write_report(f"{label}.json", None, raw=raw)
                with self.assertRaises(ValueError):
                    preview_retry(self.root, jobs, "out.json", f"{label}.json")
        with self.assertRaises(ValueError):
            preview_retry(self.root, jobs, "out.json", "missing.json")
        with self.assertRaises(ValueError):
            preview_retry(self.root, jobs, "out.json", str(ROOT / "README.md"))
        with self.assertRaises(ValueError):
            preview_retry(self.root, jobs, "out.json", "../outside.json")
        # Existing files are untouched after any failure.
        marker = self._write_report("keep.json", {"results": []})
        with self.assertRaises(ValueError):
            preview_retry(self.root, jobs, "keep.json", "missing.json")
        self.assertEqual(marker.read_text(), json.dumps({"results": []}))

    def test_retry_symlink_report_boundary(self):
        jobs = [{"name": "notes", "operation": "count-lines", "input": "notes.txt"}]
        (self.root / "sub").mkdir()
        (self.root / "sub/real.json").write_text(
            json.dumps({"results": [{"name": "notes", "status": "failed"}]}), encoding="utf-8")
        # A symlink staying inside root is followed.
        (self.root / "in-link.json").symlink_to(self.root / "sub/real.json")
        preview = preview_retry(self.root, jobs, "out.json", "in-link.json")
        self.assertEqual(preview["targets"], ["notes"])
        # A symlink whose target leaves root is rejected.
        (self.root / "out-link.json").symlink_to(ROOT / "README.md")
        with self.assertRaises(ValueError):
            preview_retry(self.root, jobs, "out.json", "out-link.json")

    def test_retry_validates_whole_plan_and_output_even_when_empty(self):
        # Even {"targets": [], "jobs": []} requires a fully valid plan and output.
        self._write_report("empty.json", {"results": []})
        bad_input = [
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
            {"name": "stray", "operation": "count-lines", "input": "../outside.txt"},
        ]
        with self.assertRaises(ValueError):
            preview_retry(self.root, bad_input, "out.json", "empty.json")
        bad_dep = [
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
            {"name": "stray", "operation": "count-lines", "input": "notes.txt", "depends_on": ["ghost"]},
        ]
        with self.assertRaises(ValueError):
            preview_retry(self.root, bad_dep, "out.json", "empty.json")
        good = [{"name": "notes", "operation": "count-lines", "input": "notes.txt"}]
        with self.assertRaises(ValueError):
            preview_retry(self.root, good, "notes.txt", "empty.json")

    def test_cli_retry_preview_flag(self):
        jobs = self._retry_jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json", "--root", str(self.root)]
        # Full run writes the report (exit 1) and retry-preview reads it back.
        run = subprocess.run(prefix + ["--output", "old.json"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 1)
        run = subprocess.run(prefix + ["--output", "old.json", "--retry-preview", "old.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        payload = json.loads(run.stdout)
        self.assertEqual(payload["targets"], ["broken", "downstream", "grandchild", "joiner"])
        self.assertEqual([row["name"] for row in payload["jobs"]],
                         ["broken", "downstream", "grandchild", "healthy", "joiner"])
        self.assertTrue(all(set(row) == {"name", "depends_on", "reason", "required_by"}
                            for row in payload["jobs"]))
        # A successful-only partial run previews an empty retry range, exit 0.
        run = subprocess.run(prefix + ["--output", "old.json", "--only", "healthy"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0)
        run = subprocess.run(prefix + ["--retry-preview", "old.json"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"targets": [], "jobs": []})
        # The default output directory is never created by a preview.
        self.assertFalse((self.root / ".results").exists())

    def test_cli_retry_preview_conflicts_and_errors(self):
        jobs = self._retry_jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json", "--root", str(self.root)]
        report = self.root / "old.json"
        report.write_text("KEEP", encoding="utf-8")
        for extra in (["--retry-preview", "old.json", "--preview"],
                      ["--retry-preview", "old.json", "--only", "healthy"],
                      ["--retry-preview", "missing.json"],
                      ["--retry-preview", "old.json"],
                      ["--retry-preview", "../outside.json"]):
            run = subprocess.run(prefix + extra, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, (extra, run.stderr))
            self.assertEqual(set(json.loads(run.stdout)), {"error"})
        self.assertEqual(report.read_text(), "KEEP")
        # An invalid plan is rejected before the report is opened.
        bad = [{"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["x"]}]
        plan.write_text(json.dumps({"jobs": bad}))
        run = subprocess.run(prefix + ["--retry-preview", "old.json"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertEqual(set(json.loads(run.stdout)), {"error"})
        self.assertEqual(report.read_text(), "KEEP")

    def test_run_retry_targets_prerequisites_and_records_this_run_only(self):
        jobs = self._retry_jobs()
        run_plan(self.root, jobs, "results/report.json")
        result = run_retry(self.root, jobs, "results/retry.json", "results/report.json")
        self.assertEqual([row["name"] for row in result],
                         ["broken", "downstream", "grandchild", "healthy", "joiner"])
        by_name = {row["name"]: row for row in result}
        self.assertEqual(by_name["broken"]["status"], "failed")
        self.assertEqual(by_name["downstream"],
                         {"name": "downstream", "status": "blocked", "blocked_by": ["broken"]})
        self.assertEqual(by_name["grandchild"],
                         {"name": "grandchild", "status": "blocked", "blocked_by": ["downstream"]})
        self.assertEqual(by_name["joiner"],
                         {"name": "joiner", "status": "blocked", "blocked_by": ["broken"]})
        # The completed shared prerequisite is executed again, exactly once.
        self.assertEqual(by_name["healthy"]["status"], "completed")
        self.assertIn("result", by_name["healthy"])
        # The unrelated completed branch is not read, not recorded.
        self.assertNotIn("unrelated", by_name)
        # The new report holds only this run's results; the old one is intact.
        written = json.loads((self.root / "results/retry.json").read_text())["results"]
        self.assertEqual(written, result)
        old = json.loads((self.root / "results/report.json").read_text())["results"]
        self.assertEqual([row["name"] for row in old],
                         ["broken", "downstream", "grandchild", "healthy", "joiner", "unrelated"])

    def test_run_retry_partial_report_pulls_unrecorded_prerequisites(self):
        jobs = self._retry_jobs()
        self._write_report("partial.json", {"results": [
            {"name": "joiner", "status": "blocked"},
            {"name": "downstream", "status": "failed"},
        ]})
        result = run_retry(self.root, jobs, "out.json", "partial.json")
        # broken and healthy were never recorded, but they are required
        # prerequisites; unrelated is neither recorded nor required.
        self.assertEqual([row["name"] for row in result],
                         ["broken", "downstream", "healthy", "joiner"])
        self.assertEqual([row["status"] for row in result],
                         ["failed", "blocked", "completed", "blocked"])
        self.assertEqual(json.loads((self.root / "out.json").read_text())["results"], result)

    def test_run_retry_depends_only_on_this_run_results(self):
        jobs = [
            {"name": "base", "operation": "count-lines", "input": "base.txt"},
            {"name": "target", "operation": "sha256", "input": "notes.txt", "depends_on": ["base"]},
        ]
        (self.root / "base.txt").write_text("a\nb\n", encoding="utf-8")
        # Historically base completed and target failed; on this rerun base's
        # input is gone, so base fails and the target is blocked by it.
        self._write_report("r.json", {"results": [
            {"name": "base", "status": "completed", "result": {"lines": 2}},
            {"name": "target", "status": "failed", "error": "old"},
        ]})
        (self.root / "base.txt").unlink()
        result = run_retry(self.root, jobs, "out.json", "r.json")
        self.assertEqual([row["name"] for row in result], ["base", "target"])
        self.assertEqual(result[0]["status"], "failed")
        self.assertEqual(result[1],
                         {"name": "target", "status": "blocked", "blocked_by": ["base"]})
        # Conversely, a historically failed prerequisite that now succeeds
        # lets its blocked downstream complete.
        (self.root / "base.txt").write_text("a\nb\n", encoding="utf-8")
        self._write_report("r2.json", {"results": [
            {"name": "base", "status": "failed", "error": "old"},
            {"name": "target", "status": "blocked", "blocked_by": ["base"]},
        ]})
        result = run_retry(self.root, jobs, "out.json", "r2.json")
        self.assertEqual([row["status"] for row in result], ["completed", "completed"])

    def test_run_retry_independent_failed_branch_does_not_stop_other_target(self):
        jobs = [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
        ]
        self._write_report("r.json", {"results": [
            {"name": "broken", "status": "failed"},
            {"name": "healthy", "status": "failed", "error": "transient"},
        ]})
        result = run_retry(self.root, jobs, "out.json", "r.json")
        self.assertEqual([row["name"] for row in result], ["broken", "healthy"])
        self.assertEqual([row["status"] for row in result], ["failed", "completed"])

    def test_run_retry_no_targets_reads_nothing_and_touches_no_files(self):
        jobs = [{"name": "ghost", "operation": "count-lines", "input": "nope.txt"}]
        for payload in ({"results": []},
                        {"results": [{"name": "ghost", "status": "completed",
                                      "result": {"lines": 1}}]}):
            self._write_report("r.json", payload)
            before = {p.name for p in self.root.iterdir()}
            self.assertEqual(run_retry(self.root, jobs, "deep/new/out.json", "r.json"), [])
            self.assertFalse((self.root / "deep").exists())
            self.assertEqual({p.name for p in self.root.iterdir()}, before)
        # An existing output file is left exactly as it was.
        marker = self._write_report("keep.json", {"results": [{"name": "kept"}]})
        before = marker.read_text()
        self.assertEqual(run_retry(self.root, jobs, "keep.json", "r.json"), [])
        self.assertEqual(marker.read_text(), before)

    def test_run_retry_report_may_equal_output(self):
        jobs = self._retry_jobs()
        run_plan(self.root, jobs, "same.json")
        result = run_retry(self.root, jobs, "same.json", "same.json")
        # Targets came from the pre-overwrite content; the file now holds
        # only this run's records.
        self.assertEqual([row["name"] for row in result],
                         ["broken", "downstream", "grandchild", "healthy", "joiner"])
        self.assertEqual(json.loads((self.root / "same.json").read_text())["results"], result)

    def test_run_retry_invalid_reports_raise_and_preserve_output(self):
        jobs = self._retry_jobs()
        run_plan(self.root, jobs, "results/report.json")
        marker = self._write_report("keep.json", {"results": []})
        raw_cases = {
            "bad-json": b"{not json",
            "unknown-name": b'{"results": [{"name": "ghost", "status": "completed"}]}',
            "bad-status": b'{"results": [{"name": "healthy", "status": "done"}]}',
        }
        for label, raw in raw_cases.items():
            with self.subTest(label=label):
                self._write_report(f"{label}.json", None, raw=raw)
                with self.assertRaises(ValueError):
                    run_retry(self.root, jobs, "keep.json", f"{label}.json")
        for bad_report in ("missing.json", "../outside.json", str(ROOT / "README.md")):
            with self.subTest(bad_report=bad_report):
                with self.assertRaises(ValueError):
                    run_retry(self.root, jobs, "keep.json", bad_report)
        self.assertEqual(marker.read_text(), json.dumps({"results": []}))

    def test_run_retry_validates_whole_plan_even_with_empty_scope(self):
        self._write_report("empty.json", {"results": []})
        bad_input = [
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
            {"name": "stray", "operation": "count-lines", "input": "../outside.txt"},
        ]
        with self.assertRaises(ValueError):
            run_retry(self.root, bad_input, "deep/out.json", "empty.json")
        bad_dep = [
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
            {"name": "stray", "operation": "count-lines", "input": "notes.txt", "depends_on": ["ghost"]},
        ]
        with self.assertRaises(ValueError):
            run_retry(self.root, bad_dep, "deep/out.json", "empty.json")
        good = [{"name": "notes", "operation": "count-lines", "input": "notes.txt"}]
        with self.assertRaises(ValueError):
            run_retry(self.root, good, "notes.txt", "empty.json")
        self.assertFalse((self.root / "deep").exists())

    def test_run_retry_output_write_failure_raises_oserror_after_running(self):
        jobs = [{"name": "notes", "operation": "count-lines", "input": "notes.txt"}]
        self._write_report("r.json", {"results": [{"name": "notes", "status": "failed"}]})
        (self.root / "blocker").write_text("x", encoding="utf-8")
        with self.assertRaises(OSError):
            run_retry(self.root, jobs, "blocker/out.json", "r.json")

    def test_cli_retry_flag_counts_scope_and_exit_code(self):
        jobs = self._retry_jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json", "--root", str(self.root)]
        run = subprocess.run(prefix + ["--output", "old.json"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 1)
        run = subprocess.run(prefix + ["--retry", "old.json", "--output", "new.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertEqual(json.loads(run.stdout),
                         {"completed": 1, "failed": 1, "blocked": 3})
        new = json.loads((self.root / "new.json").read_text())["results"]
        self.assertEqual([row["name"] for row in new],
                         ["broken", "downstream", "grandchild", "healthy", "joiner"])
        old = json.loads((self.root / "old.json").read_text())["results"]
        self.assertEqual(len(old), 6)
        # A historically failed dep-free job that now succeeds: no blocked
        # field appears when the scope has no dependencies, exit 0.
        self._write_report("one.json", {"results": [
            {"name": "healthy", "status": "failed", "error": "transient"}]})
        run = subprocess.run(prefix + ["--retry", "one.json", "--output", "one-out.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"completed": 1, "failed": 0})

    def test_cli_retry_empty_scope_outputs_zeroes_and_runs_nothing(self):
        jobs = self._retry_jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json", "--root", str(self.root)]
        self._write_report("empty.json", {"results": []})
        run = subprocess.run(prefix + ["--retry", "empty.json"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"completed": 0, "failed": 0})
        self.assertFalse((self.root / ".results").exists())
        # A report can name only completed tasks even though other plan
        # inputs are missing: without targets nothing is read.
        self._write_report("done.json", {"results": [
            {"name": "healthy", "status": "completed", "result": {"lines": 2}}]})
        run = subprocess.run(prefix + ["--retry", "done.json"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"completed": 0, "failed": 0})

    def test_cli_retry_conflicts_and_errors(self):
        jobs = self._retry_jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json", "--root", str(self.root)]
        report = self.root / "old.json"
        report.write_text("KEEP", encoding="utf-8")
        for extra in (["--retry", "old.json", "--only", "healthy"],
                      ["--retry", "old.json", "--preview"],
                      ["--retry", "old.json", "--retry-preview", "old.json"],
                      ["--retry", "missing.json"],
                      ["--retry", "../outside.json"]):
            run = subprocess.run(prefix + extra, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, (extra, run.stderr))
            self.assertEqual(set(json.loads(run.stdout)), {"error"})
        self.assertEqual(report.read_text(), "KEEP")
        bad = [{"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["x"]}]
        plan.write_text(json.dumps({"jobs": bad}))
        run = subprocess.run(prefix + ["--retry", "old.json"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertEqual(set(json.loads(run.stdout)), {"error"})
        self.assertEqual(report.read_text(), "KEEP")

    def test_compare_reports_added_removed_changed_unchanged(self):
        jobs = self._retry_jobs()
        self._write_report("before.json", {"results": [
            {"name": "broken", "status": "failed", "error": "old"},
            {"name": "downstream", "status": "blocked", "blocked_by": ["broken"]},
            {"name": "healthy", "status": "completed", "result": {"lines": 2}},
            {"name": "unrelated", "status": "completed",
             "result": {"columns": ["item", "count"], "rows": 2}},
        ]})
        self._write_report("after.json", {"results": [
            {"name": "broken", "status": "completed", "result": {"lines": 2}},
            {"name": "downstream", "status": "blocked", "blocked_by": ["broken"]},
            {"name": "healthy", "status": "completed", "result": {"lines": 2}},
            {"name": "grandchild", "status": "failed", "error": "new"},
        ]})
        compared = compare_reports(self.root, jobs, "out.json", "before.json", "after.json")
        # Plan order, tasks recorded in at least one report; joiner is in
        # neither report and never appears.
        self.assertEqual([row["name"] for row in compared["jobs"]],
                         ["broken", "downstream", "grandchild", "healthy", "unrelated"])
        self.assertEqual([row["change"] for row in compared["jobs"]],
                         ["changed", "unchanged", "added", "unchanged", "removed"])
        by_name = {row["name"]: row for row in compared["jobs"]}
        self.assertEqual(by_name["broken"],
                         {"name": "broken",
                          "before": {"status": "failed", "error": "old"},
                          "after": {"status": "completed", "result": {"lines": 2}},
                          "change": "changed"})
        self.assertEqual(by_name["grandchild"],
                         {"name": "grandchild",
                          "before": None,
                          "after": {"status": "failed", "error": "new"},
                          "change": "added"})
        self.assertEqual(by_name["unrelated"]["after"], None)
        self.assertEqual(by_name["unrelated"]["before"],
                         {"status": "completed", "result": {"columns": ["item", "count"], "rows": 2}})
        # Each entry and side carries exactly the documented fields.
        for row in compared["jobs"]:
            self.assertEqual(set(row), {"name", "before", "after", "change"})
            for side in (row["before"], row["after"]):
                if side is not None:
                    self.assertEqual(len(side), 2)
                    self.assertIn("status", side)

    def test_compare_reports_same_file_and_ignores_order_and_extra_fields(self):
        jobs = self._retry_jobs()
        self._write_report("same.json", {"results": [
            {"name": "healthy", "status": "completed", "result": {"lines": 2}, "trace": [1]},
            {"name": "broken", "status": "failed", "error": "x", "severity": 3},
        ]})
        compared = compare_reports(self.root, jobs, "out.json", "same.json", "same.json")
        self.assertEqual([row["name"] for row in compared["jobs"]], ["broken", "healthy"])
        self.assertEqual([row["change"] for row in compared["jobs"]],
                         ["unchanged", "unchanged"])
        for row in compared["jobs"]:
            self.assertEqual(row["before"], row["after"])
        # Extra fields are dropped from the output sides.
        self.assertEqual(compared["jobs"][1]["before"],
                         {"status": "completed", "result": {"lines": 2}})
        # Record order inside the reports is irrelevant.
        self._write_report("reordered.json", {"results": [
            {"name": "broken", "status": "failed", "error": "x"},
            {"name": "healthy", "status": "completed", "result": {"lines": 2}},
        ]})
        compared = compare_reports(self.root, jobs, "out.json", "same.json", "reordered.json")
        self.assertEqual([row["change"] for row in compared["jobs"]],
                         ["unchanged", "unchanged"])

    def test_compare_reports_result_json_value_semantics(self):
        jobs = [{"name": "notes", "operation": "count-lines", "input": "notes.txt"}]

        def compare(before_result, after_result):
            self._write_report("b.json", {"results": [
                {"name": "notes", "status": "completed", "result": before_result}]})
            self._write_report("a.json", {"results": [
                {"name": "notes", "status": "completed", "result": after_result}]})
            return compare_reports(self.root, jobs, "out.json", "b.json", "a.json")["jobs"][0]

        # Object key order does not matter.
        self.assertEqual(compare({"a": 1, "b": [1, 2]}, {"b": [1, 2], "a": 1})["change"],
                         "unchanged")
        # Array order matters.
        self.assertEqual(compare({"a": [1, 2]}, {"a": [2, 1]})["change"], "changed")
        # Booleans are distinct from numbers, also nested.
        self.assertEqual(compare({"ok": True}, {"ok": 1})["change"], "changed")
        self.assertEqual(compare({"ok": [False]}, {"ok": [0]})["change"], "changed")
        self.assertEqual(compare({"n": 1}, {"n": True})["change"], "changed")
        # Numbers compare numerically; nested structures recurse.
        self.assertEqual(compare({"n": 1}, {"n": 1.0})["change"], "unchanged")
        self.assertEqual(compare({"n": {"x": [1, {"y": 2}]}},
                                 {"n": {"x": [1, {"y": 2}]}})["change"], "unchanged")
        self.assertEqual(compare({"n": 1}, {"n": 2})["change"], "changed")
        self.assertEqual(compare({"a": 1}, {"a": 1, "b": 2})["change"], "changed")

    def test_compare_reports_blocked_by_set_and_declaration_order(self):
        jobs = [
            {"name": "a", "operation": "count-lines", "input": "notes.txt"},
            {"name": "b", "operation": "count-lines", "input": "notes.txt"},
            {"name": "join", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["a", "b"]},
        ]
        self._write_report("b1.json", {"results": [
            {"name": "join", "status": "blocked", "blocked_by": ["b", "a"]}]})
        self._write_report("b2.json", {"results": [
            {"name": "join", "status": "blocked", "blocked_by": ["a", "b"]}]})
        compared = compare_reports(self.root, jobs, "out.json", "b1.json", "b2.json")
        # Compared as a set; emitted in the current depends_on order.
        self.assertEqual(compared["jobs"][0]["change"], "unchanged")
        self.assertEqual(compared["jobs"][0]["before"],
                         {"status": "blocked", "blocked_by": ["a", "b"]})
        self.assertEqual(compared["jobs"][0]["after"],
                         {"status": "blocked", "blocked_by": ["a", "b"]})
        self._write_report("b3.json", {"results": [
            {"name": "join", "status": "blocked", "blocked_by": ["a"]}]})
        compared = compare_reports(self.root, jobs, "out.json", "b1.json", "b3.json")
        self.assertEqual(compared["jobs"][0]["change"], "changed")

    def test_compare_reports_failure_message_change_counts(self):
        jobs = [{"name": "notes", "operation": "count-lines", "input": "notes.txt"}]
        self._write_report("b.json", {"results": [
            {"name": "notes", "status": "failed", "error": "first"}]})
        self._write_report("a.json", {"results": [
            {"name": "notes", "status": "failed", "error": "second"}]})
        compared = compare_reports(self.root, jobs, "out.json", "b.json", "a.json")
        self.assertEqual(compared["jobs"][0]["change"], "changed")
        self._write_report("a.json", {"results": [
            {"name": "notes", "status": "failed", "error": "first"}]})
        compared = compare_reports(self.root, jobs, "out.json", "b.json", "a.json")
        self.assertEqual(compared["jobs"][0]["change"], "unchanged")

    def test_compare_reports_is_read_only(self):
        jobs = [
            {"name": "missing", "operation": "count-lines", "input": "nope.txt"},
            {"name": "ok", "operation": "sha256", "input": "notes.txt"},
        ]
        marker = self._write_report("r.json", {"results": [
            {"name": "ok", "status": "completed", "result": {"sha256": "x", "bytes": 1}}]})
        before = {p.name for p in self.root.iterdir()}
        compared = compare_reports(self.root, jobs, "deep/new/out.json", "r.json", "r.json")
        self.assertEqual([row["name"] for row in compared["jobs"]], ["ok"])
        self.assertFalse((self.root / "deep").exists())
        self.assertEqual({p.name for p in self.root.iterdir()}, before)
        self.assertEqual(marker.read_text(), json.dumps({"results": [
            {"name": "ok", "status": "completed", "result": {"sha256": "x", "bytes": 1}}]}))

    def test_compare_reports_validates_whole_plan_and_output(self):
        self._write_report("empty.json", {"results": []})
        bad_input = [
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
            {"name": "stray", "operation": "count-lines", "input": "../outside.txt"},
        ]
        with self.assertRaises(ValueError):
            compare_reports(self.root, bad_input, "out.json", "empty.json", "empty.json")
        bad_dep = [
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
            {"name": "stray", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["ghost"]},
        ]
        with self.assertRaises(ValueError):
            compare_reports(self.root, bad_dep, "out.json", "empty.json", "empty.json")
        good = [{"name": "notes", "operation": "count-lines", "input": "notes.txt"}]
        with self.assertRaises(ValueError):
            compare_reports(self.root, good, "notes.txt", "empty.json", "empty.json")

    def test_compare_reports_invalid_reports_raise_value_error(self):
        jobs = self._retry_jobs()
        self._write_report("ok.json", {"results": []})
        raw_cases = {
            "bad-encoding": b'{"results": []}\xff',
            "bad-json": b"{not json",
            "not-object": b"[1, 2]",
            "no-results": b"{}",
            "results-not-list": b'{"results": {}}',
            "entry-not-object": b'{"results": [1]}',
            "unknown-name": b'{"results": [{"name": "ghost", "status": "completed"}]}',
            "duplicate-name": b'{"results": [{"name": "healthy", "status": "failed", "error": "a"},'
                              b' {"name": "healthy", "status": "failed", "error": "b"}]}',
            "bad-status": b'{"results": [{"name": "healthy", "status": "done"}]}',
            # The status-matching field must exist with the right shape.
            "completed-no-result": b'{"results": [{"name": "healthy", "status": "completed"}]}',
            "completed-result-list": b'{"results": [{"name": "healthy", "status": "completed",'
                                     b' "result": [1]}]}',
            "completed-result-bool": b'{"results": [{"name": "healthy", "status": "completed",'
                                     b' "result": true}]}',
            "failed-no-error": b'{"results": [{"name": "healthy", "status": "failed"}]}',
            "failed-error-not-string": b'{"results": [{"name": "healthy", "status": "failed",'
                                       b' "error": 3}]}',
            "blocked-no-blocked-by": b'{"results": [{"name": "joiner", "status": "blocked"}]}',
            "blocked-empty": b'{"results": [{"name": "joiner", "status": "blocked",'
                             b' "blocked_by": []}]}',
            "blocked-duplicate": b'{"results": [{"name": "joiner", "status": "blocked",'
                                 b' "blocked_by": ["broken", "broken"]}]}',
            "blocked-non-string": b'{"results": [{"name": "joiner", "status": "blocked",'
                                  b' "blocked_by": [1]}]}',
            "blocked-not-list": b'{"results": [{"name": "joiner", "status": "blocked",'
                                b' "blocked_by": "broken"}]}',
            "blocked-non-dependency": b'{"results": [{"name": "joiner", "status": "blocked",'
                                      b' "blocked_by": ["unrelated"]}]}',
        }
        for label, raw in raw_cases.items():
            with self.subTest(label=label):
                self._write_report(f"{label}.json", None, raw=raw)
                with self.assertRaises(ValueError):
                    compare_reports(self.root, jobs, "out.json", f"{label}.json", "ok.json")
                with self.assertRaises(ValueError):
                    compare_reports(self.root, jobs, "out.json", "ok.json", f"{label}.json")
        for bad_report in ("missing.json", "../outside.json", str(ROOT / "README.md")):
            with self.subTest(bad_report=bad_report):
                with self.assertRaises(ValueError):
                    compare_reports(self.root, jobs, "out.json", bad_report, "ok.json")
        # A symlink whose target leaves root is rejected.
        (self.root / "out-link.json").symlink_to(ROOT / "README.md")
        with self.assertRaises(ValueError):
            compare_reports(self.root, jobs, "out.json", "ok.json", "out-link.json")

    def test_cli_compare_flag(self):
        jobs = self._retry_jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json", "--root", str(self.root)]
        # A real run produces the before report; an edited copy is the after.
        run = subprocess.run(prefix + ["--output", "before.json"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 1)
        before = json.loads((self.root / "before.json").read_text())["results"]
        after = [row for row in before if row["name"] != "unrelated"]
        for row in after:
            if row["name"] == "broken":
                row["status"] = "completed"
                row["result"] = {"lines": 2}
                del row["error"]
        self._write_report("after.json", {"results": after})
        run = subprocess.run(prefix + ["--compare", "before.json", "after.json"],
                             capture_output=True, text=True)
        # A valid comparison exits 0 even though records contain failures.
        self.assertEqual(run.returncode, 0, run.stderr)
        payload = json.loads(run.stdout)
        self.assertEqual([row["name"] for row in payload["jobs"]],
                         ["broken", "downstream", "grandchild", "joiner", "healthy", "unrelated"])
        by_name = {row["name"]: row for row in payload["jobs"]}
        self.assertEqual(by_name["broken"]["change"], "changed")
        self.assertEqual(by_name["unrelated"],
                         {"name": "unrelated",
                          "before": {"status": "completed",
                                     "result": {"columns": ["item", "count"], "rows": 2}},
                          "after": None, "change": "removed"})
        # Nothing was written or created by the comparison.
        self.assertFalse((self.root / ".results").exists())

    def test_cli_compare_conflicts_and_errors(self):
        jobs = self._retry_jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json", "--root", str(self.root)]
        self._write_report("ok.json", {"results": []})
        report = self.root / "old.json"
        report.write_text("KEEP", encoding="utf-8")
        for extra in (["--compare", "ok.json", "ok.json", "--preview"],
                      ["--compare", "ok.json", "ok.json", "--only", "healthy"],
                      ["--compare", "ok.json", "ok.json", "--retry-preview", "ok.json"],
                      ["--compare", "ok.json", "ok.json", "--retry", "ok.json"],
                      ["--compare", "missing.json", "ok.json"],
                      ["--compare", "ok.json", "../outside.json"],
                      ["--compare", "old.json", "ok.json"]):
            run = subprocess.run(prefix + extra, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, (extra, run.stderr))
            self.assertEqual(set(json.loads(run.stdout)), {"error"})
        self.assertEqual(report.read_text(), "KEEP")
        # The plan file is protected in compare mode as well.
        run = subprocess.run(prefix + ["--compare", "ok.json", "ok.json",
                                       "--output", "plan.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertIn("error", json.loads(run.stdout))
        # An invalid plan is rejected before any report is opened.
        bad = [{"name": "a", "operation": "count-lines", "input": "notes.txt",
                "depends_on": ["x"]}]
        plan.write_text(json.dumps({"jobs": bad}))
        run = subprocess.run(prefix + ["--compare", "ok.json", "ok.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertEqual(set(json.loads(run.stdout)), {"error"})

    def test_cli_retry_report_equals_output(self):
        jobs = self._retry_jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json", "--root", str(self.root)]
        run = subprocess.run(prefix + ["--output", "same.json"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 1)
        run = subprocess.run(prefix + ["--retry", "same.json", "--output", "same.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertEqual(json.loads(run.stdout),
                         {"completed": 1, "failed": 1, "blocked": 3})
        payload = json.loads((self.root / "same.json").read_text())["results"]
        self.assertEqual([row["name"] for row in payload],
                         ["broken", "downstream", "grandchild", "healthy", "joiner"])


if __name__ == "__main__":
    unittest.main()
