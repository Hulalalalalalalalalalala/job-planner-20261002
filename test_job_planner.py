import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from job_planner import (compare_reports, execute_job, explain_report, local_path,
                         preview_plan, preview_retry, query_history, run_plan,
                         run_retry)

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


    def _compare_jobs(self):
        return [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "downstream", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["broken"]},
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
            {"name": "newjob", "operation": "count-lines", "input": "notes.txt"},
            {"name": "gone", "operation": "count-lines", "input": "notes.txt"},
            {"name": "stable", "operation": "count-lines", "input": "notes.txt"},
        ]

    def test_compare_statuses_changes_order_and_entry_shape(self):
        jobs = self._compare_jobs()
        self._write_report("before.json", {"results": [
            {"name": "stable", "status": "completed", "result": {"lines": 2}},
            {"name": "gone", "status": "completed", "result": {"lines": 2}},
            {"name": "healthy", "status": "completed", "result": {"lines": 2}},
            {"name": "downstream", "status": "blocked", "blocked_by": ["broken"], "x": 9},
            {"name": "broken", "status": "failed", "error": "boom"},
        ]})
        self._write_report("after.json", {"results": [
            {"name": "newjob", "status": "failed", "error": "late"},
            {"name": "broken", "status": "completed", "result": {"lines": 2}},
            {"name": "downstream", "status": "blocked", "blocked_by": ["broken"]},
            {"name": "healthy", "status": "completed", "result": {"lines": 3}},
            {"name": "stable", "status": "completed", "result": {"lines": 2}},
        ]})
        result = compare_reports(self.root, jobs, "never/created.json",
                                 "before.json", "after.json")
        self.assertEqual([row["name"] for row in result["jobs"]],
                         ["broken", "downstream", "healthy", "newjob", "gone", "stable"])
        self.assertTrue(all(set(row) == {"name", "before", "after", "change"}
                            for row in result["jobs"]))
        by_name = {row["name"]: row for row in result["jobs"]}
        self.assertEqual(by_name["broken"]["change"], "changed")
        self.assertEqual(by_name["broken"]["before"],
                         {"status": "failed", "error": "boom"})
        self.assertEqual(by_name["broken"]["after"],
                         {"status": "completed", "result": {"lines": 2}})
        self.assertEqual(by_name["downstream"],
                         {"name": "downstream",
                          "before": {"status": "blocked", "blocked_by": ["broken"]},
                          "after": {"status": "blocked", "blocked_by": ["broken"]},
                          "change": "unchanged"})
        self.assertEqual(by_name["healthy"]["change"], "changed")
        self.assertEqual(by_name["newjob"]["before"], None)
        self.assertEqual(by_name["newjob"]["after"],
                         {"status": "failed", "error": "late"})
        self.assertEqual(by_name["newjob"]["change"], "added")
        self.assertEqual(by_name["gone"]["after"], None)
        self.assertEqual(by_name["gone"]["before"],
                         {"status": "completed", "result": {"lines": 2}})
        self.assertEqual(by_name["gone"]["change"], "removed")
        self.assertEqual(by_name["stable"]["change"], "unchanged")
        # A task recorded on neither side never appears.
        jobs2 = jobs + [{"name": "absent", "operation": "shell", "input": "missing.txt"}]
        names = {row["name"] for row in
                 compare_reports(self.root, jobs2, "never/created.json",
                                 "before.json", "after.json")["jobs"]}
        self.assertNotIn("absent", names)
        # The output directory is not created and inputs need not exist.
        self.assertFalse((self.root / "never").exists())

    def test_compare_same_file_and_record_order_and_extra_fields(self):
        jobs = self._compare_jobs()
        self._write_report("r.json", {"results": [
            {"name": "stable", "status": "completed", "result": {"lines": 2}, "trace": [1]},
            {"name": "broken", "status": "failed", "error": "boom", "severity": 5},
            {"name": "downstream", "status": "blocked", "blocked_by": ["broken"]},
        ]})
        result = compare_reports(self.root, jobs, "out.json", "r.json", "r.json")
        self.assertTrue(all(row["change"] == "unchanged" for row in result["jobs"]))
        self.assertEqual([row["name"] for row in result["jobs"]],
                         ["broken", "downstream", "stable"])
        self.assertTrue(all(set(row["before"]) == {"status", "error"}
                            for row in result["jobs"]
                            if row["name"] == "broken"))

    def test_compare_json_value_equality_semantics(self):
        jobs = [
            {"name": "boolnum", "operation": "count-lines", "input": "notes.txt"},
            {"name": "intfloat", "operation": "count-lines", "input": "notes.txt"},
            {"name": "keyorder", "operation": "count-lines", "input": "notes.txt"},
            {"name": "arrorder", "operation": "count-lines", "input": "notes.txt"},
            {"name": "errmsg", "operation": "count-lines", "input": "notes.txt"},
        ]
        self._write_report("b.json", {"results": [
            {"name": "boolnum", "status": "completed", "result": {"x": True}},
            {"name": "intfloat", "status": "completed", "result": {"x": 1}},
            {"name": "keyorder", "status": "completed", "result": {"a": 1, "nested": {"p": 1, "q": 2}}},
            {"name": "arrorder", "status": "completed", "result": {"xs": [1, 2]}},
            {"name": "errmsg", "status": "failed", "error": "old"},
        ]})
        self._write_report("a.json", {"results": [
            {"name": "boolnum", "status": "completed", "result": {"x": 1}},
            {"name": "intfloat", "status": "completed", "result": {"x": 1.0}},
            {"name": "keyorder", "status": "completed", "result": {"nested": {"q": 2, "p": 1}, "a": 1}},
            {"name": "arrorder", "status": "completed", "result": {"xs": [2, 1]}},
            {"name": "errmsg", "status": "failed", "error": "new"},
        ]})
        changes = {row["name"]: row["change"] for row in
                   compare_reports(self.root, jobs, "out.json", "b.json", "a.json")["jobs"]}
        self.assertEqual(changes, {"boolnum": "changed", "intfloat": "unchanged",
                                   "keyorder": "unchanged", "arrorder": "changed",
                                   "errmsg": "changed"})

    def test_compare_numbers_use_exact_decimal_value(self):
        from decimal import Decimal
        jobs = [
            {"name": "eqforms", "operation": "count-lines", "input": "notes.txt"},
            {"name": "longfrac", "operation": "count-lines", "input": "notes.txt"},
            {"name": "bigintegers", "operation": "count-lines", "input": "notes.txt"},
            {"name": "overflow", "operation": "count-lines", "input": "notes.txt"},
            {"name": "underflow", "operation": "count-lines", "input": "notes.txt"},
            {"name": "nested", "operation": "count-lines", "input": "notes.txt"},
        ]
        self._write_report("b.json", None, raw=(
            b'{"results":['
            b'{"name":"eqforms","status":"completed","result":{"v":1}},'
            b'{"name":"longfrac","status":"completed","result":{"v":0.10000000000000001}},'
            b'{"name":"bigintegers","status":"completed","result":{"v":9007199254740992.0}},'
            b'{"name":"overflow","status":"completed","result":{"v":1e400}},'
            b'{"name":"underflow","status":"completed","result":{"v":1e-400}},'
            b'{"name":"nested","status":"completed","result":{"a":[{"x":1.0}],"y":-0}}'
            b']}'))
        self._write_report("a.json", None, raw=(
            b'{"results":['
            b'{"name":"eqforms","status":"completed","result":{"v":1.0}},'
            b'{"name":"longfrac","status":"completed","result":{"v":0.1}},'
            b'{"name":"bigintegers","status":"completed","result":{"v":9007199254740993.0}},'
            b'{"name":"overflow","status":"completed","result":{"v":2e400}},'
            b'{"name":"underflow","status":"completed","result":{"v":0}},'
            b'{"name":"nested","status":"completed","result":{"a":[{"x":1e0}],"y":0}}'
            b']}'))
        changes = {row["name"]: row["change"] for row in
                   compare_reports(self.root, jobs, "out.json", "b.json", "a.json")["jobs"]}
        self.assertEqual(changes, {"eqforms": "unchanged", "longfrac": "changed",
                                   "bigintegers": "changed", "overflow": "changed",
                                   "underflow": "changed", "nested": "unchanged"})
        # 1e400 and 10e399 are the same exact decimal, while 1e0 spelling
        # variants and negative zero are all equal forms of one/zero.
        self._write_report("b2.json", None, raw=(
            b'{"results":[{"name":"overflow","status":"completed",'
            b'"result":{"v":1e400}},{"name":"eqforms","status":"completed",'
            b'"result":{"v":1e0}},{"name":"nested","status":"completed",'
            b'"result":{"a":[{"x":1.0}],"y":-0.0}}]}'))
        self._write_report("a2.json", None, raw=(
            b'{"results":[{"name":"overflow","status":"completed",'
            b'"result":{"v":10e399}},{"name":"eqforms","status":"completed",'
            b'"result":{"v":1}},{"name":"nested","status":"completed",'
            b'"result":{"a":[{"x":1}],"y":0}}]}'))
        result = compare_reports(self.root, jobs, "out.json", "b2.json", "a2.json")
        self.assertTrue(all(row["change"] == "unchanged" for row in result["jobs"]))
        # The returned sides keep the original numeric values and types:
        # integers stay int, other numbers arrive as Decimal.
        by_name = {row["name"]: row for row in result["jobs"]}
        self.assertEqual(by_name["overflow"]["before"]["result"]["v"],
                         Decimal("1E+400"))
        self.assertIsInstance(by_name["overflow"]["before"]["result"]["v"], Decimal)

    def test_compare_preserves_report_numbers_in_api_sides(self):
        from decimal import Decimal
        jobs = [{"name": "t", "operation": "count-lines", "input": "notes.txt"}]
        self._write_report("b.json", None, raw=(
            b'{"results":[{"name":"t","status":"completed","result":{'
            b'"i":9007199254740993,"f":2.5,"huge":1e400,"tiny":1e-400,'
            b'"negzero":-0,"deep":[1,1.0,1e0,{"z":0.0001}]}}]}'))
        self._write_report("a.json", {"results": [
            {"name": "t", "status": "failed", "error": "boom"}]})
        row = compare_reports(self.root, jobs, "out.json", "b.json", "a.json")["jobs"][0]
        result = row["before"]["result"]
        self.assertIsInstance(result["i"], int)
        self.assertEqual(result["i"], 9007199254740993)
        self.assertEqual(result["f"], Decimal("2.5"))
        self.assertEqual(result["huge"], Decimal("1E+400"))
        self.assertEqual(result["tiny"], Decimal("1E-400"))
        self.assertEqual(result["negzero"], Decimal("0"))
        self.assertEqual(result["deep"],
                         [1, Decimal("1.0"), Decimal("1E+0"),
                          {"z": Decimal("0.0001")}])

    def test_compare_rejects_non_json_number_constants_anywhere(self):
        jobs = self._compare_jobs()
        self._write_report("ok.json", {"results": []})
        cases = {
            "nan-in-result": b'{"results":[{"name":"healthy","status":"completed","result":{"x":NaN}}]}',
            "inf-in-result": b'{"results":[{"name":"healthy","status":"completed","result":{"x":Infinity}}]}',
            "neginf-in-result": b'{"results":[{"name":"healthy","status":"completed","result":{"x":-Infinity}}]}',
            "nan-ignored-field": b'{"results":[{"name":"healthy","status":"completed","result":{},"extra":NaN}]}',
            "inf-top-level": b'{"results":[],"x":Infinity}',
            "neginf-blocked-record": b'{"results":[{"name":"downstream","status":"blocked",'
                                     b'"blocked_by":["broken"],"y":-Infinity}]}',
        }
        for label, raw in cases.items():
            with self.subTest(label=label):
                self._write_report(f"{label}.json", None, raw=raw)
                with self.assertRaises(ValueError):
                    compare_reports(self.root, jobs, "out.json",
                                    f"{label}.json", "ok.json")
                with self.assertRaises(ValueError):
                    compare_reports(self.root, jobs, "out.json",
                                    "ok.json", f"{label}.json")
        # The same spelling inside JSON strings is ordinary text.
        self._write_report("text1.json", {"results": [
            {"name": "broken", "status": "failed", "error": "value was NaN"}]})
        self._write_report("text2.json", {"results": [
            {"name": "broken", "status": "failed", "error": "value was NaN"}]})
        result = compare_reports(self.root, jobs, "out.json", "text1.json", "text2.json")
        self.assertEqual(result["jobs"][0]["change"], "unchanged")

    def test_compare_blocked_by_is_a_set_in_declaration_order(self):
        jobs = [
            {"name": "p", "operation": "shell", "input": "notes.txt"},
            {"name": "q", "operation": "shell", "input": "notes.txt"},
            {"name": "join", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["p", "q"]},
        ]
        self._write_report("b.json", {"results": [
            {"name": "join", "status": "blocked", "blocked_by": ["q", "p"]},
            {"name": "p", "status": "failed", "error": "e"},
            {"name": "q", "status": "failed", "error": "e"},
        ]})
        self._write_report("a.json", {"results": [
            {"name": "q", "status": "failed", "error": "e"},
            {"name": "p", "status": "failed", "error": "e"},
            {"name": "join", "status": "blocked", "blocked_by": ["p", "q"]},
        ]})
        result = compare_reports(self.root, jobs, "out.json", "b.json", "a.json")
        by_name = {row["name"]: row for row in result["jobs"]}
        self.assertTrue(all(row["change"] == "unchanged" for row in result["jobs"]))
        # Both sides are emitted in current declaration order.
        self.assertEqual(by_name["join"]["before"]["blocked_by"], ["p", "q"])
        self.assertEqual(by_name["join"]["after"]["blocked_by"], ["p", "q"])
        # A genuinely different blocked_by set is a change.
        self._write_report("a2.json", {"results": [
            {"name": "p", "status": "failed", "error": "e"},
            {"name": "join", "status": "blocked", "blocked_by": ["p"]},
        ]})
        changes = {row["name"]: row["change"] for row in
                   compare_reports(self.root, jobs, "out.json", "b.json", "a2.json")["jobs"]}
        self.assertEqual(changes["join"], "changed")

    def test_compare_strict_payload_validation(self):
        jobs = self._compare_jobs()
        self._write_report("ok.json", {"results": []})
        cases = {
            "completed-no-result": '{"results":[{"name":"healthy","status":"completed"}]}',
            "completed-result-list": '{"results":[{"name":"healthy","status":"completed","result":[1]}]}',
            "completed-result-null": '{"results":[{"name":"healthy","status":"completed","result":null}]}',
            "failed-no-error": '{"results":[{"name":"broken","status":"failed"}]}',
            "failed-error-number": '{"results":[{"name":"broken","status":"failed","error":5}]}',
            "blocked-no-field": '{"results":[{"name":"downstream","status":"blocked"}]}',
            "blocked-empty": '{"results":[{"name":"downstream","status":"blocked","blocked_by":[]}]}',
            "blocked-duplicate": '{"results":[{"name":"downstream","status":"blocked","blocked_by":["broken","broken"]}]}',
            "blocked-not-direct": '{"results":[{"name":"downstream","status":"blocked","blocked_by":["healthy"]}]}',
            "blocked-unknown": '{"results":[{"name":"downstream","status":"blocked","blocked_by":["ghost"]}]}',
        }
        for label, raw in cases.items():
            with self.subTest(label=label):
                self._write_report(f"{label}.json", None, raw=raw.encode("utf-8"))
                with self.assertRaises(ValueError):
                    compare_reports(self.root, jobs, "out.json", f"{label}.json", "ok.json")

    def test_compare_report_file_and_structure_rules_match_retry(self):
        jobs = self._compare_jobs()
        run_plan(self.root, jobs, "results/report.json")
        raw_cases = {
            "bad-encoding": b'{"results": []}\xff',
            "bad-json": b"{not json",
            "not-object": b"[1, 2]",
            "no-results": b"{}",
            "results-not-list": b'{"results": {}}',
            "entry-not-object": b'{"results": [1]}',
            "no-name": b'{"results": [{"status": "completed", "result": {}}]}',
            "unknown-name": b'{"results": [{"name": "ghost", "status": "completed", "result": {}}]}',
            "duplicate-name": b'{"results": [{"name": "healthy", "status": "completed", "result": {}},'
                              b' {"name": "healthy", "status": "failed", "error": "x"}]}',
            "bad-status": b'{"results": [{"name": "healthy", "status": "done"}]}',
        }
        for label, raw in raw_cases.items():
            with self.subTest(label=label):
                self._write_report(f"{label}.json", None, raw=raw)
                with self.assertRaises(ValueError):
                    compare_reports(self.root, jobs, "out.json", f"{label}.json",
                                    "results/report.json")
        for bad in ("missing.json", "../outside.json", str(ROOT / "README.md")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    compare_reports(self.root, jobs, "out.json", bad, "results/report.json")
        # A symlink that leaves root is rejected; one staying inside is followed.
        (self.root / "sub").mkdir()
        (self.root / "sub/real.json").write_text(
            json.dumps({"results": [{"name": "healthy", "status": "completed",
                                     "result": {"lines": 2}}]}), encoding="utf-8")
        (self.root / "in-link.json").symlink_to(self.root / "sub/real.json")
        compare_reports(self.root, jobs, "out.json", "in-link.json", "in-link.json")
        (self.root / "out-link.json").symlink_to(ROOT / "README.md")
        with self.assertRaises(ValueError):
            compare_reports(self.root, jobs, "out.json", "out-link.json", "in-link.json")

    def test_compare_validates_whole_plan_and_output_without_touching_files(self):
        self._write_report("empty.json", {"results": []})
        bad_input = [
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
            {"name": "stray", "operation": "count-lines", "input": "../outside.txt"},
        ]
        with self.assertRaises(ValueError):
            compare_reports(self.root, bad_input, "out.json", "empty.json", "empty.json")
        bad_dep = [
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
            {"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["b"]},
            {"name": "b", "operation": "count-lines", "input": "notes.txt", "depends_on": ["a"]},
        ]
        with self.assertRaises(ValueError):
            compare_reports(self.root, bad_dep, "out.json", "empty.json", "empty.json")
        good = [{"name": "notes", "operation": "count-lines", "input": "notes.txt"}]
        with self.assertRaises(ValueError):
            compare_reports(self.root, good, "notes.txt", "empty.json", "empty.json")
        # No task input is read and no directory or file is created.
        before = {p.name for p in self.root.iterdir()}
        jobs = [{"name": "missing", "operation": "count-lines", "input": "nope.txt"}]
        self.assertEqual(compare_reports(self.root, jobs, "deep/new/out.json",
                                         "empty.json", "empty.json"), {"jobs": []})
        self.assertFalse((self.root / "deep").exists())
        self.assertEqual({p.name for p in self.root.iterdir()}, before)

    def test_cli_compare_flag(self):
        jobs = self._compare_jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json", "--root", str(self.root)]
        self._write_report("before.json", {"results": [
            {"name": "broken", "status": "failed", "error": "boom"},
            {"name": "gone", "status": "completed", "result": {"lines": 2}},
        ]})
        self._write_report("after.json", {"results": [
            {"name": "broken", "status": "completed", "result": {"lines": 2}},
            {"name": "newjob", "status": "blocked", "blocked_by": []},
        ]})
        # The blocked_by [] payload is invalid, so this after report must fail;
        # replace it with a valid one for the success path.
        run = subprocess.run(prefix + ["--compare", "before.json", "after.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertEqual(set(json.loads(run.stdout)), {"error"})
        self._write_report("after.json", {"results": [
            {"name": "broken", "status": "completed", "result": {"lines": 2}},
            {"name": "newjob", "status": "failed", "error": "late"},
        ]})
        run = subprocess.run(prefix + ["--compare", "before.json", "after.json"],
                             capture_output=True, text=True)
        # Failed records in either report still mean a successful comparison: exit 0.
        self.assertEqual(run.returncode, 0, run.stderr)
        payload = json.loads(run.stdout)
        self.assertEqual([row["name"] for row in payload["jobs"]],
                         ["broken", "newjob", "gone"])
        self.assertEqual([row["change"] for row in payload["jobs"]],
                         ["changed", "added", "removed"])
        # Same file for both paths is accepted.
        run = subprocess.run(prefix + ["--compare", "before.json", "before.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertTrue(all(row["change"] == "unchanged"
                            for row in json.loads(run.stdout)["jobs"]))
        # The default output path is never created.
        self.assertFalse((self.root / ".results").exists())

    def test_cli_compare_outputs_exact_json_numbers(self):
        from decimal import Decimal
        jobs = [{"name": "t", "operation": "count-lines", "input": "notes.txt"}]
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json",
                  "--root", str(self.root)]
        (self.root / "b.json").write_bytes(
            b'{"results":[{"name":"t","status":"completed","result":{'
            b'"huge":1e400,"tiny":1e-400,"long":0.10000000000000001,'
            b'"edge":9007199254740993}}]}')
        (self.root / "a.json").write_bytes(
            b'{"results":[{"name":"t","status":"completed","result":{'
            b'"huge":2e400,"tiny":0,"long":0.1,"edge":9007199254740992}}]}')
        run = subprocess.run(prefix + ["--compare", "b.json", "a.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["jobs"][0]["change"], "changed")
        # The emitted literals parse back to the exact report values and
        # stay JSON numbers: no strings, no Infinity/NaN.
        exact = json.loads(run.stdout, parse_float=Decimal)["jobs"][0]
        self.assertEqual(exact["before"]["result"]["huge"], Decimal("1E+400"))
        self.assertEqual(exact["before"]["result"]["tiny"], Decimal("1E-400"))
        self.assertEqual(exact["before"]["result"]["long"],
                         Decimal("0.10000000000000001"))
        self.assertEqual(exact["before"]["result"]["edge"], 9007199254740993)
        self.assertNotIn("Infinity", run.stdout)
        self.assertNotIn("NaN", run.stdout)
        # A NaN-bearing report is rejected with error-only JSON, exit 2.
        (self.root / "nan.json").write_bytes(
            b'{"results":[{"name":"t","status":"completed",'
            b'"result":{},"extra":NaN}]}')
        run = subprocess.run(prefix + ["--compare", "nan.json", "a.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertEqual(set(json.loads(run.stdout)), {"error"})

    def test_cli_compare_conflicts_plan_protection_and_errors(self):
        jobs = self._compare_jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json", "--root", str(self.root)]
        self._write_report("r.json", {"results": []})
        marker = self.root / "keep.json"
        marker.write_text("KEEP", encoding="utf-8")
        for extra in (["--compare", "r.json", "r.json", "--preview"],
                      ["--compare", "r.json", "r.json", "--only", "healthy"],
                      ["--compare", "r.json", "r.json", "--retry", "r.json"],
                      ["--compare", "r.json", "r.json", "--retry-preview", "r.json"],
                      ["--compare", "missing.json", "r.json"],
                      ["--compare", "r.json", "../outside.json"],
                      ["--compare", "r.json", "r.json", "--output", "notes.txt"],
                      ["--compare", "r.json", "r.json", "--output", "plan.json"]):
            run = subprocess.run(prefix + extra, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, (extra, run.stderr))
            self.assertEqual(set(json.loads(run.stdout)), {"error"})
        self.assertEqual(marker.read_text(), "KEEP")
        bad = [{"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["x"]}]
        plan.write_text(json.dumps({"jobs": bad}))
        run = subprocess.run(prefix + ["--compare", "r.json", "r.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertEqual(set(json.loads(run.stdout)), {"error"})
        self.assertEqual(marker.read_text(), "KEEP")


    def _explain_jobs(self):
        return [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "mid", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["broken"]},
            {"name": "leaf", "operation": "sha256", "input": "notes.txt",
             "depends_on": ["mid"]},
            {"name": "zfail", "operation": "shell", "input": "missing.txt"},
            {"name": "join", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["mid", "zfail"]},
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
        ]

    def test_explain_from_real_run_traces_terminal_failures(self):
        jobs = self._explain_jobs()
        run_plan(self.root, jobs, "results/report.json")
        result = explain_report(self.root, jobs, "never/created.json",
                                "results/report.json")
        self.assertEqual([row["name"] for row in result["jobs"]],
                         ["broken", "mid", "leaf", "zfail", "join"])
        self.assertTrue(all(set(row) == {"name", "status", "causes"}
                            for row in result["jobs"]))
        by_name = {row["name"]: row for row in result["jobs"]}
        fail_cause = {"name": "broken", "status": "failed",
                      "error": "unsupported operation"}
        self.assertEqual(by_name["broken"],
                         {"name": "broken", "status": "failed",
                          "causes": [dict(fail_cause)]})
        self.assertEqual(by_name["mid"]["status"], "blocked")
        self.assertEqual(by_name["mid"]["causes"], [dict(fail_cause)])
        # The chain leaf -> mid -> broken ends at the failed root only.
        self.assertEqual(by_name["leaf"]["causes"], [dict(fail_cause)])
        self.assertEqual(by_name["zfail"],
                         {"name": "zfail", "status": "failed",
                          "causes": [{"name": "zfail", "status": "failed",
                                      "error": "unsupported operation"}]})
        # join's two branches share broken via mid; causes are de-duplicated
        # and emitted in plan order (broken before zfail), not blocked_by order.
        self.assertEqual(by_name["join"],
                         {"name": "join", "status": "blocked",
                          "causes": [dict(fail_cause),
                                     {"name": "zfail", "status": "failed",
                                      "error": "unsupported operation"}]})
        self.assertTrue(all(set(cause) == {"name", "status", "error"}
                            for row in result["jobs"] for cause in row["causes"]))
        # Completed and unrecorded tasks are not jobs.
        self.assertNotIn("healthy", by_name)
        self.assertFalse((self.root / "never").exists())

    def test_explain_dedup_and_plan_order_independent_of_record_order(self):
        jobs = [
            {"name": "p", "operation": "shell", "input": "notes.txt"},
            {"name": "q", "operation": "shell", "input": "notes.txt"},
            {"name": "r", "operation": "shell", "input": "notes.txt"},
            {"name": "join", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["p", "q", "r"]},
        ]
        self._write_report("r.json", {"results": [
            {"name": "join", "status": "blocked", "blocked_by": ["r", "q", "p"],
             "note": 1},
            {"name": "r", "status": "failed", "error": "r-err", "severity": 9},
            {"name": "q", "status": "failed", "error": "q-err"},
            {"name": "p", "status": "failed", "error": "p-err"},
        ]})
        result = explain_report(self.root, jobs, "out.json", "r.json")
        self.assertEqual([row["name"] for row in result["jobs"]],
                         ["p", "q", "r", "join"])
        join = next(row for row in result["jobs"] if row["name"] == "join")
        self.assertEqual(join["causes"],
                         [{"name": "p", "status": "failed", "error": "p-err"},
                          {"name": "q", "status": "failed", "error": "q-err"},
                          {"name": "r", "status": "failed", "error": "r-err"}])

    def test_explain_traces_only_reported_blocked_by_without_inference(self):
        jobs = [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "otherfail", "operation": "shell", "input": "missing.txt"},
            {"name": "downstream", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["broken", "otherfail"]},
        ]
        # The report blames only 'broken'; the other declared dependency is
        # recorded failed but must not be inferred as a cause.
        self._write_report("r.json", {"results": [
            {"name": "downstream", "status": "blocked", "blocked_by": ["broken"]},
            {"name": "otherfail", "status": "failed", "error": "x"},
            {"name": "broken", "status": "failed", "error": "boom"},
        ]})
        result = explain_report(self.root, jobs, "out.json", "r.json")
        by_name = {row["name"]: row for row in result["jobs"]}
        self.assertEqual([c["name"] for c in by_name["downstream"]["causes"]],
                         ["broken"])
        self.assertEqual(by_name["downstream"]["causes"][0]["error"], "boom")

    def test_explain_unrecorded_terminals_partial_report(self):
        jobs = self._explain_jobs()
        # leaf and join are the only recorded jobs; every blocked_by target
        # is unrecorded, and a never-referenced task stays absent entirely.
        self._write_report("partial.json", {"results": [
            {"name": "join", "status": "blocked", "blocked_by": ["zfail", "mid"]},
            {"name": "leaf", "status": "blocked", "blocked_by": ["mid"]},
        ]})
        result = explain_report(self.root, jobs, "out.json", "partial.json")
        self.assertEqual([row["name"] for row in result["jobs"]], ["leaf", "join"])
        by_name = {row["name"]: row for row in result["jobs"]}
        self.assertEqual(by_name["leaf"],
                         {"name": "leaf", "status": "blocked",
                          "causes": [{"name": "mid", "status": "unrecorded",
                                      "error": None}]})
        self.assertEqual(by_name["join"],
                         {"name": "join", "status": "blocked",
                          "causes": [{"name": "mid", "status": "unrecorded",
                                      "error": None},
                                     {"name": "zfail", "status": "unrecorded",
                                      "error": None}]})
        # 'broken' is unrecorded and never named: it is neither a job nor a cause.
        flat = {c["name"] for row in result["jobs"] for c in row["causes"]}
        self.assertNotIn("broken", flat)
        self.assertNotIn("healthy", flat)

    def test_explain_unrecorded_beyond_recorded_blocked_chain(self):
        jobs = self._explain_jobs()
        # leaf -> mid (recorded blocked) -> broken (no record): the recorded
        # blocker is expanded past and does not itself become a cause.
        self._write_report("partial.json", {"results": [
            {"name": "leaf", "status": "blocked", "blocked_by": ["mid"]},
            {"name": "mid", "status": "blocked", "blocked_by": ["broken"]},
        ]})
        result = explain_report(self.root, jobs, "out.json", "partial.json")
        by_name = {row["name"]: row for row in result["jobs"]}
        self.assertEqual(list(by_name), ["mid", "leaf"])
        self.assertEqual(by_name["leaf"]["causes"],
                         [{"name": "broken", "status": "unrecorded",
                           "error": None}])
        self.assertEqual(by_name["mid"]["causes"],
                         [{"name": "broken", "status": "unrecorded",
                           "error": None}])

    def test_explain_blocked_by_referencing_completed_raises(self):
        jobs = self._explain_jobs()
        self._write_report("direct.json", {"results": [
            {"name": "mid", "status": "blocked", "blocked_by": ["broken"]},
            {"name": "broken", "status": "completed", "result": {"lines": 2}},
        ]})
        with self.assertRaises(ValueError):
            explain_report(self.root, jobs, "out.json", "direct.json")
        # The completed task may hide deeper in the chain.
        self._write_report("nested.json", {"results": [
            {"name": "leaf", "status": "blocked", "blocked_by": ["mid"]},
            {"name": "mid", "status": "blocked", "blocked_by": ["broken"]},
            {"name": "broken", "status": "completed", "result": {"lines": 2}},
        ]})
        with self.assertRaises(ValueError):
            explain_report(self.root, jobs, "out.json", "nested.json")

    def test_explain_empty_or_all_completed(self):
        jobs = self._explain_jobs()
        self._write_report("empty.json", {"results": []})
        self.assertEqual(explain_report(self.root, jobs, "out.json", "empty.json"),
                         {"jobs": []})
        self._write_report("done.json", {"results": [
            {"name": "healthy", "status": "completed", "result": {"lines": 1}},
            {"name": "broken", "status": "completed", "result": {"lines": 1}},
        ]})
        self.assertEqual(explain_report(self.root, jobs, "out.json", "done.json"),
                         {"jobs": []})

    def test_explain_report_may_equal_output(self):
        jobs = self._explain_jobs()
        run_plan(self.root, jobs, "same.json")
        before = (self.root / "same.json").read_text()
        result = explain_report(self.root, jobs, "same.json", "same.json")
        self.assertEqual([row["name"] for row in result["jobs"]],
                         ["broken", "mid", "leaf", "zfail", "join"])
        self.assertEqual((self.root / "same.json").read_text(), before)

    def test_explain_invalid_reports_raise_value_error(self):
        jobs = self._explain_jobs()
        run_plan(self.root, jobs, "results/report.json")
        ok = {"results": []}
        self._write_report("ok.json", ok)
        raw_cases = {
            "bad-encoding": b'{"results": []}\xff',
            "bad-json": b"{not json",
            "not-object": b"[1, 2]",
            "no-results": b"{}",
            "results-not-list": b'{"results": {}}',
            "entry-not-object": b'{"results": [1]}',
            "no-name": b'{"results": [{"status": "completed", "result": {}}]}',
            "unknown-name": b'{"results": [{"name": "ghost", "status": "failed", "error": "x"}]}',
            "duplicate-name": b'{"results": [{"name": "healthy", "status": "completed", "result": {}},'
                              b' {"name": "healthy", "status": "failed", "error": "x"}]}',
            "bad-status": b'{"results": [{"name": "healthy", "status": "done"}]}',
            "failed-no-error": b'{"results":[{"name":"broken","status":"failed"}]}',
            "failed-error-number": b'{"results":[{"name":"broken","status":"failed","error":5}]}',
            "blocked-no-field": b'{"results":[{"name":"mid","status":"blocked"}]}',
            "blocked-empty": b'{"results":[{"name":"mid","status":"blocked","blocked_by":[]}]}',
            "blocked-duplicate": b'{"results":[{"name":"join","status":"blocked",'
                                 b'"blocked_by":["mid","mid"]}]}',
            "blocked-not-direct": b'{"results":[{"name":"mid","status":"blocked",'
                                  b'"blocked_by":["healthy"]}]}',
        }
        for label, raw in raw_cases.items():
            with self.subTest(label=label):
                self._write_report(f"{label}.json", None, raw=raw)
                with self.assertRaises(ValueError):
                    explain_report(self.root, jobs, "out.json", f"{label}.json")
        for bad_report in ("missing.json", "../outside.json", str(ROOT / "README.md")):
            with self.subTest(bad_report=bad_report):
                with self.assertRaises(ValueError):
                    explain_report(self.root, jobs, "out.json", bad_report)

    def test_explain_validates_whole_plan_and_output_without_touching_files(self):
        self._write_report("empty.json", {"results": []})
        bad_input = [
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
            {"name": "stray", "operation": "count-lines", "input": "../outside.txt"},
        ]
        with self.assertRaises(ValueError):
            explain_report(self.root, bad_input, "out.json", "empty.json")
        bad_dep = [
            {"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["b"]},
            {"name": "b", "operation": "count-lines", "input": "notes.txt", "depends_on": ["a"]},
        ]
        with self.assertRaises(ValueError):
            explain_report(self.root, bad_dep, "out.json", "empty.json")
        good = [{"name": "notes", "operation": "count-lines", "input": "notes.txt"}]
        with self.assertRaises(ValueError):
            explain_report(self.root, good, "notes.txt", "empty.json")
        # Missing inputs, unknown operations and bad contents never matter;
        # no directory or file is created and inputs are not read.
        jobs = [
            {"name": "missing", "operation": "count-lines", "input": "nope.txt"},
            {"name": "bad-op", "operation": "shell", "input": "notes.txt"},
            {"name": "bad-csv", "operation": "csv-summary", "input": "notes.txt"},
        ]
        before = {p.name for p in self.root.iterdir()}
        self.assertEqual(explain_report(self.root, jobs, "deep/new/out.json",
                                        "empty.json"), {"jobs": []})
        self.assertFalse((self.root / "deep").exists())
        self.assertEqual({p.name for p in self.root.iterdir()}, before)

    def test_cli_explain_flag(self):
        jobs = self._explain_jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json", "--root", str(self.root)]
        run = subprocess.run(prefix + ["--output", "e.json"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 1)
        run = subprocess.run(prefix + ["--explain", "e.json", "--output", "e.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        payload = json.loads(run.stdout)
        self.assertEqual([row["name"] for row in payload["jobs"]],
                         ["broken", "mid", "leaf", "zfail", "join"])
        self.assertTrue(all(set(row) == {"name", "status", "causes"}
                            for row in payload["jobs"]))
        self.assertTrue(all(set(cause) == {"name", "status", "error"}
                            for row in payload["jobs"] for cause in row["causes"]))
        # Failed records still mean a successful explanation: exit 0.
        # An empty report prints {"jobs": []} and never creates the default output.
        self._write_report("empty.json", {"results": []})
        run = subprocess.run(prefix + ["--explain", "empty.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"jobs": []})
        self.assertFalse((self.root / ".results").exists())

    def test_cli_explain_conflicts_and_errors(self):
        jobs = self._explain_jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json", "--root", str(self.root)]
        self._write_report("r.json", {"results": []})
        marker = self.root / "keep.json"
        marker.write_text("KEEP", encoding="utf-8")
        for extra in (["--explain", "r.json", "--only", "healthy"],
                      ["--explain", "r.json", "--preview"],
                      ["--explain", "r.json", "--retry-preview", "r.json"],
                      ["--explain", "r.json", "--retry", "r.json"],
                      ["--explain", "r.json", "--compare", "r.json", "r.json"],
                      ["--explain", "missing.json"],
                      ["--explain", "../outside.json"],
                      ["--explain", "r.json", "--output", "notes.txt"],
                      ["--explain", "r.json", "--output", "plan.json"]):
            run = subprocess.run(prefix + extra, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, (extra, run.stderr))
            self.assertEqual(set(json.loads(run.stdout)), {"error"})
        self.assertEqual(marker.read_text(), "KEEP")
        bad = [{"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["x"]}]
        plan.write_text(json.dumps({"jobs": bad}))
        run = subprocess.run(prefix + ["--explain", "r.json"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertEqual(set(json.loads(run.stdout)), {"error"})
        self.assertEqual(marker.read_text(), "KEEP")


    def _history_jobs(self):
        return [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "downstream", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["broken"]},
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
            {"name": "lonely", "operation": "count-lines", "input": "notes.txt"},
        ]

    def test_history_order_shape_missing_and_duplicate_paths(self):
        jobs = self._history_jobs()
        run_plan(self.root, jobs, "r1.json")
        run_plan(self.root, jobs, "r2.json", targets=["healthy"])
        result = query_history(self.root, jobs, "never/created.json",
                               ["r1.json", "r2.json", "r1.json"])
        self.assertEqual([row["name"] for row in result["jobs"]],
                         ["broken", "downstream", "healthy", "lonely"])
        self.assertTrue(all(set(row) == {"name", "history"} for row in result["jobs"]))
        by_name = {row["name"]: row for row in result["jobs"]}
        self.assertTrue(all(len(row["history"]) == 3 for row in result["jobs"]))
        self.assertTrue(all(set(slot) == {"report", "record"}
                            for row in result["jobs"] for slot in row["history"]))
        self.assertEqual([slot["report"] for slot in by_name["healthy"]["history"]],
                         ["r1.json", "r2.json", "r1.json"])
        # r2 has no record of downstream; the slot is null, not copied
        # from a neighbouring report.
        self.assertEqual([slot["record"] for slot in by_name["downstream"]["history"]],
                         [{"status": "blocked", "blocked_by": ["broken"]}, None,
                          {"status": "blocked", "blocked_by": ["broken"]}])
        # Duplicate path produces an equal, duplicate history position.
        self.assertEqual(by_name["broken"]["history"][0],
                         by_name["broken"]["history"][2])
        # Payload shapes match compare_reports exactly.
        self.assertEqual(by_name["healthy"]["history"][0]["record"],
                         {"status": "completed", "result": {"lines": 2}})
        self.assertEqual(by_name["broken"]["history"][0]["record"],
                         {"status": "failed", "error": "unsupported operation"})
        # The output directory is not created and inputs need not be read.
        self.assertFalse((self.root / "never").exists())

    def test_history_unfiltered_lists_only_recorded_tasks(self):
        jobs = self._history_jobs()
        self._write_report("empty.json", {"results": []})
        self.assertEqual(query_history(self.root, jobs, "out.json", ["empty.json"]),
                         {"jobs": []})
        # A task recorded in one report is listed even with another report missing.
        self._write_report("partial.json", {"results": [
            {"name": "lonely", "status": "completed", "result": {"lines": 2}}]})
        result = query_history(self.root, jobs, "out.json",
                               ["empty.json", "partial.json"])
        self.assertEqual([row["name"] for row in result["jobs"]], ["lonely"])
        self.assertEqual([slot["record"] for slot in result["jobs"][0]["history"]],
                         [None, {"status": "completed", "result": {"lines": 2}}])

    def test_history_targets_exact_names_without_closure(self):
        jobs = self._history_jobs()
        run_plan(self.root, jobs, "r1.json")
        run_plan(self.root, jobs, "r2.json", targets=["healthy"])
        # Only the named task, not its failing prerequisite.
        result = query_history(self.root, jobs, "out.json",
                               ["r1.json", "r2.json"], targets=["downstream"])
        self.assertEqual([row["name"] for row in result["jobs"]], ["downstream"])
        # A task no report records still appears with an all-null history.
        result = query_history(self.root, jobs, "out.json",
                               ["r2.json"], targets=["broken"])
        self.assertEqual(result["jobs"],
                         [{"name": "broken",
                           "history": [{"report": "r2.json", "record": None}]}])
        # Target argument order does not change plan order.
        result = query_history(self.root, jobs, "out.json",
                               ["r1.json"], targets=["lonely", "healthy"])
        self.assertEqual([row["name"] for row in result["jobs"]],
                         ["healthy", "lonely"])

    def test_history_blocked_by_emitted_in_declaration_order(self):
        jobs = [
            {"name": "p", "operation": "shell", "input": "notes.txt"},
            {"name": "q", "operation": "shell", "input": "notes.txt"},
            {"name": "join", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["p", "q"]},
        ]
        self._write_report("r.json", {"results": [
            {"name": "join", "status": "blocked", "blocked_by": ["q", "p"]},
            {"name": "p", "status": "failed", "error": "e"},
            {"name": "q", "status": "failed", "error": "e"},
        ]})
        result = query_history(self.root, jobs, "out.json", ["r.json"])
        by_name = {row["name"]: row for row in result["jobs"]}
        self.assertEqual(by_name["join"]["history"][0]["record"],
                         {"status": "blocked", "blocked_by": ["p", "q"]})

    def test_history_invalid_reports_argument_raises(self):
        jobs = self._history_jobs()
        for bad in ([], "r1.json", None, [1], [""], ["  "]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    query_history(self.root, jobs, "out.json", bad)

    def test_history_invalid_targets_raise(self):
        self._write_report("empty.json", {"results": []})
        good = [{"name": "notes", "operation": "count-lines", "input": "notes.txt"}]
        for targets in ([], "notes", [1], ["  "], ["notes", "notes"], ["ghost"]):
            with self.subTest(targets=targets):
                with self.assertRaises(ValueError):
                    query_history(self.root, good, "out.json", ["empty.json"],
                                  targets=targets)

    def test_history_validates_reports_strictly_including_out_of_scope(self):
        jobs = self._history_jobs()
        run_plan(self.root, jobs, "results/report.json")
        self._write_report("ok.json", {"results": []})
        strict_cases = {
            "bad-encoding": b'{"results": []}\xff',
            "bad-json": b"{not json",
            "not-object": b"[1, 2]",
            "no-results": b"{}",
            "results-not-list": b'{"results": {}}',
            "entry-not-object": b'{"results": [1]}',
            "unknown-name": b'{"results": [{"name": "ghost", "status": "completed", "result": {}}]}',
            "duplicate-name": b'{"results": [{"name": "healthy", "status": "completed", "result": {}},'
                              b' {"name": "healthy", "status": "failed", "error": "x"}]}',
            "bad-status": b'{"results": [{"name": "healthy", "status": "done"}]}',
            "completed-no-result": b'{"results":[{"name":"healthy","status":"completed"}]}',
            "failed-no-error": b'{"results":[{"name":"broken","status":"failed"}]}',
            "blocked-empty": b'{"results":[{"name":"downstream","status":"blocked","blocked_by":[]}]}',
            "blocked-not-direct": b'{"results":[{"name":"downstream","status":"blocked","blocked_by":["healthy"]}]}',
        }
        for label, raw in strict_cases.items():
            with self.subTest(label=label):
                self._write_report(f"{label}.json", None, raw=raw)
                with self.assertRaises(ValueError):
                    query_history(self.root, jobs, "out.json", [f"{label}.json", "ok.json"])
        # A malformed record outside the --only scope is still checked.
        self._write_report("out-of-scope.json", {"results": [
            {"name": "healthy", "status": "completed", "result": {"lines": 1}},
            {"name": "broken", "status": "failed"},
        ]})
        with self.assertRaises(ValueError):
            query_history(self.root, jobs, "out.json", ["out-of-scope.json"],
                          targets=["healthy"])
        # All reports are validated, so a later bad report rejects the query.
        with self.assertRaises(ValueError):
            query_history(self.root, jobs, "out.json",
                          ["ok.json", "bad-json.json"])
        # Path, existence and symlink-boundary rules match compare_reports.
        for bad in ("missing.json", "../outside.json", str(ROOT / "README.md")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    query_history(self.root, jobs, "out.json", [bad])
        (self.root / "out-link.json").symlink_to(ROOT / "README.md")
        with self.assertRaises(ValueError):
            query_history(self.root, jobs, "out.json", ["out-link.json"])

    def test_history_validates_whole_plan_and_output_without_touching_files(self):
        self._write_report("empty.json", {"results": []})
        bad_input = [
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
            {"name": "stray", "operation": "count-lines", "input": "../outside.txt"},
        ]
        with self.assertRaises(ValueError):
            query_history(self.root, bad_input, "out.json", ["empty.json"])
        bad_dep = [
            {"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["b"]},
            {"name": "b", "operation": "count-lines", "input": "notes.txt", "depends_on": ["a"]},
        ]
        with self.assertRaises(ValueError):
            query_history(self.root, bad_dep, "out.json", ["empty.json"])
        good = [{"name": "notes", "operation": "count-lines", "input": "notes.txt"}]
        with self.assertRaises(ValueError):
            query_history(self.root, good, "notes.txt", ["empty.json"])
        before = {p.name for p in self.root.iterdir()}
        jobs = [{"name": "missing", "operation": "count-lines", "input": "nope.txt"}]
        self.assertEqual(query_history(self.root, jobs, "deep/new/out.json",
                                       ["empty.json"]), {"jobs": []})
        self.assertFalse((self.root / "deep").exists())
        self.assertEqual({p.name for p in self.root.iterdir()}, before)

    def test_history_report_may_equal_output(self):
        jobs = self._history_jobs()
        run_plan(self.root, jobs, "same.json")
        before = (self.root / "same.json").read_text()
        result = query_history(self.root, jobs, "same.json", ["same.json", "same.json"])
        self.assertEqual([row["name"] for row in result["jobs"]],
                         ["broken", "downstream", "healthy", "lonely"])
        self.assertTrue(all(len(row["history"]) == 2 for row in result["jobs"]))
        self.assertEqual((self.root / "same.json").read_text(), before)

    def test_cli_history_flag(self):
        jobs = self._history_jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json",
                  "--root", str(self.root)]
        subprocess.run(prefix + ["--output", "r1.json"], capture_output=True)
        subprocess.run(prefix + ["--output", "r2.json", "--only", "healthy"],
                       capture_output=True)
        # Failed/blocked records still give exit 0.
        run = subprocess.run(prefix + ["--history", "r1.json", "--history", "r2.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        payload = json.loads(run.stdout)
        self.assertEqual([row["name"] for row in payload["jobs"]],
                         ["broken", "downstream", "healthy", "lonely"])
        self.assertTrue(all(len(row["history"]) == 2 for row in payload["jobs"]))
        # --only filters to exact task names and exit stays 0 for all-null rows.
        run = subprocess.run(prefix + ["--history", "r2.json", "--only", "lonely"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout),
                         {"jobs": [{"name": "lonely",
                                    "history": [{"report": "r2.json", "record": None}]}]})
        # Empty report prints {"jobs": []} and never creates the default output.
        self._write_report("empty.json", {"results": []})
        run = subprocess.run(prefix + ["--history", "empty.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"jobs": []})
        self.assertFalse((self.root / ".results").exists())

    def test_cli_history_conflicts_and_errors(self):
        jobs = self._history_jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json",
                  "--root", str(self.root)]
        self._write_report("r.json", {"results": []})
        marker = self.root / "keep.json"
        marker.write_text("KEEP", encoding="utf-8")
        for extra in (["--history", "r.json", "--preview"],
                      ["--history", "r.json", "--retry-preview", "r.json"],
                      ["--history", "r.json", "--retry", "r.json"],
                      ["--history", "r.json", "--compare", "r.json", "r.json"],
                      ["--history", "r.json", "--explain", "r.json"],
                      ["--history", "missing.json"],
                      ["--history", "../outside.json"],
                      ["--history", "r.json", "--only", "ghost"],
                      ["--history", "r.json", "--output", "notes.txt"],
                      ["--history", "r.json", "--output", "plan.json"]):
            run = subprocess.run(prefix + extra, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, (extra, run.stderr))
            self.assertEqual(set(json.loads(run.stdout)), {"error"})
        self.assertEqual(marker.read_text(), "KEEP")
        # A bad plan is rejected before any report is opened.
        bad = [{"name": "a", "operation": "count-lines", "input": "notes.txt",
                "depends_on": ["x"]}]
        plan.write_text(json.dumps({"jobs": bad}))
        run = subprocess.run(prefix + ["--history", "missing.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertEqual(set(json.loads(run.stdout)), {"error"})
        self.assertEqual(marker.read_text(), "KEEP")


if __name__ == "__main__":
    unittest.main()
