import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from job_planner import execute_job, local_path, run_plan

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

    def test_forward_references_run_in_determined_order(self):
        jobs = [
            {"name": "all", "operation": "count-lines", "input": "notes.txt", "depends_on": ["first", "second"]},
            {"name": "second", "operation": "sha256", "input": "notes.txt", "depends_on": ["first"]},
            {"name": "first", "operation": "count-lines", "input": "notes.txt"},
        ]
        result = run_plan(self.root, jobs, "results/report.json")
        self.assertEqual([row["name"] for row in result], ["first", "second", "all"])
        self.assertTrue(all(row["status"] == "completed" for row in result))
        again = run_plan(self.root, jobs, "results/report.json")
        self.assertEqual(again, result)

    def test_failed_branch_blocks_downstream_while_unrelated_succeeds(self):
        jobs = [
            {"name": "bad", "operation": "shell", "input": "notes.txt"},
            {"name": "good", "operation": "count-lines", "input": "notes.txt"},
            {"name": "child", "operation": "sha256", "input": "notes.txt", "depends_on": ["bad"]},
            {"name": "pair", "operation": "count-lines", "input": "notes.txt", "depends_on": ["bad", "good"]},
            {"name": "grand", "operation": "count-lines", "input": "notes.txt", "depends_on": ["child", "good"]},
        ]
        result = run_plan(self.root, jobs, "results/report.json")
        by_name = {row["name"]: row for row in result}
        self.assertEqual([row["name"] for row in result], ["bad", "good", "child", "pair", "grand"])
        self.assertEqual(by_name["bad"]["status"], "failed")
        self.assertEqual(set(by_name["bad"]), {"name", "status", "error"})
        self.assertEqual(by_name["good"]["status"], "completed")
        self.assertEqual(by_name["child"], {"name": "child", "status": "blocked", "blocked_by": ["bad"]})
        self.assertEqual(by_name["pair"], {"name": "pair", "status": "blocked", "blocked_by": ["bad"]})
        self.assertEqual(by_name["grand"], {"name": "grand", "status": "blocked", "blocked_by": ["child"]})

    def test_blocked_job_does_not_read_input(self):
        jobs = [
            {"name": "bad", "operation": "shell", "input": "notes.txt"},
            {"name": "skipped", "operation": "sha256", "input": "missing.txt", "depends_on": ["bad"]},
        ]
        result = run_plan(self.root, jobs, "results/report.json")
        self.assertEqual(result[1], {"name": "skipped", "status": "blocked", "blocked_by": ["bad"]})

    def test_blocked_job_still_subject_to_report_input_check(self):
        jobs = [
            {"name": "bad", "operation": "shell", "input": "notes.txt"},
            {"name": "collides", "operation": "sha256", "input": "results/report.json", "depends_on": ["bad"]},
        ]
        with self.assertRaises(ValueError):
            run_plan(self.root, jobs, "results/report.json")

    def test_invalid_dependencies_raise_value_error(self):
        good = {"name": "good", "operation": "count-lines", "input": "notes.txt"}
        invalid_plans = [
            [{"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": "a"}],
            [{"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["ghost"]}],
            [{"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["good"]}],
            [{"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["good", "good"]}, good],
            [{"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["  "]}, good],
            [{"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": [1]}, good],
            [
                {"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["b"]},
                {"name": "b", "operation": "count-lines", "input": "notes.txt", "depends_on": ["a"]},
            ],
        ]
        for plan in invalid_plans:
            with self.assertRaises(ValueError):
                run_plan(self.root, plan, "results/report.json")

    def test_invalid_dependencies_leave_existing_report_untouched(self):
        report = self.root / "results" / "report.json"
        report.parent.mkdir(parents=True)
        report.write_text('{"kept": true}\n', encoding="utf-8")
        jobs = [{"name": "a", "operation": "count-lines", "input": "notes.txt", "depends_on": ["ghost"]}]
        with self.assertRaises(ValueError):
            run_plan(self.root, jobs, "results/report.json")
        self.assertEqual(report.read_text(), '{"kept": true}\n')

    def test_cli_blocked_summary_and_exit_code(self):
        jobs = [
            {"name": "bad", "operation": "shell", "input": "notes.txt"},
            {"name": "good", "operation": "count-lines", "input": "notes.txt"},
            {"name": "blocked", "operation": "sha256", "input": "notes.txt", "depends_on": ["bad"]},
        ]
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json", "--root", str(self.root)]
        run = subprocess.run(prefix, capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"completed": 1, "failed": 1, "blocked": 1})
        plan.write_text(json.dumps({"jobs": [{"name": "bad", "operation": "shell", "input": "notes.txt", "depends_on": []}]}))
        run = subprocess.run(prefix, capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"completed": 0, "failed": 1})
        plan.write_text(json.dumps({"jobs": [{"name": "cycle", "operation": "count-lines", "input": "notes.txt", "depends_on": ["cycle"]}]}))
        run = subprocess.run(prefix, capture_output=True, text=True)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertIn("error", json.loads(run.stdout))


if __name__ == "__main__":
    unittest.main()
