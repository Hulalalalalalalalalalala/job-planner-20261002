import json
from decimal import Decimal
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from job_planner import (compare_reports, execute_job, explain_report, local_path,
                         preview_changes, preview_history_retry, preview_plan,
                         preview_retry, query_execution, query_history, run_changes,
                         run_history_retry, run_plan, run_retry)

ROOT = Path(__file__).resolve().parent


def _reject_json_constant(value):
    raise ValueError(value)


def _loads_strict(text):
    """Parse CLI output with Decimal floats and no NaN/Infinity tokens."""
    return json.loads(text, parse_float=Decimal, parse_constant=_reject_json_constant)


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

    def _changes_jobs(self):
        return [
            {"name": "alpha", "operation": "count-lines", "input": "notes.txt"},
            {"name": "beta", "operation": "sha256", "input": "data.bin",
             "depends_on": ["alpha", "gamma"]},
            {"name": "gamma", "operation": "csv-summary", "input": "sales.csv"},
            {"name": "delta", "operation": "count-lines", "input": "sales.csv",
             "depends_on": ["beta"]},
            {"name": "unrelated", "operation": "count-lines", "input": "other.txt"},
        ]

    def test_changes_direct_hit_downstream_and_triggered_by(self):
        # alpha's input changed; beta depends on alpha and the unchanged
        # gamma, so the targets are alpha and beta (plus downstream delta)
        # while gamma joins only as a prerequisite.
        preview = preview_changes(self.root, self._changes_jobs(),
                                  "results/report.json", ["notes.txt"])
        self.assertEqual(preview["targets"], ["alpha", "beta", "delta"])
        self.assertEqual([row["name"] for row in preview["jobs"]],
                         ["alpha", "gamma", "beta", "delta"])
        by_name = {row["name"]: row for row in preview["jobs"]}
        self.assertEqual(by_name["alpha"]["triggered_by"], ["alpha"])
        self.assertEqual(by_name["beta"]["triggered_by"], ["alpha"])
        self.assertEqual(by_name["delta"]["triggered_by"], ["alpha"])
        self.assertEqual(by_name["gamma"]["triggered_by"], [])
        self.assertEqual(by_name["alpha"]["reason"], "target")
        self.assertEqual(by_name["gamma"]["reason"], "prerequisite")
        self.assertEqual(by_name["gamma"]["required_by"], ["beta", "delta"])
        self.assertEqual(by_name["beta"]["depends_on"], ["alpha", "gamma"])

    def test_changes_multiple_hits_merge_and_order_independent(self):
        jobs = self._changes_jobs()
        expected = preview_changes(self.root, jobs, "results/report.json",
                                   ["notes.txt", "sales.csv"])
        self.assertEqual(expected["targets"], ["alpha", "beta", "gamma", "delta"])
        by_name = {row["name"]: row for row in expected["jobs"]}
        self.assertEqual(by_name["beta"]["triggered_by"], ["alpha", "gamma"])
        # delta's own input changed too, so it lists itself as well.
        self.assertEqual(by_name["delta"]["triggered_by"], ["alpha", "gamma", "delta"])
        # Argument order and paths resolving to the same file change nothing.
        self.assertEqual(preview_changes(self.root, jobs, "results/report.json",
                                         ["sales.csv", "notes.txt"]), expected)
        self.assertEqual(preview_changes(self.root, jobs, "results/report.json",
                                         ["./notes.txt", "notes.txt", "sub/../sales.csv"]),
                         expected)

    def test_changes_empty_or_unmatched_inputs_and_missing_files(self):
        jobs = self._changes_jobs()
        empty = {"targets": [], "jobs": []}
        self.assertEqual(preview_changes(self.root, jobs, "results/report.json", []), empty)
        # Legal paths matching no task input produce no targets; the files
        # and even their directories need not exist.
        self.assertEqual(preview_changes(self.root, jobs, "results/report.json",
                                         ["missing.txt", "deep/none.txt"]), empty)
        # Task inputs are matched by resolved path, never read.
        self.assertEqual(preview_changes(self.root, jobs, "results/report.json",
                                         ["data.bin"])["targets"], ["beta", "delta"])

    def test_changes_invalid_inputs_raise_and_whole_plan_validated(self):
        jobs = self._changes_jobs()
        for changed in ("notes.txt", [1], ["  "], [""], [str(ROOT / "README.md")],
                        ["../outside.txt"], ["notes.txt", "../outside.txt"]):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    preview_changes(self.root, jobs, "results/report.json", changed)
        (self.root / "escape").symlink_to(ROOT / "README.md")
        with self.assertRaises(ValueError):
            preview_changes(self.root, jobs, "results/report.json", ["escape"])
        # The whole plan and output are validated even when nothing matches.
        bad_jobs = jobs + [{"name": "stray", "operation": "count-lines",
                            "input": "../outside.txt"}]
        with self.assertRaises(ValueError):
            preview_changes(self.root, bad_jobs, "results/report.json", [])
        with self.assertRaises(ValueError):
            preview_changes(self.root, jobs, "notes.txt", [])

    def test_changes_is_read_only(self):
        before = {p.name for p in self.root.iterdir()}
        preview = preview_changes(self.root, self._changes_jobs(),
                                  "results/new/deep/report.json", ["notes.txt"])
        self.assertTrue(preview["targets"])
        self.assertFalse((self.root / "results").exists())
        self.assertEqual({p.name for p in self.root.iterdir()}, before)

    def test_cli_changed_flag(self):
        jobs = self._changes_jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json",
                  "--root", str(self.root)]
        run = subprocess.run(prefix + ["--changed", "notes.txt"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        payload = json.loads(run.stdout)
        self.assertEqual(payload["targets"], ["alpha", "beta", "delta"])
        self.assertEqual([row["name"] for row in payload["jobs"]],
                         ["alpha", "gamma", "beta", "delta"])
        self.assertFalse((self.root / ".results").exists())
        # Repeated flags accumulate and --root/--output are allowed.
        run = subprocess.run(prefix + ["--changed", "notes.txt", "--changed", "sales.csv",
                                       "--output", ".results/other.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["targets"],
                         ["alpha", "beta", "gamma", "delta"])
        self.assertFalse((self.root / ".results").exists())
        # An unmatched change prints the empty object and still exits 0.
        run = subprocess.run(prefix + ["--changed", "missing.txt"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"targets": [], "jobs": []})
        # Conflicts and validation failures print error JSON and exit 2.
        for extra in (["--changed", "notes.txt", "--only", "alpha"],
                      ["--changed", "notes.txt", "--preview"],
                      ["--changed", "notes.txt", "--retry-preview", "r.json"],
                      ["--changed", "notes.txt", "--retry", "r.json"],
                      ["--changed", "notes.txt", "--compare", "a.json", "b.json"],
                      ["--changed", "notes.txt", "--explain", "r.json"],
                      ["--changed", "notes.txt", "--history", "r.json"],
                      ["--changed", "../outside.txt"]):
            with self.subTest(extra=extra):
                run = subprocess.run(prefix + extra, capture_output=True, text=True)
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertIn("error", json.loads(run.stdout))


    def test_run_changes_scope_order_and_report_match_preview(self):
        (self.root / "data.bin").write_bytes(b"\x00\x01")
        jobs = self._changes_jobs()
        result = run_changes(self.root, jobs, "results/report.json", ["notes.txt"])
        # Same scope and processing order as the preview for the same input;
        # the unrelated branch is neither read nor recorded.
        preview = preview_changes(self.root, jobs, "results/report.json", ["notes.txt"])
        self.assertEqual([row["name"] for row in result],
                         [row["name"] for row in preview["jobs"]])
        self.assertEqual([row["name"] for row in result],
                         ["alpha", "gamma", "beta", "delta"])
        self.assertTrue(all(row["status"] == "completed" for row in result))
        report = json.loads((self.root / "results/report.json").read_text())["results"]
        self.assertEqual(report, result)
        # A narrower rerun replaces the old report content entirely.
        result = run_changes(self.root, jobs, "results/report.json", ["data.bin"])
        self.assertEqual([row["name"] for row in result], ["alpha", "gamma", "beta", "delta"])
        result = run_changes(self.root, jobs, "results/report.json", ["sales.csv"])
        self.assertEqual([row["name"] for row in result],
                         ["alpha", "gamma", "beta", "delta"])
        report = json.loads((self.root / "results/report.json").read_text())["results"]
        self.assertEqual(report, result)

    def test_run_changes_merges_paths_and_ignores_order(self):
        (self.root / "data.bin").write_bytes(b"\x00\x01")
        jobs = self._changes_jobs()
        expected = run_changes(self.root, jobs, "results/a.json",
                               ["notes.txt", "sales.csv"])
        self.assertEqual([row["name"] for row in expected],
                         ["alpha", "gamma", "beta", "delta"])
        # Argument order, repeats and aliases of the same file change nothing.
        self.assertEqual(run_changes(self.root, jobs, "results/b.json",
                                     ["sales.csv", "notes.txt"]), expected)
        self.assertEqual(run_changes(self.root, jobs, "results/b.json",
                                     ["./notes.txt", "notes.txt", "sub/../sales.csv"]),
                         expected)

    def test_run_changes_failure_blocks_downstream_independent_continues(self):
        jobs = [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "stray", "operation": "count-lines", "input": "missing.txt",
             "depends_on": ["broken"]},
            {"name": "joiner", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["broken", "healthy"]},
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
            {"name": "unrelated", "operation": "count-lines", "input": "other.txt"},
        ]
        result = run_changes(self.root, jobs, "results/report.json", ["notes.txt"])
        self.assertEqual([row["name"] for row in result],
                         ["broken", "stray", "healthy", "joiner"])
        by_name = {row["name"]: row for row in result}
        self.assertEqual(by_name["broken"]["status"], "failed")
        self.assertIn("error", by_name["broken"])
        self.assertEqual(by_name["healthy"]["status"], "completed")
        # Blocked tasks never read their input: stray's missing file stays
        # unread and blocked_by keeps declaration order.
        self.assertEqual(by_name["stray"],
                         {"name": "stray", "status": "blocked", "blocked_by": ["broken"]})
        self.assertEqual(by_name["joiner"],
                         {"name": "joiner", "status": "blocked", "blocked_by": ["broken"]})
        # The unrelated branch is out of scope: no record, no read.
        self.assertNotIn("unrelated", by_name)

    def test_run_changes_empty_or_unmatched_runs_nothing(self):
        jobs = [{"name": "ghost", "operation": "count-lines", "input": "nope.txt"}]
        for changed in ([], ["missing.txt"], ["deep/none.txt"]):
            before = {p.name for p in self.root.iterdir()}
            self.assertEqual(run_changes(self.root, jobs, "deep/new/out.json", changed), [])
            self.assertFalse((self.root / "deep").exists())
            self.assertEqual({p.name for p in self.root.iterdir()}, before)
        # An existing output file is left exactly as it was.
        marker = self._write_report("keep.json", {"results": [{"name": "kept"}]})
        before = marker.read_text()
        self.assertEqual(run_changes(self.root, jobs, "keep.json", []), [])
        self.assertEqual(marker.read_text(), before)

    def test_run_changes_invalid_inputs_and_whole_plan_validated(self):
        jobs = self._changes_jobs()
        for changed in ("notes.txt", [1], ["  "], [""], [str(ROOT / "README.md")],
                        ["../outside.txt"], ["notes.txt", "../outside.txt"]):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    run_changes(self.root, jobs, "results/report.json", changed)
        (self.root / "escape").symlink_to(ROOT / "README.md")
        with self.assertRaises(ValueError):
            run_changes(self.root, jobs, "results/report.json", ["escape"])
        # The whole plan and output are validated even when nothing matches.
        bad_jobs = jobs + [{"name": "stray", "operation": "count-lines",
                            "input": "../outside.txt"}]
        with self.assertRaises(ValueError):
            run_changes(self.root, bad_jobs, "results/report.json", [])
        bad_dep = jobs + [{"name": "stray", "operation": "count-lines",
                           "input": "notes.txt", "depends_on": ["ghost"]}]
        with self.assertRaises(ValueError):
            run_changes(self.root, bad_dep, "results/report.json", [])
        with self.assertRaises(ValueError):
            run_changes(self.root, jobs, "notes.txt", [])
        self.assertFalse((self.root / "results").exists())

    def test_run_changes_output_write_failure_raises_oserror_after_running(self):
        jobs = [{"name": "notes", "operation": "count-lines", "input": "notes.txt"}]
        (self.root / "blocker").write_text("x", encoding="utf-8")
        with self.assertRaises(OSError):
            run_changes(self.root, jobs, "blocker/out.json", ["notes.txt"])

    def test_cli_run_changed_flag(self):
        jobs = [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "downstream", "operation": "count-lines", "input": "missing.txt",
             "depends_on": ["broken"]},
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
            {"name": "unrelated", "operation": "count-lines", "input": "other.txt"},
        ]
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json",
                  "--root", str(self.root)]
        run = subprocess.run(prefix + ["--run-changed", "notes.txt",
                                       "--output", "out.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertEqual(json.loads(run.stdout),
                         {"completed": 1, "failed": 1, "blocked": 1})
        report = json.loads((self.root / "out.json").read_text())["results"]
        self.assertEqual([row["name"] for row in report],
                         ["broken", "downstream", "healthy"])
        # Repeated flags accumulate; a fully successful scope exits 0.
        (self.root / "data.bin").write_bytes(b"\x00\x01")
        plan.write_text(json.dumps({"jobs": self._changes_jobs()}))
        run = subprocess.run(prefix + ["--run-changed", "notes.txt",
                                       "--run-changed", "sales.csv",
                                       "--output", "all.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout),
                         {"completed": 4, "failed": 0, "blocked": 0})
        # An unmatched change prints zeroes, exits 0 and touches nothing.
        run = subprocess.run(prefix + ["--run-changed", "none.txt",
                                       "--output", "deep/out.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"completed": 0, "failed": 0})
        self.assertFalse((self.root / "deep").exists())
        # Conflicts and validation failures print error-only JSON, exit 2
        # and leave existing reports untouched.
        marker = self._write_report("keep.json", {"results": []})
        for extra in (["--run-changed", "notes.txt", "--only", "alpha"],
                      ["--run-changed", "notes.txt", "--preview"],
                      ["--run-changed", "notes.txt", "--retry-preview", "r.json"],
                      ["--run-changed", "notes.txt", "--retry", "r.json"],
                      ["--run-changed", "notes.txt", "--compare", "a.json", "b.json"],
                      ["--run-changed", "notes.txt", "--explain", "r.json"],
                      ["--run-changed", "notes.txt", "--history", "r.json"],
                      ["--run-changed", "notes.txt", "--changed", "notes.txt"],
                      ["--run-changed", "../outside.txt"]):
            with self.subTest(extra=extra):
                run = subprocess.run(prefix + extra, capture_output=True, text=True)
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertEqual(set(json.loads(run.stdout)), {"error"})
        self.assertEqual(marker.read_text(), json.dumps({"results": []}))

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


    def _deep_chain_jobs(self, n=2000, reverse=False):
        order = range(n - 1, -1, -1) if reverse else range(n)
        return [{"name": f"j{i}", "operation": "count-lines", "input": "notes.txt",
                 "depends_on": [f"j{i - 1}"] if i else []} for i in order]

    def test_deep_chain_validates_without_recursion_configuration(self):
        import sys
        n = 2000
        limit_before = sys.getrecursionlimit()
        # The forward-reference layout (tail first) is the deepest DFS case.
        jobs = self._deep_chain_jobs(n, reverse=True)
        preview = preview_plan(self.root, jobs, "results/report.json")
        self.assertEqual([row["name"] for row in preview["jobs"]],
                         [f"j{i}" for i in range(n)])
        result = run_plan(self.root, jobs, "results/report.json",
                          targets=[f"j{n - 1}"])
        self.assertEqual(len(result), n)
        self.assertEqual([row["name"] for row in result],
                         [f"j{i}" for i in range(n)])
        self.assertTrue(all(row["status"] == "completed" for row in result))
        self.assertEqual(len({row["name"] for row in result}), n)
        # Reordering a legal graph cannot make validation fail; the
        # plan-ordered spelling of the same graph validates too.
        ordered = self._deep_chain_jobs(n)
        self.assertEqual(len(preview_plan(self.root, ordered, "o.json")["jobs"]), n)
        self.assertEqual(sys.getrecursionlimit(), limit_before)

    def test_deep_chain_with_shared_prerequisite_and_join(self):
        n = 2000
        jobs = self._deep_chain_jobs(n)
        jobs.append({"name": "join", "operation": "sha256", "input": "sales.csv",
                     "depends_on": [f"j{n - 1}", "j0"]})
        preview = preview_plan(self.root, jobs, "results/report.json",
                               targets=["join"])
        names = [row["name"] for row in preview["jobs"]]
        self.assertEqual(names, [f"j{i}" for i in range(n)] + ["join"])
        self.assertEqual(preview["jobs"][-1]["reason"], "target")
        result = run_plan(self.root, jobs, "results/report.json")
        self.assertEqual(len(result), n + 1)
        self.assertTrue(all(row["status"] == "completed" for row in result))

    def test_cycles_rejected_everywhere_including_unselected_branches(self):
        n = 2000
        report = "results/report.json"
        run_plan(self.root, self._deep_chain_jobs(n), report)

        def cyclic(where):
            if where == "tail":
                jobs = self._deep_chain_jobs(n)
                jobs += [
                    {"name": "c1", "operation": "count-lines", "input": "notes.txt",
                     "depends_on": ["c2"]},
                    {"name": "c2", "operation": "count-lines", "input": "notes.txt",
                     "depends_on": ["c1"]},
                ]
                return jobs
            if where == "deep":
                jobs = self._deep_chain_jobs(n, reverse=True)
                jobs[500]["depends_on"] = [f"j{n - 2 - 500}", "ca"]
                jobs += [
                    {"name": "ca", "operation": "count-lines", "input": "notes.txt",
                     "depends_on": ["cb"]},
                    {"name": "cb", "operation": "count-lines", "input": "notes.txt",
                     "depends_on": ["ca"]},
                ]
                return jobs
            jobs = self._deep_chain_jobs(n)
            jobs += [
                {"name": "loose1", "operation": "count-lines", "input": "notes.txt",
                 "depends_on": ["loose2"]},
                {"name": "loose2", "operation": "count-lines", "input": "notes.txt",
                 "depends_on": ["loose1"]},
            ]
            return jobs

        for where in ("tail", "deep", "unselected"):
            jobs = cyclic(where)
            with self.subTest(where=where):
                with self.assertRaises(ValueError):
                    run_plan(self.root, jobs, "x.json")
                with self.assertRaises(ValueError):
                    preview_plan(self.root, jobs, "x.json")
                with self.assertRaises(ValueError):
                    preview_plan(self.root, jobs, "x.json", targets=["j0"])
                with self.assertRaises(ValueError):
                    compare_reports(self.root, jobs, "x.json", report, report)
                with self.assertRaises(ValueError):
                    explain_report(self.root, jobs, "x.json", report)
                with self.assertRaises(ValueError):
                    query_history(self.root, jobs, "x.json", [report])
                with self.assertRaises(ValueError):
                    query_history(self.root, jobs, "x.json", [report],
                                  targets=["j0"])
                # Validation happens before the historical report is opened.
                with self.assertRaises(ValueError):
                    preview_retry(self.root, jobs, "x.json", "missing.json")
                with self.assertRaises(ValueError):
                    run_retry(self.root, jobs, "x.json", "missing.json")
        self.assertFalse((self.root / "x.json").exists())

    def test_cli_deep_chain_succeeds_and_cycle_exits_2_cleanly(self):
        n = 2000
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": self._deep_chain_jobs(n, reverse=True)}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json",
                  "--root", str(self.root)]
        run = subprocess.run(prefix + ["--output", "deep.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout),
                         {"completed": n, "failed": 0, "blocked": 0})
        self.assertFalse(run.stderr)
        cyclic = self._deep_chain_jobs(n) + [
            {"name": "c1", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["c2"]},
            {"name": "c2", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["c1"]},
        ]
        plan.write_text(json.dumps({"jobs": cyclic}))
        for extra in ([], ["--preview"], ["--preview", "--only", "j0"],
                      ["--compare", "deep.json", "deep.json"],
                      ["--explain", "deep.json"], ["--history", "deep.json"],
                      ["--retry-preview", "deep.json"], ["--retry", "deep.json"]):
            run = subprocess.run(prefix + ["--output", "should/not.json"] + extra,
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, (extra, run.stderr))
            self.assertEqual(set(json.loads(run.stdout)), {"error"})
            # A RecursionError would surface as a traceback on stderr.
            self.assertNotIn("RecursionError", run.stderr)
            self.assertNotIn("Traceback", run.stderr)
        self.assertFalse((self.root / "should").exists())


class HistoryRetryPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "notes.txt").write_text("one\ntwo\n", encoding="utf-8")

    def _write_report(self, relative, payload):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _jobs(self):
        return [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "downstream", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["broken"]},
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
            {"name": "lonely", "operation": "count-lines", "input": "notes.txt"},
            {"name": "needs-healthy", "operation": "sha256", "input": "notes.txt",
             "depends_on": ["healthy"]},
        ]

    def test_latest_record_per_task_selects_targets_and_sources(self):
        jobs = self._jobs()
        # Oldest: broken fails (downstream blocked), healthy completes.
        self._write_report("r1.json", {"results": [
            {"name": "broken", "status": "failed", "error": "boom"},
            {"name": "downstream", "status": "blocked", "blocked_by": ["broken"]},
            {"name": "healthy", "status": "completed", "result": {"lines": 2}},
        ]})
        # Middle: broken recovered, healthy broke; downstream unrecorded here.
        self._write_report("r2.json", {"results": [
            {"name": "broken", "status": "completed", "result": {"lines": 2}},
            {"name": "healthy", "status": "failed", "error": "nope"},
        ]})
        # Newest: healthy recovered; nothing else recorded, so broken's
        # completion and downstream's old blocked state both stand.
        self._write_report("r3.json", {"results": [
            {"name": "healthy", "status": "completed", "result": {"lines": 2}},
        ]})
        result = preview_history_retry(self.root, jobs, "out.json",
                                       ["r1.json", "r2.json", "r3.json"])
        self.assertEqual(set(result), {"targets", "jobs", "sources"})
        # Only downstream survives: latest status failed/blocked per task,
        # missing later records never clear it; plan order is used.
        self.assertEqual(result["targets"], ["downstream"])
        self.assertEqual(result["sources"], [
            {"name": "downstream", "status": "blocked",
             "report": "r1.json", "index": 0}])
        # The closure matches preview_plan with those targets exactly.
        expected = preview_plan(self.root, jobs, "out.json",
                                targets=result["targets"])
        self.assertEqual(result["jobs"], expected["jobs"])

    def test_later_failure_overrides_completion_and_plan_order_rules(self):
        jobs = self._jobs()
        # Records appear shuffled; output ordering must follow the plan.
        self._write_report("old.json", {"results": [
            {"name": "healthy", "status": "completed", "result": {"lines": 2}},
            {"name": "lonely", "status": "completed", "result": {"lines": 2}},
        ]})
        self._write_report("new.json", {"results": [
            {"name": "lonely", "status": "failed", "error": "x"},
            {"name": "healthy", "status": "failed", "error": "y"},
        ]})
        result = preview_history_retry(self.root, jobs, "out.json",
                                       ["old.json", "new.json"])
        self.assertEqual(result["targets"], ["healthy", "lonely"])
        self.assertEqual(result["sources"], [
            {"name": "healthy", "status": "failed",
             "report": "new.json", "index": 1},
            {"name": "lonely", "status": "failed",
             "report": "new.json", "index": 1}])
        names = [row["name"] for row in result["jobs"]]
        # Closure includes needs-healthy? No: it is downstream, not a
        # prerequisite; unrelated branches stay out.
        self.assertNotIn("needs-healthy", names)
        self.assertNotIn("broken", names)
        self.assertNotIn("downstream", names)
        self.assertEqual(names, ["healthy", "lonely"])
        reasons = {row["name"]: row["reason"] for row in result["jobs"]}
        self.assertEqual(reasons, {"healthy": "target", "lonely": "target"})

    def test_completed_and_unrecorded_prerequisites_still_included(self):
        jobs = self._jobs()
        self._write_report("r1.json", {"results": [
            {"name": "broken", "status": "completed", "result": {"lines": 2}},
            {"name": "downstream", "status": "failed", "error": "e"},
            {"name": "healthy", "status": "completed", "result": {"lines": 2}},
        ]})
        result = preview_history_retry(self.root, jobs, "out.json", ["r1.json"])
        self.assertEqual(result["targets"], ["downstream"])
        names = [row["name"] for row in result["jobs"]]
        # broken completed in history but is a prerequisite, so it is included;
        # needs-healthy is a downstream task and must not appear.
        self.assertEqual(names, ["broken", "downstream"])
        entry = next(row for row in result["jobs"] if row["name"] == "broken")
        self.assertEqual(entry["reason"], "prerequisite")
        self.assertEqual(entry["required_by"], ["downstream"])
        target_entry = next(row for row in result["jobs"]
                            if row["name"] == "downstream")
        self.assertEqual(target_entry["reason"], "target")
        self.assertEqual(target_entry["required_by"], ["downstream"])

    def test_shared_prerequisite_appears_once(self):
        jobs = [
            {"name": "p", "operation": "shell", "input": "notes.txt"},
            {"name": "a", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["p"]},
            {"name": "b", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["p"]},
        ]
        self._write_report("r.json", {"results": [
            {"name": "a", "status": "failed", "error": "x"},
            {"name": "b", "status": "blocked", "blocked_by": ["p"]},
        ]})
        result = preview_history_retry(self.root, jobs, "out.json", ["r.json"])
        self.assertEqual(result["targets"], ["a", "b"])
        self.assertEqual([row["name"] for row in result["jobs"]], ["p", "a", "b"])
        p_entry = next(row for row in result["jobs"] if row["name"] == "p")
        self.assertEqual(p_entry["required_by"], ["a", "b"])
        self.assertEqual([s["name"] for s in result["sources"]], ["a", "b"])

    def test_no_targets_gives_three_empty_arrays(self):
        jobs = self._jobs()
        self._write_report("empty.json", {"results": []})
        self.assertEqual(preview_history_retry(self.root, jobs, "out.json",
                                               ["empty.json"]),
                         {"targets": [], "jobs": [], "sources": []})
        # All completed, or only never-targeted tasks recorded.
        self._write_report("done.json", {"results": [
            {"name": "healthy", "status": "completed", "result": {"lines": 2}},
            {"name": "lonely", "status": "completed", "result": {"lines": 2}},
        ]})
        self.assertEqual(preview_history_retry(self.root, jobs, "out.json",
                                               ["done.json"]),
                         {"targets": [], "jobs": [], "sources": []})

    def test_duplicate_report_path_keeps_separate_later_position(self):
        jobs = self._jobs()
        run_plan(self.root, jobs, "same.json")
        # same content twice: the second position overrides the first, and
        # sources point at index 1 with the path verbatim.
        result = preview_history_retry(self.root, jobs, "out.json",
                                       ["same.json", "same.json"])
        targets = result["targets"]
        self.assertEqual(targets, ["broken", "downstream"])
        self.assertTrue(all(s["report"] == "same.json" and s["index"] == 1
                            for s in result["sources"]))
        self.assertEqual([s["name"] for s in result["sources"]], targets)

    def test_report_path_verbatim_in_sources(self):
        jobs = self._jobs()
        run_plan(self.root, jobs, "results/r.json")
        result = preview_history_retry(self.root, jobs, "out.json",
                                       ["results/r.json"])
        self.assertEqual(result["sources"][0]["report"], "results/r.json")

    def test_invalid_reports_argument_raises(self):
        jobs = self._jobs()
        self._write_report("empty.json", {"results": []})
        for bad in ([], "r1.json", None, [1], [""], ["  "]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    preview_history_retry(self.root, jobs, "out.json", bad)

    def test_all_reports_validated_including_unselected_and_overridden(self):
        jobs = self._jobs()
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
            "nan-extra": b'{"results": [], "extra": NaN}',
            "inf-nested": b'{"results":[{"name":"healthy","status":"completed",'
                          b'"result":{"n":Infinity}}]}',
            "neg-inf-ignored-field": b'{"results":[{"name":"healthy","status":"completed",'
                                     b'"result":{},"weird":-Infinity}]}',
        }
        # A malformed report at any position rejects everything.
        for label, raw in strict_cases.items():
            with self.subTest(label=label):
                self._write_report(f"{label}.json", {"results": []})
                (self.root / f"{label}.json").write_bytes(raw)
                with self.assertRaises(ValueError):
                    preview_history_retry(self.root, jobs, "out.json",
                                          [f"{label}.json", "ok.json"])
                with self.assertRaises(ValueError):
                    preview_history_retry(self.root, jobs, "out.json",
                                          ["ok.json", f"{label}.json"])
        # An overridden old bad record is still validated, even though a newer
        # report would make the task a non-target.
        self._write_report("old-bad.json", {"results": [
            {"name": "healthy", "status": "completed"}]})
        self._write_report("new-good.json", {"results": [
            {"name": "healthy", "status": "completed", "result": {"lines": 2}}]})
        with self.assertRaises(ValueError):
            preview_history_retry(self.root, jobs, "out.json",
                                  ["old-bad.json", "new-good.json"])
        # Records outside the eventual target scope are checked too.
        self._write_report("out-of-scope.json", {"results": [
            {"name": "lonely", "status": "failed"}]})
        with self.assertRaises(ValueError):
            preview_history_retry(self.root, jobs, "out.json",
                                  ["out-of-scope.json"])
        # Path, existence and symlink-boundary rules match query_history.
        for bad in ("missing.json", "../outside.json", str(ROOT / "README.md")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    preview_history_retry(self.root, jobs, "out.json", [bad])
        (self.root / "out-link.json").symlink_to(ROOT / "README.md")
        with self.assertRaises(ValueError):
            preview_history_retry(self.root, jobs, "out.json", ["out-link.json"])

    def test_legal_numbers_accepted_including_beyond_float_range(self):
        jobs = self._jobs()
        (self.root / "big.json").write_bytes(
            b'{"results":[{"name":"healthy","status":"completed",'
            b'"result":{"n":1e400}},{"name":"lonely","status":"failed",'
            b'"error":"NaN Infinity -Infinity as strings are fine"}]}')
        result = preview_history_retry(self.root, jobs, "out.json", ["big.json"])
        self.assertEqual(result["targets"], ["lonely"])
        self.assertEqual(result["sources"], [
            {"name": "lonely", "status": "failed",
             "report": "big.json", "index": 0}])

    def test_whole_plan_and_output_validated_first_without_touching_files(self):
        self._write_report("r.json", {"results": []})
        bad_input = [
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
            {"name": "stray", "operation": "count-lines", "input": "../outside.txt"},
        ]
        with self.assertRaises(ValueError):
            preview_history_retry(self.root, bad_input, "out.json", ["r.json"])
        bad_dep = [
            {"name": "a", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["b"]},
            {"name": "b", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["a"]},
        ]
        with self.assertRaises(ValueError):
            preview_history_retry(self.root, bad_dep, "out.json", ["r.json"])
        good = [{"name": "notes", "operation": "count-lines", "input": "notes.txt"}]
        with self.assertRaises(ValueError):
            preview_history_retry(self.root, good, "notes.txt", ["r.json"])
        # Plan failure precedes report opening: a missing report is not the
        # error that surfaces first; either way it is a ValueError with no
        # partial result.
        with self.assertRaises(ValueError):
            preview_history_retry(self.root, bad_dep, "out.json", ["missing.json"])
        before = {p.name for p in self.root.iterdir()}
        jobs = [{"name": "missing", "operation": "count-lines", "input": "nope.txt"}]
        self.assertEqual(preview_history_retry(self.root, jobs,
                                               "deep/new/out.json", ["r.json"]),
                         {"targets": [], "jobs": [], "sources": []})
        self.assertFalse((self.root / "deep").exists())
        self.assertEqual({p.name for p in self.root.iterdir()}, before)

    def test_report_may_equal_output_and_is_not_modified(self):
        jobs = self._jobs()
        run_plan(self.root, jobs, "same.json")
        before = (self.root / "same.json").read_text()
        result = preview_history_retry(self.root, jobs, "same.json",
                                       ["same.json", "same.json"])
        self.assertEqual(result["targets"], ["broken", "downstream"])
        self.assertEqual((self.root / "same.json").read_text(), before)

    def test_cli_history_retry_flag(self):
        jobs = self._jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json",
                  "--root", str(self.root)]
        run_plan(self.root, jobs, "r1.json")
        self._write_report("r2.json", {"results": [
            {"name": "broken", "status": "completed", "result": {"lines": 2}},
            {"name": "downstream", "status": "completed",
             "result": {"lines": 2}},
        ]})
        # Failed/blocked records still give exit 0.
        run = subprocess.run(prefix + ["--history-retry", "r1.json",
                                       "--history-retry", "r2.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        payload = json.loads(run.stdout)
        self.assertEqual(set(payload), {"targets", "jobs", "sources"})
        # r2 recovered the r1 failure branch, so nothing is targeted.
        self.assertEqual(payload, {"targets": [], "jobs": [], "sources": []})
        # An old failure with no later record is still selected.
        run = subprocess.run(prefix + ["--history-retry", "r1.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        payload = json.loads(run.stdout)
        self.assertEqual(payload["targets"], ["broken", "downstream"])
        self.assertEqual(payload["sources"], [
            {"name": "broken", "status": "failed",
             "report": "r1.json", "index": 0},
            {"name": "downstream", "status": "blocked",
             "report": "r1.json", "index": 0}])
        self.assertEqual([row["name"] for row in payload["jobs"]],
                         ["broken", "downstream"])
        # Empty results never create the default output.
        self._write_report("empty.json", {"results": []})
        run = subprocess.run(prefix + ["--history-retry", "empty.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout),
                         {"targets": [], "jobs": [], "sources": []})
        self.assertFalse((self.root / ".results").exists())

    def test_cli_history_retry_conflicts_and_errors(self):
        jobs = self._jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json",
                  "--root", str(self.root)]
        self._write_report("r.json", {"results": []})
        marker = self.root / "keep.json"
        marker.write_text("KEEP", encoding="utf-8")
        for extra in (["--history-retry", "r.json", "--only", "healthy"],
                      ["--history-retry", "r.json", "--record-reasons"],
                      ["--history-retry", "r.json", "--preview"],
                      ["--history-retry", "r.json", "--retry-preview", "r.json"],
                      ["--history-retry", "r.json", "--retry", "r.json"],
                      ["--history-retry", "r.json",
                       "--compare", "r.json", "r.json"],
                      ["--history-retry", "r.json", "--explain", "r.json"],
                      ["--history-retry", "r.json", "--history", "r.json"],
                      ["--history-retry", "r.json", "--execution", "r.json"],
                      ["--history-retry", "r.json", "--changed", "notes.txt"],
                      ["--history-retry", "r.json", "--run-changed", "notes.txt"],
                      ["--history-retry", "missing.json"],
                      ["--history-retry", "../outside.json"],
                      ["--history-retry", "r.json", "--output", "notes.txt"],
                      ["--history-retry", "r.json", "--output", "plan.json"]):
            run = subprocess.run(prefix + extra, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, (extra, run.stderr))
            self.assertEqual(set(json.loads(run.stdout)), {"error"})
        self.assertEqual(marker.read_text(), "KEEP")
        # A bad plan is rejected before any report is opened.
        bad = [{"name": "a", "operation": "count-lines", "input": "notes.txt",
                "depends_on": ["x"]}]
        plan.write_text(json.dumps({"jobs": bad}))
        run = subprocess.run(prefix + ["--history-retry", "missing.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertEqual(set(json.loads(run.stdout)), {"error"})
        self.assertEqual(marker.read_text(), "KEEP")
        # A NaN-bearing report rejects at the CLI with error-only JSON.
        plan.write_text(json.dumps({"jobs": jobs}))
        (self.root / "nan.json").write_bytes(
            b'{"results": [], "x": NaN}')
        run = subprocess.run(prefix + ["--history-retry", "nan.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertEqual(set(json.loads(run.stdout)), {"error"})


class RunHistoryRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "notes.txt").write_text("one\ntwo\n", encoding="utf-8")
        (self.root / "sales.csv").write_text("item,count\nbook,2\n", encoding="utf-8")

    def _write_report(self, relative, payload, raw=None):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if raw is not None:
            path.write_bytes(raw)
        else:
            path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _jobs(self):
        return [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "downstream", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["broken"]},
            {"name": "grandchild", "operation": "sha256", "input": "notes.txt",
             "depends_on": ["downstream"]},
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
            {"name": "needs-healthy", "operation": "sha256", "input": "sales.csv",
             "depends_on": ["healthy"]},
            {"name": "lonely", "operation": "csv-summary", "input": "sales.csv"},
        ]

    def test_scope_matches_preview_and_writes_only_this_run(self):
        jobs = self._jobs()
        self._write_report("r1.json", {"results": [
            {"name": "broken", "status": "failed", "error": "old boom"},
            {"name": "downstream", "status": "blocked",
             "blocked_by": ["broken"]},
            {"name": "grandchild", "status": "blocked",
             "blocked_by": ["downstream"]},
            {"name": "healthy", "status": "completed", "result": {"lines": 99}},
            {"name": "needs-healthy", "status": "completed",
             "result": {"columns": ["x"], "rows": 1}},
            {"name": "lonely", "status": "completed",
             "result": {"columns": ["x"], "rows": 1}},
        ]})
        self._write_report("r2.json", {"results": [
            {"name": "healthy", "status": "failed", "error": "transient"},
        ]})
        preview = preview_history_retry(self.root, jobs, "out.json",
                                        ["r1.json", "r2.json"])
        result = run_history_retry(self.root, jobs, "out.json",
                                   ["r1.json", "r2.json"])
        self.assertEqual([row["name"] for row in result],
                         [row["name"] for row in preview["jobs"]])
        self.assertEqual([row["name"] for row in result],
                         ["broken", "downstream", "grandchild", "healthy"])
        by_name = {row["name"]: row for row in result}
        # This run's own failure text, never the historical error string.
        self.assertEqual(by_name["broken"]["status"], "failed")
        self.assertEqual(by_name["broken"]["error"], "unsupported operation")
        self.assertEqual(by_name["downstream"],
                         {"name": "downstream", "status": "blocked",
                          "blocked_by": ["broken"]})
        self.assertEqual(by_name["grandchild"],
                         {"name": "grandchild", "status": "blocked",
                          "blocked_by": ["downstream"]})
        # healthy failed in the newest report but succeeds now; it is a
        # target, so this run completes it.
        self.assertEqual(by_name["healthy"]["status"], "completed")
        self.assertEqual(by_name["healthy"]["result"], {"lines": 2})
        # Unrelated branches are neither read nor recorded.
        self.assertNotIn("needs-healthy", by_name)
        self.assertNotIn("lonely", by_name)
        written = json.loads((self.root / "out.json").read_text())
        self.assertEqual(set(written), {"results"})
        self.assertEqual(written["results"], result)
        # The history reports are untouched.
        self.assertEqual(json.loads((self.root / "r1.json").read_text())["results"][0],
                         {"name": "broken", "status": "failed", "error": "old boom"})

    def test_missing_later_records_keep_old_status_and_overrides_apply(self):
        jobs = self._jobs()
        # r1: broken fails, downstream blocked, healthy completes.
        self._write_report("r1.json", {"results": [
            {"name": "broken", "status": "failed", "error": "x"},
            {"name": "downstream", "status": "blocked",
             "blocked_by": ["broken"]},
            {"name": "healthy", "status": "completed", "result": {"lines": 2}},
        ]})
        # r2: broken recovered; downstream omitted, so its old blocked stands.
        self._write_report("r2.json", {"results": [
            {"name": "broken", "status": "completed", "result": {"lines": 2}},
        ]})
        # r3: healthy's record only; nothing changes for broken/downstream.
        self._write_report("r3.json", {"results": [
            {"name": "healthy", "status": "completed", "result": {"lines": 2}},
        ]})
        result = run_history_retry(self.root, jobs, "out.json",
                                   ["r1.json", "r2.json", "r3.json"])
        # Only downstream survives as a target; broken is re-executed as its
        # prerequisite (and fails again on this run), healthy is out of scope.
        self.assertEqual([row["name"] for row in result],
                         ["broken", "downstream"])
        self.assertEqual([row["status"] for row in result],
                         ["failed", "blocked"])

    def test_duplicate_report_path_keeps_separate_later_position(self):
        jobs = self._jobs()
        run_plan(self.root, jobs, "same.json")
        # First position (same content) would target broken/downstream; the
        # second identical position overrides at index 1 with no change in
        # targets, and sources ordering is the preview's concern.
        result = run_history_retry(self.root, jobs, "out.json",
                                   ["same.json", "same.json"])
        self.assertEqual([row["name"] for row in result],
                         ["broken", "downstream", "grandchild"])
        preview = preview_history_retry(self.root, jobs, "out.json",
                                        ["same.json", "same.json"])
        self.assertTrue(all(s["index"] == 1 and s["report"] == "same.json"
                            for s in preview["sources"]))

    def test_dependency_decisions_use_only_this_run_results(self):
        jobs = [
            {"name": "base", "operation": "count-lines", "input": "base.txt"},
            {"name": "target", "operation": "sha256", "input": "notes.txt",
             "depends_on": ["base"]},
        ]
        (self.root / "base.txt").write_text("a\nb\n", encoding="utf-8")
        # Historically base completed and target failed; now base's input is
        # gone, so base fails this run and blocks the target.
        self._write_report("r.json", {"results": [
            {"name": "base", "status": "completed", "result": {"lines": 2}},
            {"name": "target", "status": "failed", "error": "old"},
        ]})
        (self.root / "base.txt").unlink()
        result = run_history_retry(self.root, jobs, "out.json", ["r.json"])
        self.assertEqual([row["name"] for row in result], ["base", "target"])
        self.assertEqual(result[0]["status"], "failed")
        self.assertEqual(result[1],
                         {"name": "target", "status": "blocked",
                          "blocked_by": ["base"]})
        # Conversely, a historically failed base that now succeeds lets the
        # blocked downstream complete.
        (self.root / "base.txt").write_text("a\nb\n", encoding="utf-8")
        self._write_report("r2.json", {"results": [
            {"name": "base", "status": "failed", "error": "old"},
            {"name": "target", "status": "blocked",
             "blocked_by": ["base"]},
        ]})
        result = run_history_retry(self.root, jobs, "out.json", ["r2.json"])
        self.assertEqual([row["status"] for row in result],
                         ["completed", "completed"])

    def test_blocked_task_never_reads_its_input(self):
        jobs = [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "stray", "operation": "count-lines", "input": "missing.txt",
             "depends_on": ["broken"]},
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
        ]
        self._write_report("r.json", {"results": [
            {"name": "broken", "status": "failed", "error": "x"},
            {"name": "stray", "status": "failed", "error": "old missing read"},
            {"name": "healthy", "status": "failed", "error": "transient"},
        ]})
        result = run_history_retry(self.root, jobs, "out.json", ["r.json"])
        self.assertEqual([row["name"] for row in result],
                         ["broken", "stray", "healthy"])
        by_name = {row["name"]: row for row in result}
        self.assertEqual(by_name["stray"],
                         {"name": "stray", "status": "blocked",
                          "blocked_by": ["broken"]})
        self.assertEqual(by_name["healthy"]["status"], "completed")

    def test_no_targets_reads_nothing_creates_nothing_and_keeps_output(self):
        # Every input is missing, yet with no targets nothing is read.
        jobs = [{"name": "ghost", "operation": "count-lines", "input": "nope.txt"},
                {"name": "other", "operation": "sha256", "input": "absent.bin"}]
        self._write_report("empty.json", {"results": []})
        self._write_report("done.json", {"results": [
            {"name": "ghost", "status": "completed", "result": {"lines": 1}},
            {"name": "other", "status": "completed", "result": {"sha256": "x"}},
        ]})
        for reports in (["empty.json"], ["done.json", "empty.json"]):
            before = {p.name for p in self.root.iterdir()}
            self.assertEqual(run_history_retry(self.root, jobs,
                                               "deep/new/out.json", reports), [])
            self.assertFalse((self.root / "deep").exists())
            self.assertEqual({p.name for p in self.root.iterdir()}, before)
        marker = self._write_report("keep.json", {"results": [{"name": "kept"}]})
        marker_text = marker.read_text()
        self.assertEqual(run_history_retry(self.root, jobs, "keep.json",
                                           ["done.json"]), [])
        self.assertEqual(marker.read_text(), marker_text)

    def test_output_may_equal_a_history_report_and_uses_pre_overwrite_content(self):
        jobs = self._jobs()
        run_plan(self.root, jobs, "same.json")
        old = json.loads((self.root / "same.json").read_text())["results"]
        result = run_history_retry(self.root, jobs, "same.json", ["same.json"])
        # Targets came from the pre-overwrite content; the file now holds
        # only this run's records.
        self.assertEqual([row["name"] for row in result],
                         ["broken", "downstream", "grandchild"])
        doc = json.loads((self.root / "same.json").read_text())
        self.assertEqual(doc["results"], result)
        self.assertNotIn("execution", doc)
        self.assertEqual(len(old), 6)

    def test_alias_of_history_report_equal_to_output_resolves_same_file(self):
        jobs = self._jobs()
        run_plan(self.root, jobs, "same.json")
        result = run_history_retry(self.root, jobs, "sub/../same.json",
                                   ["./same.json"])
        self.assertEqual([row["name"] for row in result],
                         ["broken", "downstream", "grandchild"])
        self.assertEqual(json.loads((self.root / "same.json").read_text())["results"],
                         result)

    def test_record_reasons_records_retry_execution_without_sources(self):
        jobs = self._jobs()
        run_plan(self.root, jobs, "r1.json")
        self._write_report("r2.json", {"results": [
            {"name": "broken", "status": "completed", "result": {"lines": 2}},
        ]})
        reports = ["r1.json", "r2.json"]
        result = run_history_retry(self.root, jobs, "out.json", reports,
                                   record_reasons=True)
        doc = json.loads((self.root / "out.json").read_text())
        self.assertEqual(set(doc), {"results", "execution"})
        execution = doc["execution"]
        self.assertEqual(set(execution), {"mode", "targets", "jobs"})
        preview = preview_history_retry(self.root, jobs, "out.json", reports)
        self.assertEqual(execution["mode"], "retry")
        self.assertEqual(execution["targets"], preview["targets"])
        self.assertEqual(execution["targets"], ["downstream", "grandchild"])
        self.assertEqual(execution["jobs"], preview["jobs"])
        self.assertNotIn("sources", execution)
        self.assertEqual([job["name"] for job in execution["jobs"]],
                         [row["name"] for row in result])
        # The existing reasons query reads it back exactly.
        self.assertEqual(query_execution(self.root, jobs, "out.json", "out.json"),
                         {"execution": execution})
        # And without the flag the report has no execution object.
        run_history_retry(self.root, jobs, "plain.json", reports)
        self.assertEqual(set(json.loads(
            (self.root / "plain.json").read_text())), {"results"})

    def test_record_reasons_with_output_equal_to_report_uses_history_content(self):
        jobs = self._jobs()
        run_plan(self.root, jobs, "same.json")
        preview = preview_history_retry(self.root, jobs, "same.json",
                                        ["same.json", "same.json"])
        run_history_retry(self.root, jobs, "same.json",
                          ["same.json", "same.json"], record_reasons=True)
        execution = json.loads(
            (self.root / "same.json").read_text())["execution"]
        self.assertEqual(execution["mode"], "retry")
        self.assertEqual(execution["targets"], preview["targets"])
        self.assertEqual(execution["jobs"], preview["jobs"])

    def test_record_reasons_off_by_default_and_empty_scope_writes_nothing(self):
        jobs = self._jobs()
        self._write_report("empty.json", {"results": []})
        before = {p.name for p in self.root.iterdir()}
        self.assertEqual(run_history_retry(self.root, jobs, "deep/out.json",
                                           ["empty.json"], record_reasons=True),
                         [])
        self.assertFalse((self.root / "deep").exists())
        self.assertEqual({p.name for p in self.root.iterdir()}, before)

    def test_non_bool_record_reasons_raises_value_error(self):
        jobs = self._jobs()
        self._write_report("r.json", {"results": []})
        for bad in (0, 1, "true", None, [], {}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    run_history_retry(self.root, jobs, "x.json", ["r.json"],
                                      record_reasons=bad)
        self.assertFalse((self.root / "x.json").exists())

    def test_invalid_reports_argument_raises(self):
        jobs = self._jobs()
        self._write_report("empty.json", {"results": []})
        for bad in ([], "r1.json", None, [1], [""], ["  "]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    run_history_retry(self.root, jobs, "out.json", bad)

    def test_all_reports_validated_strictly_and_files_preserved(self):
        jobs = self._jobs()
        self._write_report("ok.json", {"results": []})
        marker = self._write_report("keep.json", {"results": []})
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
            "nan-extra": b'{"results": [], "extra": NaN}',
        }
        for label, raw in strict_cases.items():
            with self.subTest(label=label):
                self._write_report(f"{label}.json", None, raw=raw)
                with self.assertRaises(ValueError):
                    run_history_retry(self.root, jobs, "keep.json",
                                      [f"{label}.json", "ok.json"])
                with self.assertRaises(ValueError):
                    run_history_retry(self.root, jobs, "keep.json",
                                      ["ok.json", f"{label}.json"])
        # An overridden old bad record is validated even though a newer
        # report makes the task a non-target.
        self._write_report("old-bad.json", {"results": [
            {"name": "healthy", "status": "completed"}]})
        self._write_report("new-good.json", {"results": [
            {"name": "healthy", "status": "completed", "result": {"lines": 2}}]})
        with self.assertRaises(ValueError):
            run_history_retry(self.root, jobs, "keep.json",
                              ["old-bad.json", "new-good.json"])
        # Records outside the eventual scope are checked too.
        self._write_report("out-of-scope.json", {"results": [
            {"name": "lonely", "status": "failed"}]})
        with self.assertRaises(ValueError):
            run_history_retry(self.root, jobs, "keep.json",
                              ["out-of-scope.json"])
        # Path, existence and symlink-boundary rules.
        for bad in ("missing.json", "../outside.json", str(ROOT / "README.md")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    run_history_retry(self.root, jobs, "keep.json", [bad])
        (self.root / "out-link.json").symlink_to(ROOT / "README.md")
        with self.assertRaises(ValueError):
            run_history_retry(self.root, jobs, "keep.json", ["out-link.json"])
        self.assertEqual(marker.read_text(), json.dumps({"results": []}))

    def test_whole_plan_validated_before_any_report_is_opened(self):
        self._write_report("r.json", {"results": []})
        bad_input = [
            {"name": "healthy", "operation": "count-lines", "input": "notes.txt"},
            {"name": "stray", "operation": "count-lines", "input": "../outside.txt"},
        ]
        with self.assertRaises(ValueError):
            run_history_retry(self.root, bad_input, "out.json", ["r.json"])
        bad_dep = [
            {"name": "a", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["b"]},
            {"name": "b", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["a"]},
        ]
        with self.assertRaises(ValueError):
            run_history_retry(self.root, bad_dep, "out.json", ["r.json"])
        good = [{"name": "notes", "operation": "count-lines",
                 "input": "notes.txt"}]
        with self.assertRaises(ValueError):
            run_history_retry(self.root, good, "notes.txt", ["r.json"])
        with self.assertRaises(ValueError):
            run_history_retry(self.root, bad_dep, "out.json", ["missing.json"])
        self.assertFalse((self.root / "out.json").exists())

    def test_output_write_failure_raises_oserror_after_running(self):
        jobs = [{"name": "notes", "operation": "count-lines", "input": "notes.txt"}]
        self._write_report("r.json", {"results": [
            {"name": "notes", "status": "failed", "error": "x"}]})
        (self.root / "blocker").write_text("x", encoding="utf-8")
        with self.assertRaises(OSError):
            run_history_retry(self.root, jobs, "blocker/out.json", ["r.json"])

    def test_cli_run_history_retry_counts_scope_and_exit_codes(self):
        jobs = self._jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json",
                  "--root", str(self.root)]
        run = subprocess.run(prefix + ["--output", "r1.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 1)
        # r2 recovers the healthy branch but leaves r1's failed chain as the
        # latest record for those tasks.
        self._write_report("r2.json", {"results": [
            {"name": "healthy", "status": "completed", "result": {"lines": 2}},
            {"name": "lonely", "status": "completed",
             "result": {"columns": ["a"], "rows": 1}},
        ]})
        run = subprocess.run(prefix + ["--run-history-retry", "r1.json",
                                       "--run-history-retry", "r2.json",
                                       "--output", "new.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertEqual(json.loads(run.stdout),
                         {"completed": 0, "failed": 1, "blocked": 2})
        new = json.loads((self.root / "new.json").read_text())["results"]
        self.assertEqual([row["name"] for row in new],
                         ["broken", "downstream", "grandchild"])
        old = json.loads((self.root / "r1.json").read_text())["results"]
        self.assertEqual(len(old), 6)
        # A scope where the historical failure now succeeds exits 0; the
        # blocked summary field is absent without dependencies in scope.
        self._write_report("one.json", {"results": [
            {"name": "healthy", "status": "failed", "error": "transient"}]})
        run = subprocess.run(prefix + ["--run-history-retry", "one.json",
                                       "--output", "one-out.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"completed": 1, "failed": 0})

    def test_cli_run_history_retry_empty_scope_outputs_zeroes(self):
        jobs = self._jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json",
                  "--root", str(self.root)]
        self._write_report("empty.json", {"results": []})
        run = subprocess.run(prefix + ["--run-history-retry", "empty.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"completed": 0, "failed": 0})
        self.assertFalse((self.root / ".results").exists())

    def test_cli_run_history_retry_record_reasons(self):
        jobs = self._jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json",
                  "--root", str(self.root)]
        run = subprocess.run(prefix + ["--output", "r1.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 1)
        run = subprocess.run(prefix + ["--run-history-retry", "r1.json",
                                       "--output", "new.json",
                                       "--record-reasons"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stderr)
        doc = json.loads((self.root / "new.json").read_text())
        self.assertEqual(doc["execution"]["mode"], "retry")
        self.assertEqual(doc["execution"]["targets"],
                         ["broken", "downstream", "grandchild"])

    def test_cli_run_history_retry_conflicts_and_errors(self):
        jobs = self._jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json",
                  "--root", str(self.root)]
        self._write_report("r.json", {"results": []})
        marker = self.root / "keep.json"
        marker.write_text("KEEP", encoding="utf-8")
        for extra in (["--run-history-retry", "r.json", "--only", "healthy"],
                      ["--run-history-retry", "r.json", "--preview"],
                      ["--run-history-retry", "r.json",
                       "--retry-preview", "r.json"],
                      ["--run-history-retry", "r.json", "--retry", "r.json"],
                      ["--run-history-retry", "r.json",
                       "--compare", "r.json", "r.json"],
                      ["--run-history-retry", "r.json", "--explain", "r.json"],
                      ["--run-history-retry", "r.json", "--history", "r.json"],
                      ["--run-history-retry", "r.json",
                       "--history-retry", "r.json"],
                      ["--run-history-retry", "r.json", "--execution", "r.json"],
                      ["--run-history-retry", "r.json",
                       "--changed", "notes.txt"],
                      ["--run-history-retry", "r.json",
                       "--run-changed", "notes.txt"],
                      ["--run-history-retry", "missing.json"],
                      ["--run-history-retry", "../outside.json"],
                      ["--run-history-retry", "r.json",
                       "--output", "notes.txt"],
                      ["--run-history-retry", "r.json", "--output", "plan.json"]):
            with self.subTest(extra=extra):
                run = subprocess.run(prefix + extra, capture_output=True, text=True)
                self.assertEqual(run.returncode, 2, (extra, run.stderr))
                self.assertEqual(set(json.loads(run.stdout)), {"error"})
        self.assertEqual(marker.read_text(), "KEEP")
        # A bad plan is rejected before any report is opened.
        bad = [{"name": "a", "operation": "count-lines", "input": "notes.txt",
                "depends_on": ["x"]}]
        plan.write_text(json.dumps({"jobs": bad}))
        run = subprocess.run(prefix + ["--run-history-retry", "missing.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertEqual(set(json.loads(run.stdout)), {"error"})
        self.assertEqual(marker.read_text(), "KEEP")
        # A NaN-bearing report rejects with error-only JSON before execution.
        plan.write_text(json.dumps({"jobs": jobs}))
        (self.root / "nan.json").write_bytes(b'{"results": [], "x": NaN}')
        run = subprocess.run(prefix + ["--run-history-retry", "nan.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertEqual(set(json.loads(run.stdout)), {"error"})
        self.assertEqual(marker.read_text(), "KEEP")


class RecordReasonsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "notes.txt").write_text("one\ntwo\n", encoding="utf-8")
        (self.root / "sales.csv").write_text("item,count\nbook,2\npen,4\n", encoding="utf-8")

    def _write_report(self, relative, payload):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _jobs(self):
        return [
            {"name": "final", "operation": "sha256", "input": "notes.txt",
             "depends_on": ["mid", "indirect"]},
            {"name": "mid", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["base"]},
            {"name": "indirect", "operation": "csv-summary", "input": "sales.csv",
             "depends_on": ["base"]},
            {"name": "base", "operation": "sha256", "input": "notes.txt"},
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "down", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["broken"]},
        ]

    def _change_jobs(self):
        return [
            {"name": "alpha", "operation": "count-lines", "input": "notes.txt"},
            {"name": "beta", "operation": "sha256", "input": "data.bin",
             "depends_on": ["alpha", "gamma"]},
            {"name": "gamma", "operation": "csv-summary", "input": "sales.csv"},
            {"name": "delta", "operation": "count-lines", "input": "sales.csv",
             "depends_on": ["beta"]},
            {"name": "unrelated", "operation": "count-lines", "input": "other.txt"},
        ]

    def test_default_off_keeps_report_and_return_unchanged(self):
        jobs = self._jobs()
        result = run_plan(self.root, jobs, "r.json")
        text = (self.root / "r.json").read_text()
        self.assertEqual(text, json.dumps({"results": result}, ensure_ascii=False,
                                          indent=2) + "\n")
        self.assertEqual(set(json.loads(text)), {"results"})
        # Explicit False is identical to omitting the option.
        self.assertEqual(run_plan(self.root, jobs, "r2.json", record_reasons=False),
                         result)
        self.assertEqual((self.root / "r2.json").read_text(),
                         (self.root / "r.json").read_text())

    def test_all_mode_records_execution_aligned_with_preview_and_results(self):
        jobs = self._jobs()
        result = run_plan(self.root, jobs, "r.json", record_reasons=True)
        doc = json.loads((self.root / "r.json").read_text())
        self.assertEqual(set(doc), {"results", "execution"})
        self.assertEqual(doc["results"], result)
        execution = doc["execution"]
        self.assertEqual(set(execution), {"mode", "targets", "jobs"})
        self.assertEqual(execution["mode"], "all")
        self.assertEqual(execution["targets"], [job["name"] for job in jobs])
        preview = preview_plan(self.root, jobs, "r.json")
        self.assertEqual(execution["jobs"], preview["jobs"])
        self.assertTrue(all(job["reason"] == "all" and job["required_by"] == []
                            for job in execution["jobs"]))
        # Same names, same order as results; the shared prerequisite appears once.
        self.assertEqual([job["name"] for job in execution["jobs"]],
                         [row["name"] for row in result])
        self.assertEqual(len(execution["jobs"]),
                         len({job["name"] for job in execution["jobs"]}))

    def test_only_mode_is_plan_ordered_even_when_every_job_is_selected(self):
        jobs = self._jobs()
        all_names = [job["name"] for job in jobs]
        # Targets deliberately out of plan order and covering every job:
        # the mode stays "only" and targets come back in plan order.
        result = run_plan(self.root, jobs, "r.json",
                          targets=list(reversed(all_names)), record_reasons=True)
        execution = json.loads((self.root / "r.json").read_text())["execution"]
        self.assertEqual(execution["mode"], "only")
        self.assertEqual(execution["targets"], all_names)
        preview = preview_plan(self.root, jobs, "r.json",
                               targets=list(reversed(all_names)))
        self.assertEqual(execution["jobs"], preview["jobs"])
        self.assertEqual([job["name"] for job in execution["jobs"]],
                         [row["name"] for row in result])

    def test_only_mode_reasons_required_by_and_shared_prerequisite(self):
        jobs = self._jobs()
        run_plan(self.root, jobs, "r.json", targets=["final", "mid"],
                 record_reasons=True)
        execution = json.loads((self.root / "r.json").read_text())["execution"]
        by_name = {job["name"]: job for job in execution["jobs"]}
        self.assertEqual(execution["targets"], ["final", "mid"])
        self.assertEqual([job["name"] for job in execution["jobs"]],
                         ["base", "mid", "indirect", "final"])
        self.assertEqual(by_name["base"]["reason"], "prerequisite")
        self.assertEqual(by_name["base"]["required_by"], ["final", "mid"])
        self.assertEqual(by_name["mid"]["reason"], "target")
        self.assertEqual(by_name["final"]["reason"], "target")
        self.assertEqual(by_name["indirect"]["reason"], "prerequisite")
        # Swapping target argument order changes nothing.
        run_plan(self.root, jobs, "r2.json", targets=["mid", "final"],
                 record_reasons=True)
        self.assertEqual(json.loads((self.root / "r2.json").read_text())["execution"],
                         execution)

    def test_failed_and_blocked_jobs_keep_their_selection_reasons(self):
        jobs = [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "down", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["broken"]},
            {"name": "ok", "operation": "count-lines", "input": "notes.txt"},
        ]
        result = run_plan(self.root, jobs, "r.json", targets=["down", "ok"],
                          record_reasons=True)
        self.assertEqual([row["status"] for row in result],
                         ["failed", "blocked", "completed"])
        execution = json.loads((self.root / "r.json").read_text())["execution"]
        by_name = {job["name"]: job for job in execution["jobs"]}
        # The blocked task is still an explicit target with its preview entry;
        # the failing prerequisite stays a prerequisite, outcome or not.
        self.assertEqual(by_name["down"]["reason"], "target")
        self.assertEqual(by_name["down"]["required_by"], ["down"])
        self.assertEqual(by_name["broken"]["reason"], "prerequisite")
        self.assertEqual(by_name["broken"]["required_by"], ["down"])
        self.assertNotIn("status", by_name["down"])
        self.assertEqual([job["name"] for job in execution["jobs"]],
                         [row["name"] for row in result])

    def test_retry_mode_aligns_with_preview_retry(self):
        jobs = self._jobs()
        run_plan(self.root, jobs, "old.json")
        result = run_retry(self.root, jobs, "new.json", "old.json",
                           record_reasons=True)
        execution = json.loads((self.root / "new.json").read_text())["execution"]
        self.assertEqual(execution["mode"], "retry")
        preview = preview_retry(self.root, jobs, "new.json", "old.json")
        self.assertEqual(execution["targets"], preview["targets"])
        self.assertEqual(execution["targets"], ["broken", "down"])
        self.assertEqual(execution["jobs"], preview["jobs"])
        self.assertEqual([job["name"] for job in execution["jobs"]],
                         [row["name"] for row in result])
        # The blocked task recorded this run still carries its target reason.
        down_job = next(job for job in execution["jobs"] if job["name"] == "down")
        self.assertEqual(down_job["reason"], "target")
        # The old report keeps no execution object.
        self.assertEqual(set(json.loads((self.root / "old.json").read_text())),
                         {"results"})

    def test_retry_report_equal_to_output_uses_pre_overwrite_content(self):
        jobs = [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "down", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["broken"]},
            {"name": "ok", "operation": "count-lines", "input": "notes.txt"},
        ]
        run_plan(self.root, jobs, "same.json")
        before_preview = preview_retry(self.root, jobs, "same.json", "same.json")
        result = run_retry(self.root, jobs, "same.json", "same.json",
                           record_reasons=True)
        doc = json.loads((self.root / "same.json").read_text())
        self.assertEqual(doc["execution"]["mode"], "retry")
        self.assertEqual(doc["execution"]["targets"], before_preview["targets"])
        self.assertEqual(doc["execution"]["jobs"], before_preview["jobs"])
        self.assertEqual([job["name"] for job in doc["execution"]["jobs"]],
                         [row["name"] for row in result])

    def test_changes_mode_aligns_with_preview_changes(self):
        (self.root / "data.bin").write_bytes(b"\x00\x01")
        jobs = self._change_jobs()
        result = run_changes(self.root, jobs, "r.json", ["notes.txt"],
                             record_reasons=True)
        execution = json.loads((self.root / "r.json").read_text())["execution"]
        self.assertEqual(execution["mode"], "changes")
        preview = preview_changes(self.root, jobs, "r.json", ["notes.txt"])
        self.assertEqual(execution["targets"], preview["targets"])
        self.assertEqual(execution["jobs"], preview["jobs"])
        self.assertEqual([job["name"] for job in execution["jobs"]],
                         [row["name"] for row in result])
        by_name = {job["name"]: job for job in execution["jobs"]}
        self.assertEqual(by_name["alpha"]["triggered_by"], ["alpha"])
        self.assertEqual(by_name["gamma"]["triggered_by"], [])

    def test_changes_mode_target_order_and_path_aliases_change_nothing(self):
        (self.root / "data.bin").write_bytes(b"\x00\x01")
        jobs = self._change_jobs()
        first = run_changes(self.root, jobs, "a.json", ["notes.txt", "sales.csv"],
                            record_reasons=True)
        first_exec = json.loads((self.root / "a.json").read_text())["execution"]
        self.assertEqual(first_exec["targets"],
                         ["alpha", "beta", "gamma", "delta"])
        for paths in (["sales.csv", "notes.txt"],
                      ["./notes.txt", "notes.txt", "sub/../sales.csv"]):
            result = run_changes(self.root, jobs, "b.json", paths,
                                 record_reasons=True)
            doc = json.loads((self.root / "b.json").read_text())
            self.assertEqual(result, first)
            self.assertEqual(doc["execution"], first_exec)

    def test_empty_retry_and_changes_scopes_write_nothing_even_with_flag(self):
        jobs = self._jobs()
        self._write_report("empty.json", {"results": []})
        before = {p.name for p in self.root.iterdir()}
        self.assertEqual(run_retry(self.root, jobs, "deep/new.json", "empty.json",
                                   record_reasons=True), [])
        self.assertFalse((self.root / "deep").exists())
        self.assertEqual({p.name for p in self.root.iterdir()}, before)
        self.assertEqual(run_changes(self.root, jobs, "deep/new.json", ["nope.txt"],
                                     record_reasons=True), [])
        self.assertFalse((self.root / "deep").exists())
        self.assertEqual({p.name for p in self.root.iterdir()}, before)
        # An existing report is left byte-for-byte untouched.
        marker = self._write_report("keep.json", {"results": [{"name": "kept"}]})
        marker_text = marker.read_text()
        self.assertEqual(run_changes(self.root, jobs, "keep.json", [],
                                     record_reasons=True), [])
        self.assertEqual(marker.read_text(), marker_text)

    def test_non_bool_record_reasons_raises_value_error(self):
        jobs = self._jobs()
        run_plan(self.root, jobs, "old.json")
        for bad in (0, 1, "true", "false", None, [], {}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    run_plan(self.root, jobs, "x.json", record_reasons=bad)
                with self.assertRaises(ValueError):
                    run_retry(self.root, jobs, "x.json", "old.json",
                              record_reasons=bad)
                with self.assertRaises(ValueError):
                    run_changes(self.root, jobs, "x.json", ["notes.txt"],
                                record_reasons=bad)
        self.assertFalse((self.root / "x.json").exists())

    def test_invalid_plan_or_targets_still_raise_with_flag_and_write_nothing(self):
        jobs = self._jobs()
        with self.assertRaises(ValueError):
            run_plan(self.root, jobs, "r.json", targets=["ghost"],
                     record_reasons=True)
        bad = jobs + [{"name": "stray", "operation": "count-lines",
                       "input": "../outside.txt"}]
        with self.assertRaises(ValueError):
            run_plan(self.root, bad, "r.json", record_reasons=True)
        with self.assertRaises(ValueError):
            run_changes(self.root, bad, "r.json", [], record_reasons=True)
        self.assertFalse((self.root / "r.json").exists())

    def test_report_readers_treat_execution_as_an_extra_field(self):
        from job_planner import compare_reports, explain_report, query_history
        jobs = [
            {"name": "broken", "operation": "shell", "input": "notes.txt"},
            {"name": "down", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["broken"]},
            {"name": "ok", "operation": "count-lines", "input": "notes.txt"},
        ]
        run_plan(self.root, jobs, "r.json", record_reasons=True)
        # Every read-only entry point handles the execution-bearing report
        # exactly as before.
        compared = compare_reports(self.root, jobs, "out.json", "r.json", "r.json")
        self.assertTrue(all(row["change"] == "unchanged" for row in compared["jobs"]))
        explained = explain_report(self.root, jobs, "out.json", "r.json")
        self.assertEqual([row["name"] for row in explained["jobs"]],
                         ["broken", "down"])
        history = query_history(self.root, jobs, "out.json", ["r.json"])
        self.assertEqual([row["name"] for row in history["jobs"]],
                         ["broken", "down", "ok"])
        retry = preview_retry(self.root, jobs, "out.json", "r.json")
        self.assertEqual(retry["targets"], ["broken", "down"])
        # A retry without the flag strips execution back out of the new report.
        run_retry(self.root, jobs, "r2.json", "r.json")
        self.assertEqual(set(json.loads((self.root / "r2.json").read_text())),
                         {"results"})

    def test_recorded_execution_is_deterministic(self):
        jobs = self._jobs()
        run_plan(self.root, jobs, "a.json", targets=["final"], record_reasons=True)
        run_plan(self.root, jobs, "b.json", targets=["final"], record_reasons=True)
        exec_a = json.loads((self.root / "a.json").read_text())["execution"]
        exec_b = json.loads((self.root / "b.json").read_text())["execution"]
        self.assertEqual(exec_a, exec_b)

    def test_cli_record_reasons_flag_all_modes(self):
        jobs = self._jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json",
                  "--root", str(self.root)]
        run = subprocess.run(prefix + ["--output", "all.json", "--record-reasons"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertEqual(json.loads(run.stdout),
                         {"completed": 4, "failed": 1, "blocked": 1})
        doc = json.loads((self.root / "all.json").read_text())
        self.assertEqual(doc["execution"]["mode"], "all")
        self.assertEqual(doc["execution"]["targets"],
                         [job["name"] for job in jobs])
        # --only keeps the normal summary while recording mode "only".
        run = subprocess.run(prefix + ["--output", "only.json",
                                       "--only", "final", "--record-reasons"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout),
                         {"completed": 4, "failed": 0, "blocked": 0})
        doc = json.loads((self.root / "only.json").read_text())
        self.assertEqual(doc["execution"]["mode"], "only")
        self.assertEqual(doc["execution"]["targets"], ["final"])
        # Retry mode.
        run = subprocess.run(prefix + ["--retry", "all.json", "--output",
                                       "retry.json", "--record-reasons"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertEqual(json.loads((self.root / "retry.json").read_text())
                         ["execution"]["mode"], "retry")
        # Change mode.
        run = subprocess.run(prefix + ["--run-changed", "notes.txt", "--output",
                                       "chg.json", "--record-reasons"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertEqual(json.loads((self.root / "chg.json").read_text())
                         ["execution"]["mode"], "changes")
        # Without the flag none of these reports carry execution.
        run = subprocess.run(prefix + ["--output", "plain.json"],
                             capture_output=True, text=True)
        self.assertEqual(set(json.loads((self.root / "plain.json").read_text())),
                         {"results"})

    def test_cli_record_reasons_conflicts_exit_2_error_only(self):
        jobs = self._jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        self._write_report("r.json", {"results": []})
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json",
                  "--root", str(self.root)]
        marker = self._write_report("keep.json", {"results": []})
        marker_text = marker.read_text()
        for extra in (["--record-reasons", "--preview"],
                      ["--record-reasons", "--retry-preview", "r.json"],
                      ["--record-reasons", "--changed", "notes.txt"],
                      ["--record-reasons", "--compare", "r.json", "r.json"],
                      ["--record-reasons", "--explain", "r.json"],
                      ["--record-reasons", "--history", "r.json"],
                      ["--preview", "--record-reasons"],
                      ["--compare", "r.json", "r.json", "--record-reasons"]):
            with self.subTest(extra=extra):
                run = subprocess.run(prefix + extra, capture_output=True, text=True)
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertEqual(set(json.loads(run.stdout)), {"error"})
        self.assertEqual(marker.read_text(), marker_text)

    def test_cli_record_reasons_empty_scopes_output_zeroes_and_touch_nothing(self):
        jobs = self._jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        self._write_report("empty.json", {"results": []})
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json",
                  "--root", str(self.root)]
        run = subprocess.run(prefix + ["--retry", "empty.json", "--record-reasons"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"completed": 0, "failed": 0})
        self.assertFalse((self.root / ".results").exists())
        run = subprocess.run(prefix + ["--run-changed", "nope.txt",
                                       "--record-reasons"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"completed": 0, "failed": 0})
        self.assertFalse((self.root / ".results").exists())


class QueryExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "notes.txt").write_text("one\ntwo\n", encoding="utf-8")
        (self.root / "sales.csv").write_text("item,count\nbook,2\npen,4\n", encoding="utf-8")

    def _write_report(self, relative, payload):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _jobs(self):
        return [
            {"name": "base", "operation": "sha256", "input": "notes.txt"},
            {"name": "mid", "operation": "count-lines", "input": "notes.txt",
             "depends_on": ["base"]},
            {"name": "final", "operation": "csv-summary", "input": "sales.csv",
             "depends_on": ["mid"]},
        ]

    def _only_report(self):
        """A valid hand-written report carrying an ``only`` execution."""
        return {
            "results": [
                {"name": "base", "status": "completed", "result": {"bytes": 8}},
                {"name": "mid", "status": "completed", "result": {"lines": 2}},
            ],
            "execution": {
                "mode": "only",
                "targets": ["mid"],
                "jobs": [
                    {"name": "base", "depends_on": [], "reason": "prerequisite",
                     "required_by": ["mid"]},
                    {"name": "mid", "depends_on": ["base"], "reason": "target",
                     "required_by": ["mid"]},
                ],
            },
        }

    def test_round_trip_returns_exactly_the_recorded_execution(self):
        jobs = RecordReasonsTests._jobs(self)
        run_plan(self.root, jobs, "all.json", record_reasons=True)
        run_plan(self.root, jobs, "only.json", targets=["final", "mid"],
                 record_reasons=True)
        run_retry(self.root, jobs, "retry.json", "all.json", record_reasons=True)
        run_changes(self.root, jobs, "chg.json", ["notes.txt"], record_reasons=True)
        for name in ("all.json", "only.json", "retry.json", "chg.json"):
            with self.subTest(name=name):
                recorded = json.loads((self.root / name).read_text())["execution"]
                queried = query_execution(self.root, jobs, "out.json", name)
                self.assertEqual(queried, {"execution": recorded})
                # Only the execution field is returned.
                self.assertEqual(list(queried), ["execution"])

    def test_failed_and_blocked_records_do_not_rewrite_reasons(self):
        jobs = RecordReasonsTests._jobs(self)
        run_plan(self.root, jobs, "r.json", record_reasons=True)
        recorded = json.loads((self.root / "r.json").read_text())["execution"]
        statuses = {row["name"]: row["status"]
                    for row in json.loads((self.root / "r.json").read_text())["results"]}
        self.assertEqual(statuses["broken"], "failed")
        self.assertEqual(statuses["down"], "blocked")
        queried = query_execution(self.root, jobs, "out.json", "r.json")["execution"]
        self.assertEqual(queried, recorded)
        by_name = {job["name"]: job for job in queried["jobs"]}
        self.assertEqual(by_name["broken"]["reason"], "all")
        self.assertEqual(by_name["down"]["reason"], "all")

    def test_missing_field_and_empty_results_return_null(self):
        jobs = self._jobs()
        run_plan(self.root, jobs, "plain.json")
        self.assertEqual(query_execution(self.root, jobs, "out.json", "plain.json"),
                         {"execution": None})
        self._write_report("empty.json", {"results": []})
        self.assertEqual(query_execution(self.root, jobs, "out.json", "empty.json"),
                         {"execution": None})
        # Empty results yield null even when an execution object is present.
        self._write_report("empty-exec.json",
                           {"results": [], "execution": {"mode": "all",
                                                         "targets": ["base"],
                                                         "jobs": []}})
        self.assertEqual(query_execution(self.root, jobs, "out.json", "empty-exec.json"),
                         {"execution": None})

    def test_null_and_non_object_execution_raise(self):
        jobs = self._jobs()
        report = self._only_report()
        for bad in (None, "only", 3, ["mode"], True):
            with self.subTest(bad=bad):
                payload = json.loads(json.dumps(report))
                payload["execution"] = bad
                self._write_report("bad.json", payload)
                with self.assertRaises(ValueError):
                    query_execution(self.root, jobs, "out.json", "bad.json")

    def test_invalid_mode_and_target_violations_raise(self):
        jobs = self._jobs()
        report = self._only_report()
        bad_executions = []
        for mode in (None, "everything", "", 1):
            execution = json.loads(json.dumps(report["execution"]))
            if mode is None:
                del execution["mode"]
            else:
                execution["mode"] = mode
            bad_executions.append(execution)
        for targets in (None, [], "mid", [""], ["mid", "mid"], ["ghost"], [3]):
            execution = json.loads(json.dumps(report["execution"]))
            if targets is None:
                del execution["targets"]
            else:
                execution["targets"] = targets
            bad_executions.append(execution)
        for execution in bad_executions:
            with self.subTest(execution=execution):
                payload = self._only_report()
                payload["execution"] = execution
                self._write_report("bad.json", payload)
                with self.assertRaises(ValueError):
                    query_execution(self.root, jobs, "out.json", "bad.json")

    def test_jobs_must_match_results_in_name_and_order(self):
        jobs = self._jobs()
        base = self._only_report()
        variants = []
        # Missing / empty / non-list jobs.
        for value in (None, [], "jobs"):
            execution = json.loads(json.dumps(base["execution"]))
            if value is None:
                del execution["jobs"]
            else:
                execution["jobs"] = value
            variants.append(execution)
        # Swapped order, renamed entry, extra entry, missing entry, non-object.
        swapped = json.loads(json.dumps(base["execution"]))
        swapped["jobs"] = [swapped["jobs"][1], swapped["jobs"][0]]
        variants.append(swapped)
        renamed = json.loads(json.dumps(base["execution"]))
        renamed["jobs"][0]["name"] = "final"
        variants.append(renamed)
        extra = json.loads(json.dumps(base["execution"]))
        extra["jobs"].append({"name": "final", "depends_on": ["mid"],
                              "reason": "prerequisite", "required_by": ["mid"]})
        variants.append(extra)
        short = json.loads(json.dumps(base["execution"]))
        short["jobs"] = short["jobs"][:1]
        variants.append(short)
        non_object = json.loads(json.dumps(base["execution"]))
        non_object["jobs"][0] = "base"
        variants.append(non_object)
        # A target no execution job covers.
        uncovered = json.loads(json.dumps(base["execution"]))
        uncovered["targets"] = ["mid", "final"]
        variants.append(uncovered)
        for execution in variants:
            with self.subTest(execution=execution):
                payload = self._only_report()
                payload["execution"] = execution
                self._write_report("bad.json", payload)
                with self.assertRaises(ValueError):
                    query_execution(self.root, jobs, "out.json", "bad.json")

    def test_depends_on_must_match_the_current_declaration(self):
        jobs = self._jobs()
        for depends_on in ([], ["mid"], ["base", "base"], "base"):
            with self.subTest(depends_on=depends_on):
                payload = self._only_report()
                payload["execution"]["jobs"][1]["depends_on"] = depends_on
                self._write_report("bad.json", payload)
                with self.assertRaises(ValueError):
                    query_execution(self.root, jobs, "out.json", "bad.json")

    def test_reason_correspondences_are_enforced(self):
        jobs = self._jobs()
        # "only" mode: a target recorded as prerequisite, and vice versa.
        payload = self._only_report()
        payload["execution"]["jobs"][1]["reason"] = "prerequisite"
        self._write_report("bad.json", payload)
        with self.assertRaises(ValueError):
            query_execution(self.root, jobs, "out.json", "bad.json")
        payload = self._only_report()
        payload["execution"]["jobs"][0]["reason"] = "target"
        self._write_report("bad.json", payload)
        with self.assertRaises(ValueError):
            query_execution(self.root, jobs, "out.json", "bad.json")
        payload = self._only_report()
        payload["execution"]["jobs"][0]["reason"] = "all"
        self._write_report("bad.json", payload)
        with self.assertRaises(ValueError):
            query_execution(self.root, jobs, "out.json", "bad.json")

    def test_all_mode_rules(self):
        jobs = self._jobs()
        run_plan(self.root, jobs, "all.json", record_reasons=True)
        recorded = json.loads((self.root / "all.json").read_text())["execution"]
        self.assertEqual(query_execution(self.root, jobs, "out.json", "all.json"),
                         {"execution": recorded})
        results = json.loads((self.root / "all.json").read_text())["results"]
        # Targets not covering the whole plan.
        payload = {"results": results,
                   "execution": {**recorded, "targets": ["base", "mid"]}}
        self._write_report("bad.json", payload)
        with self.assertRaises(ValueError):
            query_execution(self.root, jobs, "out.json", "bad.json")
        # A non-"all" reason.
        execution = json.loads(json.dumps(recorded))
        execution["jobs"][0]["reason"] = "target"
        self._write_report("bad.json", {"results": results, "execution": execution})
        with self.assertRaises(ValueError):
            query_execution(self.root, jobs, "out.json", "bad.json")
        # A nonempty required_by.
        execution = json.loads(json.dumps(recorded))
        execution["jobs"][0]["required_by"] = ["base"]
        self._write_report("bad.json", {"results": results, "execution": execution})
        with self.assertRaises(ValueError):
            query_execution(self.root, jobs, "out.json", "bad.json")

    def test_changes_mode_requires_triggered_by_referencing_targets(self):
        jobs = RecordReasonsTests._change_jobs(self)
        (self.root / "data.bin").write_bytes(b"x")
        (self.root / "other.txt").write_text("y\n", encoding="utf-8")
        run_changes(self.root, jobs, "chg.json", ["notes.txt"], record_reasons=True)
        doc = json.loads((self.root / "chg.json").read_text())
        self.assertEqual(doc["execution"]["mode"], "changes")
        self.assertEqual(query_execution(self.root, jobs, "out.json", "chg.json"),
                         {"execution": doc["execution"]})
        results = doc["results"]
        execution = doc["execution"]
        # Missing triggered_by.
        broken = json.loads(json.dumps(execution))
        for job in broken["jobs"]:
            del job["triggered_by"]
        self._write_report("bad.json", {"results": results, "execution": broken})
        with self.assertRaises(ValueError):
            query_execution(self.root, jobs, "out.json", "bad.json")
        # triggered_by naming a non-target.
        broken = json.loads(json.dumps(execution))
        broken["jobs"][0]["triggered_by"] = ["unrelated"]
        self._write_report("bad.json", {"results": results, "execution": broken})
        with self.assertRaises(ValueError):
            query_execution(self.root, jobs, "out.json", "bad.json")

    def test_name_lists_reject_bad_types_duplicates_and_unknown_references(self):
        jobs = self._jobs()
        for value in ("mid", ["mid", "mid"], ["ghost"], [3], [""]):
            with self.subTest(value=value):
                payload = self._only_report()
                payload["execution"]["jobs"][0]["required_by"] = value
                self._write_report("bad.json", payload)
                with self.assertRaises(ValueError):
                    query_execution(self.root, jobs, "out.json", "bad.json")

    def test_extra_fields_ignored_and_recorded_values_preserved(self):
        jobs = self._jobs()
        payload = self._only_report()
        execution = payload["execution"]
        execution["recorded_at"] = "never"
        execution["jobs"][0]["triggered_by"] = ["mid"]  # extra in only mode
        execution["jobs"][0]["note"] = {"anything": True}
        # required_by lists are trusted, not recomputed: an empty list where
        # the graph would imply one, and an unusual but valid order, survive.
        execution["jobs"][0]["required_by"] = []
        payload["results"][0]["extra"] = "ignored"
        self._write_report("rich.json", payload)
        queried = query_execution(self.root, jobs, "out.json", "rich.json")
        self.assertEqual(queried, {"execution": {
            "mode": "only",
            "targets": ["mid"],
            "jobs": [
                {"name": "base", "depends_on": [], "reason": "prerequisite",
                 "required_by": []},
                {"name": "mid", "depends_on": ["base"], "reason": "target",
                 "required_by": ["mid"]},
            ]}})

    def test_partial_report_is_legal_and_report_rules_are_strict(self):
        jobs = self._jobs()
        # The hand-written only-report covers a partial run and is legal.
        self._write_report("partial.json", self._only_report())
        queried = query_execution(self.root, jobs, "out.json", "partial.json")
        self.assertEqual(queried["execution"]["mode"], "only")
        # compare-grade report violations still raise.
        for results in ([{"name": "ghost", "status": "completed", "result": {}}],
                        [{"name": "base", "status": "completed"}],
                        [{"name": "base", "status": "done"}],
                        [{"name": "base"}, {"name": "base"}]):
            with self.subTest(results=results):
                payload = self._only_report()
                payload["results"] = results
                self._write_report("bad.json", payload)
                with self.assertRaises(ValueError):
                    query_execution(self.root, jobs, "out.json", "bad.json")
        for relative in ("missing.json", str(self.root / "elsewhere.json"),
                         "../outside.json"):
            with self.subTest(relative=relative):
                with self.assertRaises(ValueError):
                    query_execution(self.root, jobs, "out.json", relative)
        (self.root / "binary.json").write_bytes(b"\xff\xfe")
        with self.assertRaises(ValueError):
            query_execution(self.root, jobs, "out.json", "binary.json")
        self._write_report("notjson.json", None)
        (self.root / "notjson.json").write_text("{oops", encoding="utf-8")
        with self.assertRaises(ValueError):
            query_execution(self.root, jobs, "out.json", "notjson.json")

    def test_query_is_read_only_and_report_may_equal_output(self):
        jobs = self._jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        run_plan(self.root, jobs, "same.json", record_reasons=True)
        before = (self.root / "same.json").read_text()
        queried = query_execution(self.root, jobs, "same.json", "same.json")
        self.assertEqual(queried["execution"]["mode"], "all")
        self.assertEqual((self.root / "same.json").read_text(), before)
        self.assertFalse((self.root / ".results").exists())
        # Plan and output protection rules still apply.
        with self.assertRaises(ValueError):
            query_execution(self.root, jobs, "notes.txt", "same.json")
        bad = jobs + [{"name": "dup", "operation": "sha256", "input": "notes.txt"},
                      {"name": "dup", "operation": "sha256", "input": "notes.txt"}]
        with self.assertRaises(ValueError):
            query_execution(self.root, bad, "out.json", "same.json")

    def test_cli_execution_query_and_conflicts(self):
        jobs = self._jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json",
                  "--root", str(self.root)]
        run = subprocess.run(prefix + ["--output", "r.json", "--record-reasons"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        recorded = json.loads((self.root / "r.json").read_text())["execution"]
        run = subprocess.run(prefix + ["--execution", "r.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"execution": recorded})
        # report may equal --output.
        run = subprocess.run(prefix + ["--execution", "r.json", "--output", "r.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        # An old report without the field yields null.
        self._write_report("old.json", {"results": []})
        run = subprocess.run(prefix + ["--execution", "old.json"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), {"execution": None})
        # Nothing was created or written by the queries.
        self.assertFalse((self.root / ".results").exists())
        # Conflicts and invalid executions print error-only JSON and exit 2.
        self._write_report("bad.json", {"results": [{"name": "base",
                                                     "status": "completed",
                                                     "result": {}}],
                                        "execution": None})
        for extra in (["--execution", "r.json", "--only", "base"],
                      ["--execution", "r.json", "--record-reasons"],
                      ["--execution", "r.json", "--preview"],
                      ["--execution", "r.json", "--retry-preview", "r.json"],
                      ["--execution", "r.json", "--retry", "r.json"],
                      ["--execution", "r.json", "--compare", "r.json", "r.json"],
                      ["--execution", "r.json", "--explain", "r.json"],
                      ["--execution", "r.json", "--history", "r.json"],
                      ["--execution", "r.json", "--changed", "notes.txt"],
                      ["--execution", "r.json", "--run-changed", "notes.txt"],
                      ["--execution", "bad.json"],
                      ["--execution", "missing.json"]):
            with self.subTest(extra=extra):
                run = subprocess.run(prefix + extra, capture_output=True, text=True)
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertEqual(set(json.loads(run.stdout)), {"error"})


class NonJsonConstantTests(unittest.TestCase):
    """Bare NaN/Infinity/-Infinity tokens are invalid JSON for every reader."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "notes.txt").write_text("one\ntwo\n", encoding="utf-8")

    def _write(self, relative, raw):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return path

    def _jobs(self):
        return [
            {"name": "a", "operation": "count-lines", "input": "notes.txt"},
            {"name": "b", "operation": "sha256", "input": "notes.txt",
             "depends_on": ["a"]},
        ]

    # Every syntactic position rejects; bytes carry the bare constants.
    def _constant_reports(self):
        return {
            "nested-result": b'{"results":[{"name":"a","status":"completed",'
                            b'"result":{"x":[1,{"y":NaN}]}}]}',
            "entry-extra": b'{"results":[{"name":"a","status":"completed",'
                           b'"result":{},"junk":Infinity}]}',
            "toplevel-extra": b'{"results":[],"junk":-Infinity}',
            "empty-results": b'{"results":[NaN]}',
            "would-retry": b'{"results":[{"name":"a","status":"failed",'
                           b'"error":"boom","junk":Infinity}]}',
        }

    def test_preview_retry_rejects_bare_constants_everywhere(self):
        jobs = self._jobs()
        for label, raw in self._constant_reports().items():
            with self.subTest(label=label):
                self._write(f"{label}.json", raw)
                with self.assertRaises(ValueError):
                    preview_retry(self.root, jobs, "new/out.json", f"{label}.json")
        # No directory for the output path is created by the failed preview.
        self.assertFalse((self.root / "new").exists())

    def test_preview_retry_accepts_strings_and_legal_numbers(self):
        jobs = self._jobs()
        # The same words as JSON strings stay legal, including error text.
        self._write("strings.json",
                    b'{"results":['
                    b'{"name":"a","status":"completed","result":{"x":"NaN"}},'
                    b'{"name":"b","status":"failed","error":"got Infinity/-Infinity"}]}')
        preview = preview_retry(self.root, jobs, "new/out.json", "strings.json")
        self.assertEqual(preview["targets"], ["b"])
        # Legal-but-huge numbers keep the retry entry's lenient parsing.
        self._write("numbers.json",
                    b'{"results":[{"name":"a","status":"completed",'
                    b'"result":{"big":1e400,"tiny":1e-400,'
                    b'"long":0.123456789012345678901234567890123456789,'
                    b'"int":90071992547409931}}]}')
        self.assertEqual(preview_retry(self.root, jobs, "new/out.json", "numbers.json"),
                         {"targets": [], "jobs": []})

    def test_run_retry_rejects_without_running_or_touching_files(self):
        jobs = self._jobs()
        marker = self._write("keep.json", b"KEEP-CONTENT")
        for label, raw in self._constant_reports().items():
            with self.subTest(label=label):
                report = self._write(f"{label}.json", raw)
                with self.assertRaises(ValueError):
                    run_retry(self.root, jobs, "keep.json", f"{label}.json")
                # The invalid report itself is untouched and no task ran.
                self.assertEqual(report.read_bytes(), raw)
        # report == output: the original invalid content is preserved...
        same = self._write("same.json", self._constant_reports()["nested-result"])
        with self.assertRaises(ValueError):
            run_retry(self.root, jobs, "same.json", "same.json")
        self.assertEqual(same.read_bytes(), self._constant_reports()["nested-result"])
        # ...also with record_reasons enabled, which cannot change the failure.
        with self.assertRaises(ValueError):
            run_retry(self.root, jobs, "same.json", "same.json", record_reasons=True)
        self.assertEqual(same.read_bytes(), self._constant_reports()["nested-result"])
        # Nothing ran, no output directory was made and other files are intact.
        self.assertFalse((self.root / "new").exists())
        self.assertEqual(marker.read_bytes(), b"KEEP-CONTENT")
        # A string-form report still retries normally; the failed record is
        # the only target and its dependent is not pulled in.
        self._write("strings.json",
                    b'{"results":[{"name":"a","status":"failed",'
                    b'"error":"value NaN"}]}')
        result = run_retry(self.root, jobs, "out.json", "strings.json")
        self.assertEqual([row["name"] for row in result], ["a"])

    def test_explain_rejects_bare_constants_but_keeps_string_text(self):
        jobs = self._jobs()
        for label, raw in (
            ("nested.json", b'{"results":[{"name":"a","status":"completed",'
                            b'"result":{"deep":{"x":NaN}}},{"name":"b",'
                            b'"status":"blocked","blocked_by":["a"]}]}'),
            ("entry-extra.json", b'{"results":[{"name":"a","status":"failed",'
                                 b'"error":"e","junk":Infinity}]}'),
            ("toplevel.json", b'{"results":[],"junk":-Infinity}'),
            ("empty.json", b'{"results":[{"junk":NaN}]}'),
        ):
            with self.subTest(label=label):
                self._write(label, raw)
                with self.assertRaises(ValueError):
                    explain_report(self.root, jobs, "new/out.json", label)
        self.assertFalse((self.root / "new").exists())
        # Words inside the failure string are returned verbatim.
        self._write("strings.json",
                    b'{"results":[{"name":"a","status":"failed",'
                    b'"error":"value was NaN and Infinity too"}]}')
        explained = explain_report(self.root, jobs, "out.json", "strings.json")
        self.assertEqual(explained["jobs"][0]["causes"][0]["error"],
                         "value was NaN and Infinity too")
        # Strict payloads with legal out-of-range exponents explain fine.
        self._write("numbers.json",
                    b'{"results":[{"name":"a","status":"completed",'
                    b'"result":{"big":1e400,"tiny":1e-400}}]}')
        self.assertEqual(explain_report(self.root, jobs, "out.json", "numbers.json"),
                         {"jobs": []})

    def test_query_execution_rejects_constants_even_when_it_would_return_null(self):
        jobs = self._jobs()
        valid_all = (
            b'{"results":[{"name":"a","status":"completed","result":{"lines":2}},'
            b'{"name":"b","status":"completed","result":{"sha256":"x","bytes":8}}],'
            b'"execution":{"mode":"all","targets":["a","b"],"jobs":['
            b'{"name":"a","depends_on":[],"reason":"all","required_by":[]},'
            b'{"name":"b","depends_on":["a"],"reason":"all","required_by":[]}]}}'
        )
        self._write("valid.json", valid_all)
        self.assertEqual(query_execution(self.root, jobs, "out.json", "valid.json")
                         ["execution"]["mode"], "all")
        for label, raw in (
            ("nested.json", b'{"results":[{"name":"a","status":"completed",'
                            b'"result":{"x":NaN}}]}'),
            ("entry-extra.json", b'{"results":[{"name":"a","status":"completed",'
                                 b'"result":{},"junk":Infinity}]}'),
            ("toplevel.json", b'{"results":[],"junk":-Infinity}'),
            ("no-exec.json", b'{"results":[],"junk":NaN}'),
            ("empty-null.json", b'{"results":[],"execution":null,"junk":Infinity}'),
            ("exec-extra.json",
             b'{"results":[{"name":"a","status":"completed","result":{"lines":2}},'
             b'{"name":"b","status":"completed","result":{"sha256":"x","bytes":8}}],'
             b'"execution":{"mode":"all","targets":["a","b"],"jobs":['
             b'{"name":"a","depends_on":[],"reason":"all","required_by":[]},'
             b'{"name":"b","depends_on":["a"],"reason":"all","required_by":[],'
             b'"junk":NaN}]}}'),
        ):
            with self.subTest(label=label):
                self._write(label, raw)
                with self.assertRaises(ValueError):
                    query_execution(self.root, jobs, "new/out.json", label)
        self.assertFalse((self.root / "new").exists())
        # Strings (including failure text) do not trigger the rule.
        self._write("strings.json",
                    b'{"results":[{"name":"a","status":"failed","error":"NaN"}]}')
        self.assertEqual(query_execution(self.root, jobs, "out.json", "strings.json"),
                         {"execution": None})

    def test_compare_and_history_keep_rejecting_constants(self):
        jobs = self._jobs()
        self._write("ok.json", b'{"results":[]}')
        for label, raw in self._constant_reports().items():
            report = self._write(f"c-{label}.json", raw)
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    compare_reports(self.root, jobs, "out.json", f"c-{label}.json", "ok.json")
                with self.assertRaises(ValueError):
                    query_history(self.root, jobs, "out.json", [f"c-{label}.json"])
                self.assertEqual(report.read_bytes(), raw)

    def test_cli_four_entries_exit_2_error_only_and_preserve_files(self):
        jobs = self._jobs()
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"jobs": jobs}))
        prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json",
                  "--root", str(self.root)]
        self._write("nan.json",
                    b'{"results":[{"name":"a","status":"completed",'
                    b'"result":{"x":NaN}}]}')
        keep = self._write("keep.json", b"KEEP")
        cases = [
            ["--retry-preview", "nan.json"],
            ["--retry", "nan.json", "--output", "new/out.json"],
            ["--retry", "nan.json", "--output", "new/out.json", "--record-reasons"],
            ["--explain", "nan.json", "--output", "new/out.json"],
            ["--execution", "nan.json", "--output", "new/out.json"],
        ]
        for extra in cases:
            with self.subTest(extra=extra):
                run = subprocess.run(prefix + extra, capture_output=True, text=True)
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertEqual(set(json.loads(run.stdout)), {"error"})
                self.assertIn("not valid JSON", run.stdout)
        # No run, no created directory, existing file untouched, inputs intact.
        self.assertFalse((self.root / "new").exists())
        self.assertEqual(keep.read_bytes(), b"KEEP")
        # The string form is legal for every entry.
        self._write("strings.json",
                    b'{"results":[{"name":"a","status":"completed",'
                    b'"result":{"x":"NaN"},"note":"Infinity"},'
                    b'{"name":"b","status":"completed","result":{}}]}')
        for extra in (["--retry-preview", "strings.json"],
                      ["--explain", "strings.json"],
                      ["--execution", "strings.json"]):
            with self.subTest(extra=extra):
                run = subprocess.run(prefix + extra, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)


class DeepResultTests(unittest.TestCase):
    """Legal ~600-level nested results never hit the recursion limit."""

    DEPTH = 600

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "notes.txt").write_text("one\ntwo\n", encoding="utf-8")
        self.jobs = [
            {"name": "t", "operation": "count-lines", "input": "notes.txt"},
            {"name": "u", "operation": "sha256", "input": "notes.txt"},
        ]
        self.plan = self.root / "plan.json"
        self.plan.write_text(json.dumps({"jobs": self.jobs}))
        self.prefix = [sys.executable, str(ROOT / "job_planner.py"), "plan.json",
                       "--root", str(self.root)]

    def _deep_text(self, leaf, shape="object", name="t"):
        if shape == "object":
            inner = '{"a":' * self.DEPTH + leaf + '}' * self.DEPTH
        elif shape == "array":
            # The result root object plus DEPTH nested arrays.
            inner = '{"v":' + '[' * self.DEPTH + leaf + ']' * self.DEPTH + '}'
        else:
            pairs = self.DEPTH // 2
            inner = '{"a":[' * pairs + leaf + ']}' * pairs
        return ('{"results":[{"name":' + json.dumps(name)
                + ',"status":"completed","result":' + inner + '}]}')

    def _decorated_text(self, leaf):
        # DEPTH nested objects, with a sibling string marker at level 251.
        return ('{"results":[{"name":"t","status":"completed","result":'
                + '{"a":' * 250 + '{"note":"mid","a":'
                + '{"a":' * 349 + leaf + '}' * 350 + '}' * 250 + '}]}')

    def _write_raw(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    @staticmethod
    def _follow(value):
        if isinstance(value, dict):
            if "a" in value:
                return value["a"]
            if "v" in value:
                return value["v"]
            return None
        return value[0]

    def _descend(self, value):
        depth = 0
        marker = None
        while isinstance(value, (dict, list)):
            if isinstance(value, dict) and "note" in value:
                marker = value["note"]
            value = self._follow(value)
            if value is None:
                break
            depth += 1
        return depth, value, marker

    def test_compare_deep_shapes_unchanged_and_leaf_changed(self):
        for shape in ("object", "array", "alternating"):
            with self.subTest(shape=shape):
                self._write_raw(f"{shape}.json", self._deep_text("1", shape))
                self._write_raw(f"{shape}-str.json",
                                self._deep_text('"deep"', shape))
                same = compare_reports(self.root, self.jobs, "out.json",
                                       f"{shape}.json", f"{shape}.json")
                self.assertEqual([row["change"] for row in same["jobs"]],
                                 ["unchanged"])
                changed = compare_reports(self.root, self.jobs, "out.json",
                                          f"{shape}.json", f"{shape}-str.json")
                row = changed["jobs"][0]
                self.assertEqual(row["change"], "changed")
                depth, before_leaf, _ = self._descend(row["before"]["result"])
                depth_after, after_leaf, _ = self._descend(row["after"]["result"])
                self.assertEqual(before_leaf, 1)
                self.assertEqual(after_leaf, "deep")
                self.assertGreaterEqual(depth, self.DEPTH)
                self.assertEqual(depth_after, depth)

    def test_compare_deep_exact_number_and_json_semantics(self):
        pairs = {
            "int-float": ("1", "1.0", "unchanged"),
            "int-exp": ("1", "1e0", "unchanged"),
            "exponents": ("1e400", "10e399", "unchanged"),
            "neg-zero": ("-0", "0", "unchanged"),
            "double-noise": ("0.10000000000000001", "0.1", "changed"),
            "huge": ("1e400", "2e400", "changed"),
            "tiny": ("1e-400", "0", "changed"),
            "big-integers": ("9007199254740992", "9007199254740993", "changed"),
            "bool-number": ("true", "1", "changed"),
        }
        for label, (before, after, change) in pairs.items():
            with self.subTest(label=label):
                self._write_raw("b.json", self._deep_text(before))
                self._write_raw("a.json", self._deep_text(after))
                result = compare_reports(self.root, self.jobs, "out.json",
                                         "b.json", "a.json")
                self.assertEqual(result["jobs"][0]["change"], change)
        # Object key order at the deepest level is irrelevant; array order
        # stays significant, both 600 containers down.
        ordered = ('{"z":' + '[' * 599 + '{"p":1,"q":2}' + ']' * 599 + '}')
        reordered = ('{"z":' + '[' * 599 + '{"q":2,"p":1}' + ']' * 599 + '}')
        swapped_arrays = ('{"z":' + '[' * 599 + '[1,2]' + ']' * 599 + '}')
        swapped = ('{"z":' + '[' * 599 + '[2,1]' + ']' * 599 + '}')
        for label, b, a, change in (("key-order", ordered, reordered, "unchanged"),
                                    ("array-order", swapped_arrays, swapped, "changed")):
            with self.subTest(label=label):
                self._write_raw("b.json",
                                '{"results":[{"name":"t","status":"completed",'
                                '"result":' + b + '}]}')
                self._write_raw("a.json",
                                '{"results":[{"name":"t","status":"completed",'
                                '"result":' + a + '}]}')
                result = compare_reports(self.root, self.jobs, "out.json",
                                         "b.json", "a.json")
                self.assertEqual(result["jobs"][0]["change"], change)

    def test_compare_deep_preserves_layers_strings_and_numbers(self):
        leaf = '{"tip":"h\\u00e9llo \\u6df1","n":0.10000000000000001,' \
               '"big":1e400,"tiny":1e-400}'
        self._write_raw("b.json", self._decorated_text(leaf))
        self._write_raw("a.json", self._decorated_text(leaf))
        result = compare_reports(self.root, self.jobs, "out.json",
                                 "b.json", "a.json")
        self.assertEqual(result["jobs"][0]["change"], "unchanged")
        for side in ("before", "after"):
            depth, _, marker = self._descend(result["jobs"][0][side]["result"])
            self.assertEqual(marker, "mid")
            self.assertEqual(depth, self.DEPTH)
        tip = result["jobs"][0]["after"]["result"]
        for _ in range(self.DEPTH):
            tip = self._follow(tip)
        self.assertEqual(tip["tip"], "héllo 深")
        self.assertIsInstance(tip["n"], Decimal)
        self.assertEqual(tip["n"], Decimal("0.10000000000000001"))
        self.assertEqual(tip["big"], Decimal("1e400"))
        self.assertNotEqual(tip["big"], Decimal("2e400"))
        self.assertNotEqual(tip["tiny"], 0)
        self.assertNotIsInstance(tip["big"], float)
        self.assertNotIsInstance(tip["big"], str)

    def test_history_deep_order_duplicates_missing_and_only(self):
        self._write_raw("deep.json", self._deep_text("1"))
        self._write_raw("flat.json",
                        '{"results":[{"name":"u","status":"completed",'
                        '"result":{"n":1}}]}')
        result = query_history(self.root, self.jobs, "out.json",
                               ["deep.json", "flat.json", "deep.json"])
        self.assertEqual([row["name"] for row in result["jobs"]], ["t", "u"])
        t = result["jobs"][0]["history"]
        u = result["jobs"][1]["history"]
        self.assertEqual([slot["report"] for slot in t],
                         ["deep.json", "flat.json", "deep.json"])
        self.assertIsNone(t[1]["record"])
        self.assertEqual(t[0], t[2])
        depth, leaf, _ = self._descend(t[0]["record"]["result"])
        self.assertEqual(depth, self.DEPTH)
        self.assertEqual(leaf, 1)
        self.assertEqual([slot["record"] for slot in u],
                         [None, {"status": "completed", "result": {"n": 1}}, None])
        # --only selects exact task names without expanding anything, and a
        # task no report records keeps an all-null history.
        only_t = query_history(self.root, self.jobs, "out.json",
                               ["deep.json", "flat.json"], targets=["t"])
        self.assertEqual([row["name"] for row in only_t["jobs"]], ["t"])
        only_u = query_history(self.root, self.jobs, "out.json",
                               ["deep.json"], targets=["u"])
        self.assertEqual(only_u["jobs"],
                         [{"name": "u",
                           "history": [{"report": "deep.json", "record": None}]}])

    def test_recursion_limit_unchanged_after_deep_queries(self):
        import sys
        from job_planner import _compare_dumps
        before = sys.getrecursionlimit()
        for shape in ("object", "array", "alternating"):
            self._write_raw(f"{shape}.json", self._deep_text("1", shape))
        compare_reports(self.root, self.jobs, "out.json",
                        "object.json", "array.json")
        query_history(self.root, self.jobs, "out.json",
                      ["object.json", "alternating.json", "object.json"])
        value = Decimal("1")
        for _ in range(self.DEPTH // 2):
            value = {"a": [value]}
        encoded = _compare_dumps({"result": value})
        self.assertTrue(_loads_strict(encoded)["result"] is not None)
        self.assertEqual(sys.getrecursionlimit(), before)

    def test_cli_compare_deep_outputs_full_json_exit_0(self):
        for shape in ("object", "array", "alternating"):
            with self.subTest(shape=shape):
                self._write_raw(f"{shape}.json", self._deep_text("1", shape))
                self._write_raw(f"{shape}-str.json",
                                self._deep_text('"deep"', shape))
                run = subprocess.run(
                    self.prefix + ["--compare", f"{shape}.json", f"{shape}.json"],
                    capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertFalse(run.stderr)
                self.assertNotIn("Infinity", run.stdout)
                payload = _loads_strict(run.stdout)
                self.assertEqual(payload["jobs"][0]["change"], "unchanged")
                depth, leaf, _ = self._descend(
                    payload["jobs"][0]["after"]["result"])
                self.assertGreaterEqual(depth, self.DEPTH)
                self.assertEqual(leaf, Decimal("1"))
                run = subprocess.run(
                    self.prefix + ["--compare", f"{shape}.json", f"{shape}-str.json"],
                    capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                payload = json.loads(run.stdout, parse_float=Decimal)
                self.assertEqual(payload["jobs"][0]["change"], "changed")

    def test_cli_history_deep_outputs_full_json_exit_0(self):
        self._write_raw("deep.json", self._deep_text("1"))
        self._write_raw("flat.json",
                        '{"results":[{"name":"u","status":"completed",'
                        '"result":{"n":1}}]}')
        run = subprocess.run(
            self.prefix + ["--history", "deep.json", "--history", "flat.json",
                           "--history", "deep.json"],
            capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertFalse(run.stderr)
        self.assertNotIn("Infinity", run.stdout)
        payload = json.loads(run.stdout, parse_float=Decimal,
                             parse_constant=lambda c: (_ for _ in ()).throw(
                                 ValueError(c)))
        t = next(row for row in payload["jobs"] if row["name"] == "t")
        self.assertIsNone(t["history"][1]["record"])
        depth, leaf, _ = self._descend(t["history"][0]["record"]["result"])
        self.assertEqual(depth, self.DEPTH)
        self.assertEqual(leaf, Decimal("1"))
        self.assertEqual(t["history"][0], t["history"][2])
        # --only keeps the filter exact and still exits 0 on an all-null row.
        run = subprocess.run(
            self.prefix + ["--history", "deep.json", "--only", "u"],
            capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout),
                         {"jobs": [{"name": "u", "history": [
                             {"report": "deep.json", "record": None}]}]})

    def test_deep_invalid_reports_still_raise_and_exit_2(self):
        good = self._deep_text("1")
        self._write_raw("good.json", good)
        deep_nan = self._deep_text("NaN")
        truncated = good[:-8]
        self._write_raw("nan.json", deep_nan)
        self._write_raw("truncated.json", truncated)
        path = self.root / "bad-utf8.json"
        path.write_text(good, encoding="utf-8")
        with path.open("ab") as handle:
            handle.write(b"\xff")
        for bad in ("nan.json", "truncated.json", "bad-utf8.json"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    compare_reports(self.root, self.jobs, "out.json",
                                    bad, "good.json")
                with self.assertRaises(ValueError):
                    query_history(self.root, self.jobs, "out.json", [bad])
                for extra in (["--compare", bad, "good.json"],
                              ["--history", bad]):
                    run = subprocess.run(self.prefix + extra,
                                         capture_output=True, text=True)
                    self.assertEqual(run.returncode, 2, run.stderr)
                    self.assertEqual(set(json.loads(run.stdout)), {"error"})
                    self.assertFalse(run.stderr)
        # The words as strings at depth remain legal.
        self._write_raw("strings.json", self._deep_text('"NaN and Infinity"'))
        run = subprocess.run(
            self.prefix + ["--compare", "strings.json", "strings.json"],
            capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["jobs"][0]["change"], "unchanged")


if __name__ == "__main__":
    unittest.main()