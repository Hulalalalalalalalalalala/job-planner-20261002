# Job Planner

Run a small deterministic file-task plan and persist each result. Python 3.10+ and the standard library are sufficient.

`python3 job_planner.py samples/plan.json` runs three included jobs: line count, SHA-256 and CSV dimensions. It prints completed/failed counts and writes `.results/latest.json`. The plan refers to `samples/notes.txt` and `samples/sales.csv` inside the chosen root. Use `--root` to set that directory and `--output` for a relative report path.

Supported operations are count-lines, sha256 and csv-summary. Jobs run sequentially in plan order; results have no timestamps so identical inputs produce identical report content. Duplicate job names are rejected. Per-job file or operation errors are recorded and subsequent jobs continue. Exit codes are 0 when all jobs complete, 1 when any job fails, and 2 for an invalid plan or report path.

The API is `execute_job(root, job)` and `run_plan(root, jobs, output)`. All input, plan and report paths must be relative and resolve inside root, including symlinks. Reports cannot replace a job input; the CLI also protects its plan. Operations are implemented directly in Python: no shell commands, subprocess execution, schedules or retries are available. This is a local trusted-user utility rather than an OS sandbox; it does not isolate concurrent filesystem changes.

Run `python3 -B -m unittest -v` for deterministic outputs, failure records, path boundaries, symlinks and CLI validation.
