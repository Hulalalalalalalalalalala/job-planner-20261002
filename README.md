# Job Planner

Run a small deterministic file-task plan and persist each result. Python 3.10+ and the standard library are sufficient.

`python3 job_planner.py samples/plan.json` runs three included jobs: line count, SHA-256 and CSV dimensions. It prints completed/failed counts and writes `.results/latest.json`. The plan refers to `samples/notes.txt` and `samples/sales.csv` inside the chosen root. Use `--root` to set that directory and `--output` for a relative report path.

Supported operations are count-lines, sha256 and csv-summary. A job may declare `"depends_on"`: a list of other job names (exact match, forward references allowed) that must complete before it runs. Missing or empty means no dependency. Among jobs whose direct dependencies already have records, the one earliest in plan order runs next; jobs without dependencies take part as usual. Results have no timestamps so identical inputs produce identical report content. Duplicate job names and invalid dependencies (wrong types, blank or duplicate entries, unknown names, self references, cycles) are rejected before any task runs and leave any existing report untouched.

Per-job file or operation errors are recorded as `failed` and subsequent jobs continue; the failed job's downstream is not read from disk and is recorded as `blocked` with only `name`, `status` and `blocked_by` (every failed or blocked direct dependency, in `depends_on` order), and blocking propagates onward. Each job appears once in results, ordered by actual processing order. Exit codes are 0 when all jobs complete, 1 when any job fails or is blocked, and 2 for an invalid plan or report path. When the plan has any nonempty `depends_on`, the CLI summary also reports a `blocked` count; plans without dependencies keep the original summary fields and ordering.

The API is `execute_job(root, job)` and `run_plan(root, jobs, output)`. All input, plan and report paths must be relative and resolve inside root, including symlinks. Reports cannot replace a job input; the CLI also protects its plan. Operations are implemented directly in Python: no shell commands, subprocess execution, schedules or retries are available. This is a local trusted-user utility rather than an OS sandbox; it does not isolate concurrent filesystem changes.

Run `python3 -B -m unittest -v` for deterministic outputs, failure records, path boundaries, symlinks and CLI validation.
