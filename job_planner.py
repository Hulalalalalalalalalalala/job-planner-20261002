"""Run named deterministic file tasks within a repository directory."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


def local_path(root, relative):
    root = Path(root).resolve()
    if Path(relative).is_absolute():
        raise ValueError("only relative repository paths are allowed")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("path escapes repository")
    return path


def execute_job(root, job):
    operation = job["operation"]
    if operation not in ("count-lines", "sha256", "csv-summary"):
        raise ValueError("unsupported operation")
    source = local_path(root, job["input"])
    content = source.read_bytes()
    if operation == "sha256":
        return {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    if operation == "count-lines":
        return {"lines": len(content.decode("utf-8").splitlines())}
    rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows[1:]):
        raise ValueError("CSV must have a header and consistent row widths")
    return {"columns": rows[0], "rows": len(rows) - 1}


def _validate_dependencies(jobs):
    """Validate depends_on fields and return {name: [dependency names]}.

    References are matched exactly against plan job names and may point
    forward; cycles are rejected before anything runs.
    """
    names = {job["name"] for job in jobs}
    deps_by_name = {}
    for job in jobs:
        name = job["name"]
        deps = job.get("depends_on", [])
        if not isinstance(deps, list):
            raise ValueError(f"job {name!r}: depends_on must be a list")
        if any(not isinstance(dep, str) or not dep.strip() for dep in deps):
            raise ValueError(f"job {name!r}: depends_on entries must be nonblank strings")
        if len(set(deps)) != len(deps):
            raise ValueError(f"job {name!r}: depends_on contains a duplicate")
        if name in deps:
            raise ValueError(f"job {name!r} cannot depend on itself")
        unknown = [dep for dep in deps if dep not in names]
        if unknown:
            raise ValueError(f"job {name!r}: unknown dependency {unknown[0]!r}")
        deps_by_name[name] = list(deps)

    state = {}

    def visit(node):
        if state.get(node) == 1:
            raise ValueError("dependency cycle detected")
        if state.get(node) == 2:
            return
        state[node] = 1
        for dep in deps_by_name[node]:
            visit(dep)
        state[node] = 2

    for name in deps_by_name:
        visit(name)
    return deps_by_name


def _select_jobs(jobs, targets):
    """Return the plan-ordered sublist for targets plus all their prerequisites.

    Targets match job names exactly; the whole plan must already be validated.
    """
    if targets is None:
        return list(jobs)
    if not isinstance(targets, list) or not targets:
        raise ValueError("targets must be a nonempty list of job names")
    if any(not isinstance(target, str) or not target.strip() for target in targets):
        raise ValueError("targets must be nonblank strings")
    if len(set(targets)) != len(targets):
        raise ValueError("targets contain a duplicate")
    names = {job["name"] for job in jobs}
    unknown = [target for target in targets if target not in names]
    if unknown:
        raise ValueError(f"unknown target {unknown[0]!r}")
    deps_by_name = {job["name"]: job.get("depends_on", []) for job in jobs}
    keep = set()
    stack = list(targets)
    while stack:
        name = stack.pop()
        if name not in keep:
            keep.add(name)
            stack.extend(deps_by_name[name])
    return [job for job in jobs if job["name"] in keep]


def _validate_plan(root, jobs, output):
    """Validate names, dependencies and all paths; return deps map and report path.

    The whole plan is checked, including unselected branches, before any
    scope is chosen. Inputs are resolved for boundary checks but never read;
    nothing is created.
    """
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("plan requires at least one job")
    names = [job["name"] for job in jobs]
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("job names must be nonempty and unique")
    deps_by_name = _validate_dependencies(jobs)
    report_path = local_path(root, output)
    if any(report_path == local_path(root, job["input"]) for job in jobs):
        raise ValueError("report cannot overwrite a task input")
    return deps_by_name, report_path


def _run_order(selected, deps_by_name):
    """Return selected job names in the earliest-ready processing order.

    Each pass emits the earliest-in-plan pending job whose direct
    dependencies were already emitted; the validated acyclic graph and the
    full prerequisite closure guarantee this matches an actual run.
    """
    pending = {job["name"] for job in selected}
    ordered = []
    while pending:
        ready = [job for job in selected
                 if job["name"] in pending
                 and all(dep not in pending for dep in deps_by_name[job["name"]])]
        name = ready[0]["name"]
        pending.discard(name)
        ordered.append(name)
    return ordered


def _preview_entries(jobs, deps_by_name, targets):
    """Build run-ordered preview entries for targets plus every prerequisite.

    ``targets=None`` describes the whole plan with reason ``"all"``;
    otherwise explicit targets are ``"target"`` and every other included
    job is ``"prerequisite"``. Shared prerequisites appear once and
    required_by lists are walked in plan order, independently of target
    argument order.
    """
    selected = _select_jobs(jobs, targets)
    ordered = _run_order(selected, deps_by_name)
    explicit = set(targets) if targets is not None else set()
    required_by = {name: [] for name in ordered}
    if targets is not None:
        # Walk explicit targets in original plan order so required_by lists
        # are independent of the target argument order.
        for job in selected:
            target = job["name"]
            if target not in explicit:
                continue
            stack = [target]
            seen = set()
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                required_by[node].append(target)
                stack.extend(deps_by_name[node])
    return [
        {"name": name,
         "depends_on": list(deps_by_name[name]),
         "reason": "all" if targets is None else ("target" if name in explicit else "prerequisite"),
         "required_by": required_by[name]}
        for name in ordered
    ]


def _load_report_entries(root, report, names):
    """Read a report file into its raw ``results`` entries.

    Only the report file is read. The path must be relative to root and,
    after resolving symlinks, stay inside root. The report must be a UTF-8
    JSON object whose ``results`` is a list; per-entry name, status and
    payload checks are left to the caller so each entry point can enforce
    its own rules. Any violation raises ValueError.
    """
    path = local_path(root, report)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"report {report!r} is missing or unreadable") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"report {report!r} is not valid UTF-8") from exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"report {report!r} is not valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("report must be a JSON object")
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("report results must be a list")
    return results


def _read_retry_report(root, report, names):
    """Read and validate a historical run report; return {name: status}.

    Names must match the current plan exactly, with no repeats, and one of
    the three statuses; other entry fields are ignored. Any violation
    raises ValueError.
    """
    results = _load_report_entries(root, report, names)
    statuses = {}
    for entry in results:
        if not isinstance(entry, dict):
            raise ValueError("each report result must be a JSON object")
        name = entry.get("name")
        if not isinstance(name, str) or name not in names:
            raise ValueError(f"report contains unknown task {name!r}")
        if name in statuses:
            raise ValueError(f"report repeats task {name!r}")
        status = entry.get("status")
        if status not in ("completed", "failed", "blocked"):
            raise ValueError(f"report task {name!r} has invalid status {status!r}")
        statuses[name] = status
    return statuses


def _read_compare_report(root, report, deps_by_name):
    """Read and validate a report for comparison; return {name: side entry}.

    Structure, names, statuses and the repeat rule match the retry report,
    but every status must carry its matching payload: ``completed`` needs an
    object ``result``, ``failed`` a string ``error`` and ``blocked`` a
    nonempty, duplicate-free list naming only the job's declared direct
    dependencies. Extra entry fields are ignored. ``blocked_by`` is
    returned in current declaration order so callers compare it as a set
    independently of report record order. Any violation raises ValueError.
    """
    names = set(deps_by_name)
    results = _load_report_entries(root, report, names)
    records = {}
    for entry in results:
        if not isinstance(entry, dict):
            raise ValueError("each report result must be a JSON object")
        name = entry.get("name")
        if not isinstance(name, str) or name not in names:
            raise ValueError(f"report contains unknown task {name!r}")
        if name in records:
            raise ValueError(f"report repeats task {name!r}")
        status = entry.get("status")
        if status not in ("completed", "failed", "blocked"):
            raise ValueError(f"report task {name!r} has invalid status {status!r}")
        if status == "completed":
            result = entry.get("result")
            if not isinstance(result, dict):
                raise ValueError(
                    f"report task {name!r}: completed record requires an object result")
            records[name] = {"status": "completed", "result": result}
        elif status == "failed":
            error = entry.get("error")
            if not isinstance(error, str):
                raise ValueError(
                    f"report task {name!r}: failed record requires a string error")
            records[name] = {"status": "failed", "error": error}
        else:
            blocked_by = entry.get("blocked_by")
            if (not isinstance(blocked_by, list) or not blocked_by
                    or any(not isinstance(dep, str) for dep in blocked_by)
                    or len(set(blocked_by)) != len(blocked_by)):
                raise ValueError(
                    f"report task {name!r}: blocked record requires a nonempty list "
                    "of unique dependency names")
            declared = deps_by_name[name]
            if any(dep not in declared for dep in blocked_by):
                raise ValueError(
                    f"report task {name!r}: blocked_by names a task that is not a "
                    "declared direct dependency")
            ordered = [dep for dep in declared if dep in set(blocked_by)]
            records[name] = {"status": "blocked", "blocked_by": ordered}
    return records


def _json_equal(a, b):
    """Compare parsed JSON values the way JSON itself defines equality.

    Booleans are not numbers (``true`` differs from ``1``), object key
    order is irrelevant, array order is significant, and nesting is
    compared recursively.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    return a == b


def compare_reports(root, jobs, output, before, after):
    """Compare two historical run reports without running anything.

    Returns ``{"jobs": [...]}`` in current plan order covering every task
    recorded in at least one report; each entry has ``name``, ``before``,
    ``after`` and ``change``. A side without a record is ``null``; present
    sides keep only ``status`` and the matching payload (``result`` for
    completed, ``error`` for failed, ``blocked_by`` for blocked, the last
    in current dependency declaration order). ``change`` is ``added``,
    ``removed``, ``changed`` (status or payload differs, including the
    failure message) or ``unchanged``. Result values compare as JSON
    (booleans differ from numbers, object key order is ignored, array
    order matters) and ``blocked_by`` compares as a set.

    The whole plan and output path are validated exactly like a run, but
    nothing is executed, created or written and task inputs are never
    read; ``before`` and ``after`` may name the same file. Any plan,
    output or report violation raises ValueError.
    """
    deps_by_name, _report_path = _validate_plan(root, jobs, output)
    before_records = _read_compare_report(root, before, deps_by_name)
    after_records = _read_compare_report(root, after, deps_by_name)
    compared = []
    for job in jobs:
        name = job["name"]
        old = before_records.get(name)
        new = after_records.get(name)
        if old is None and new is None:
            continue
        if old is None:
            change = "added"
        elif new is None:
            change = "removed"
        elif _json_equal(old, new):
            change = "unchanged"
        else:
            change = "changed"
        compared.append({"name": name, "before": old, "after": new,
                         "change": change})
    return {"jobs": compared}


def preview_plan(root, jobs, output, targets=None):
    """Describe what run_plan would do, without executing or touching files.

    Validates the whole plan exactly like a run, then returns
    ``{"jobs": [...]}`` covering the targets and every prerequisite, each
    once, in run processing order. Jobs carry name, depends_on (declaration
    order, [] when absent), reason ("all" without targets, otherwise
    "target"/"prerequisite") and required_by (explicit targets needing the
    job, in original plan order). Inputs are never read and no report is
    written, so missing files, unknown operations and bad file contents do
    not fail a preview; validation failures still raise ValueError.
    """
    deps_by_name, _report_path = _validate_plan(root, jobs, output)
    return {"jobs": _preview_entries(jobs, deps_by_name, targets)}


def preview_retry(root, jobs, output, report):
    """Preview a retry from a historical run report, without touching files.

    Tasks recorded as ``failed`` or ``blocked`` in the report become the
    retry targets; the returned jobs cover those targets and every
    prerequisite (including prerequisites the report marked completed,
    since a manual rerun reprocesses them), in plan run order with the
    same fields as ``preview_plan`` with targets. The report may cover a
    partial run: unrecorded tasks join the preview only when they are
    required prerequisites. Nothing is executed or written and job inputs
    are never read; only the plan and the named report are read.
    """
    deps_by_name, _report_path = _validate_plan(root, jobs, output)
    statuses = _read_retry_report(root, report, {job["name"] for job in jobs})
    targets = [job["name"] for job in jobs
               if statuses.get(job["name"]) in ("failed", "blocked")]
    if not targets:
        return {"targets": [], "jobs": []}
    return {"targets": targets, "jobs": _preview_entries(jobs, deps_by_name, targets)}


def _run_selected(root, selected, deps_by_name, report_path):
    """Execute an already-selected plan-ordered job list and write its report.

    Processing order, failure records and blocked propagation are shared by
    every execution mode: each pass runs the earliest ready job, a failing
    or blocked direct dependency blocks dependants without reading their
    input, and only this run's results drive dependency decisions. The
    report contains exactly these results, replacing any prior content.
    """
    results = []
    records = {}
    pending = {job["name"] for job in selected}
    while pending:
        ready = [i for i, job in enumerate(selected)
                 if job["name"] in pending
                 and all(dep in records for dep in deps_by_name[job["name"]])]
        job = selected[ready[0]]
        name = job["name"]
        pending.discard(name)
        deps = deps_by_name[name]
        blocked_by = [dep for dep in deps if records[dep] in ("failed", "blocked")]
        if blocked_by:
            result = {"name": name, "status": "blocked", "blocked_by": blocked_by}
        else:
            try:
                result = {"name": name, "status": "completed", "result": execute_job(root, job)}
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result = {"name": name, "status": "failed", "error": str(exc)}
        records[name] = result["status"]
        results.append(result)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return results


def run_plan(root, jobs, output, targets=None):
    deps_by_name, report_path = _validate_plan(root, jobs, output)
    selected = _select_jobs(jobs, targets)
    return _run_selected(root, selected, deps_by_name, report_path)


def run_retry(root, jobs, output, report):
    """Execute a manual retry derived from a historical run report.

    Target selection mirrors ``preview_retry``: every task the report
    records as ``failed`` or ``blocked`` becomes a target in current plan
    order, and the run covers the targets plus every direct and indirect
    prerequisite — prerequisites the report marked ``completed`` are
    re-executed too, and shared prerequisites run once. The report may
    cover a partial run; unrecorded tasks join only when they are required
    prerequisites and unrelated branches are neither read nor recorded.

    With no retry targets the empty list is returned without reading task
    inputs, creating directories or touching the report. Otherwise this
    run's result list is returned and written to ``output`` only — old
    records are never copied. ``report`` may equal ``output``; targets are
    taken from the content read before the overwrite. Execution order,
    failure records and ``blocked_by`` propagation are exactly
    ``run_plan``'s, driven solely by this run's results.

    The whole plan (including unselected branches and an empty scope) and
    the output path are validated first; report problems raise ValueError
    exactly as in ``preview_retry``, and creating the output directory or
    writing the report may raise OSError after tasks have run.
    """
    deps_by_name, report_path = _validate_plan(root, jobs, output)
    statuses = _read_retry_report(root, report, {job["name"] for job in jobs})
    targets = [job["name"] for job in jobs
               if statuses.get(job["name"]) in ("failed", "blocked")]
    if not targets:
        return []
    selected = _select_jobs(jobs, targets)
    return _run_selected(root, selected, deps_by_name, report_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=".results/latest.json")
    parser.add_argument("--only", action="append", default=None, metavar="NAME",
                        help="run only this job and its prerequisites (repeatable)")
    parser.add_argument("--preview", action="store_true",
                        help="print the jobs that would run with reasons, then exit 0")
    parser.add_argument("--retry-preview", metavar="REPORT",
                        help="print targets and jobs a retry of REPORT would run, then exit 0")
    parser.add_argument("--retry", metavar="REPORT",
                        help="rerun the report's failed/blocked tasks with their prerequisites")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="compare two reports read-only and print per-task changes")
    args = parser.parse_args()
    try:
        if args.compare is not None and (args.preview or args.only is not None
                                         or args.retry_preview is not None
                                         or args.retry is not None):
            raise ValueError("--compare cannot be combined with --only, --preview, "
                             "--retry-preview or --retry")
        if args.retry_preview is not None and (args.preview or args.only is not None):
            raise ValueError("--retry-preview cannot be combined with --preview or --only")
        if args.retry is not None and (args.preview or args.only is not None
                                       or args.retry_preview is not None):
            raise ValueError("--retry cannot be combined with --only, --preview or --retry-preview")
        plan = local_path(args.root, args.plan)
        if plan == local_path(args.root, args.output):
            raise ValueError("report cannot overwrite its plan")
        jobs = json.loads(plan.read_text(encoding="utf-8"))["jobs"]
        if args.compare is not None:
            print(json.dumps(compare_reports(args.root, jobs, args.output,
                                             args.compare[0], args.compare[1]),
                             ensure_ascii=False, indent=2))
            return 0
        if args.retry_preview is not None:
            print(json.dumps(preview_retry(args.root, jobs, args.output, args.retry_preview),
                             ensure_ascii=False, indent=2))
            return 0
        if args.preview:
            print(json.dumps(preview_plan(args.root, jobs, args.output, targets=args.only),
                             ensure_ascii=False, indent=2))
            return 0
        if args.retry is not None:
            results = run_retry(args.root, jobs, args.output, args.retry)
        else:
            results = run_plan(args.root, jobs, args.output, targets=args.only)
        summary = {"completed": sum(row["status"] == "completed" for row in results),
                   "failed": sum(row["status"] == "failed" for row in results)}
        declared_deps = {job["name"]: job.get("depends_on", []) for job in jobs}
        if any(declared_deps[row["name"]] for row in results):
            summary["blocked"] = sum(row["status"] == "blocked" for row in results)
        print(json.dumps(summary))
        return 1 if any(row["status"] in ("failed", "blocked") for row in results) else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
