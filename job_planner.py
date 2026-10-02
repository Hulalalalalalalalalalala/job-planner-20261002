"""Run named deterministic file tasks within a repository directory."""
import argparse
import csv
import hashlib
import io
import json
from decimal import Decimal
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
    for root in deps_by_name:
        if root in state:
            continue
        state[root] = 1
        # Iterative three-color DFS: a legal chain may span thousands of
        # forward-referencing jobs, so the walk cannot depend on Python's
        # recursion depth. Each frame is (node, index of its next dependency).
        stack = [(root, 0)]
        while stack:
            node, index = stack[-1]
            deps = deps_by_name[node]
            if index >= len(deps):
                state[node] = 2
                stack.pop()
                continue
            dep = deps[index]
            stack[-1] = (node, index + 1)
            if state.get(dep) == 1:
                raise ValueError("dependency cycle detected")
            if dep not in state:
                state[dep] = 1
                stack.append((dep, 0))
    return deps_by_name


def _validate_target_names(jobs, targets):
    """Validate targets the way every target-taking entry point does.

    Returns the target names as given; callers decide whether the
    prerequisite closure is expanded. The list must be nonempty and hold
    unique nonblank strings naming current plan jobs.
    """
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
    return list(targets)


def _select_jobs(jobs, targets):
    """Return the plan-ordered sublist for targets plus all their prerequisites.

    Targets match job names exactly; the whole plan must already be validated.
    """
    if targets is None:
        return list(jobs)
    _validate_target_names(jobs, targets)
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


def _reject_json_constant(value):
    """Reject the non-JSON constants ``NaN``/``Infinity``/``-Infinity``.

    ``json.loads`` accepts these by default; used as ``parse_constant`` they
    raise instead, even when the token sits in an otherwise ignored field.
    """
    raise ValueError(f"invalid JSON constant {value}")


def _load_report_entries(root, report, names, exact_numbers=False):
    """Read a report file into its raw ``results`` entries.

    Only the report file is read. The path must be relative to root and,
    after resolving symlinks, stay inside root. The report must be a UTF-8
    JSON object whose ``results`` is a list; per-entry name, status and
    payload checks are left to the caller so each entry point can enforce
    its own rules. Any violation raises ValueError.

    With ``exact_numbers`` every JSON number is kept as the ``Decimal`` of
    its literal text so comparisons can use the value the report actually
    expressed instead of a lossy float, and the non-JSON constants
    ``NaN``/``Infinity``/``-Infinity`` are rejected anywhere in the file.
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
        if exact_numbers:
            data = json.loads(text, parse_float=Decimal, parse_constant=_reject_json_constant)
        else:
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


def _read_compare_report(root, report, deps_by_name, exact_numbers=False):
    """Read and validate a report for comparison; return {name: side entry}.

    Structure, names, statuses and the repeat rule match the retry report,
    but every status must carry its matching payload: ``completed`` needs an
    object ``result``, ``failed`` a string ``error`` and ``blocked`` a
    nonempty, duplicate-free list naming only the job's declared direct
    dependencies. Extra entry fields are ignored. ``blocked_by`` is
    returned in current declaration order so callers compare it as a set
    independently of report record order. Any violation raises ValueError.

    Only the compare and history entry points pass ``exact_numbers``:
    then every JSON number is kept as ``Decimal`` and non-JSON numeric
    constants are rejected, while explain keeps ordinary float parsing.
    """
    names = set(deps_by_name)
    results = _load_report_entries(root, report, names, exact_numbers=exact_numbers)
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

    Booleans are not numbers (``true`` differs from ``1``), numbers never
    equal strings, object key order is irrelevant, array order is
    significant, and nesting is compared recursively. Compare-side
    reports keep every number as the ``Decimal`` of its literal text, so
    numbers compare by their exact decimal value: ``1``, ``1.0`` and
    ``1e0`` are equal and negative zero equals zero, while values that
    merely share a double-precision representation (``0.1`` vs
    ``0.10000000000000001``, ``9007199254740992`` vs
    ``9007199254740993``) are not, and this stays correct beyond float
    range (``1e400`` vs ``2e400``, ``1e-400`` vs ``0`` differ;
    ``1e400`` equals ``10e399``).
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float, Decimal)) and isinstance(b, (int, float, Decimal)):
        return Decimal(a) == Decimal(b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    return a == b


def _encode_compare_json(value, level=0):
    """Serialize compare/history output, writing ``Decimal`` numbers literally.

    Compare- and history-side reports parse every JSON number into the
    ``Decimal`` of its literal text, so a plain ``json.dumps`` would turn
    the value into a string or round it through float. This emits each
    ``Decimal`` via ``str`` as a raw, finite JSON number — fixed point or
    ``E`` exponent as ``Decimal`` chooses, still an exact legal number —
    rather than a string, a rounded float or an infinity; trailing zeros
    and exponent spelling are not guaranteed. Nested containers are
    indented exactly like ``json.dumps(..., indent=2)``.
    """
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        inner = "\n" + "  " * (level + 1)
        pieces = [inner + _encode_compare_json(item, level + 1) for item in value]
        closer = "\n" + "  " * level
        return "[" + ",".join(pieces) + closer + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        inner = "\n" + "  " * (level + 1)
        pieces = [inner + json.dumps(str(key), ensure_ascii=False) + ": "
                  + _encode_compare_json(item, level + 1)
                  for key, item in value.items()]
        closer = "\n" + "  " * level
        return "{" + ",".join(pieces) + closer + "}"
    raise TypeError(f"cannot serialize {type(value).__name__} in compare output")


def _compare_dumps(value):
    """Serialize compare/history output with exact numeric values preserved."""
    return _encode_compare_json(value)


def compare_reports(root, jobs, output, before, after):
    """Compare two historical run reports without running anything.

    Returns ``{"jobs": [...]}`` in current plan order covering every task
    recorded in at least one report; each entry has ``name``, ``before``,
    ``after`` and ``change``. A side without a record is ``null``; present
    sides keep only ``status`` and the matching payload (``result`` for
    completed, ``error`` for failed, ``blocked_by`` for blocked, the last
    in current dependency declaration order). ``change`` is ``added``,
    ``removed``, ``changed`` (status or payload differs, including the
    failure message) or ``unchanged``. Numbers nested anywhere in
    ``result`` compare by the exact decimal value each report expresses:
    ``1``, ``1.0`` and ``1e0`` are equal and negative zero equals zero,
    but ``0.10000000000000001`` differs from ``0.1``,
    ``9007199254740993`` from ``9007199254740992``, ``2e400`` from
    ``1e400`` and ``1e-400`` from ``0`` (while ``1e400`` equals
    ``10e399``); otherwise JSON value equality applies — booleans differ
    from numbers, numbers differ from numeric strings, object key order
    is ignored, array order matters — and ``blocked_by`` compares as a
    set. Present sides preserve the report's numbers as numbers
    (``Decimal``) rather than strings or rounded floats.

    The whole plan and output path are validated exactly like a run, but
    nothing is executed, created or written and task inputs are never
    read; ``before`` and ``after`` may name the same file. Both reports
    are fully read and validated (syntactically legal long and
    out-of-float-range exponents included) before the comparison is
    returned; ``NaN``, ``Infinity`` and ``-Infinity`` are rejected even
    inside ignored extra fields, though the same words inside strings
    are fine. Any plan, output or report violation raises ValueError.
    """
    deps_by_name, _report_path = _validate_plan(root, jobs, output)
    before_records = _read_compare_report(root, before, deps_by_name,
                                          exact_numbers=True)
    after_records = _read_compare_report(root, after, deps_by_name,
                                         exact_numbers=True)
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


def explain_report(root, jobs, output, report):
    """Explain read-only why a report's failed or blocked tasks did not pass.

    Returns ``{"jobs": [...]}`` in current plan order covering exactly the
    tasks the report records as ``failed`` or ``blocked``; each entry has
    only ``name``, ``status`` and ``causes``. A ``failed`` task has one
    cause — itself, ``status`` ``failed`` with the report's error string
    untouched. A ``blocked`` task follows that report's ``blocked_by``
    chains level by level: a ``failed`` record is a terminal cause, a
    ``blocked`` record is expanded through its own ``blocked_by``, and a
    task with no record is a terminal ``unrecorded`` cause with
    ``error`` ``null``. Only the report is traced — current file contents
    and other declared dependencies are never inferred. A ``blocked_by``
    that names a task the report records as ``completed`` is impossible
    and raises ValueError. Causes are de-duplicated by task name and
    emitted in current plan order, independently of report record order;
    unrecorded tasks appear only as terminal causes, never as jobs.

    The whole plan, every input path, the output path (``report`` may
    equal it) and the report are validated exactly like ``compare_reports``
    — including its strict payload checks — before any tracing, so a
    partial report is valid and unrecorded branches are covered by plan
    validation. Nothing is executed, created or written and task inputs
    are never read.
    """
    deps_by_name, _report_path = _validate_plan(root, jobs, output)
    records = _read_compare_report(root, report, deps_by_name)

    def terminal_causes(name):
        """Trace one recorded blocked task to its de-duplicated terminal causes.

        Traversal order and repeat filtering are internal; the caller
        orders the resulting names by the current plan.
        """
        terminals = []
        seen = set()
        stack = list(records[name]["blocked_by"])
        while stack:
            dep = stack.pop(0)
            if dep in seen:
                continue
            seen.add(dep)
            record = records.get(dep)
            if record is None:
                terminals.append(dep)
            elif record["status"] == "failed":
                terminals.append(dep)
            elif record["status"] == "blocked":
                stack.extend(record["blocked_by"])
            else:
                raise ValueError(
                    f"report task {name!r}: blocked_by names {dep!r}, "
                    "which the report records as completed")
        return terminals

    plan_order = [job["name"] for job in jobs]
    explained = []
    for name in plan_order:
        record = records.get(name)
        if record is None or record["status"] == "completed":
            continue
        if record["status"] == "failed":
            causes = [{"name": name, "status": "failed",
                       "error": record["error"]}]
        else:
            names = set(terminal_causes(name))
            causes = []
            for cause_name in plan_order:
                if cause_name not in names:
                    continue
                cause = records.get(cause_name)
                if cause is None:
                    causes.append({"name": cause_name, "status": "unrecorded",
                                   "error": None})
                else:
                    causes.append({"name": cause_name, "status": "failed",
                                   "error": cause["error"]})
        explained.append({"name": name, "status": record["status"],
                          "causes": causes})
    return {"jobs": explained}


def query_history(root, jobs, output, reports, targets=None):
    """Show each task's record across several reports, read-only.

    Returns ``{"jobs": [...]}`` in current plan order; each entry has only
    ``name`` and ``history``, a list as long as ``reports`` with entries in
    the exact order the report paths were given — duplicates create
    duplicate positions and nothing is sorted or scanned. Each history
    entry has only ``report`` (the path verbatim) and ``record``: ``null``
    when the report has no record for the task, otherwise ``status`` plus
    its matching payload, shaped and ordered exactly like a
    ``compare_reports`` side (``result`` object, ``error`` string or
    ``blocked_by`` in current dependency declaration order); extra fields
    are ignored. A missing record is never a failure and is never filled
    from a neighbouring report. Numbers nested anywhere in a
    ``completed`` record's ``result`` keep the exact decimal value the
    report expresses, exactly as in ``compare_reports``: they are
    returned as ``Decimal`` — never rounded to float, collapsed to zero
    or infinity, or turned into strings — so ``0.10000000000000001``,
    ``9007199254740993.0``, ``1e400`` and ``1e-400`` survive verbatim,
    and ``NaN``/``Infinity``/``-Infinity`` constants anywhere in a report
    (even in ignored extra fields) raise ValueError while the same words
    inside strings are fine.

    Without targets only tasks appearing in at least one report are
    listed, so all-empty reports give ``{"jobs": []}``. With targets the
    listed tasks are exactly those named — the prerequisite closure is
    not expanded, argument order does not reorder anything, and a task no
    report ever records still appears with an all-``null`` history.

    The whole plan, every input path and the output path are validated
    first, exactly like ``compare_reports``, then every report is fully
    validated under that entry point's strict rules — including records
    outside the target scope — so no partial history is returned on
    error. ``reports`` must be a nonempty list of nonblank path strings
    (repeats allowed); it or an illegal target raises ValueError too.
    Nothing is executed, created or written, task inputs are never read,
    and a report path may equal ``output``.
    """
    deps_by_name, _report_path = _validate_plan(root, jobs, output)
    if not isinstance(reports, list) or not reports:
        raise ValueError("reports must be a nonempty list of report paths")
    if any(not isinstance(report, str) or not report.strip() for report in reports):
        raise ValueError("report paths must be nonblank strings")
    if targets is not None:
        _validate_target_names(jobs, targets)
        wanted = list(targets)
    else:
        wanted = None
    report_records = []
    for report in reports:
        records = _read_compare_report(root, report, deps_by_name,
                                       exact_numbers=True)
        report_records.append(records)
    history_jobs = []
    for job in jobs:
        name = job["name"]
        if wanted is not None:
            if name not in wanted:
                continue
        elif not any(name in records for records in report_records):
            continue
        history = [{"report": report, "record": records.get(name)}
                   for report, records in zip(reports, report_records)]
        history_jobs.append({"name": name, "history": history})
    return {"jobs": history_jobs}


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


def _changed_targets(root, jobs, deps_by_name, changed_inputs):
    """Return (directly changed task names, affected target names), plan-ordered.

    Shared by ``preview_changes`` and ``run_changes`` so both always select
    the same scope for the same input. ``changed_inputs`` must be a list of
    nonblank strings; each path is resolved like a plan input path, so
    absolute paths and symlink escapes raise ValueError. Matching uses
    resolved paths only: files need not exist and are never read, paths
    resolving to the same file are merged, and argument order never
    changes the result. Targets are the directly changed tasks plus every
    task downstream of them along declared dependencies.
    """
    if not isinstance(changed_inputs, list):
        raise ValueError("changed_inputs must be a list of paths")
    if any(not isinstance(path, str) or not path.strip() for path in changed_inputs):
        raise ValueError("changed inputs must be nonblank strings")
    resolved = {local_path(root, path) for path in changed_inputs}
    hits = [job["name"] for job in jobs
            if local_path(root, job["input"]) in resolved]
    if not hits:
        return [], []
    dependents = {name: [] for name in deps_by_name}
    for name, deps in deps_by_name.items():
        for dep in deps:
            dependents[dep].append(name)
    affected = set(hits)
    stack = list(hits)
    while stack:
        node = stack.pop()
        for child in dependents[node]:
            if child not in affected:
                affected.add(child)
                stack.append(child)
    return hits, [job["name"] for job in jobs if job["name"] in affected]


def _change_entries(jobs, deps_by_name, hits, targets):
    """Preview entries for a change run, with the ``triggered_by`` field added.

    Shared by ``preview_changes`` and the recorded ``execution`` block of
    ``run_changes`` so both always express the same reasons. ``hits`` are
    the directly changed tasks and ``targets`` the hits plus every
    downstream task, both plan-ordered; the entries otherwise match
    ``preview_plan`` with targets and gain ``triggered_by`` — the directly
    changed tasks reaching each job along dependency edges, in plan order
    (``[]`` for jobs included only as prerequisites).
    """
    entries = _preview_entries(jobs, deps_by_name, targets)
    dependents = {name: [] for name in deps_by_name}
    for name, deps in deps_by_name.items():
        for dep in deps:
            dependents[dep].append(name)
    reachable = {name: set() for name in targets}
    for hit in hits:
        seen = set()
        stack = [hit]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            reachable[node].add(hit)
            stack.extend(dependents[node])
    for entry in entries:
        entry["triggered_by"] = [hit for hit in hits
                                 if hit in reachable.get(entry["name"], set())]
    return entries


def preview_changes(root, jobs, output, changed_inputs):
    """Preview the tasks affected by changed input files, without touching files.

    ``changed_inputs`` is a list of root-relative path strings (an empty
    list is allowed); each is resolved like any plan path — absolute paths
    and symlink escapes raise ValueError — and paths resolving to the same
    file are merged, so argument order never changes the result. Matching
    uses resolved paths only: files need not exist and are never read, and
    a legal path matching no task input simply produces no targets.

    Returns ``{"targets": [...], "jobs": [...]}``. Targets are the tasks
    whose resolved input path was directly changed plus every task
    downstream of them along declared dependencies, each once, in original
    plan order. Jobs cover the targets and all their prerequisites, each
    once, in the usual preview processing order with the same depends_on,
    reason and required_by semantics as ``preview_plan`` with targets,
    plus ``triggered_by``: the directly changed tasks that can reach the
    job along dependency edges, in original plan order, including the job
    itself when directly changed; jobs included only as prerequisites get
    ``[]``. With no changed inputs or no matching task the result is
    ``{"targets": [], "jobs": []}``.

    The whole plan, every input path and the output path are validated
    exactly like a run before the scope is chosen, even for an empty
    result. Nothing is executed, created or written, no report is read and
    task inputs are never read.
    """
    deps_by_name, _report_path = _validate_plan(root, jobs, output)
    hits, targets = _changed_targets(root, jobs, deps_by_name, changed_inputs)
    if not targets:
        return {"targets": [], "jobs": []}
    entries = _change_entries(jobs, deps_by_name, hits, targets)
    return {"targets": targets, "jobs": entries}


def _reasons_enabled(record_reasons):
    """The record_reasons option accepts booleans only; default off."""
    if not isinstance(record_reasons, bool):
        raise ValueError("record_reasons must be a boolean")
    return record_reasons


def _run_selected(root, selected, deps_by_name, report_path, reasons=None):
    """Execute an already-selected plan-ordered job list and write its report.

    Processing order, failure records and blocked propagation are shared by
    every execution mode: each pass runs the earliest ready job, a failing
    or blocked direct dependency blocks dependants without reading their
    input, and only this run's results drive dependency decisions. The
    report contains exactly these results, replacing any prior content.

    When ``reasons`` is the planned ``{"mode", "targets", "jobs"}`` block
    it is recorded under the report's top-level ``execution`` key alongside
    ``results``; the reasons are fixed before execution, so failed and
    blocked tasks keep their selected reason and the block is never
    rewritten from the run's outcomes. The jobs list the results share
    names and order with, each once.
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
    report = {"results": results}
    if reasons is not None:
        report["execution"] = reasons
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return results


def run_plan(root, jobs, output, targets=None, record_reasons=False):
    """Execute the plan (or targets plus prerequisites) and write its report.

    With ``record_reasons`` off (the default) behavior, the returned list
    and the report are exactly the historical ones. With it on and at
    least one processed job, the report additionally carries a top-level
    ``execution`` object with only ``mode`` (``"all"`` without targets,
    otherwise ``"only"`` even when the targets cover the whole plan),
    ``targets`` (every plan job for ``all``, else the explicit targets in
    plan order) and ``jobs`` — the ``preview_plan`` entries for the same
    targets, same names and order as the returned results. ``record_reasons``
    must be a boolean; any other type raises ValueError.
    """
    _reasons_enabled(record_reasons)
    deps_by_name, report_path = _validate_plan(root, jobs, output)
    selected = _select_jobs(jobs, targets)
    reasons = None
    if record_reasons and selected:
        mode = "all" if targets is None else "only"
        plan_targets = [job["name"] for job in jobs] if targets is None \
            else [job["name"] for job in jobs if job["name"] in set(targets)]
        reasons = {"mode": mode, "targets": plan_targets,
                   "jobs": _preview_entries(jobs, deps_by_name, targets)}
    return _run_selected(root, selected, deps_by_name, report_path, reasons)


def run_retry(root, jobs, output, report, record_reasons=False):
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

    With ``record_reasons`` on and at least one retry target, the report
    additionally carries the top-level ``execution`` block with
    ``mode: "retry"``, ``targets`` the failed/blocked tasks in current
    plan order (taken from the report before any overwrite, so
    ``report == output`` still records the pre-overwrite reasons) and
    ``jobs`` the ``preview_retry`` entries for the same targets, sharing
    names and order with the returned results. With no retry targets no
    report is written regardless of the option. ``record_reasons`` must
    be a boolean; any other type raises ValueError.
    """
    _reasons_enabled(record_reasons)
    deps_by_name, report_path = _validate_plan(root, jobs, output)
    statuses = _read_retry_report(root, report, {job["name"] for job in jobs})
    targets = [job["name"] for job in jobs
               if statuses.get(job["name"]) in ("failed", "blocked")]
    if not targets:
        return []
    selected = _select_jobs(jobs, targets)
    reasons = {"mode": "retry", "targets": targets,
               "jobs": _preview_entries(jobs, deps_by_name, targets)} \
        if record_reasons else None
    return _run_selected(root, selected, deps_by_name, report_path, reasons)


def run_changes(root, jobs, output, changed_inputs, record_reasons=False):
    """Execute the tasks affected by changed input files.

    Target selection mirrors ``preview_changes``: the tasks whose resolved
    input path was directly changed plus every task downstream of them
    become targets in current plan order, and the run covers the targets
    plus every direct and indirect prerequisite, shared prerequisites
    running once. ``changed_inputs`` follows the same rules as the
    preview: a list of root-relative path strings (an empty list is
    allowed), resolved like plan input paths — absolute paths and symlink
    escapes raise ValueError — matched by resolved path only, so files
    need not exist, duplicates, aliases and argument order change
    nothing, and a legal path matching no task input produces no targets.

    Execution order, failure records and ``blocked_by`` propagation are
    exactly ``run_plan``'s: input read failures, unknown operations and
    invalid contents are recorded ``failed`` with their error, independent
    branches keep running, blocked tasks never read their input, and
    dependency decisions use only this run's results. The returned list
    holds this run's records in actual processing order and is written to
    ``output`` under ``results``, replacing any prior content; tasks
    outside the scope are not read, produce no records and cannot affect
    the outcome.

    With no changed inputs or no matching task the empty list is returned
    without reading task inputs, creating directories or touching the
    report. The whole plan (including unselected branches and an empty
    scope) and the output path are validated before the scope is chosen;
    any plan, dependency, path or ``changed_inputs`` violation raises
    ValueError before anything runs. Creating the output directory or
    writing the report may raise OSError after tasks have run; executed
    tasks are not undone.

    With ``record_reasons`` on and at least one target, the report
    additionally carries the top-level ``execution`` block with
    ``mode: "changes"``, ``targets`` the directly changed tasks plus
    their downstream tasks in plan order and ``jobs`` exactly the
    ``preview_changes`` entries (including ``triggered_by``) for the same
    inputs, sharing names and order with the returned results; duplicate
    and alias paths change neither targets nor reasons. With no match no
    report is written regardless of the option. ``record_reasons`` must
    be a boolean; any other type raises ValueError.
    """
    _reasons_enabled(record_reasons)
    deps_by_name, report_path = _validate_plan(root, jobs, output)
    hits, targets = _changed_targets(root, jobs, deps_by_name, changed_inputs)
    if not targets:
        return []
    selected = _select_jobs(jobs, targets)
    reasons = {"mode": "changes", "targets": targets,
               "jobs": _change_entries(jobs, deps_by_name, hits, targets)} \
        if record_reasons else None
    return _run_selected(root, selected, deps_by_name, report_path, reasons)


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
    parser.add_argument("--explain", metavar="REPORT",
                        help="explain read-only why the report's failed/blocked tasks did not pass")
    parser.add_argument("--history", action="append", default=None, metavar="REPORT",
                        help="show each task's record across REPORT (repeatable), read-only")
    parser.add_argument("--changed", action="append", default=None, metavar="PATH",
                        help="preview tasks affected by changed input PATH (repeatable), read-only")
    parser.add_argument("--run-changed", action="append", default=None, metavar="PATH",
                        help="run tasks affected by changed input PATH (repeatable)")
    parser.add_argument("--record-reasons", action="store_true",
                        help="record why tasks entered this run in the report's execution block")
    args = parser.parse_args()
    try:
        if args.record_reasons and (args.preview or args.retry_preview is not None
                                    or args.compare is not None
                                    or args.explain is not None
                                    or args.history is not None
                                    or args.changed is not None):
            raise ValueError("--record-reasons cannot be combined with --preview, "
                             "--retry-preview, --compare, --explain, --history or --changed")
        if args.run_changed is not None and (args.only is not None or args.preview
                                             or args.retry_preview is not None
                                             or args.retry is not None
                                             or args.compare is not None
                                             or args.explain is not None
                                             or args.history is not None
                                             or args.changed is not None):
            raise ValueError("--run-changed cannot be combined with --only, --preview, "
                             "--retry-preview, --retry, --compare, --explain, "
                             "--history or --changed")
        if args.changed is not None and (args.only is not None or args.preview
                                         or args.retry_preview is not None
                                         or args.retry is not None
                                         or args.compare is not None
                                         or args.explain is not None
                                         or args.history is not None):
            raise ValueError("--changed cannot be combined with --only, --preview, "
                             "--retry-preview, --retry, --compare, --explain or --history")
        if args.history is not None and (args.preview or args.retry_preview is not None
                                         or args.retry is not None
                                         or args.compare is not None
                                         or args.explain is not None):
            raise ValueError("--history cannot be combined with --preview, "
                             "--retry-preview, --retry, --compare or --explain")
        if args.explain is not None and (args.only is not None or args.preview
                                         or args.retry_preview is not None
                                         or args.retry is not None
                                         or args.compare is not None
                                         or args.history is not None):
            raise ValueError("--explain cannot be combined with --only, --preview, "
                             "--retry-preview, --retry, --compare or --history")
        if args.compare is not None and (args.preview or args.only is not None
                                         or args.retry_preview is not None
                                         or args.retry is not None
                                         or args.history is not None):
            raise ValueError("--compare cannot be combined with --only, --preview, "
                             "--retry-preview, --retry or --history")
        if args.retry_preview is not None and (args.preview or args.only is not None
                                               or args.history is not None):
            raise ValueError("--retry-preview cannot be combined with --preview, --only or --history")
        if args.retry is not None and (args.preview or args.only is not None
                                       or args.retry_preview is not None
                                       or args.history is not None):
            raise ValueError("--retry cannot be combined with --only, --preview, "
                             "--retry-preview or --history")
        plan = local_path(args.root, args.plan)
        if plan == local_path(args.root, args.output):
            raise ValueError("report cannot overwrite its plan")
        jobs = json.loads(plan.read_text(encoding="utf-8"))["jobs"]
        if args.changed is not None:
            print(json.dumps(preview_changes(args.root, jobs, args.output, args.changed),
                             ensure_ascii=False, indent=2))
            return 0
        if args.history is not None:
            print(_compare_dumps(query_history(args.root, jobs, args.output,
                                               args.history, targets=args.only)))
            return 0
        if args.compare is not None:
            print(_compare_dumps(compare_reports(args.root, jobs, args.output,
                                                 args.compare[0], args.compare[1])))
            return 0
        if args.explain is not None:
            print(json.dumps(explain_report(args.root, jobs, args.output, args.explain),
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
            results = run_retry(args.root, jobs, args.output, args.retry,
                                record_reasons=args.record_reasons)
        elif args.run_changed is not None:
            results = run_changes(args.root, jobs, args.output, args.run_changed,
                                  record_reasons=args.record_reasons)
        else:
            results = run_plan(args.root, jobs, args.output, targets=args.only,
                               record_reasons=args.record_reasons)
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
