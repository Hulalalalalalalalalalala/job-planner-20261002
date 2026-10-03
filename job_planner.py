"""Run named deterministic file tasks within a repository directory."""
import argparse
import csv
import hashlib
import io
import json
import os
import stat
import tempfile
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


def _load_report_data(root, report, exact_numbers=False):
    """Read a report file into its parsed top-level JSON object.

    Only the report file is read. The path must be relative to root and,
    after resolving symlinks, stay inside root. The report must be a UTF-8
    JSON object whose ``results`` is a list; per-entry name, status and
    payload checks are left to the caller so each entry point can enforce
    its own rules. Any violation raises ValueError.

    With ``exact_numbers`` every JSON number is kept as the ``Decimal`` of
    its literal text so comparisons can use the value the report actually
    expressed instead of a lossy float. Either way the non-JSON constants
    ``NaN``/``Infinity``/``-Infinity`` are rejected anywhere in the file,
    even inside an otherwise ignored field, while the same words written
    as JSON strings are untouched.
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
            data = json.loads(text, parse_float=Decimal,
                              parse_constant=_reject_json_constant)
        else:
            data = json.loads(text, parse_constant=_reject_json_constant)
    except ValueError as exc:
        raise ValueError(f"report {report!r} is not valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("report must be a JSON object")
    if not isinstance(data.get("results"), list):
        raise ValueError("report results must be a list")
    return data


def _load_report_entries(root, report, names, exact_numbers=False):
    """Read a report file into its raw ``results`` entries.

    Follows ``_load_report_data`` exactly and returns only its ``results``
    list; ``names`` is accepted for caller symmetry and not needed here.
    """
    return _load_report_data(root, report, exact_numbers=exact_numbers)["results"]


def _read_retry_report(root, report, names):
    """Read and validate a historical run report; return {name: status}.

    Names must match the current plan exactly, with no repeats, and one of
    the three statuses; other entry fields are ignored. A bare
    ``NaN``/``Infinity``/``-Infinity`` constant anywhere in the file is
    still invalid JSON, even inside an ignored field or an empty results
    list; the same words inside strings are accepted. Any violation
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


def _validate_compare_entries(results, deps_by_name):
    """Validate raw ``results`` entries under the compare report's strict rules.

    Every entry must be an object with a known, unrepeated task name and
    one of the three statuses, and every status must carry its matching
    payload: ``completed`` needs an object ``result``, ``failed`` a string
    ``error`` and ``blocked`` a nonempty, duplicate-free list naming only
    the job's declared direct dependencies. Extra entry fields are
    ignored. Returns {name: side entry} with ``blocked_by`` in current
    declaration order so callers compare it as a set independently of
    report record order. Any violation raises ValueError.
    """
    names = set(deps_by_name)
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
    then every JSON number is kept as ``Decimal``; explain and the other
    readers keep ordinary float parsing, but all readers reject non-JSON
    numeric constants.
    """
    results = _load_report_entries(root, report, set(deps_by_name),
                                   exact_numbers=exact_numbers)
    return _validate_compare_entries(results, deps_by_name)


def _json_equal(a, b):
    """Compare parsed JSON values the way JSON itself defines equality.

    Booleans are not numbers (``true`` differs from ``1``), numbers never
    equal strings, object key order is irrelevant, array order is
    significant, and nesting is compared to arbitrary depth. The walk uses
    an explicit stack rather than Python's call stack, so a legal report
    whose result nests hundreds of objects or arrays cannot trigger a
    RecursionError, and the caller's recursion limit is never changed.
    Compare-side reports keep every number as the ``Decimal`` of its
    literal text, so numbers compare by their exact decimal value: ``1``,
    ``1.0`` and ``1e0`` are equal and negative zero equals zero, while
    values that merely share a double-precision representation
    (``0.1`` vs ``0.10000000000000001``, ``9007199254740992`` vs
    ``9007199254740993``) are not, and this stays correct beyond float
    range (``1e400`` vs ``2e400``, ``1e-400`` vs ``0`` differ;
    ``1e400`` equals ``10e399``).
    """
    pending = [(a, b)]
    while pending:
        x, y = pending.pop()
        if isinstance(x, bool) or isinstance(y, bool):
            if not (isinstance(x, bool) and isinstance(y, bool) and x == y):
                return False
            continue
        if isinstance(x, (int, float, Decimal)) and isinstance(y, (int, float, Decimal)):
            if Decimal(x) != Decimal(y):
                return False
            continue
        if isinstance(x, dict) and isinstance(y, dict):
            if x.keys() != y.keys():
                return False
            pending.extend((x[k], y[k]) for k in x)
            continue
        if isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                return False
            pending.extend(zip(x, y))
            continue
        if x != y:
            return False
    return True


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

    Containers are walked with an explicit frame stack rather than Python
    recursion, so serializing a result hundreds of levels deep (through the
    already-deeper CLI call stack) cannot raise RecursionError and the
    caller's recursion limit is never touched.
    """
    parts = []
    # Pending-container frames: [kind, items, level, next_index]; kind 0 is
    # a list/tuple (items are values) and kind 1 is a dict (items are
    # (key, value) pairs). The value currently being rendered is ``v``.
    frames = []
    v = value
    while True:
        if v is None:
            parts.append("null")
        elif v is True:
            parts.append("true")
        elif v is False:
            parts.append("false")
        elif isinstance(v, str):
            parts.append(json.dumps(v, ensure_ascii=False))
        elif isinstance(v, Decimal):
            parts.append(str(v))
        elif isinstance(v, int):
            parts.append(str(v))
        elif isinstance(v, float):
            parts.append(repr(v))
        elif isinstance(v, (list, tuple)):
            if not v:
                parts.append("[]")
            else:
                parts.append("[")
                frames.append([0, v, len(frames), 0])
        elif isinstance(v, dict):
            if not v:
                parts.append("{}")
            else:
                parts.append("{")
                frames.append([1, list(v.items()), len(frames), 0])
        else:
            raise TypeError(f"cannot serialize {type(v).__name__} in compare output")
        # Descend into the nearest pending child, or close finished frames.
        while True:
            if not frames:
                return "".join(parts)
            kind, items, frame_level, index = frames[-1]
            if index < len(items):
                frames[-1][3] += 1
                if index:
                    parts.append(",")
                parts.append("\n" + "  " * (frame_level + 1))
                if kind == 1:
                    key, v = items[index]
                    parts.append(json.dumps(str(key), ensure_ascii=False) + ": ")
                else:
                    v = items[index]
                break
            parts.append("\n" + "  " * frame_level + ("]" if kind == 0 else "}"))
            frames.pop()


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
    — including its strict payload checks and the rejection of bare
    ``NaN``/``Infinity``/``-Infinity`` constants anywhere in the file
    (the same words inside strings stay valid, including inside the
    reported error text) — before any tracing, so a partial report is
    valid and unrecorded branches are covered by plan validation. Nothing
    is executed, created or written and task inputs are never read.
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


def _validate_execution_name_list(value, target_names, field, name):
    """Check a recorded ``required_by``/``triggered_by`` list against the targets.

    Only the type, duplicates and references are enforced: every entry
    must be a string naming one of the execution's targets and no entry
    may repeat. The recorded values themselves are trusted — the
    historical selection is never recomputed — and returned unchanged.
    """
    if not isinstance(value, list):
        raise ValueError(f"execution job {name!r}: {field} must be a list")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"execution job {name!r}: {field} entries must be strings")
    if len(set(value)) != len(value):
        raise ValueError(f"execution job {name!r}: {field} contains a duplicate")
    unknown = [item for item in value if item not in target_names]
    if unknown:
        raise ValueError(f"execution job {name!r}: {field} names {unknown[0]!r}, "
                         "which is not an execution target")


def query_execution(root, jobs, output, report):
    """Return a report's recorded execution reasons, read-only.

    Returns ``{"execution": ...}`` where the value is exactly the
    ``execution`` object a ``record_reasons`` run saved in the report —
    ``mode``, ``targets`` and ``jobs`` — reduced to the published reason
    fields: each job keeps only ``name``, ``depends_on``, ``reason`` and
    ``required_by``, plus ``triggered_by`` for ``changes`` mode; every
    array keeps the order the report recorded and every value is returned
    as recorded, never recomputed from the current plan or the run's
    outcomes (failed and blocked records do not rewrite reasons). A
    report without an ``execution`` field — one written before reasons
    could be recorded or with the option off — and a report with empty
    ``results`` both return ``{"execution": None}``.

    The whole plan, every input path and the output path (``report`` may
    equal it) are validated exactly like ``compare_reports``, and the
    report's ``results`` must satisfy that entry point's strict rules —
    so a partial report is valid. When ``execution`` is present it must
    be an object: ``mode`` one of ``all``/``only``/``retry``/``changes``;
    ``targets`` a nonempty, duplicate-free list of current plan job
    names; ``jobs`` a nonempty list of objects whose names and order
    match ``results`` exactly and which cover every target. Each job's
    ``depends_on`` must equal the current declaration, ``required_by``
    may only reference targets, ``changes`` jobs must also carry a
    ``triggered_by`` referencing only targets, and no task-name list may
    repeat. For ``all`` the targets must cover the whole plan and every
    job needs reason ``all`` with an empty ``required_by``; otherwise
    each job's reason must be ``target`` or ``prerequisite`` according to
    whether it is a target. Beyond types, references and these
    correspondences the recorded reason lists are not checked against the
    dependency graph — the historical selection is preserved, not
    recomputed. Missing fields, wrong types, unknown references, an
    illegal mode, a broken correspondence or an ``execution`` that is
    ``null`` all raise ValueError; extra fields are ignored.

    The query reads only the plan and the named report: nothing is
    executed, task inputs are never read, and no directory is created or
    file written. A missing, unreadable, non-UTF-8 or invalid-JSON
    report — including one containing a bare
    ``NaN``/``Infinity``/``-Infinity`` constant anywhere, even inside
    ``execution`` or an ignored extra field and even when ``results`` is
    empty or ``execution`` absent; the same words inside strings are
    fine — an illegal payload, or an absolute or symlink-escaping path
    raises ValueError.
    """
    deps_by_name, _report_path = _validate_plan(root, jobs, output)
    data = _load_report_data(root, report)
    results = data["results"]
    _validate_compare_entries(results, deps_by_name)
    if "execution" not in data:
        return {"execution": None}
    execution = data["execution"]
    if execution is None:
        raise ValueError("report execution must be an object")
    if not results:
        return {"execution": None}
    if not isinstance(execution, dict):
        raise ValueError("report execution must be an object")
    mode = execution.get("mode")
    if mode not in ("all", "only", "retry", "changes"):
        raise ValueError(f"report execution has invalid mode {mode!r}")
    targets = _validate_target_names(jobs, execution.get("targets"))
    target_names = set(targets)
    exec_jobs = execution.get("jobs")
    if not isinstance(exec_jobs, list) or not exec_jobs:
        raise ValueError("execution jobs must be a nonempty list of objects")
    if len(exec_jobs) != len(results):
        raise ValueError("execution jobs must match the report results one to one")
    kept = []
    for entry, result in zip(exec_jobs, results):
        if not isinstance(entry, dict):
            raise ValueError("each execution job must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or name != result["name"]:
            raise ValueError(
                "execution jobs must match the report results in name and order")
        depends_on = entry.get("depends_on")
        if depends_on != deps_by_name[name]:
            raise ValueError(
                f"execution job {name!r}: depends_on differs from the current "
                "declaration")
        reason = entry.get("reason")
        required_by = entry.get("required_by")
        _validate_execution_name_list(required_by, target_names,
                                       "required_by", name)
        if mode == "all":
            if reason != "all":
                raise ValueError(
                    f"execution job {name!r}: all mode requires reason 'all'")
            if required_by:
                raise ValueError(
                    f"execution job {name!r}: all mode requires an empty "
                    "required_by")
        else:
            expected = "target" if name in target_names else "prerequisite"
            if reason != expected:
                raise ValueError(
                    f"execution job {name!r}: reason must be {expected!r}")
        job = {"name": name, "depends_on": list(depends_on), "reason": reason,
               "required_by": list(required_by)}
        if mode == "changes":
            triggered_by = entry.get("triggered_by")
            _validate_execution_name_list(triggered_by, target_names,
                                          "triggered_by", name)
            job["triggered_by"] = list(triggered_by)
        kept.append(job)
    job_names = {job["name"] for job in kept}
    missing = [target for target in targets if target not in job_names]
    if missing:
        raise ValueError(
            f"execution target {missing[0]!r} is missing from execution jobs")
    if mode == "all" and target_names != set(deps_by_name):
        raise ValueError("all mode execution targets must cover the whole plan")
    return {"execution": {"mode": mode, "targets": list(targets), "jobs": kept}}


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
    are never read; only the plan and the named report are read. Like
    every report reader this rejects the non-JSON constants
    ``NaN``/``Infinity``/``-Infinity`` anywhere in the file — even in
    ignored fields and with an empty ``results`` — while the same words
    inside strings are harmless; syntactically legal numbers such as
    ``1e400`` keep their existing, lenient parsing here.
    """
    deps_by_name, _report_path = _validate_plan(root, jobs, output)
    statuses = _read_retry_report(root, report, {job["name"] for job in jobs})
    targets = [job["name"] for job in jobs
               if statuses.get(job["name"]) in ("failed", "blocked")]
    if not targets:
        return {"targets": [], "jobs": []}
    return {"targets": targets, "jobs": _preview_entries(jobs, deps_by_name, targets)}


def _read_history_reports(root, deps_by_name, reports):
    """Validate ``reports`` and return one validated record map per report.

    Shared by the history-retry entry points: the list must be nonempty and
    hold nonblank path strings, and every report is fully validated under
    the compare report's strict rules — including out-of-scope and
    overridden records — before any selection happens.
    """
    if not isinstance(reports, list) or not reports:
        raise ValueError("reports must be a nonempty list of report paths")
    if any(not isinstance(report, str) or not report.strip() for report in reports):
        raise ValueError("report paths must be nonblank strings")
    return [_read_compare_report(root, report, deps_by_name,
                                 exact_numbers=True)
            for report in reports]


def _history_retry_selection(jobs, reports, report_records):
    """Return the retry range's ``(targets, sources)`` from validated records.

    Reports are applied strictly in the given order (repeats keep separate
    positions) and each task is judged by the last report recording it — a
    later report missing the task never clears an older status. Targets are
    the tasks whose latest record is ``failed`` or ``blocked``, in current
    plan order; sources parallels targets with each target's latest status,
    report path and zero-based report index.
    """
    latest = {}
    for index, (report, records) in enumerate(zip(reports, report_records)):
        for name, record in records.items():
            latest[name] = {"status": record["status"],
                            "report": report, "index": index}
    targets = []
    sources = []
    for job in jobs:
        source = latest.get(job["name"])
        if source is not None and source["status"] in ("failed", "blocked"):
            targets.append(job["name"])
            sources.append({"name": job["name"], "status": source["status"],
                            "report": source["report"], "index": source["index"]})
    return targets, sources


def _history_retry_targets(root, jobs, deps_by_name, reports):
    """Validate ``reports`` and return the retry range's ``(targets, sources)``.

    Shared by ``preview_history_retry`` and ``run_history_retry`` so both
    always select the same range: reports are read strictly in the given
    order (repeats keep separate positions) and each task is judged by the
    last report recording it — a later report missing the task never clears
    an older status. Targets are the tasks whose latest record is
    ``failed`` or ``blocked``, in current plan order; sources parallels
    targets with each target's latest status, report path and zero-based
    report index. Every report is fully validated, including out-of-scope
    and overridden records.
    """
    report_records = _read_history_reports(root, deps_by_name, reports)
    return _history_retry_selection(jobs, reports, report_records)


def _validate_max_failures(max_failures):
    """Validate the failure limit: only ``None`` or a non-bool positive int."""
    if max_failures is None:
        return None
    if (isinstance(max_failures, bool) or not isinstance(max_failures, int)
            or max_failures < 1):
        raise ValueError("max_failures must be None or a positive integer")
    return max_failures


def _failure_counts(jobs, report_records):
    """Return each task's failure count across validated report records.

    Reports are applied in the given order, so a repeated report path counts
    at every position: a ``failed`` record adds one to the task's count, a
    ``completed`` record resets it to zero, and a ``blocked`` or missing
    record leaves it untouched.
    """
    counts = {job["name"]: 0 for job in jobs}
    for records in report_records:
        for name, record in records.items():
            if record["status"] == "completed":
                counts[name] = 0
            elif record["status"] == "failed":
                counts[name] += 1
    return counts


def preview_history_retry(root, jobs, output, reports, max_failures=None):
    """Preview a retry range from several historical reports, read-only.

    ``reports`` is a nonempty list of nonblank root-relative report paths,
    ordered oldest to newest by the caller; repeats keep separate positions
    and nothing is sorted by name or modification time and no other report
    is scanned. Each task is judged by the last report (highest index) that
    contains a record of it: a later report missing the task never clears
    an older status. Tasks whose latest record is ``failed`` or ``blocked``
    become retry targets, in current plan order; tasks whose latest record
    is ``completed`` and tasks no report ever records are not targets.

    Returns exactly ``{"targets": [...], "jobs": [...], "sources": [...]}``.
    ``jobs`` cover the targets and every direct and indirect prerequisite
    — completed or never-recorded prerequisites included, shared
    prerequisites once, unrelated branches omitted — with the fields,
    processing order and reasons of ``preview_plan`` with those targets.
    ``sources`` corresponds to ``targets`` one to one in the same order;
    each entry has only ``name`` (the target), ``status`` (its latest
    status), ``report`` (the source path verbatim) and ``index`` (that
    report's zero-based position in ``reports``). With no targets all
    three arrays are empty. The history only selects the range; the
    previewed run's dependency decisions would still follow the ordinary
    run rules.

    ``max_failures`` is ``None`` (the default, no limit, and the result
    keeps the three-field shape above) or a non-boolean positive integer;
    any other value raises ValueError. With a limit the result gains a
    fourth field, ``excluded``: every task starts at zero and the reports
    are walked in the given order — a repeated report counts at each of
    its positions — with a ``failed`` record adding one to the task's
    count, a ``completed`` record resetting it to zero, and a ``blocked``
    or missing record changing nothing. A candidate whose own count, or
    the count of any of its direct or indirect prerequisites, has reached
    the limit is excluded instead of targeted; each excluded entry has
    only ``name`` (the candidate), ``failures`` (its own count) and
    ``limited_by`` (every task that caused the exclusion — the candidate
    itself included when its own count reached the limit — in plan order
    without repeats). ``targets``, ``excluded`` and each ``limited_by``
    follow plan order; ``jobs`` and ``sources`` cover only the remaining
    targets, exactly as without a limit. With no candidates all four
    arrays are empty; when every candidate is excluded only ``excluded``
    is nonempty.

    The whole plan, every input path and the output path are validated
    first, exactly like ``query_history``, and then every report is fully
    validated under that entry point's path, UTF-8, structure, name,
    status and strict payload rules — including non-target tasks,
    overridden older records and reports after the newest record, with or
    without a failure limit and whether or not any target survives — so
    an illegal ``max_failures``, ``reports``, plan or report raises
    ValueError and no partial result is returned. A bare
    ``NaN``/``Infinity``/``-Infinity`` constant anywhere in any report is
    rejected while the same words inside strings are fine; syntactically
    legal JSON numbers are accepted. Nothing is executed, created or
    written, task inputs are never read, and a report path may equal
    ``output``.
    """
    deps_by_name, _report_path = _validate_plan(root, jobs, output)
    max_failures = _validate_max_failures(max_failures)
    report_records = _read_history_reports(root, deps_by_name, reports)
    targets, sources = _history_retry_selection(jobs, reports, report_records)
    if max_failures is None:
        if not targets:
            return {"targets": [], "jobs": [], "sources": []}
        return {"targets": targets,
                "jobs": _preview_entries(jobs, deps_by_name, targets),
                "sources": sources}
    counts = _failure_counts(jobs, report_records)
    plan_order = [job["name"] for job in jobs]
    kept_targets = []
    kept_sources = []
    excluded = []
    for target, source in zip(targets, sources):
        closure = {target}
        stack = [target]
        while stack:
            node = stack.pop()
            for dep in deps_by_name[node]:
                if dep not in closure:
                    closure.add(dep)
                    stack.append(dep)
        limited_by = [name for name in plan_order
                      if name in closure and counts[name] >= max_failures]
        if limited_by:
            excluded.append({"name": target, "failures": counts[target],
                             "limited_by": limited_by})
        else:
            kept_targets.append(target)
            kept_sources.append(source)
    entries = (_preview_entries(jobs, deps_by_name, kept_targets)
               if kept_targets else [])
    return {"targets": kept_targets, "jobs": entries,
            "sources": kept_sources, "excluded": excluded}


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
    entries = _preview_entries(jobs, deps_by_name, targets)
    # triggered_by: directly changed tasks reaching each target, plan order.
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
    return {"targets": targets, "jobs": entries}


def _apply_output_mode(fd, previous_mode):
    """Give the staged temporary file the permission bits the output should have.

    An existing output's mode is preserved; a fresh output gets the mode a
    plain create would have — 0666 minus the process umask — instead of
    the 0600 ``mkstemp`` forces. Permission bits are a Unix concept:
    platforms without a file-descriptor chmod interface, or where it is
    not implemented, skip this step and follow their normal file
    semantics — a missing or unimplemented interface is not a permission
    denial. A real denial from the interface itself is a filesystem error
    and propagates as OSError.
    """
    fchmod = getattr(os, "fchmod", None)
    if fchmod is None:
        return
    if previous_mode is not None:
        mode = previous_mode
    else:
        umask = os.umask(0)
        os.umask(umask)
        mode = 0o666 & ~umask
    try:
        fchmod(fd, mode)
    except NotImplementedError:
        pass


def _save_report(report_path, report):
    """Serialize ``report`` and atomically replace the output file with it.

    The report is fully serialized and UTF-8 encoded before the filesystem
    is touched: a report that cannot become one complete UTF-8 JSON
    document raises ValueError and nothing is created, changed or removed.
    The encoded bytes are then written to a temporary file in the same
    directory and moved over the target with ``os.replace``, so a
    filesystem error while creating the directory, creating the temporary
    file, setting its permissions, writing the content or replacing the
    output raises OSError (including when the output path already is a
    directory) with the pre-run output file preserved byte-for-byte — or
    still absent when there was none — and no temporary file or open
    handle left behind; empty parent directories created for the attempt
    may remain. ``report_path`` is already symlink-resolved, so a legal
    in-root symlink output updates the resolved target while the link
    itself is kept. The saved file keeps the permissions a plain
    overwrite would have: on Unix an existing output's mode is preserved
    and a new file follows the process umask, while platforms without a
    file-descriptor permission interface (such as Windows) follow their
    normal file semantics — see ``_apply_output_mode``.
    """
    try:
        data = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"report cannot be encoded as UTF-8 JSON: {exc}") from exc
    try:
        previous_mode = stat.S_IMODE(os.stat(report_path).st_mode)
    except OSError:
        previous_mode = None
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=report_path.parent,
                                     prefix=report_path.name + ".",
                                     suffix=".tmp")
    try:
        try:
            _apply_output_mode(fd, previous_mode)
            with os.fdopen(fd, "wb") as handle:
                fd = None  # the file object now owns the descriptor
                handle.write(data)
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        os.replace(temporary, report_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _run_selected(root, selected, deps_by_name, report_path, execution=None):
    """Execute an already-selected plan-ordered job list and write its report.

    Processing order, failure records and blocked propagation are shared by
    every execution mode: each pass runs the earliest ready job, a failing
    or blocked direct dependency blocks dependants without reading their
    input, and only this run's results drive dependency decisions. The
    report contains exactly these results, replacing any prior content;
    with ``execution`` set it is recorded alongside them as the top-level
    ``execution`` object explaining why this run happened. Saving follows
    ``_save_report``: only a successful save means the new results have
    fully replaced the old ones, and a failed save raises with the
    previous output file (or its absence) unchanged.
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
    if execution is not None:
        report["execution"] = execution
    _save_report(report_path, report)
    return results


def _require_bool(record_reasons):
    """Validate the record_reasons option: only a plain bool is accepted."""
    if not isinstance(record_reasons, bool):
        raise ValueError("record_reasons must be a boolean")
    return record_reasons


def _execution_for_run(root, jobs, output, mode, targets_or_report, changed_inputs=None):
    """Build the recorded ``execution`` object from the public previews.

    The mode's corresponding preview entry point computes both ``targets``
    and ``jobs`` exactly as users see them, so the recorded reasons can
    never drift from preview semantics and stay independent of this run's
    results: failed and blocked jobs keep the selection reasons they had
    before execution. ``targets_or_report`` is the explicit target list
    for ``"only"`` or the historical report path for ``"retry"``;
    ``changed_inputs`` only matters for ``"changes"``. Callers guarantee
    the scope is nonempty, so the previews' empty-list cases never
    surface here.
    """
    if mode == "all":
        preview = preview_plan(root, jobs, output, targets=None)
        execution_targets = [job["name"] for job in jobs]
    elif mode == "only":
        preview = preview_plan(root, jobs, output, targets=targets_or_report)
        chosen = set(targets_or_report)
        execution_targets = [job["name"] for job in jobs if job["name"] in chosen]
    elif mode == "retry":
        preview = preview_retry(root, jobs, output, targets_or_report)
        execution_targets = preview["targets"]
    else:
        preview = preview_changes(root, jobs, output, changed_inputs)
        execution_targets = preview["targets"]
    return {"mode": mode, "targets": execution_targets, "jobs": preview["jobs"]}


def run_plan(root, jobs, output, targets=None, record_reasons=False):
    """Execute the whole plan, or targets with every prerequisite.

    With ``record_reasons=True`` the report additionally carries a
    top-level ``execution`` object — ``{"mode", "targets", "jobs"}`` —
    describing why this run happened: ``mode`` is ``"all"`` without
    targets (an explicit selection naming every job still records
    ``"only"``), and ``targets`` lists this run's targets in current plan
    order — every plan job for ``"all"``, the explicit targets for
    ``"only"``, the report's failed/blocked tasks for ``"retry"`` and the
    directly changed tasks plus their downstream dependants for
    ``"changes"`` — and ``jobs`` are exactly the matching public preview
    entries (``preview_plan``, ``preview_retry`` or ``preview_changes``),
    same names and order as the run's ``results``. Reasons reflect the
    plan as selected, never this run's outcomes.
    ``record_reasons`` must be a bool; other values raise ValueError. The
    returned list, CLI statistics and exit codes are unchanged and only
    ``output`` is ever written; with an empty scope no report is written
    and no ``execution`` is recorded.

    Saving the report is atomic: a report that cannot be encoded as one
    complete UTF-8 JSON document raises ValueError, and a filesystem error
    while creating the output directory, saving the content or replacing
    the output raises OSError (the output path already being a directory
    included) after tasks have run. Either failure leaves the pre-run
    output file byte-for-byte intact — or still absent when there was
    none — with no partial or extra files left behind (empty parent
    directories created for the attempt may remain); executed tasks are
    not undone and a failed save never turns their results into failures.
    Only a successful save means the new results have fully replaced the
    old ones.
    """
    _require_bool(record_reasons)
    deps_by_name, report_path = _validate_plan(root, jobs, output)
    selected = _select_jobs(jobs, targets)
    execution = None
    if record_reasons and selected:
        mode = "all" if targets is None else "only"
        execution = _execution_for_run(root, jobs, output, mode, targets)
    return _run_selected(root, selected, deps_by_name, report_path, execution)


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
    exactly as in ``preview_retry`` — including a bare
    ``NaN``/``Infinity``/``-Infinity`` constant anywhere in the report,
    which stops the retry before any task input is read, directory
    created or file written (the same words inside strings are fine).
    Saving follows ``run_plan``'s atomic rule: an unencodable report
    raises ValueError and a filesystem error while creating the output
    directory, saving the content or replacing the output raises OSError,
    in both cases after tasks have run, with the pre-run output file
    preserved byte-for-byte (or still absent when there was none), no
    partial or extra files left behind and executed tasks not undone — so
    when ``report`` equals ``output`` the historical report stays
    queryable after a failed save, and only a successful save replaces it
    with this run's results.

    With ``record_reasons=True`` and at least one retry target, the
    written report additionally carries the top-level ``execution``
    object with ``mode`` ``"retry"``; ``targets`` are the report's
    failed/blocked tasks in current plan order and ``jobs`` are exactly
    ``preview_retry``'s entries, same names and order as ``results``. The
    reasons are computed from the report content read before the
    overwrite, so when ``report`` equals ``output`` they still reflect
    the historical run; outcomes of this run never rewrite them. With no
    retry targets nothing is written and no ``execution`` is recorded.
    ``record_reasons`` must be a bool; other values raise ValueError.
    """
    _require_bool(record_reasons)
    deps_by_name, report_path = _validate_plan(root, jobs, output)
    statuses = _read_retry_report(root, report, {job["name"] for job in jobs})
    targets = [job["name"] for job in jobs
               if statuses.get(job["name"]) in ("failed", "blocked")]
    if not targets:
        return []
    selected = _select_jobs(jobs, targets)
    execution = (_execution_for_run(root, jobs, output, "retry", report)
                 if record_reasons else None)
    return _run_selected(root, selected, deps_by_name, report_path, execution)


def run_history_retry(root, jobs, output, reports, record_reasons=False):
    """Execute a manual retry range derived from several historical reports.

    Target selection and scope mirror ``preview_history_retry``: ``reports``
    is a nonempty list of nonblank root-relative report paths interpreted
    oldest to newest in the order given (repeats keep separate positions
    and nothing is sorted or scanned), each task is judged by the last
    report recording it — a later report missing the task never clears an
    older status — and tasks whose latest record is ``failed`` or
    ``blocked`` become targets in current plan order. The run covers the
    targets plus every direct and indirect prerequisite — prerequisites the
    history marked ``completed`` or never recorded are re-executed too, and
    shared prerequisites run once; unrelated branches are neither read nor
    recorded.

    Execution order, failure records and ``blocked_by`` propagation are
    exactly ``run_plan``'s, driven solely by this run's results: input read
    failures, unknown operations and invalid contents are recorded
    ``failed``, independent branches keep running and blocked tasks never
    read their input. The returned list holds this run's records in actual
    processing order and is written to ``output`` under ``results``,
    replacing any prior content — old records are never copied. A report
    path may equal ``output`` or name the same file another way; the range
    is chosen from all the historical content read before the overwrite.

    With no targets the empty list is returned without reading task inputs,
    creating directories or touching the report. The whole plan (including
    unselected branches and an empty scope), every input path and the
    output path are validated first, and then every report is fully
    validated under ``preview_history_retry``'s path, UTF-8, structure,
    name, status and strict payload rules — including out-of-scope and
    overridden older records — so an illegal ``reports``, plan or report
    raises ValueError before anything runs and no file changes. Saving
    follows ``run_plan``'s atomic rule: an unencodable report raises
    ValueError and a filesystem error while creating the output
    directory, saving the content or replacing the output raises OSError,
    in both cases after tasks have run, with the pre-run output file
    preserved byte-for-byte (or still absent when there was none), no
    partial or extra files left behind and executed tasks not undone — so
    a historical report aliasing ``output`` stays queryable after a
    failed save, and only a successful save replaces it with this run's
    results.

    With ``record_reasons=True`` and at least one target, the written
    report additionally carries the top-level ``execution`` object with
    ``mode`` ``"retry"``; ``targets`` and ``jobs`` are exactly
    ``preview_history_retry``'s (``sources`` is not recorded), in the same
    names and order as ``results``, computed from the report contents read
    before any overwrite, and the reasons query reads them back. With no
    targets nothing is written and no ``execution`` is recorded.
    ``record_reasons`` must be a bool; other values raise ValueError.
    """
    _require_bool(record_reasons)
    deps_by_name, report_path = _validate_plan(root, jobs, output)
    targets, _sources = _history_retry_targets(root, jobs, deps_by_name, reports)
    if not targets:
        return []
    selected = _select_jobs(jobs, targets)
    execution = ({"mode": "retry", "targets": list(targets),
                  "jobs": _preview_entries(jobs, deps_by_name, targets)}
                 if record_reasons else None)
    return _run_selected(root, selected, deps_by_name, report_path, execution)


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
    ValueError before anything runs. Saving follows ``run_plan``'s atomic
    rule: an unencodable report raises ValueError and a filesystem error
    while creating the output directory, saving the content or replacing
    the output raises OSError, in both cases after tasks have run, with
    the pre-run output file preserved byte-for-byte (or still absent when
    there was none), no partial or extra files left behind and executed
    tasks not undone.

    With ``record_reasons=True`` and at least one target, the written
    report additionally carries the top-level ``execution`` object with
    ``mode`` ``"changes"``; ``targets`` are the directly changed tasks
    plus every downstream dependent (each once, in plan order, matched by
    resolved path so argument order and aliases of the same file change
    nothing) and ``jobs`` are exactly ``preview_changes``' entries —
    including ``triggered_by`` — same names and order as ``results``.
    Reasons are chosen before execution and are never rewritten by this
    run's outcomes. With no match nothing is written and no
    ``execution`` is recorded. ``record_reasons`` must be a bool; other
    values raise ValueError.
    """
    _require_bool(record_reasons)
    deps_by_name, report_path = _validate_plan(root, jobs, output)
    _hits, targets = _changed_targets(root, jobs, deps_by_name, changed_inputs)
    if not targets:
        return []
    selected = _select_jobs(jobs, targets)
    execution = (_execution_for_run(root, jobs, output, "changes", None,
                                    changed_inputs=changed_inputs)
                 if record_reasons else None)
    return _run_selected(root, selected, deps_by_name, report_path, execution)


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
    parser.add_argument("--history-retry", action="append", default=None,
                        metavar="REPORT",
                        help="preview a retry range from the newest matching record "
                             "across REPORT (repeatable, oldest to newest), read-only")
    parser.add_argument("--max-failures", default=None, metavar="N",
                        help="with --history-retry, exclude candidates whose "
                             "consecutive failure count (or a prerequisite's) "
                             "reached N, a positive integer")
    parser.add_argument("--run-history-retry", action="append", default=None,
                        metavar="REPORT",
                        help="rerun the retry range selected from the newest matching "
                             "record across REPORT (repeatable, oldest to newest)")
    parser.add_argument("--execution", metavar="REPORT",
                        help="print the execution reasons recorded in REPORT, read-only")
    parser.add_argument("--changed", action="append", default=None, metavar="PATH",
                        help="preview tasks affected by changed input PATH (repeatable), read-only")
    parser.add_argument("--run-changed", action="append", default=None, metavar="PATH",
                        help="run tasks affected by changed input PATH (repeatable)")
    parser.add_argument("--record-reasons", action="store_true",
                        help="record why this run happened in the report's execution object")
    args = parser.parse_args()
    try:
        if args.run_history_retry is not None and (args.only is not None
                                                   or args.preview
                                                   or args.retry_preview is not None
                                                   or args.retry is not None
                                                   or args.compare is not None
                                                   or args.explain is not None
                                                   or args.history is not None
                                                   or args.history_retry is not None
                                                   or args.execution is not None
                                                   or args.changed is not None
                                                   or args.run_changed is not None):
            raise ValueError("--run-history-retry cannot be combined with --only, "
                             "--preview, --retry-preview, --retry, --compare, "
                             "--explain, --history, --history-retry, --execution, "
                             "--changed or --run-changed")
        if args.history_retry is not None and (args.only is not None
                                               or args.record_reasons
                                               or args.preview
                                               or args.retry_preview is not None
                                               or args.retry is not None
                                               or args.compare is not None
                                               or args.explain is not None
                                               or args.history is not None
                                               or args.execution is not None
                                               or args.changed is not None
                                               or args.run_changed is not None):
            raise ValueError("--history-retry cannot be combined with --only, "
                             "--record-reasons, --preview, --retry-preview, --retry, "
                             "--compare, --explain, --history, --execution, "
                             "--changed or --run-changed")
        if args.max_failures is not None and args.history_retry is None:
            raise ValueError("--max-failures can only be used with --history-retry")
        max_failures = None
        if args.max_failures is not None:
            try:
                max_failures = int(args.max_failures)
            except ValueError:
                raise ValueError("--max-failures must be a positive integer")
            if max_failures < 1:
                raise ValueError("--max-failures must be a positive integer")
        if args.execution is not None and (args.only is not None or args.record_reasons
                                           or args.preview or args.retry_preview is not None
                                           or args.retry is not None
                                           or args.compare is not None
                                           or args.explain is not None
                                           or args.history is not None
                                           or args.changed is not None
                                           or args.run_changed is not None):
            raise ValueError("--execution cannot be combined with --only, "
                             "--record-reasons, --preview, --retry-preview, --retry, "
                             "--compare, --explain, --history, --changed or --run-changed")
        if args.record_reasons and (args.preview or args.retry_preview is not None
                                    or args.changed is not None or args.compare is not None
                                    or args.explain is not None or args.history is not None):
            raise ValueError("--record-reasons cannot be combined with --preview, "
                             "--retry-preview, --changed, --compare, --explain or --history")
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
        if args.history_retry is not None:
            print(json.dumps(preview_history_retry(args.root, jobs, args.output,
                                                   args.history_retry,
                                                   max_failures=max_failures),
                             ensure_ascii=False, indent=2))
            return 0
        if args.compare is not None:
            print(_compare_dumps(compare_reports(args.root, jobs, args.output,
                                                 args.compare[0], args.compare[1])))
            return 0
        if args.explain is not None:
            print(json.dumps(explain_report(args.root, jobs, args.output, args.explain),
                             ensure_ascii=False, indent=2))
            return 0
        if args.execution is not None:
            print(json.dumps(query_execution(args.root, jobs, args.output,
                                             args.execution),
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
        elif args.run_history_retry is not None:
            results = run_history_retry(args.root, jobs, args.output,
                                        args.run_history_retry,
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
