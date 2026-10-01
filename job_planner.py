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
    names = [job["name"] for job in jobs]
    name_set = set(names)
    deps_by_name = {}
    for job in jobs:
        raw = job.get("depends_on", [])
        if not isinstance(raw, list):
            raise ValueError("depends_on must be a list of job names")
        deps = []
        for dep in raw:
            if not isinstance(dep, str) or not dep.strip():
                raise ValueError("depends_on entries must be nonempty strings")
            deps.append(dep)
        if len(set(deps)) != len(deps):
            raise ValueError("depends_on entries must not repeat")
        name = job["name"]
        if name in deps:
            raise ValueError("job cannot depend on itself")
        unknown = next((dep for dep in deps if dep not in name_set), None)
        if unknown is not None:
            raise ValueError(f"unknown dependency: {unknown}")
        deps_by_name[name] = deps
    color = {name: 0 for name in names}

    def visit(node):
        color[node] = 1
        for dep in deps_by_name[node]:
            if color[dep] == 1:
                raise ValueError("dependency cycle detected")
            if color[dep] == 0:
                visit(dep)
        color[node] = 2

    for name in names:
        if color[name] == 0:
            visit(name)
    return deps_by_name


def run_plan(root, jobs, output):
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("plan requires at least one job")
    names = [job["name"] for job in jobs]
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("job names must be nonempty and unique")
    deps_by_name = _validate_dependencies(jobs)
    report_path = local_path(root, output)
    if any(report_path == local_path(root, job["input"]) for job in jobs):
        raise ValueError("report cannot overwrite a task input")
    results = []
    records = {}
    pending = dict(zip(names, jobs))
    while pending:
        job = next(
            candidate
            for candidate in jobs
            if candidate["name"] in pending
            and all(dep in records for dep in deps_by_name[candidate["name"]])
        )
        name = job["name"]
        blocked_by = [
            dep
            for dep in deps_by_name[name]
            if records[dep]["status"] in ("failed", "blocked")
        ]
        if blocked_by:
            record = {"name": name, "status": "blocked", "blocked_by": blocked_by}
        else:
            try:
                record = {"name": name, "status": "completed", "result": execute_job(root, job)}
            except (OSError, ValueError, KeyError, TypeError) as exc:
                record = {"name": name, "status": "failed", "error": str(exc)}
        records[name] = record
        results.append(record)
        del pending[name]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=".results/latest.json")
    args = parser.parse_args()
    try:
        plan = local_path(args.root, args.plan)
        if plan == local_path(args.root, args.output):
            raise ValueError("report cannot overwrite its plan")
        jobs = json.loads(plan.read_text(encoding="utf-8"))["jobs"]
        results = run_plan(args.root, jobs, args.output)
        summary = {"completed": sum(row["status"] == "completed" for row in results), "failed": sum(row["status"] == "failed" for row in results)}
        if any(job.get("depends_on") for job in jobs):
            summary["blocked"] = sum(row["status"] == "blocked" for row in results)
        print(json.dumps(summary))
        return 1 if any(row["status"] in ("failed", "blocked") for row in results) else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
