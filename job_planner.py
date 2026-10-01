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


def run_plan(root, jobs, output):
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("plan requires at least one job")
    names = [job["name"] for job in jobs]
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("job names must be nonempty and unique")
    report_path = local_path(root, output)
    if any(report_path == local_path(root, job["input"]) for job in jobs):
        raise ValueError("report cannot overwrite a task input")
    results = []
    for job in jobs:
        try:
            result = {"name": job["name"], "status": "completed", "result": execute_job(root, job)}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result = {"name": job["name"], "status": "failed", "error": str(exc)}
        results.append(result)
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
        print(json.dumps({"completed": sum(row["status"] == "completed" for row in results), "failed": sum(row["status"] == "failed" for row in results)}))
        return 1 if any(row["status"] == "failed" for row in results) else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
