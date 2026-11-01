"""Immutable evidence records, paired regression gates, and portable reports."""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from experiments.source import registry

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def default_seed(path: Path) -> int:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "report":
            for keyword in node.keywords:
                if keyword.arg == "seed":
                    return int(ast.literal_eval(keyword.value))
    raise ValueError(f"No declared default seed in {path}")


def plan(
    *,
    run_id: str,
    candidate: dict,
    baseline: dict,
    evidence_sha: str,
    executor_sha: str,
    root: Path = ROOT,
    identifiers: list[str] | None = None,
    steps: int | None = None,
) -> dict:
    entries = registry(root)
    selected = identifiers if identifiers is not None else list(entries)
    if not selected or len(set(selected)) != len(selected) or set(selected) - entries.keys():
        raise ValueError("Select distinct registered experiment IDs")
    tasks = []
    for identifier in selected:
        script = root / entries[identifier]["script"]
        for role in ("baseline", "candidate"):
            tasks.append(
                {
                    "id": identifier,
                    "role": role,
                    "seed": default_seed(script),
                    "accelerator": "cpu",
                    "attempt": 1,
                    "steps": steps,
                    "script": entries[identifier]["script"],
                    "script_sha256": digest(script),
                    "executor_sha": executor_sha,
                }
            )
    return {
        "schema": 2,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - lab host runs Python 3.10
        "evidence_sha": evidence_sha,
        "executor_sha": executor_sha,
        "policy": "paired-passing-checks-v1",
        "candidate": candidate,
        "baseline": baseline,
        "full_catalog": set(selected) == set(entries),
        "mode": "full" if steps is None else "smoke",
        "tasks": tasks,
    }


def task_key(task: dict) -> str:
    return f"{task['role']}/{task['id']}/{task['accelerator']}/seed-{task['seed']}/attempt-{task['attempt']}"


def execute(task: dict, *, root: Path = ROOT, timeout: int = 1800) -> dict:
    """Run one experiment in a subprocess; capture partial errors without altering archives."""
    started = time.monotonic()
    record = {"task": task, "result": None, "error": None, "log": ""}
    try:
        script = root / task["script"]
        if digest(script) != task["script_sha256"]:
            raise ValueError("Experiment digest differs from manifest")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            env = dict(
                os.environ,
                PYTHONPATH=str(root / "experiments"),
                EVIDENCE_JSON_OUTPUT=str(output),
                HF_HUB_OFFLINE="1",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                POLARS_MAX_THREADS="2",
                TOKENIZERS_PARALLELISM="false",
                PYTHONHASHSEED=str(task["seed"]),
            )
            command = [
                sys.executable,
                str(script),
                "--seed",
                str(task["seed"]),
                "--accelerator",
                task["accelerator"],
                "--threads",
                "1",
            ]
            if task["steps"] is not None:
                command += ["--steps", str(task["steps"])]
            result = subprocess.run(
                command,
                cwd=root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
            record["log"] = result.stdout
            if output.is_file():
                record["result"] = json.loads(output.read_text())
            if result.returncode:
                record["error"] = f"Experiment exited with status {result.returncode}"
            elif record["result"] is None:
                record["error"] = "Experiment produced no result"
    except subprocess.TimeoutExpired as exception:
        record["error"] = "Experiment timed out"
        record["log"] = (
            (exception.stdout or b"").decode(errors="replace")
            if isinstance(exception.stdout, bytes)
            else (exception.stdout or "")
        )
    except Exception:
        record["error"] = traceback.format_exc()
    record["duration_seconds"] = round(time.monotonic() - started, 3)
    return record


def collect(output: Path, record: dict) -> None:
    key = task_key(record["task"])
    record = dict(record)
    log = record.pop("log", "")
    write_json(output / "results" / f"{key}.json", record)
    logpath = output / "logs" / f"{key}.txt"
    logpath.parent.mkdir(parents=True, exist_ok=True)
    with logpath.open("x") as handle:
        handle.write(log)


def evaluate(manifest: dict, records: list[dict], *, root: Path = ROOT) -> dict:
    expected = {task_key(task): task for task in manifest["tasks"]}
    seen = {}
    errors = []
    catalog = registry(root)
    if manifest.get("schema") != 2 or not manifest.get("executor_sha"):
        errors.append("Expected a current manifest with an independently pinned executor")
    if any(task.get("executor_sha") != manifest.get("executor_sha") for task in manifest["tasks"]):
        errors.append("Task executor differs from the manifest")
    if len(expected) != len(manifest["tasks"]):
        errors.append("Manifest contains duplicate tasks")
    if {task["id"] for task in manifest["tasks"]} != set(catalog):
        errors.append("Manifest does not cover the complete registered catalog")
    if len(expected) != 2 * len(catalog):
        errors.append("Every experiment needs both baseline and candidate execution")
    for identifier in catalog:
        if {task["role"] for task in manifest["tasks"] if task["id"] == identifier} != {"baseline", "candidate"}:
            errors.append(f"Missing baseline or candidate task: {identifier}")
    for record in records:
        key = task_key(record["task"])
        if key not in expected or key in seen or record["task"] != expected.get(key):
            errors.append(f"Unexpected, duplicate, or altered task: {key}")
            continue
        seen[key] = record
        result = record.get("result")
        if record.get("error") or not result or result.get("outcome") == "error":
            errors.append(f"Execution error: {key}: {record.get('error') or (result or {}).get('error')}")
            continue
        checks = result.get("checks")
        if not isinstance(checks, dict) or not checks or any(type(value) is not bool for value in checks.values()):
            errors.append(f"Invalid checks: {key}")
        if result.get("seed") != record["task"]["seed"] or result.get("accelerator") != record["task"]["accelerator"]:
            errors.append(f"Run identity differs from task: {key}")
        if result.get("mode") != manifest["mode"]:
            errors.append(f"Run budget differs from manifest: {key}")
        task = record["task"]
        identity = manifest.get(task["role"], {})
        provenance = result.get("provenance", {})
        for field, wanted in (
            ("evidence_revision", manifest["evidence_sha"]),
            ("relflow_revision", identity.get("revision")),
            ("wheel_sha256", identity.get("wheel_sha256")),
            ("script_sha256", task["script_sha256"]),
        ):
            if wanted is None or provenance.get(field) != wanted:
                errors.append(f"Provenance differs from manifest: {key}: {field}")
    errors += [f"Missing result: {key}" for key in sorted(expected.keys() - seen.keys())]
    regressions = []
    limitations = []
    for task in manifest["tasks"]:
        if task["role"] != "candidate":
            continue
        candidate = seen.get(task_key(task), {}).get("result") or {}
        baseline = seen.get(task_key(dict(task, role="baseline")), {}).get("result") or {}
        bc, cc = baseline.get("checks", {}), candidate.get("checks", {})
        if set(bc) != set(cc):
            errors.append(f"Check set differs for {task['id']}")
        for check, passed in bc.items():
            if passed and cc.get(check) is False:
                regressions.append({"id": task["id"], "seed": task["seed"], "check": check})
            if passed is False:
                limitations.append(
                    {"id": task["id"], "seed": task["seed"], "check": check, "candidate_passed": cc.get(check)}
                )
    if manifest["mode"] != "full" or not manifest["full_catalog"]:
        errors.append("Smoke or partial-catalog execution cannot satisfy the required gate")
    return {
        "schema": 1,
        "run_id": manifest["run_id"],
        "complete": len(seen) == len(expected),
        "passed": not errors and not regressions,
        "errors": errors,
        "regressions": regressions,
        "baseline_unmet_checks": limitations,
        "expected_tasks": len(expected),
        "received_tasks": len(seen),
    }


def finish(output: Path, *, root: Path = ROOT) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    records = [json.loads(path.read_text()) for path in sorted((output / "results").rglob("*.json"))]
    summary = evaluate(manifest, records, root=root)
    write_json(output / "summary.json", summary)
    rows = []
    for record in records:
        result = record.get("result") or {}
        checks = result.get("checks", {})
        task = record["task"]
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in (
                    task["role"],
                    task["id"],
                    task["seed"],
                    result.get("outcome", "error"),
                    f"{sum(checks.values())}/{len(checks)}",
                    record.get("duration_seconds"),
                )
            )
            + "</tr>"
        )
    report = (
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>RelFlow evidence</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}"
        "td,th{padding:.5rem;border-bottom:1px solid #ccc}pre{white-space:pre-wrap}</style>"
        f"<h1>Evidence: {html.escape(manifest['run_id'])}</h1>"
        f"<p>Required gate: {'passed' if summary['passed'] else 'failed'}</p>"
        f"<pre>{html.escape(json.dumps(summary, indent=2))}</pre><table>"
        "<tr><th>Role</th><th>Experiment</th><th>Seed</th><th>Outcome</th><th>Checks</th><th>Seconds</th></tr>"
        + "".join(rows)
        + "</table></html>"
    )
    (output / "report.html").write_text(report)
    checksums = {
        path.relative_to(output).as_posix(): digest(path) for path in sorted(output.rglob("*")) if path.is_file()
    }
    write_json(output / "checksums.json", checksums)
    return summary


def bundle(source: Path, destination: Path) -> None:
    """Pack only regular files with relative paths; no shared mutable result file."""
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file() and not path.is_symlink():
                archive.add(path, arcname=path.relative_to(source), recursive=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("output", type=Path)
    pack = sub.add_parser("bundle")
    pack.add_argument("source", type=Path)
    pack.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        summary = json.loads((args.output / "summary.json").read_text())
        raise SystemExit(0 if summary["passed"] else 1)
    bundle(args.source, args.destination)


if __name__ == "__main__":
    main()
