"""Immutable evidence records, paired regression gates, and portable reports."""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from experiments.source import declaration, frontmatter, metadata, registry

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def wheel_payload(path: Path) -> str:
    """Bind all wheel payload bytes while ignoring ZIP transport timestamps."""
    with zipfile.ZipFile(path) as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]
        names = [item.filename for item in files]
        if len(set(names)) != len(names) or sum(item.file_size for item in files) > 1024**3:
            raise ValueError("Wheel payload has duplicate members or exceeds 1 GiB")
        return identity(
            {
                item.filename: {
                    "sha256": hashlib.sha256(archive.read(item)).hexdigest(),
                    "mode": item.external_attr,
                    "system": item.create_system,
                }
                for item in files
            }
        )


def suite_identity(root: Path) -> str:
    """Bind suite files and modes; normalize only the two selected result fields."""
    scripts = {entry["script"] for entry in registry(root, measurements=False).values()}
    sources = {}
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise ValueError("Execution identity requires regular suite sources")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Execution identity rejects special suite files")
        name = path.relative_to(root).as_posix()
        content = path.read_bytes()
        if name in scripts:
            text = content.decode()
            header = metadata(text)
            header.pop("evidence-result", None)
            header.pop("evidence-report", None)
            content = frontmatter(text, header).encode()
        sources[name] = {"sha256": hashlib.sha256(content).hexdigest(), "mode": path.stat().st_mode & 0o777}
    return identity(sources)


def runtime_inputs(root: Path) -> dict:
    """Bind experiment code and shared runtime files, excluding publication and test tooling."""
    inputs = {}
    for path in sorted((root / "experiments").rglob("*")):
        name = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts:
            continue
        if path.is_symlink():
            raise ValueError(f"Execution inputs require regular files: {name}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Execution inputs reject special files: {name}")
        inputs[name] = {"sha256": digest(path), "mode": path.stat().st_mode & 0o777}
    return inputs


def execution_key(manifest: dict, task: dict, execution: dict) -> str:
    """An execution equivalence key, never a replacement for original provenance."""
    return identity(
        {
            "schema": 1,
            "wheel_payload": manifest[task["role"]]["payload_sha256"],
            "requirements": manifest["requirements_sha256"],
            "suite": task.get("inputs_sha256", manifest["suite_sha256"]),
            "executor": manifest["executor_sha"],
            "policy": manifest["policy"],
            "environment": execution["identity"],
            "task": {key: value for key, value in task.items() if key not in {"role", "script_sha256"}},
        }
    )


def integrity(reports: Path) -> None:
    """Validate the complete immutable report inventory, including every task log."""
    if reports.is_symlink() or not reports.is_dir():
        raise ValueError("Retained report must be a regular directory")
    files = {}
    total = 0
    for path in reports.rglob("*"):
        if path.is_symlink():
            raise ValueError("Retained reports must contain only regular files")
        if path.is_file() and path != reports / "checksums.json":
            total += path.stat().st_size
            if total > 256 * 1024 * 1024:
                raise ValueError("Retained baseline proof exceeds 256 MiB")
            files[path.relative_to(reports).as_posix()] = digest(path)
    if json.loads((reports / "checksums.json").read_text()) != files:
        raise ValueError("Retained report checksum inventory differs from its files")
    manifest = json.loads((reports / "manifest.json").read_text())
    for task in manifest["tasks"]:
        if not (reports / "logs" / (task_key(task) + ".txt")).is_file():
            raise ValueError("Retained report is missing a task log")


def resolved(
    manifest: dict,
    records: list[dict],
    reports: Path,
    *,
    root: Path,
    reuse_depth: int = 0,
    validated_sources: dict | None = None,
) -> list[dict]:
    """Resolve baseline slots to unchanged fresh originals, validating source runs first."""
    if reuse_depth >= 2:
        raise ValueError("Baseline proof nesting exceeds two retained source runs")
    sources = validated_sources if validated_sources is not None else {}
    values = []
    for record in records:
        if "reuse" not in record:
            values.append(record)
            continue
        task = record["task"]
        reference = record["reuse"]
        name = reference.get("report", "")
        parts = Path(name).parts
        if (
            task["role"] != "baseline"
            or set(record) != {"task", "reuse"}
            or len(parts) != 2
            or parts[0] != "reused"
            or len(parts[1]) != 64
            or any(value not in "0123456789abcdef" for value in parts[1])
        ):
            raise ValueError("Only baseline slots may reference retained source reports")
        source = reports / name
        if name not in sources:
            integrity(source)
            if digest(source / "checksums.json") != parts[1]:
                raise ValueError("Retained source checksum identity differs from its reference")
            archived_summary = json.loads((source / "summary.json").read_text())
            if not archived_summary.get("complete") or not archived_summary.get("passed"):
                raise ValueError("Baseline reuse cannot promote an incomplete or failed original run")
            original = json.loads((source / "manifest.json").read_text())
            originals = [json.loads(path.read_text()) for path in sorted((source / "results").rglob("*.json"))]
            json.dumps(originals, allow_nan=False)
            summary = evaluate(original, originals, root=source / "source", reports=source, reuse_depth=reuse_depth + 1)
            if not summary["passed"] or not summary["complete"]:
                raise ValueError("Baseline reuse requires a complete passing original run")
            if original.get("suite_sha256") != suite_identity(source / "source"):
                raise ValueError("Retained suite identity differs from its source")
            for original_task in original["tasks"]:
                if digest(source / "source" / original_task["script"]) != original_task["script_sha256"]:
                    raise ValueError("Retained original script differs from its execution")
            for role in ("baseline", "candidate"):
                wheels = list((source / "artifacts" / role).glob("*.whl"))
                if len(wheels) != 1 or digest(wheels[0]) != original[role]["wheel_sha256"]:
                    raise ValueError("Retained original wheel differs from its execution")
                if wheel_payload(wheels[0]) != original[role]["payload_sha256"]:
                    raise ValueError("Retained wheel payload identity differs")
            if digest(source / "artifacts/requirements.txt") != original["requirements_sha256"]:
                raise ValueError("Retained original requirements differ")
            sources[name] = (
                original,
                {task_key(value["task"]): value for value in originals},
                json.loads((source / "execution.json").read_text()),
            )
        original, originals, environment = sources[name]
        value = originals.get(reference.get("task"))
        current_environment = json.loads((reports / "execution.json").read_text())
        if (
            value is None
            or "reuse" in value
            or execution_key(original, value["task"], environment) != execution_key(manifest, task, current_environment)
        ):
            raise ValueError("Reused baseline execution identity differs from the current slot")
        # The original object and its provenance remain untouched in the retained source.
        values.append(
            {
                "task": task,
                "result": value["result"],
                "error": value.get("error"),
                "duration_seconds": value.get("duration_seconds"),
                "reused_from": reference,
            }
        )
    return values


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def default_seed(path: Path) -> int:
    source = path.read_text()
    declared = declaration(source)
    if declared is not None:
        return declared.seed
    # Archived suites retain their original report(...) declaration and hashes.
    for node in ast.walk(ast.parse(source)):
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
    entries = registry(root, measurements=False)
    selected = identifiers if identifiers is not None else list(entries)
    if not selected or len(set(selected)) != len(selected) or set(selected) - entries.keys():
        raise ValueError("Select distinct registered experiment IDs")
    tasks = []
    inputs = runtime_inputs(root)
    scripts = {entry["script"] for entry in entries.values()}
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
                    "inputs_sha256": identity(
                        {
                            name: value
                            for name, value in inputs.items()
                            if name not in scripts or name == entries[identifier]["script"]
                        }
                    ),
                    "executor_sha": executor_sha,
                }
            )
    return {
        "schema": 3,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "evidence_sha": evidence_sha,
        "executor_sha": executor_sha,
        "policy": "paired-passing-checks-v1",
        "candidate": candidate,
        "baseline": baseline,
        "full_catalog": set(selected) == set(entries),
        "mode": "full" if steps is None else "smoke",
        "tasks": tasks,
        "runtime_sha256": identity(inputs),
    }


def task_key(task: dict) -> str:
    return f"{task['role']}/{task['id']}/{task['accelerator']}/seed-{task['seed']}/attempt-{task['attempt']}"


def execute(task: dict, *, root: Path = ROOT, timeout: int = 1800, observations: Path | None = None) -> dict:
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
                PYTHONPATH=os.pathsep.join([str(root / "experiments"), str(Path(__file__).parent)]),
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
            if observations is not None:
                env["EVIDENCE_TASK"] = json.dumps(task)
                command = [
                    sys.executable,
                    str(Path(__file__).parent / "diagnostics/capture.py"),
                    "--source",
                    str(script),
                    "--output",
                    str(observations / "raw"),
                    "--archive",
                    str(observations / "bundle.zip"),
                    "--",
                    *command[2:],
                ]
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
            if observations is not None and (observations / "bundle.zip").is_file():
                from observations import digest as archive_digest
                from observations import read as read_observations

                path = observations / "bundle.zip"
                captured = read_observations(path)
                manifest = captured["manifest"]
                if captured["record"] != record["result"] or manifest["task"] != task:
                    raise ValueError("Proof observations differ from the canonical measured task")
                if manifest["capture_sha256"] != digest(Path(__file__).parent / "diagnostics/capture.py"):
                    raise ValueError("Proof observations differ from the trusted recorder")
                record["observations"] = {
                    "schema": 1,
                    "sha256": archive_digest(path),
                    "bytes": path.stat().st_size,
                    "manifest_sha256": captured["manifest_sha256"],
                    "capture_sha256": manifest["capture_sha256"],
                }
                shutil.rmtree(observations / "raw")
            elif observations is not None and record["result"] is not None:
                raise ValueError("Measured task produced no observation archive")
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


def verify_execution(manifest: dict, records: list[dict], reports: Path | None) -> list[str]:
    """Check hosted cohorts while retaining historical single-host report semantics."""
    if manifest.get("backend") != "github-hosted-docker":
        return []
    if reports is None:
        return ["Hosted execution requires retained shard identities and launch journals"]
    from hosted import verify_execution as verify_hosted

    return verify_hosted(manifest, records, reports)


def evaluate(
    manifest: dict, records: list[dict], *, root: Path = ROOT, reports: Path | None = None, reuse_depth: int = 0
) -> dict:
    reuse_errors = []
    reused_keys = set()
    if any("reuse" in record for record in records):
        try:
            if manifest.get("schema") != 3 or reports is None:
                raise ValueError("Reused baseline slots require a schema3 report directory")
            if manifest.get("suite_sha256") != suite_identity(root):
                raise ValueError("Current suite differs from the execution identity")
            reused_keys = {task_key(record["task"]) for record in records if "reuse" in record}
            records = resolved(manifest, records, reports, root=root, reuse_depth=reuse_depth)
        except (ValueError, KeyError, OSError, TypeError) as error:
            reuse_errors.append(f"Invalid baseline reuse: {error}")
            reused_keys.clear()
    expected = {task_key(task): task for task in manifest["tasks"]}
    seen = {}
    errors = reuse_errors + verify_execution(manifest, records, reports)
    catalog = registry(root, measurements=False)
    if "runtime_sha256" in manifest:
        inputs = runtime_inputs(root)
        scripts = {entry["script"] for entry in catalog.values()}
        if manifest["runtime_sha256"] != identity(inputs):
            errors.append("Experiment runtime inputs differ from the manifest")
        for task in manifest["tasks"]:
            if task.get("script") not in scripts:
                errors.append(f"Unknown experiment source path: {task.get('script')}")
            elif task.get("inputs_sha256") != identity(
                {name: value for name, value in inputs.items() if name not in scripts or name == task["script"]}
            ):
                errors.append(f"Experiment execution inputs differ: {task['id']}")
    if manifest.get("schema") not in {2, 3} or not manifest.get("executor_sha"):
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
        if key in reused_keys:
            continue
        task = record["task"]
        package = manifest.get(task["role"], {})
        provenance = result.get("provenance", {})
        for field, wanted in (
            ("evidence_revision", manifest["evidence_sha"]),
            ("relflow_revision", package.get("revision")),
            ("wheel_sha256", package.get("wheel_sha256")),
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
        **(
            {"reused_baseline_tasks": len(reused_keys), "fresh_tasks": len(seen) - len(reused_keys)}
            if manifest.get("schema") == 3
            else {}
        ),
    }


def finish(output: Path, *, root: Path = ROOT) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    records = [json.loads(path.read_text()) for path in sorted((output / "results").rglob("*.json"))]
    summary = evaluate(manifest, records, root=root, reports=output)
    write_json(output / "summary.json", summary)
    rows = []
    displayed = resolved(manifest, records, output, root=root) if summary["passed"] else records
    for record in displayed:
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
                    (record.get("reused_from") or {}).get("report", "fresh"),
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
        "<tr><th>Role</th><th>Experiment</th><th>Seed</th><th>Outcome</th><th>Checks</th><th>Seconds</th><th>Execution source</th></tr>"
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
