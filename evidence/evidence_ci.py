"""Immutable evidence records, paired regression gates, and portable reports."""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import os
import re
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

from experiments.source import declaration, describe, registry

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
    """Bind the exact suite files and modes without importing experiment code."""
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
        sources[name] = {"sha256": hashlib.sha256(content).hexdigest(), "mode": path.stat().st_mode & 0o777}
    return identity(sources)


def runtime_inputs(root: Path) -> dict:
    """Bind every experiment and shared runtime file, excluding publication tooling."""
    inputs = {}
    paths = list((root / "experiments").rglob("*"))
    for path in sorted(paths):
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


def script_paths(root: Path, entries: dict) -> set[str]:
    """Identify experiment scripts that must be isolated from one another."""
    return {entry["script"] for entry in entries.values()}


def baseline_catalog(revision: str) -> set[str]:
    """Read proof identities from the exact baseline Git tree without executing it."""
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Baseline catalog requires a full Git revision")
    try:
        data = subprocess.check_output(
            ["git", "archive", revision, "evidence/experiments"], cwd=ROOT.parent, stderr=subprocess.PIPE
        )
        entries = set()
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for member in archive:
                parts = Path(member.name).parts
                if len(parts) < 3 or parts[:2] != ("evidence", "experiments") or not parts[-1].endswith(".py"):
                    continue
                if not member.isfile():
                    raise ValueError(f"Baseline experiment must be a regular file: {member.name}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Cannot read baseline experiment: {member.name}")
                source = stream.read().decode()
                if declaration(source) is None and not re.search(r"^# evidence-id:", source, re.MULTILINE):
                    continue
                identifier = describe(source)["evidence-id"]
                if identifier in entries:
                    raise ValueError(f"Duplicate baseline experiment: {identifier}")
                entries.add(identifier)
        if not entries:
            raise ValueError("Baseline catalog is empty")
        return entries
    except (subprocess.CalledProcessError, OSError, tarfile.TarError, UnicodeError) as error:
        raise ValueError(f"Cannot read baseline catalog at {revision}") from error


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
    baseline_ids: set[str] | None = None,
) -> dict:
    entries = registry(root, measurements=False)
    selected = identifiers if identifiers is not None else list(entries)
    if not selected or len(set(selected)) != len(selected) or set(selected) - entries.keys():
        raise ValueError("Select distinct registered experiment IDs")
    tasks = []
    inputs = runtime_inputs(root)
    scripts = script_paths(root, entries)
    introduced = set(entries) - baseline_ids if baseline_ids is not None else set()
    for identifier in selected:
        script = root / entries[identifier]["script"]
        for role in ("candidate",) if identifier in introduced else ("baseline", "candidate"):
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
        "policy": "introduced-passing-checks-v1" if introduced else "paired-passing-checks-v1",
        "candidate": candidate,
        "baseline": baseline,
        "full_catalog": set(selected) == set(entries),
        "mode": "full" if steps is None else "smoke",
        "tasks": tasks,
        "runtime_sha256": identity(inputs),
        **({"introduced": sorted(introduced), "retired": sorted(baseline_ids - set(entries))} if introduced else {}),
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
    """Verify shard receipts for a hosted execution manifest."""
    if manifest.get("backend") != "github-hosted-docker":
        return []
    if reports is None:
        return ["Hosted execution requires retained shard identities and launch journals"]
    from hosted import verify_execution as verify_hosted

    return verify_hosted(manifest, records, reports)


def evaluate(manifest: dict, records: list[dict], *, root: Path = ROOT, reports: Path | None = None) -> dict:
    expected = {task_key(task): task for task in manifest["tasks"]}
    seen = {}
    errors = verify_execution(manifest, records, reports)
    catalog = registry(root, measurements=False)
    introduced = set()
    if manifest.get("policy") not in {"paired-passing-checks-v1", "introduced-passing-checks-v1"}:
        errors.append("Unknown Evidence comparison policy")
    if manifest.get("policy") == "introduced-passing-checks-v1":
        try:
            baseline_ids = baseline_catalog(manifest["baseline"]["revision"])
            introduced = set(catalog) - baseline_ids
            planned = plan(
                run_id=manifest["run_id"],
                candidate=manifest["candidate"],
                baseline=manifest["baseline"],
                evidence_sha=manifest["evidence_sha"],
                executor_sha=manifest["executor_sha"],
                root=root,
                baseline_ids=baseline_ids,
            )
            if any(manifest.get(key) != planned.get(key) for key in ("policy", "introduced", "retired", "tasks")):
                raise ValueError("Task plan differs from the baseline catalog")
        except (ValueError, KeyError, OSError, TypeError) as error:
            errors.append(f"Invalid baseline catalog: {error}")
    elif "introduced" in manifest or "retired" in manifest:
        errors.append("Introduced proofs require the catalog comparison policy")
    if "runtime_sha256" in manifest:
        inputs = runtime_inputs(root)
        scripts = script_paths(root, catalog)
        if manifest["runtime_sha256"] != identity(inputs):
            errors.append("Experiment runtime inputs differ from the manifest")
        for task in manifest["tasks"]:
            if task.get("script") not in scripts:
                errors.append(f"Unknown experiment source path: {task.get('script')}")
            elif task.get("inputs_sha256") != identity(
                {name: value for name, value in inputs.items() if name not in scripts or name == task["script"]}
            ):
                errors.append(f"Experiment execution inputs differ: {task['id']}")
    if manifest.get("schema") != 3 or not manifest.get("executor_sha"):
        errors.append("Expected a current manifest with an independently pinned executor")
    if any(task.get("executor_sha") != manifest.get("executor_sha") for task in manifest["tasks"]):
        errors.append("Task executor differs from the manifest")
    if len(expected) != len(manifest["tasks"]):
        errors.append("Manifest contains duplicate tasks")
    if {task["id"] for task in manifest["tasks"]} != set(catalog):
        errors.append("Manifest does not cover the complete registered catalog")
    for identifier in catalog:
        roles = {task["role"] for task in manifest["tasks"] if task["id"] == identifier}
        wanted = {"candidate"} if identifier in introduced else {"baseline", "candidate"}
        if roles != wanted:
            errors.append(f"Missing or unexpected task roles: {identifier}")
    for record in records:
        if record.keys() & {"reuse", "reused_from"}:
            errors.append("Evidence requires fresh measured records, without result-reuse references")
            continue
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
    new_unmet = []
    for task in manifest["tasks"]:
        if task["role"] != "candidate":
            continue
        if task["id"] in introduced:
            candidate = seen.get(task_key(task), {}).get("result") or {}
            new_unmet.extend(
                {"id": task["id"], "seed": task["seed"], "check": name}
                for name, passed in candidate.get("checks", {}).items()
                if passed is False
            )
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
        "passed": not errors and not regressions and not new_unmet,
        "errors": errors,
        "regressions": regressions,
        "baseline_unmet_checks": limitations,
        **({"introduced_unmet_checks": new_unmet} if introduced else {}),
        "expected_tasks": len(expected),
        "received_tasks": len(seen),
    }


def finish(output: Path, *, root: Path = ROOT) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    records = [json.loads(path.read_text()) for path in sorted((output / "results").rglob("*.json"))]
    summary = evaluate(manifest, records, root=root, reports=output)
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
