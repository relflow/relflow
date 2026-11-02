"""Run paired evidence on lab with one disposable Docker container per task."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import shlex
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

from evidence_ci import (
    ROOT,
    collect,
    digest,
    evaluate,
    execution_key,
    finish,
    identity,
    integrity,
    plan,
    resolved,
    suite_identity,
    task_key,
    wheel_payload,
    write_json,
)


def image_definition(torch_version: str) -> str:
    return f"""FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30
RUN python -m pip install --no-cache-dir torch=={torch_version}+cpu --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt pyyaml==6.0.3
ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=2 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"""


def environment(requirements: Path, workers: int) -> dict:
    """Build/inspect the trusted runtime once for both reuse probes and execution."""
    version = next(
        (
            line.split("==", 1)[1].split(";", 1)[0].split()[0].rstrip("\\")
            for line in requirements.read_text().splitlines()
            if line.startswith("torch==")
        ),
        None,
    )
    if not version:
        raise ValueError("The runtime requirements must pin Torch")
    recipe = image_definition(version)
    image = "relflow-evidence:" + identity({"lock": digest(requirements), "recipe": recipe})[:20]
    with tempfile.TemporaryDirectory(prefix="evidence-image-") as temporary:
        context = Path(temporary)
        (context / "Dockerfile").write_text(recipe)
        shutil.copy2(requirements, context / "requirements.txt")
        subprocess.run(["docker", "build", "--tag", image, str(context)], check=True)
    image_id = subprocess.check_output(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True).strip()
    processor = Path("/proc/cpuinfo")
    wanted = {
        "vendor_id",
        "cpu family",
        "model",
        "model name",
        "stepping",
        "microcode",
        "flags",
        "cpu cores",
        "siblings",
    }
    cpu = (
        sorted(
            {line.strip() for line in processor.read_text().splitlines() if line.partition(":")[0].strip() in wanted}
        )
        if processor.is_file()
        else [platform.processor()]
    )
    runtime = {
        "schema": 1,
        "image_id": image_id,
        "recipe_sha256": hashlib.sha256(recipe.encode()).hexdigest(),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "node": platform.node(),
            "cpu_count": os.cpu_count(),
            "memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            "uid": os.getuid(),
            "gid": os.getgid(),
            "cpu": cpu,
        },
        "docker": subprocess.check_output(["docker", "version", "--format", "{{.Server.Version}}"], text=True).strip(),
        "resources": {
            "workers": workers,
            "cpu_per_task": 2,
            "memory_per_task": "4g",
            "pids": 512,
            "tmpfs": "2g",
            "network": "none",
            "read_only": True,
            "cap_drop": "ALL",
            "no_new_privileges": True,
            "task_timeout_seconds": 1850,
            "experiment_timeout_seconds": 1700,
        },
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "POLARS_MAX_THREADS": "2",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "threads": 1,
        },
    }
    return {
        "identity": runtime,
        "image_id": image_id,
        "image": image,
        "workers": workers,
        "cpu_per_task": 2,
        "memory_per_task": "4g",
    }


def cached(
    cache: Path,
    output: Path,
    manifest: dict,
    task: dict,
    execution: dict,
    suite: Path,
    validated_sources: dict | None = None,
) -> dict | None:
    """A corrupt, stale, or overly deep proof is a miss; workers never see the cache."""
    key = execution_key(manifest, task, execution)
    entry = cache / "keys" / (key + ".json")
    if not entry.is_file() or entry.is_symlink():
        return None
    copied = None
    try:
        reference = json.loads(entry.read_text())
        name = reference["report"]
        if len(name) != 64 or any(value not in "0123456789abcdef" for value in name):
            raise ValueError("Invalid cached report identity")
        source = cache / "reports" / name
        target = output / "reused" / name
        if not target.exists():
            integrity(source)
            retained_bytes = sum(path.stat().st_size for path in (output / "reused").rglob("*") if path.is_file())
            source_bytes = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
            if retained_bytes + source_bytes > 256 * 1024 * 1024:
                raise ValueError("Combined baseline proofs exceed 256 MiB")
            shutil.copytree(source, target)
            copied = target
        record = {"task": task, "reuse": {"report": "reused/" + name, "task": reference["task"]}}
        resolved(manifest, [record], output, root=suite, validated_sources=validated_sources)
        return record
    except (ValueError, KeyError, OSError, TypeError) as error:
        if copied is not None:
            shutil.rmtree(copied)
        print(f"Baseline cache miss for {task['id']}: {error}", flush=True)
        return None


def prune(cache: Path, *, max_reports: int = 8, max_bytes: int = 1024**3) -> None:
    """Bound host cache storage; retained gate reports already contain their own proofs."""
    reports = cache / "reports"
    entries = [
        path
        for path in reports.iterdir()
        if path.is_dir()
        and not path.is_symlink()
        and len(path.name) == 64
        and all(value in "0123456789abcdef" for value in path.name)
    ]
    total = 0
    kept = set()
    for path in sorted(entries, key=lambda path: path.stat().st_mtime_ns, reverse=True):
        size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        if len(kept) < max_reports and total + size <= max_bytes:
            total += size
            kept.add(path.name)
        else:
            shutil.rmtree(path)
    for entry in (cache / "keys").glob("*.json"):
        if entry.is_symlink():
            continue
        try:
            if json.loads(entry.read_text()).get("report") not in kept:
                entry.unlink()
        except (ValueError, OSError, AttributeError):
            entry.unlink(missing_ok=True)


def retain(cache: Path, output: Path, manifest: dict, execution: dict) -> None:
    """Seed only fresh originals from a complete passing run after collection."""
    integrity(output)
    records = [json.loads(path.read_text()) for path in sorted((output / "results").rglob("*.json"))]
    if not json.loads((output / "summary.json").read_text()).get("passed"):
        return
    if not evaluate(manifest, records, root=output / "source", reports=output)["passed"]:
        return
    # The reference preserves the complete immutable source report, deduplicated per run.
    name = digest(output / "checksums.json")
    destination = cache / "reports" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix="pending-") as temporary:
            pending = Path(temporary) / "report"
            shutil.copytree(output, pending)
            pending.rename(destination)
    for path in sorted((output / "results").rglob("*.json")):
        record = json.loads(path.read_text())
        if "reuse" in record:
            continue
        task = record["task"]
        key = execution_key(manifest, task, execution)
        entry = cache / "keys" / (key + ".json")
        entry.parent.mkdir(parents=True, exist_ok=True)
        # Only the index is replaceable; execution reports and records remain immutable.
        with tempfile.TemporaryDirectory(dir=entry.parent, prefix="pending-") as temporary:
            pending = Path(temporary) / "entry.json"
            write_json(pending, {"report": name, "task": task_key(task)})
            pending.replace(entry)
    prune(cache)


def run(options: argparse.Namespace) -> None:
    suite = options.suite_root.resolve()
    output = options.output.resolve()
    if output == suite or suite in output.parents:
        raise ValueError("Store lab reports outside the read-only suite source tree")
    output.mkdir(parents=True, exist_ok=False)
    marker = output.parent / (output.name + ".pid")
    with marker.open("x") as handle:
        handle.write(str(os.getpid()) + "\n")
    candidate = {
        "revision": options.candidate_sha,
        "wheel_sha256": digest(options.candidate_wheel),
        "payload_sha256": wheel_payload(options.candidate_wheel),
    }
    baseline = {
        "revision": options.baseline_sha,
        "wheel_sha256": digest(options.baseline_wheel),
        "payload_sha256": wheel_payload(options.baseline_wheel),
    }
    manifest = plan(
        run_id=output.name,
        candidate=candidate,
        baseline=baseline,
        evidence_sha=options.evidence_sha,
        executor_sha=options.executor_sha,
        root=suite,
        identifiers=options.ids,
        steps=options.steps,
    )
    manifest["backend"] = "lab-docker"
    manifest["requirements_sha256"] = digest(options.requirements)
    manifest["suite_sha256"] = suite_identity(suite)
    write_json(output / "manifest.json", manifest)
    shutil.copytree(suite, output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    wheels = {"candidate": options.candidate_wheel.resolve(), "baseline": options.baseline_wheel.resolve()}
    for role, wheel in wheels.items():
        target = output / "artifacts" / role / wheel.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wheel, target)
    shutil.copy2(options.requirements, output / "artifacts/requirements.txt")
    label = hashlib.sha256(output.name.encode()).hexdigest()[:20]
    cache = Path.home() / "relflow-evidence/cache/v1"
    try:
        execution = environment(options.requirements, options.workers)
        image = execution["image_id"]
        write_json(output / "execution.json", execution)

        def one(task: dict) -> dict:
            scratch = output / "tasks" / task_key(task)
            scratch.mkdir(parents=True)
            write_json(scratch / "task.json", task)
            role = task["role"]
            name = "evidence-" + label + "-" + hashlib.sha256(task_key(task).encode()).hexdigest()[:12]
            wheel = wheels[role]
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                name,
                "--label",
                "relflow.evidence=" + label,
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "512",
                "--cpus",
                "2",
                "--memory",
                "4g",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=2g,mode=1777",
                "--mount",
                f"type=bind,src={ROOT},dst=/executor,readonly",
                "--mount",
                f"type=bind,src={suite},dst=/suite,readonly",
                "--mount",
                f"type=bind,src={wheel},dst=/wheels/{wheel.name},readonly",
                "--mount",
                f"type=bind,src={scratch},dst=/results",
                "--env",
                "HOME=/tmp/home",
                "--env",
                "USER=evidence",
                "--env",
                "LOGNAME=evidence",
                "--env",
                "EVIDENCE_REVISION=" + options.evidence_sha,
                "--env",
                "EVIDENCE_RELFLOW_REVISION=" + manifest[role]["revision"],
                "--env",
                "EVIDENCE_WHEEL_SHA256=" + manifest[role]["wheel_sha256"],
                image,
                "sh",
                "-c",
                "python -m venv --system-site-packages /tmp/runtime && "
                "/tmp/runtime/bin/python -m pip install --no-deps "
                + shlex.quote("/wheels/" + wheel.name)
                + " && exec /tmp/runtime/bin/python /executor/lab_worker.py /results/task.json",
            ]
            try:
                process = subprocess.run(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1850, check=False
                )
                record_path = scratch / "record.json"
                if record_path.is_symlink():
                    raise ValueError("Container result must not be a symbolic link")
                if record_path.is_file():
                    if record_path.stat().st_size > 64 * 1024 * 1024:
                        raise ValueError("Container result exceeds the 64 MiB record bound")
                    record = json.loads(record_path.read_text())
                    if record.get("task") != task:
                        raise ValueError("Container result identity differs from its planned task")
                    if process.returncode:
                        record["error"] = f"Container exited with status {process.returncode}"
                        record["log"] = process.stdout + "\n" + record.get("log", "")
                    return record
                return {
                    "task": task,
                    "result": None,
                    "error": f"Container exited {process.returncode} without a result",
                    "log": process.stdout,
                }
            except (subprocess.TimeoutExpired, ValueError, OSError) as error:
                return {"task": task, "result": None, "error": str(error), "log": ""}
            finally:
                subprocess.run(
                    ["docker", "rm", "--force", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                )

        tasks = []
        validated_sources = {}
        for task in manifest["tasks"]:
            record = None
            if task["role"] == "baseline" and options.steps is None and not options.force_fresh:
                record = cached(cache, output, manifest, task, execution, suite, validated_sources)
            if record is None:
                tasks.append(task)
            else:
                collect(output, record)
                print(f"Reused baseline {task['id']} from {record['reuse']['report']}", flush=True)
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=options.workers)
        try:
            futures = [pool.submit(one, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                collect(output, record)
                print(
                    f"Collected {record['task']['role']} {record['task']['id']}: "
                    f"{(record.get('result') or {}).get('outcome', 'error')}",
                    flush=True,
                )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            active = subprocess.check_output(
                ["docker", "ps", "--quiet", "--filter", "label=relflow.evidence=" + label], text=True
            ).split()
            if active:
                subprocess.run(["docker", "rm", "--force", *active], check=False)
    finally:
        try:
            summary = finish(output, root=suite)
            print(json.dumps(summary, indent=2), flush=True)
        finally:
            marker.unlink(missing_ok=True)
    if not summary["passed"]:
        raise SystemExit(1)
    try:
        retain(cache, output, manifest, execution)
    except (OSError, ValueError) as error:
        print(f"Passing run was not cached: {error}", flush=True)


def main() -> None:
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    for signum in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(signum, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-wheel", required=True, type=Path)
    parser.add_argument("--baseline-wheel", required=True, type=Path)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--baseline-sha", required=True)
    parser.add_argument("--evidence-sha", required=True)
    parser.add_argument("--executor-sha", required=True)
    parser.add_argument("--suite-root", required=True, type=Path)
    parser.add_argument("--requirements", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--force-fresh", action="store_true")
    parser.add_argument("--probe", action="store_true")
    options = parser.parse_args()
    if not 1 <= options.workers <= 16:
        parser.error("workers must be between 1 and 16")
    if options.steps is not None and options.steps < 1:
        parser.error("steps must be positive")
    if options.probe:
        write_json(options.output, environment(options.requirements, options.workers))
    else:
        run(options)


if __name__ == "__main__":
    main()
