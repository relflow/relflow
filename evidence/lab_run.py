"""Run paired evidence on lab with one disposable Docker container per task."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

from evidence_ci import ROOT, collect, digest, finish, plan, task_key, write_json


def image_definition(torch_version: str) -> str:
    return f"""FROM python:3.12-slim
RUN python -m pip install --no-cache-dir torch=={torch_version}+cpu --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt pyyaml==6.0.3
ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=2 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"""


def run(options: argparse.Namespace) -> None:
    suite = options.suite_root.resolve()
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    marker = output.parent / (output.name + ".pid")
    with marker.open("x") as handle:
        handle.write(str(os.getpid()) + "\n")
    candidate = {"revision": options.candidate_sha, "wheel_sha256": digest(options.candidate_wheel)}
    baseline = {"revision": options.baseline_sha, "wheel_sha256": digest(options.baseline_wheel)}
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
    write_json(output / "manifest.json", manifest)
    wheels = {"candidate": options.candidate_wheel.resolve(), "baseline": options.baseline_wheel.resolve()}
    for role, wheel in wheels.items():
        target = output / "artifacts" / role / wheel.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wheel, target)
    shutil.copy2(options.requirements, output / "artifacts/requirements.txt")
    version = next(
        (
            line.split("==", 1)[1].split(";", 1)[0].split()[0].rstrip("\\")
            for line in options.requirements.read_text().splitlines()
            if line.startswith("torch==")
        ),
        None,
    )
    if not version:
        raise ValueError("The runtime requirements must pin Torch")
    image = "relflow-evidence:" + digest(options.requirements)[:20]
    label = hashlib.sha256(output.name.encode()).hexdigest()[:20]
    try:
        with tempfile.TemporaryDirectory(prefix="evidence-image-") as temporary:
            context = Path(temporary)
            (context / "Dockerfile").write_text(image_definition(version))
            shutil.copy2(options.requirements, context / "requirements.txt")
            subprocess.run(["docker", "build", "--tag", image, str(context)], check=True)
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True
        ).strip()
        write_json(
            output / "execution.json",
            {"image_id": image_id, "workers": options.workers, "cpu_per_task": 2, "memory_per_task": "4g"},
        )

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

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=options.workers)
        try:
            futures = [pool.submit(one, task) for task in manifest["tasks"]]
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
    options = parser.parse_args()
    if not 1 <= options.workers <= 16:
        parser.error("workers must be between 1 and 16")
    if options.steps is not None and options.steps < 1:
        parser.error("steps must be positive")
    run(options)


if __name__ == "__main__":
    main()
