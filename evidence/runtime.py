"""Build verified images and execute isolated Evidence tasks on GitHub runners."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from evidence_ci import ROOT, digest, identity, write_json


def image_definition(torch_version: str) -> str:
    return f"""FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30
RUN python -m pip install --no-cache-dir torch=={torch_version}+cpu --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt pyyaml==6.0.3
RUN python -m pip check
ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=2 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"""


def runtime_recipe(requirements: Path) -> str:
    """Resolve one CPU runtime recipe from the exact supplied requirements."""
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
    return image_definition(version.removesuffix("+cpu"))


def environment(
    requirements: Path, workers: int, *, image_id: str | None = None, build_cache: Path | None = None
) -> dict:
    """Build or inspect the trusted runtime shared by all hosted tasks."""
    recipe = runtime_recipe(requirements)
    image = "relflow-evidence:" + identity({"lock": digest(requirements), "recipe": recipe})[:20]
    if image_id is None:
        with tempfile.TemporaryDirectory(prefix="evidence-image-") as temporary:
            context = Path(temporary)
            (context / "Dockerfile").write_text(recipe)
            shutil.copy2(requirements, context / "requirements.txt")
            command = ["docker", "build", "--tag", image]
            if build_cache is not None:
                pending = build_cache.with_name(build_cache.name + ".next")
                if pending.exists():
                    shutil.rmtree(pending)
                command = ["docker", "buildx", "build", "--load", "--tag", image]
                if (build_cache / "index.json").is_file():
                    command += ["--cache-from", f"type=local,src={build_cache}"]
                command += ["--cache-to", f"type=local,dest={pending},mode=max"]
            subprocess.run([*command, str(context)], check=True)
            if build_cache is not None:
                if build_cache.exists():
                    shutil.rmtree(build_cache)
                pending.rename(build_cache)
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True
        ).strip()
    else:
        actual = subprocess.check_output(
            ["docker", "image", "inspect", image_id, "--format", "{{.Id}}"], text=True
        ).strip()
        if actual != image_id:
            raise ValueError("Loaded dependency image differs from its trusted identity")
        subprocess.run(["docker", "image", "tag", image_id, image], check=True)
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
            "runtime_layout": "installed-wheel-v1",
            "input_layout": "declared-files-v1",
            "scheduler": "longest-first-v1",
            "task_timeout_seconds": 1850,
            "experiment_timeout_seconds": 1700,
            "observation_capture": "py-return-v1",
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


def wheel_image(execution: dict, wheel: Path) -> str:
    """Install the exact wheel once in a cached layer; tasks still get fresh containers."""
    recipe = (
        f"FROM {execution['image']}\n"
        f"COPY {wheel.name} /tmp/{wheel.name}\n"
        "RUN python -m pip install --no-deps --no-index " + shlex.quote("/tmp/" + wheel.name) + "\n"
    )
    tag = (
        "relflow-evidence-wheel:"
        + identity({"base": execution["image_id"], "wheel": digest(wheel), "recipe": recipe})[:20]
    )
    with tempfile.TemporaryDirectory(prefix="evidence-wheel-") as temporary:
        context = Path(temporary)
        (context / "Dockerfile").write_text(recipe)
        shutil.copy2(wheel, context / wheel.name)
        subprocess.run(["docker", "build", "--network", "none", "--tag", tag, str(context)], check=True)
    return subprocess.check_output(["docker", "image", "inspect", tag, "--format", "{{.Id}}"], text=True).strip()


def isolate(suite: Path, destination: Path, task: dict, inputs: dict, scripts: set[str]) -> Path:
    """Expose exactly the selected experiment and shared files bound by its task key."""
    selected = {name: value for name, value in inputs.items() if name not in scripts or name == task["script"]}
    if identity(selected) != task["inputs_sha256"]:
        raise ValueError(f"Execution inputs differ from the planned task: {task['id']}")
    destination.mkdir(parents=True)
    for name, expected in selected.items():
        source = suite / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Execution inputs require regular files: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        actual = {"sha256": digest(target), "mode": target.stat().st_mode & 0o777}
        if actual != expected:
            raise ValueError(f"Execution input changed after planning: {name}")
    return destination


def container(
    task: dict,
    *,
    execution: dict,
    scratch: Path,
    view: Path,
    run_id: str,
    evidence_sha: str,
    package: dict,
    observations: Path | None = None,
) -> dict:
    """Execute one immutable launch with the shared Docker isolation policy.

    A launch journal records Docker status separately from the experiment's
    result. Callers may retry only a Docker 125 launch without a result; every
    launch keeps its own directory and journal.
    """
    launched = time.monotonic()
    started_ns = time.monotonic_ns()
    label = hashlib.sha256(run_id.encode()).hexdigest()[:20]
    name = "evidence-" + label + "-" + hashlib.sha256(str(scratch).encode()).hexdigest()[:12]
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
        f"type=bind,src={view},dst=/suite,readonly",
        "--mount",
        f"type=bind,src={scratch},dst=/results",
        "--env",
        "HOME=/tmp/home",
        "--env",
        "USER=evidence",
        "--env",
        "LOGNAME=evidence",
        "--env",
        "EVIDENCE_REVISION=" + evidence_sha,
        "--env",
        "EVIDENCE_RELFLOW_REVISION=" + package["revision"],
        "--env",
        "EVIDENCE_WHEEL_SHA256=" + package["wheel_sha256"],
    ]
    if observations is not None:
        observations.mkdir(parents=True, exist_ok=False)
        command += [
            "--mount",
            f"type=bind,src={observations},dst=/observations",
            "--env",
            "EVIDENCE_OBSERVATIONS=/observations",
            "--env",
            "EVIDENCE_RUN_ID=" + run_id,
        ]
    command += [
        execution["role_images"][task["role"]],
        "python",
        "/executor/worker.py",
        "/results/task.json",
    ]
    code = None
    timed_out = False
    record_path = scratch / "record.json"
    try:
        process = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1850, check=False
        )
        code = process.returncode
        log = process.stdout
        if record_path.is_symlink():
            raise ValueError("Container result must not be a symbolic link")
        if record_path.is_file():
            if record_path.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("Container result exceeds the 64 MiB record bound")
            record = json.loads(record_path.read_text())
            if record.get("task") != task:
                raise ValueError("Container result identity differs from its planned task")
            if code:
                record["error"] = f"Container exited with status {code}"
                record["log"] = log + "\n" + record.get("log", "")
            return record
        return {"task": task, "result": None, "error": f"Container exited {code} without a result", "log": log}
    except subprocess.TimeoutExpired as error:
        timed_out = True
        log = error.stdout or ""
        if isinstance(log, bytes):
            log = log.decode(errors="replace")
        return {"task": task, "result": None, "error": "Container launch timed out", "log": log}
    except (ValueError, OSError) as error:
        return {"task": task, "result": None, "error": str(error), "log": ""}
    finally:
        subprocess.run(
            ["docker", "rm", "--force", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        )
        write_json(
            scratch / "launch.json",
            {
                "schema": 1,
                "task": task,
                "container": name,
                "docker_exit_code": code,
                "image_id": execution["role_images"][task["role"]],
                "started_ns": started_ns,
                "finished_ns": time.monotonic_ns(),
                "timed_out": timed_out,
                "has_record": record_path.is_file(),
                "record_sha256": digest(record_path)
                if record_path.is_file() and not record_path.is_symlink()
                else None,
                "duration_seconds": round(time.monotonic() - launched, 3),
            },
        )


def render_plots(reports: Path, observations: Path, output: Path) -> dict:
    """Execute the selected renderer only inside a bounded offline container."""
    recipe = (
        "FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30\n"
        "RUN python -m pip install --no-cache-dir matplotlib==3.10.7 pyarrow==24.0.0 numpy==2.4.6 pyyaml==6.0.3\n"
    )
    image = "relflow-evidence-plots:" + hashlib.sha256(recipe.encode()).hexdigest()[:20]
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="evidence-plot-image-") as directory:
        (Path(directory) / "Dockerfile").write_text(recipe)
        subprocess.run(["docker", "build", "--tag", image, directory], check=True)
    image_id = subprocess.check_output(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True).strip()
    with tempfile.TemporaryDirectory(prefix="evidence-plot-work-") as staging:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--cpus",
                "2",
                "--memory",
                "4g",
                "--pids-limit",
                "256",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=2g,mode=1777",
                "--mount",
                f"type=bind,src={reports.resolve()},dst=/reports,readonly",
                "--mount",
                f"type=bind,src={(reports / 'source').resolve()},dst=/workspace/evidence,readonly",
                "--mount",
                f"type=bind,src={observations.resolve()},dst=/observations,readonly",
                "--mount",
                f"type=bind,src={output.resolve()},dst=/output",
                "--mount",
                f"type=bind,src={staging},dst=/work",
                "--workdir",
                "/workspace",
                "--env",
                "PYTHONPATH=/workspace:/workspace/evidence",
                "--env",
                "TMPDIR=/work",
                "--env",
                "MPLCONFIGDIR=/tmp/matplotlib",
                "--env",
                "OMP_NUM_THREADS=1",
                "--env",
                "OPENBLAS_NUM_THREADS=1",
                "--env",
                "USER=evidence",
                "--env",
                "LOGNAME=evidence",
                image_id,
                "python",
                "-m",
                "evidence.diagnostics.build",
                "--reports",
                "/reports",
                "--observations",
                "/observations",
                "--output",
                "/output/figures",
                "--archive",
                "/output/proof-visualizations.zip",
            ],
            check=True,
            timeout=1700,
        )
    return {"schema": 1, "image_id": image_id, "recipe_sha256": hashlib.sha256(recipe.encode()).hexdigest()}
