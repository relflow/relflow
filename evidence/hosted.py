"""Run complete paired evidence across bounded GitHub-hosted Docker workers."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from evidence_ci import (
    collect,
    digest,
    finish,
    plan,
    runtime_inputs,
    suite_identity,
    task_key,
    wheel_payload,
    write_json,
)
from experiments.source import registry
from lab_run import container, environment, isolate, runtime_recipe, wheel_image

BACKEND = "github-hosted-docker"
PROFILE = {
    "schema": 1,
    "shards": 8,
    "pair_workers": 2,
    "cpu_per_task": 2,
    "memory_per_task": "4g",
    "pids": 512,
    "tmpfs": "2g",
    "network": "none",
    "read_only": True,
    "cap_drop": "ALL",
    "no_new_privileges": True,
    "scheduler": "paired-longest-first-shards-v1",
    "launch_attempts": 3,
    "task_timeout_seconds": 1850,
    "experiment_timeout_seconds": 1700,
    "runtime_layout": "installed-wheel-v1",
    "input_layout": "declared-files-v1",
}
THREADS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "POLARS_MAX_THREADS": "2",
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "threads": 1,
}
LIMIT = 256 * 1024 * 1024


def assignments(manifest: dict, suite: Path) -> list[list[str]]:
    """Balance whole pairs using immutable historical durations as hints only."""
    durations = {}
    for identifier, entry in registry(suite).items():
        value = (entry["run"] or {}).get("duration_seconds", 0)
        durations[identifier] = value if type(value) in {int, float} and math.isfinite(value) and value > 0 else 0
    identifiers = {task["id"] for task in manifest["tasks"]}
    buckets: list[list[str]] = [[] for _ in range(PROFILE["shards"])]
    weights = [0.0] * len(buckets)
    for identifier in sorted(identifiers, key=lambda value: (-durations.get(value, 0), value)):
        index = min(range(len(buckets)), key=lambda value: (weights[value], len(buckets[value]), value))
        buckets[index].append(identifier)
        weights[index] += 2 * durations.get(identifier, 0)
    return buckets


def inventory(root: Path) -> dict[str, str]:
    """Bound and hash every regular file, rejecting aliases and special files."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Evidence artifacts require a regular directory")
    result = {}
    size = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Evidence artifacts cannot contain symbolic links")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Evidence artifacts cannot contain special files")
        if path == root / "checksums.json":
            continue
        size += path.stat().st_size
        if size > LIMIT:
            raise ValueError("Evidence artifact exceeds the 256 MiB bound")
        result[path.relative_to(root).as_posix()] = digest(path)
    return result


def seal(root: Path) -> None:
    write_json(root / "checksums.json", inventory(root))


def validate_inventory(root: Path) -> None:
    if json.loads((root / "checksums.json").read_text()) != inventory(root):
        raise ValueError("Evidence artifact checksum inventory differs from its files")


def context(request: dict) -> dict:
    """Bind each job to the original prepare attempt; partial reruns fail closed."""
    current = {
        "run_id": int(os.environ.get("GITHUB_RUN_ID", "0")),
        "run_attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "0")),
    }
    if any(current[name] != request[name] for name in current):
        raise ValueError("Evidence belongs to a different GitHub attempt; re-run all jobs to prepare a fresh plan")
    return {
        **current,
        "job": os.environ.get("GITHUB_JOB", ""),
        "runner": os.environ.get("RUNNER_NAME", ""),
        "image_os": os.environ.get("ImageOS", ""),
        "image_version": os.environ.get("ImageVersion", ""),
    }


def load_plan(root: Path) -> dict:
    validate_inventory(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("backend") != BACKEND or manifest.get("profile") != PROFILE:
        raise ValueError("Evidence plan differs from the reviewed hosted execution profile")
    suite = root / "source"
    if manifest.get("suite_sha256") != suite_identity(suite):
        raise ValueError("Evidence plan source differs from its immutable suite identity")
    expected = plan(
        run_id=manifest["run_id"],
        candidate=manifest["candidate"],
        baseline=manifest["baseline"],
        evidence_sha=manifest["evidence_sha"],
        executor_sha=manifest["executor_sha"],
        root=suite,
    )
    for name in ("tasks", "policy", "mode", "full_catalog", "runtime_sha256"):
        if manifest.get(name) != expected[name]:
            raise ValueError(f"Evidence plan changed canonical tasks or budgets: {name}")
    if manifest.get("shards") != assignments(manifest, suite):
        raise ValueError("Evidence shard assignment differs from the deterministic paired plan")
    for role in ("baseline", "candidate"):
        wheels = list((root / "artifacts" / role).glob("*.whl"))
        if len(wheels) != 1 or digest(wheels[0]) != manifest[role]["wheel_sha256"]:
            raise ValueError(f"Evidence {role} wheel differs from its planned bytes")
        if wheel_payload(wheels[0]) != manifest[role]["payload_sha256"]:
            raise ValueError(f"Evidence {role} wheel differs from its planned runtime payload")
    if digest(root / "artifacts/requirements.txt") != manifest["requirements_sha256"]:
        raise ValueError("Evidence requirements differ from the planned runtime")
    dependency = manifest.get("dependency_image")
    recipe = hashlib.sha256(runtime_recipe(root / "artifacts/requirements.txt").encode()).hexdigest()
    if (
        dependency != json.loads((root / "artifacts/runtime.json").read_text())
        or dependency.get("requirements_sha256") != manifest["requirements_sha256"]
        or dependency.get("recipe_sha256") != recipe
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", dependency.get("image_id", ""))
        or not re.fullmatch(r"[0-9a-f]{64}", dependency.get("archive_sha256", ""))
        or dependency.get("torch_cuda") is not None
        or not dependency.get("torch_version", "").endswith("+cpu")
        or not isinstance(dependency.get("packages"), dict)
    ):
        raise ValueError("Dependency image differs from its exact CPU recipe, inventory, or trusted prepare inputs")
    request = json.loads((root / "artifacts/request.json").read_text())
    match = re.fullmatch(r"gh-(\d+)-(\d+)", manifest["run_id"])
    if (
        match is None
        or [request.get("run_id"), request.get("run_attempt")] != [int(value) for value in match.groups()]
        or request.get("candidate") != manifest["candidate"]["revision"]
        or request.get("baseline") != manifest["baseline"]["revision"]
        or request.get("suite") != manifest["evidence_sha"]
    ):
        raise ValueError("Evidence plan request differs from its exact source identities")
    return manifest


def create_plan(options: argparse.Namespace) -> None:
    output = options.output.resolve()
    suite = options.suite_root.resolve()
    if output == suite or suite in output.parents:
        raise ValueError("Store plans outside selected experiment sources")
    suite_identity(suite)
    inventory(options.inputs)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(suite, output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(options.inputs, output / "artifacts")
    if getattr(options, "runtime_manifest", None) is not None:
        shutil.copy2(options.runtime_manifest, output / "artifacts/runtime.json")
    packages = {}
    for role, revision in (("candidate", options.candidate_sha), ("baseline", options.baseline_sha)):
        wheels = list((output / "artifacts" / role).glob("*.whl"))
        if len(wheels) != 1 or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"Evidence {role} requires one wheel and a full source revision")
        packages[role] = {
            "revision": revision,
            "wheel_sha256": digest(wheels[0]),
            "payload_sha256": wheel_payload(wheels[0]),
        }
    manifest = plan(
        run_id=options.run_id,
        candidate=packages["candidate"],
        baseline=packages["baseline"],
        evidence_sha=options.evidence_sha,
        executor_sha=options.executor_sha,
        root=output / "source",
    )
    manifest.update(
        backend=BACKEND,
        profile=PROFILE,
        requirements_sha256=digest(output / "artifacts/requirements.txt"),
        suite_sha256=suite_identity(output / "source"),
    )
    manifest["shards"] = assignments(manifest, output / "source")
    manifest["dependency_image"] = json.loads((output / "artifacts/runtime.json").read_text())
    write_json(output / "manifest.json", manifest)
    seal(output)
    load_plan(output)
    context(json.loads((output / "artifacts/request.json").read_text()))


def setup(operation, output: Path):
    """Retry bounded setup only, before any measurement has been launched."""
    for attempt in range(1, 4):
        started = time.monotonic()
        try:
            value = operation()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
            write_json(
                output / f"attempt-{attempt}.json",
                {"attempt": attempt, "error": str(error), "seconds": time.monotonic() - started},
            )
            if attempt == 3:
                raise
        else:
            write_json(
                output / f"attempt-{attempt}.json",
                {"attempt": attempt, "error": None, "seconds": time.monotonic() - started},
            )
            return value
    raise AssertionError("Unreachable setup attempt")


def stream_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def image(options: argparse.Namespace) -> None:
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    execution = setup(
        lambda: environment(options.requirements, PROFILE["pair_workers"], build_cache=options.build_cache),
        output / "setup",
    )
    probe = setup(
        lambda: subprocess.check_output(
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
                "2g",
                "--pids-limit",
                "128",
                execution["image"],
                "python",
                "-c",
                "import importlib.metadata as m,json,torch; assert torch.version.cuda is None; "
                "print(json.dumps({'torch_version':str(torch.__version__),'torch_cuda':torch.version.cuda,"
                "'packages':dict(sorted((d.metadata['Name'].lower(),d.version) for d in m.distributions()))},sort_keys=True))",
            ],
            text=True,
            timeout=120,
        ),
        output / "probe",
    )
    if len(probe) > 1024 * 1024:
        raise ValueError("Runtime package inventory exceeds one MiB")
    packages = json.loads(probe)
    archive = output / "runtime.tar"
    setup(
        lambda: subprocess.run(["docker", "save", "--output", str(archive), execution["image"]], check=True),
        output / "save",
    )
    if archive.stat().st_size > 6 * 1024**3:
        raise ValueError("Dependency image exceeds the 6 GiB archive bound")
    write_json(
        output / "runtime.json",
        {
            "schema": 1,
            "image_id": execution["image_id"],
            "archive_name": archive.name,
            "archive_sha256": stream_digest(archive),
            "recipe_sha256": execution["identity"]["recipe_sha256"],
            "requirements_sha256": digest(options.requirements),
            **packages,
        },
    )


def launch(
    task: dict, *, output: Path, suite: Path, manifest: dict, execution: dict, execution_sha256: str, shard: int
) -> dict:
    """Retry a Docker 125 without a record, preserving every launch and log."""
    root = output / "tasks" / task_key(task)
    inputs = runtime_inputs(suite)
    scripts = {entry["script"] for entry in registry(suite, measurements=False).values()}
    journals = []
    for attempt in range(1, PROFILE["launch_attempts"] + 1):
        scratch = root / f"launch-{attempt}"
        scratch.mkdir(parents=True)
        write_json(scratch / "task.json", task)
        view = isolate(suite, output / "views" / task_key(task) / f"launch-{attempt}", task, inputs, scripts)
        try:
            record = container(
                task,
                execution=execution,
                scratch=scratch,
                view=view,
                run_id=manifest["run_id"],
                evidence_sha=manifest["evidence_sha"],
                package=manifest[task["role"]],
            )
        finally:
            shutil.rmtree(view)
        collect(scratch / "captured", record)
        journal_path = scratch / "launch.json"
        journal = json.loads(journal_path.read_text())
        journals.append({"path": journal_path.relative_to(output).as_posix(), "sha256": digest(journal_path)})
        retry = journal["docker_exit_code"] == 125 and not journal["has_record"] and not journal["timed_out"]
        if not retry or attempt == PROFILE["launch_attempts"]:
            write_json(root / "selection.json", {"task": task, "launches": journals, "selected": attempt})
            record.update(
                shard=shard, execution_sha256=execution_sha256, launch=journal_path.relative_to(output).as_posix()
            )
            collect(output, record)
            return record
    raise AssertionError("Unreachable launch attempt")


def run(options: argparse.Namespace) -> None:
    started = time.monotonic()
    source = options.plan.resolve()
    manifest = load_plan(source)
    request = json.loads((source / "artifacts/request.json").read_text())
    runner = context(request)
    if type(options.shard) is not int or not 0 <= options.shard < PROFILE["shards"]:
        raise ValueError("Evidence shard must be an integer from zero through seven")
    output = options.output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Store shard outputs outside the immutable plan")
    output.mkdir(parents=True, exist_ok=False)
    requirements = source / "artifacts/requirements.txt"
    if options.runtime_image is not None:
        archive = options.runtime_image.resolve()
        if archive.is_symlink() or not archive.is_file() or archive.stat().st_size > 6 * 1024**3:
            raise ValueError("Dependency image archive is missing, linked, or oversized")
        if stream_digest(archive) != options.runtime_sha256 or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", options.runtime_id or ""
        ):
            raise ValueError("Dependency image differs from the trusted prepare outputs")
        if (
            options.runtime_sha256 != manifest["dependency_image"]["archive_sha256"]
            or options.runtime_id != manifest["dependency_image"]["image_id"]
        ):
            raise ValueError("Dependency image differs from its immutable plan attestation")
        setup(lambda: subprocess.run(["docker", "load", "--input", str(archive)], check=True), output / "setup/load")
        execution = setup(
            lambda: environment(requirements, PROFILE["pair_workers"], image_id=options.runtime_id),
            output / "setup/probe",
        )
    else:
        execution = setup(lambda: environment(requirements, PROFILE["pair_workers"]), output / "setup/environment")
    execution["identity"]["resources"]["scheduler"] = PROFILE["scheduler"]
    execution["runner"] = runner
    execution["role_images"] = {}
    for role in ("baseline", "candidate"):
        wheel = next((source / "artifacts" / role).glob("*.whl"))
        execution["role_images"][role] = setup(lambda: wheel_image(execution, wheel), output / "setup" / role)
    check_runtime(execution, manifest["dependency_image"], options.shard)
    write_json(output / "execution.json", execution)
    fingerprint = digest(output / "execution.json")
    assigned = manifest["shards"][options.shard]
    write_json(
        output / "shard.json",
        {
            "schema": 1,
            "index": options.shard,
            "manifest_sha256": digest(source / "manifest.json"),
            "ids": assigned,
            "execution_sha256": fingerprint,
        },
    )

    def pair(identifier: str) -> None:
        for role in ("baseline", "candidate"):
            task = next(task for task in manifest["tasks"] if task["id"] == identifier and task["role"] == role)
            launch(
                task,
                output=output,
                suite=source / "source",
                manifest=manifest,
                execution=execution,
                execution_sha256=fingerprint,
                shard=options.shard,
            )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=PROFILE["pair_workers"]) as pool:
            futures = [pool.submit(pair, identifier) for identifier in assigned]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    finally:
        write_json(output / "timing.json", {"wall_seconds": round(time.monotonic() - started, 3)})
        seal(output)


def check_runtime(actual: dict, dependency: dict, index: int) -> str:
    """Validate the complete container and actual-host contract before fitting."""
    runtime = actual["identity"]
    if (
        not {"system", "release", "version", "machine", "node", "cpu_count", "memory_bytes", "uid", "gid", "cpu"}
        <= runtime.get("host", {}).keys()
    ):
        raise ValueError("Shard is missing actual host identity fields")
    if not runtime["host"]["node"] or not runtime["host"]["cpu"] or not runtime.get("docker"):
        raise ValueError("Shard is missing actual host, CPU, or Docker identification")
    if runtime["image_id"] != dependency["image_id"] or runtime["recipe_sha256"] != dependency["recipe_sha256"]:
        raise ValueError("Shard used another dependency image or runtime recipe than trusted prepare")
    if runtime.get("environment") != THREADS:
        raise ValueError("Shard changed the declared thread or offline environment")
    if set(actual.get("role_images", {})) != {"baseline", "candidate"} or any(
        not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value)
        for value in actual["role_images"].values()
    ):
        raise ValueError("Shard wheel images require exact Docker content identities")
    resources = runtime["resources"]
    for name in (
        "cpu_per_task",
        "memory_per_task",
        "pids",
        "tmpfs",
        "network",
        "read_only",
        "cap_drop",
        "no_new_privileges",
        "scheduler",
        "task_timeout_seconds",
        "experiment_timeout_seconds",
        "runtime_layout",
        "input_layout",
    ):
        if resources.get(name) != PROFILE[name]:
            raise ValueError(f"Shard {index} changed container policy: {name}")
    if resources.get("workers") != PROFILE["pair_workers"] or runtime["host"]["system"] != "Linux":
        raise ValueError("Hosted shards require two pair workers on Linux")
    if runtime["host"]["cpu_count"] < 4 or runtime["host"]["memory_bytes"] < 8 * 1024**3:
        raise ValueError("Hosted shard cannot sustain the reviewed CPU and memory limits")
    return runtime["image_id"]


def verify_execution(manifest: dict, records: list[dict], reports: Path) -> list[str]:
    """Validate per-host pairing and every immutable launch before gating learning."""
    if manifest.get("backend") != BACKEND:
        return []
    errors = []
    try:
        if manifest.get("profile") != PROFILE or manifest.get("shards") != assignments(manifest, reports / "source"):
            raise ValueError("Hosted execution profile or shard plan differs from the reviewed policy")
        execution = json.loads((reports / "execution.json").read_text())
        if execution.get("schema") != 2 or execution.get("identity") != {"backend": BACKEND, "profile": PROFILE}:
            raise ValueError("Hosted execution requires a sharded identity without a fabricated common host")
        shards = execution.get("shards", [])
        if len(shards) != PROFILE["shards"] or {entry["index"] for entry in shards} != set(range(PROFILE["shards"])):
            raise ValueError("Hosted execution is missing or duplicating shard identities")
        request = json.loads((reports / "artifacts/request.json").read_text())
        dependency = manifest["dependency_image"]
        if (
            dependency != json.loads((reports / "artifacts/runtime.json").read_text())
            or dependency["requirements_sha256"] != manifest["requirements_sha256"]
            or dependency["requirements_sha256"] != digest(reports / "artifacts/requirements.txt")
            or dependency["recipe_sha256"]
            != hashlib.sha256(runtime_recipe(reports / "artifacts/requirements.txt").encode()).hexdigest()
        ):
            raise ValueError("Hosted runtime attestation differs from the exact requirements and recipe")
        expected_tasks = {task_key(task): task for task in manifest["tasks"]}
        seen = set()
        base_images = set()
        for entry in shards:
            index = entry["index"]
            directory = reports / "shards" / f"shard-{index}"
            validate_inventory(directory)
            receipt = json.loads((directory / "shard.json").read_text())
            actual = json.loads((directory / "execution.json").read_text())
            if entry != {"index": index, "execution_sha256": digest(directory / "execution.json"), "identity": actual}:
                raise ValueError("Shard execution identity differs from its retained original")
            if receipt != {
                "schema": 1,
                "index": index,
                "manifest_sha256": digest(reports / "manifest.json"),
                "ids": manifest["shards"][index],
                "execution_sha256": entry["execution_sha256"],
            }:
                raise ValueError("Shard receipt differs from its frozen assignment or source manifest")
            for name in ("run_id", "run_attempt"):
                if actual.get("runner", {}).get(name) != request[name]:
                    raise ValueError("Mixed GitHub attempts cannot form one report; re-run all jobs")
            base_images.add(check_runtime(actual, dependency, index))
            local = [json.loads(path.read_text()) for path in sorted((directory / "results").rglob("*.json"))]
            wanted = {key for key, task in expected_tasks.items() if task["id"] in manifest["shards"][index]}
            if len(local) != len(wanted) or {task_key(record["task"]) for record in local} != wanted:
                raise ValueError("Shard records are missing, duplicated, or outside their assignment")
            intervals = []
            bounds = {}
            for record in local:
                task = record["task"]
                key = task_key(task)
                if (
                    key in seen
                    or task != expected_tasks[key]
                    or record.get("shard") != index
                    or record.get("execution_sha256") != entry["execution_sha256"]
                ):
                    raise ValueError("Task differs from its exact plan or actual shard host")
                seen.add(key)
                if record not in records:
                    raise ValueError("Collected record differs from the immutable shard result")
                taskroot = directory / "tasks" / key
                selection = json.loads((taskroot / "selection.json").read_text())
                launches = selection.get("launches", [])
                if (
                    selection.get("task") != task
                    or selection.get("selected") != len(launches)
                    or not 1 <= len(launches) <= PROFILE["launch_attempts"]
                ):
                    raise ValueError("Launch selection is not a bounded immutable attempt chain")
                first, last = None, None
                for offset, reference in enumerate(launches, 1):
                    journal_path = taskroot / f"launch-{offset}/launch.json"
                    if reference != {
                        "path": journal_path.relative_to(directory).as_posix(),
                        "sha256": digest(journal_path),
                    }:
                        raise ValueError("Launch journal differs from its retained digest or path")
                    journal = json.loads(journal_path.read_text())
                    if journal.get("task") != task or journal.get("image_id") != actual["role_images"][task["role"]]:
                        raise ValueError("Launch used another task or wheel image")
                    start, end = journal["started_ns"], journal["finished_ns"]
                    if type(start) is not int or type(end) is not int or end < start:
                        raise ValueError("Launch timing interval is invalid")
                    intervals.extend(((start, 1), (end, -1)))
                    first = start if first is None else first
                    last = end
                    raw = journal_path.parent / "record.json"
                    if journal.get("has_record") != raw.is_file() or journal.get("record_sha256") != (
                        digest(raw) if raw.is_file() else None
                    ):
                        raise ValueError("Launch record differs from its preserved original")
                    if offset != len(launches) and not (
                        journal.get("docker_exit_code") == 125
                        and not journal.get("has_record")
                        and not journal.get("timed_out")
                    ):
                        raise ValueError("A completed fit, learning error, timeout, or OOM was retried")
                if record.get("launch") != launches[-1]["path"]:
                    raise ValueError("Collected task did not select its last declared launch")
                captured = taskroot / f"launch-{len(launches)}/captured/results/{key}.json"
                original = json.loads(captured.read_text())
                raw_path = taskroot / f"launch-{len(launches)}/record.json"
                if journal["has_record"]:
                    raw = json.loads(raw_path.read_text())
                    if {key: value for key, value in raw.items() if key not in {"log", "error"}} != {
                        key: value for key, value in original.items() if key != "error"
                    }:
                        raise ValueError("Selected measurement differs from its raw worker record")
                    if journal["docker_exit_code"] == 0:
                        captured_log = taskroot / f"launch-{len(launches)}/captured/logs/{key}.txt"
                        if original.get("error") != raw.get("error") or captured_log.read_text() != raw.get("log", ""):
                            raise ValueError("Selected measurement changed the original worker error or log")
                    elif not original.get("error"):
                        raise ValueError("A failed container cannot produce a successful measurement")
                elif original.get("result") is not None or not original.get("error"):
                    raise ValueError("A launch without a worker result cannot fabricate a passing measurement")
                if record != dict(
                    original, shard=index, execution_sha256=entry["execution_sha256"], launch=launches[-1]["path"]
                ):
                    raise ValueError("Collected task rewrote the selected original measurement")
                if (directory / "logs" / f"{key}.txt").read_bytes() != (
                    taskroot / f"launch-{len(launches)}/captured/logs/{key}.txt"
                ).read_bytes():
                    raise ValueError("Collected task rewrote the selected original log")
                bounds[task["id"], task["role"]] = first, last
            active = 0
            for _, change in sorted(intervals):
                active += change
                if not 0 <= active <= PROFILE["pair_workers"]:
                    raise ValueError("Shard exceeded its active-container bound")
            for identifier in manifest["shards"][index]:
                if bounds[identifier, "baseline"][1] > bounds[identifier, "candidate"][0]:
                    raise ValueError("Candidate started before its paired baseline finished")
        if len(base_images) != 1 or seen != set(expected_tasks) or len(records) != len(expected_tasks):
            raise ValueError("Shards differ in dependency image or do not form the exact full task union")
    except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
        errors.append(f"Invalid hosted execution: {error}")
    return errors


def collect_shards(options: argparse.Namespace) -> None:
    source = options.plan.resolve()
    manifest = load_plan(source)
    context(json.loads((source / "artifacts/request.json").read_text()))
    output = options.output.resolve()
    if (
        output == source
        or source in output.parents
        or output == options.shards.resolve()
        or options.shards.resolve() in output.parents
    ):
        raise ValueError("Store collected reports outside immutable plan and shard inputs")
    shutil.copytree(source, output)
    (output / "checksums.json").unlink()
    receipts = sorted(options.shards.rglob("shard.json"))
    if len(receipts) != PROFILE["shards"]:
        raise ValueError("The complete eight-shard matrix is required; re-run all jobs after infrastructure failures")
    entries = []
    seen = set()
    records = []
    for path in receipts:
        directory = path.parent
        validate_inventory(directory)
        receipt = json.loads(path.read_text())
        index = receipt["index"]
        if type(index) is not int or index in seen or not 0 <= index < PROFILE["shards"]:
            raise ValueError("Unexpected or duplicate shard artifact")
        seen.add(index)
        target = output / "shards" / f"shard-{index}"
        shutil.copytree(directory, target)
        actual = json.loads((directory / "execution.json").read_text())
        entries.append({"index": index, "execution_sha256": digest(directory / "execution.json"), "identity": actual})
        for result in sorted((directory / "results").rglob("*.json")):
            record = json.loads(result.read_text())
            key = task_key(record["task"])
            log = (directory / "logs" / f"{key}.txt").read_text()
            collect(output, dict(record, log=log))
            records.append(record)
    write_json(
        output / "execution.json",
        {
            "schema": 2,
            "identity": {"backend": BACKEND, "profile": PROFILE},
            "shards": sorted(entries, key=lambda item: item["index"]),
        },
    )
    errors = verify_execution(manifest, records, output)
    if errors:
        write_json(output / "collection-errors.json", errors)
    summary = finish(output, root=output / "source")
    if not summary["passed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("plan")
    for name in ("suite-root", "inputs", "output"):
        prepare.add_argument("--" + name, required=True, type=Path)
    for name in ("candidate-sha", "baseline-sha", "evidence-sha", "executor-sha", "run-id"):
        prepare.add_argument("--" + name, required=True)
    prepare.add_argument("--runtime-manifest", type=Path)
    build = commands.add_parser("image")
    for name in ("requirements", "output"):
        build.add_argument("--" + name, required=True, type=Path)
    build.add_argument("--build-cache", type=Path)
    worker = commands.add_parser("run")
    for name in ("plan", "output"):
        worker.add_argument("--" + name, required=True, type=Path)
    worker.add_argument("--shard", required=True, type=int)
    worker.add_argument("--runtime-image", type=Path)
    worker.add_argument("--runtime-sha256")
    worker.add_argument("--runtime-id")
    gather = commands.add_parser("collect")
    for name in ("plan", "shards", "output"):
        gather.add_argument("--" + name, required=True, type=Path)
    options = parser.parse_args()
    {"plan": create_plan, "image": image, "run": run, "collect": collect_shards}[options.command](options)


if __name__ == "__main__":
    main()
