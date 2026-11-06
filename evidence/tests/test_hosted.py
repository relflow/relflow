"""Hosted sharding must preserve the paired learning contract and fail closed."""

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from threading import Barrier, Lock

import hosted
import lab_run
import pytest
from evidence_ci import digest, plan, registry, task_key, write_json


def wheel(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("relflow/__init__.py", content)
    return path


def source(root, count=16):
    family = root / "experiments/family"
    family.mkdir(parents=True)
    for index in range(count):
        identifier = f"P{901 + index}"
        (family / f"proof_{index}.py").write_text(
            f"# %% [markdown]\n# ---\n# evidence-id: {identifier}\n# title: Signal {index}\n# ---\n"
            f"# %%\nPROOF_ID = {identifier!r}\nreport(seed={701 + index})\n"
        )
    (root / "experiments/reporting.py").write_text("# declared shared reporter fixture\n")
    return root


@pytest.fixture
def planned(tmp_path, monkeypatch, request):
    suite = source(tmp_path / "suite", count=getattr(request, "param", 16))
    inputs = tmp_path / "inputs"
    wheel(inputs / "baseline/relflow-0.4.5-py3-none-any.whl", "baseline runtime\n")
    wheel(inputs / "candidate/relflow-0.4.5-py3-none-any.whl", "candidate runtime\n")
    (inputs / "requirements.txt").write_text("torch==2.12.0\n")
    (inputs / "runtime.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "image_id": "sha256:" + "e" * 64,
                "archive_name": "runtime.tar",
                "archive_sha256": "f" * 64,
                "recipe_sha256": hashlib.sha256(
                    lab_run.runtime_recipe(inputs / "requirements.txt").encode()
                ).hexdigest(),
                "requirements_sha256": digest(inputs / "requirements.txt"),
                "torch_version": "2.12.0+cpu",
                "torch_cuda": None,
                "packages": {"torch": "2.12.0+cpu"},
            }
        )
    )
    request = {
        "candidate": "a" * 40,
        "baseline": "b" * 40,
        "suite": "c" * 40,
        "run_id": 123,
        "run_attempt": 1,
        "event": "push",
    }
    (inputs / "request.json").write_text(json.dumps(request))
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_REPOSITORY", "relflow/relflow")
    output = tmp_path / "plan"
    hosted.create_plan(
        argparse.Namespace(
            suite_root=suite,
            inputs=inputs,
            candidate_sha=request["candidate"],
            baseline_sha=request["baseline"],
            evidence_sha=request["suite"],
            executor_sha="d" * 40,
            run_id="gh-123-1",
            output=output,
        )
    )
    return output, json.loads((output / "manifest.json").read_text()), suite


def test_eight_shards_cover_every_original_pair_without_changing_the_protocol(planned):
    output, manifest, suite = planned
    before = deepcopy(manifest)
    shards = hosted.assignments(manifest, suite)
    assert len(shards) == 8
    assert all(shards)
    identifiers = [identifier for shard in shards for identifier in shard]
    assert Counter(identifiers) == Counter(registry(suite).keys())
    assert len(identifiers) == 16
    assert shards == hosted.assignments(manifest, suite)
    assert manifest == before
    original = plan(
        run_id=manifest["run_id"],
        candidate=manifest["candidate"],
        baseline=manifest["baseline"],
        evidence_sha=manifest["evidence_sha"],
        executor_sha=manifest["executor_sha"],
        root=suite,
    )
    assert manifest["tasks"] == original["tasks"]
    assert all(task["steps"] is None and task["attempt"] == 1 for task in manifest["tasks"])
    for shard in shards:
        tasks = [task for task in manifest["tasks"] if task["id"] in shard]
        for identifier in shard:
            pair = [task for task in tasks if task["id"] == identifier]
            assert {task["role"] for task in pair} == {"baseline", "candidate"}
            assert len({task["seed"] for task in pair}) == 1
    assert (output / "artifacts/requirements.txt").read_bytes() == b"torch==2.12.0\n"


def test_hosted_planning_reads_selected_suite_as_data_without_importing_its_helpers(tmp_path):
    suite = source(tmp_path / "suite")
    (suite / "experiments/source.py").write_text("raise RuntimeError('untrusted host import')\n")
    manifest = plan(
        run_id="data-only",
        candidate={},
        baseline={},
        evidence_sha="suite",
        executor_sha="executor",
        root=suite,
    )
    shards = hosted.assignments(manifest, suite)
    assert set(identifier for shard in shards for identifier in shard) == set(registry(suite))


def test_container_keeps_the_original_caps_and_excludes_runner_credentials(tmp_path, monkeypatch):
    scratch, view = tmp_path / "result", tmp_path / "input"
    scratch.mkdir()
    view.mkdir()
    task = {"id": "P901", "role": "candidate"}
    commands = []
    monkeypatch.setenv("GITHUB_TOKEN", "a-test-value-that-must-never-be-forwarded")

    def docker(command, **options):
        commands.append(command)
        return subprocess.CompletedProcess(command, 125, stdout="daemon temporarily unavailable")

    monkeypatch.setattr(lab_run.subprocess, "run", docker)
    record = lab_run.container(
        task,
        execution={"role_images": {"candidate": "sha256:exact-installed-image"}},
        scratch=scratch,
        view=view,
        run_id="gh-123-1-shard-0",
        evidence_sha="c" * 40,
        package={"revision": "a" * 40, "wheel_sha256": "f" * 64},
    )
    command = commands[0]
    for flag, expected in (
        ("--network", "none"),
        ("--cpus", "2"),
        ("--memory", "4g"),
        ("--pids-limit", "512"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
    ):
        assert command[command.index(flag) + 1] == expected
    assert "--read-only" in command
    assert "sha256:exact-installed-image" in command
    mounts = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--mount"]
    assert any(f"src={view},dst=/suite,readonly" in mount for mount in mounts)
    assert any(f"src={scratch},dst=/results" in mount for mount in mounts)
    assert all("docker.sock" not in mount and "dst=/home" not in mount for mount in mounts)
    assert all("GITHUB_TOKEN" not in value for value in command)
    assert "without a result" in record["error"]
    journal = json.loads((scratch / "launch.json").read_text())
    assert journal["docker_exit_code"] == 125 and journal["has_record"] is False
    assert commands[-1][:3] == ["docker", "rm", "--force"]


def fresh(manifest, task, *, passed=True):
    return {
        "task": task,
        "error": None,
        "log": f"original {task['role']} measurement\n",
        "result": {
            "mode": "full",
            "seed": task["seed"],
            "accelerator": "cpu",
            "outcome": "met" if passed else "not_met",
            "checks": {"held-out signal": passed, "existing baseline limitation": False},
            "metrics": {"score": 0.8 if passed else 0.2},
            "provenance": {
                "evidence_revision": manifest["evidence_sha"],
                "relflow_revision": manifest[task["role"]]["revision"],
                "wheel_sha256": manifest[task["role"]]["wheel_sha256"],
                "script_sha256": task["script_sha256"],
            },
        },
    }


def environment(index):
    image_id = "sha256:" + "e" * 64
    return {
        "identity": {
            "schema": 1,
            "image_id": image_id,
            "recipe_sha256": hashlib.sha256(lab_run.image_definition("2.12.0").encode()).hexdigest(),
            "host": {
                "system": "Linux",
                "release": "fixture-kernel",
                "version": "fixture-version",
                "machine": "x86_64",
                "node": f"host-{index}",
                "cpu_count": 4,
                "memory_bytes": 16 * 1024**3,
                "uid": 1001,
                "gid": 1001,
                "cpu": [f"model name: shard CPU {index}"],
            },
            "docker": "27.5.1",
            "resources": {
                "workers": 2,
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
            },
            "environment": {
                "threads": 1,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "POLARS_MAX_THREADS": "2",
                "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
        },
        "image_id": image_id,
        "image": "relflow-evidence:fixture",
        "workers": 2,
        "cpu_per_task": 2,
        "memory_per_task": "4g",
    }


@pytest.fixture
def shards(planned, tmp_path, monkeypatch):
    prepared, manifest, suite = planned
    state = {"index": 0, "candidate_passes": True, "launches": []}

    def docker(command, **options):
        assert command[:2] in (["docker", "run"], ["docker", "rm"]), command
        if command[1] == "run":
            mount = next(value for value in command if value.endswith(",dst=/results"))
            scratch = Path(mount.removeprefix("type=bind,src=").removesuffix(",dst=/results"))
            task = json.loads((scratch / "task.json").read_text())
            passed = task["role"] == "baseline" or state["candidate_passes"]
            write_json(scratch / "record.json", fresh(manifest, task, passed=passed))
            state["launches"].append(deepcopy(task))
        return subprocess.CompletedProcess(command, 0, stdout="original Docker log\n")

    monkeypatch.setattr(lab_run.subprocess, "run", docker)
    monkeypatch.setattr(hosted, "environment", lambda *args, **kwargs: environment(state["index"]))
    monkeypatch.setattr(hosted, "wheel_image", lambda execution, wheel: "sha256:" + digest(wheel))
    root = tmp_path / "shards"

    def execute(*, candidate_passes=True):
        state["candidate_passes"] = candidate_passes
        for index in range(8):
            state["index"] = index
            monkeypatch.setenv("RUNNER_NAME", f"runner-{index}")
            monkeypatch.setenv("GITHUB_JOB", "shard")
            monkeypatch.setenv("ImageOS", "ubuntu24")
            monkeypatch.setenv("ImageVersion", "fixture")
            hosted.run(
                argparse.Namespace(
                    plan=prepared,
                    shard=index,
                    output=root / f"downloaded-{index}",
                    runtime_image=None,
                    runtime_sha256=None,
                    runtime_id=None,
                )
            )
        return root

    return prepared, manifest, suite, root, execute, state


def collected(prepared, shards, output):
    try:
        hosted.collect_shards(argparse.Namespace(plan=prepared, shards=shards, output=output))
    except SystemExit as error:
        assert error.code == 1
    return json.loads((output / "summary.json").read_text())


def reseal(root):
    (root / "checksums.json").unlink()
    hosted.seal(root)


def rewrite(root, path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    reseal(root)


def change_execution(shard, change):
    execution_path = shard / "execution.json"
    execution = json.loads(execution_path.read_text())
    change(execution)
    execution_path.write_text(json.dumps(execution))
    fingerprint = digest(execution_path)
    receipt_path = shard / "shard.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["execution_sha256"] = fingerprint
    receipt_path.write_text(json.dumps(receipt))
    for path in (shard / "results").rglob("*.json"):
        record = json.loads(path.read_text())
        record["execution_sha256"] = fingerprint
        path.write_text(json.dumps(record))
    reseal(shard)


def test_full_collection_preserves_eight_actual_hosts_and_all_original_measurements(shards, tmp_path):
    prepared, manifest, suite, root, execute, state = shards
    execute()
    output = tmp_path / "collected"
    summary = collected(prepared, root, output)
    assert summary["passed"] and summary["complete"]
    assert summary["fresh_tasks"] == 32 and summary["reused_baseline_tasks"] == 0
    assert len(state["launches"]) == 32
    execution = json.loads((output / "execution.json").read_text())
    assert "host" not in execution["identity"]
    assert len({item["identity"]["identity"]["host"]["node"] for item in execution["shards"]}) == 8
    for shard in root.iterdir():
        index = json.loads((shard / "shard.json").read_text())["index"]
        for path in (shard / "results").rglob("*.json"):
            relative = path.relative_to(shard)
            assert (output / relative).read_bytes() == path.read_bytes()
            assert (output / "shards" / f"shard-{index}" / relative).read_bytes() == path.read_bytes()
    assert {task_key(task) for task in state["launches"]} == {task_key(task) for task in manifest["tasks"]}


def test_completed_failed_learning_checks_are_never_retried_and_only_collector_fails(shards, tmp_path):
    prepared, manifest, suite, root, execute, state = shards
    execute(candidate_passes=False)
    assert len(state["launches"]) == len(manifest["tasks"])
    assert all((root / f"downloaded-{index}/checksums.json").is_file() for index in range(8))
    summary = collected(prepared, root, tmp_path / "failed-learning")
    assert summary["complete"] and not summary["passed"]
    assert len(summary["regressions"]) == 16


@pytest.mark.parametrize("planned", [8], indirect=True)
def test_same_host_baseline_and_candidate_can_overlap_with_two_active_containers(shards, tmp_path, monkeypatch):
    prepared, manifest, suite, root, execute, state = shards
    original = lab_run.subprocess.run
    lock = Lock()
    active, overlaps = set(), []
    peak = 0

    def capture():
        with lock:
            overlaps.append(set(active))

    barrier = Barrier(2, action=capture, timeout=5)

    def docker(command, **options):
        nonlocal peak
        if command[:2] != ["docker", "run"]:
            return original(command, **options)
        mount = next(value for value in command if value.endswith(",dst=/results"))
        scratch = Path(mount.removeprefix("type=bind,src=").removesuffix(",dst=/results"))
        task = json.loads((scratch / "task.json").read_text())
        coordinate = task["id"], task["role"]
        with lock:
            active.add(coordinate)
            peak = max(peak, len(active))
        try:
            barrier.wait()
            return original(command, **options)
        finally:
            with lock:
                active.remove(coordinate)

    monkeypatch.setattr(lab_run.subprocess, "run", docker)
    execute()
    assert not active and peak == 2
    assert len(overlaps) == 8
    for overlapping in overlaps:
        assert len({identifier for identifier, role in overlapping}) == 1
        assert {role for identifier, role in overlapping} == {"baseline", "candidate"}
    summary = collected(prepared, root, tmp_path / "overlapping-roles")
    assert summary["passed"] and summary["complete"]
    assert summary["fresh_tasks"] == len(manifest["tasks"]) == 16


def test_resealed_launch_times_cannot_exceed_two_active_containers(shards, tmp_path):
    prepared, manifest, suite, root, execute, state = shards
    execute()
    shard = root / "downloaded-0"
    for journal_path in (shard / "tasks").rglob("launch.json"):
        journal = json.loads(journal_path.read_text())
        journal.update(started_ns=10, finished_ns=20)
        journal_path.write_text(json.dumps(journal))
        selection_path = journal_path.parent.parent / "selection.json"
        selection = json.loads(selection_path.read_text())
        selection["launches"][0]["sha256"] = digest(journal_path)
        selection_path.write_text(json.dumps(selection))
    reseal(shard)
    summary = collected(prepared, root, tmp_path / "oversubscribed-shard")
    assert not summary["passed"]
    assert any("active-container bound" in error for error in summary["errors"])


@pytest.mark.parametrize("fault", ["missing-shard", "duplicate-shard", "unexpected-shard", "changed-bytes"])
def test_collection_rejects_missing_duplicate_or_tampered_artifacts(shards, tmp_path, fault):
    prepared, manifest, suite, root, execute, state = shards
    execute()
    shard = root / "downloaded-0"
    if fault == "missing-shard":
        shutil.rmtree(shard)
    elif fault == "duplicate-shard":
        shutil.copytree(shard, root / "duplicate")
    elif fault == "unexpected-shard":
        receipt_path = shard / "shard.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["index"] = 8
        rewrite(shard, receipt_path, receipt)
    else:
        next((shard / "logs").rglob("*.txt")).write_text("tampered log\n")
    with pytest.raises((ValueError, FileExistsError), match="shard|checksum|artifact"):
        hosted.collect_shards(argparse.Namespace(plan=prepared, shards=root, output=tmp_path / "invalid"))


@pytest.mark.parametrize("fault", ["task-seed", "task-budget", "wrong-shard", "wrong-host", "record-provenance"])
def test_recomputed_artifact_hashes_cannot_authorize_altered_task_or_provenance(shards, tmp_path, fault):
    prepared, manifest, suite, root, execute, state = shards
    execute()
    shard = root / "downloaded-0"
    path = next((shard / "results/candidate").rglob("*.json"))
    record = json.loads(path.read_text())
    original_key = task_key(record["task"])
    if fault == "task-seed":
        record["task"]["seed"] += 1
    elif fault == "task-budget":
        record["task"]["steps"] = 2
    elif fault == "wrong-shard":
        record["shard"] = 1
    elif fault == "wrong-host":
        record["execution_sha256"] = "0" * 64
    else:
        record["result"]["provenance"]["wheel_sha256"] = "0" * 64
    if fault == "task-seed":
        changed_key = task_key(record["task"])
        changed_path = shard / "results" / f"{changed_key}.json"
        changed_path.parent.mkdir(parents=True, exist_ok=True)
        path.rename(changed_path)
        path = changed_path
        changed_log = shard / "logs" / f"{changed_key}.txt"
        changed_log.parent.mkdir(parents=True, exist_ok=True)
        (shard / "logs" / f"{original_key}.txt").rename(changed_log)
    rewrite(shard, path, record)
    summary = collected(prepared, root, tmp_path / "invalid-record")
    assert not summary["passed"] and summary["errors"]


@pytest.mark.parametrize(
    "fault",
    [
        "threads",
        "network",
        "cpu-cap",
        "task-timeout",
        "experiment-timeout",
        "input-layout",
        "base-image",
        "common-base-image",
        "wheel-image",
        "attempt",
    ],
)
def test_rehashed_execution_changes_cannot_weaken_policy_or_mix_hosts(shards, tmp_path, fault):
    prepared, manifest, suite, root, execute, state = shards
    execute()

    def change(execution):
        runtime = execution["identity"]
        if fault == "threads":
            runtime["environment"]["OMP_NUM_THREADS"] = "4"
        elif fault == "network":
            runtime["resources"]["network"] = "host"
        elif fault == "cpu-cap":
            runtime["resources"]["cpu_per_task"] = 4
        elif fault in {"task-timeout", "experiment-timeout"}:
            runtime["resources"][fault.replace("-", "_") + "_seconds"] = 2
        elif fault == "input-layout":
            runtime["resources"]["input_layout"] = "whole-repository"
        elif fault in {"base-image", "common-base-image"}:
            runtime["image_id"] = "sha256:" + "0" * 64
        elif fault == "wheel-image":
            execution["role_images"]["candidate"] = "sha256:" + "0" * 64
        else:
            execution["runner"]["run_attempt"] = 2

    for index in range(8 if fault == "common-base-image" else 1):
        change_execution(root / f"downloaded-{index}", change)
    summary = collected(prepared, root, tmp_path / "invalid-execution")
    assert not summary["passed"] and summary["errors"]


@pytest.mark.parametrize("fault", ["raw-worker-result", "captured-result", "captured-and-collected-log"])
def test_resealed_artifacts_cannot_rewrite_the_selected_worker_measurement(shards, tmp_path, fault):
    prepared, manifest, suite, root, execute, state = shards
    execute()
    shard = root / "downloaded-0"
    record_path = next((shard / "results/candidate").rglob("*.json"))
    record = json.loads(record_path.read_text())
    key = task_key(record["task"])
    taskroot = shard / "tasks" / key
    launch = taskroot / "launch-1"
    if fault == "raw-worker-result":
        raw_path = launch / "record.json"
        raw = json.loads(raw_path.read_text())
        raw["result"]["metrics"]["score"] = 0.999
        raw_path.write_text(json.dumps(raw))
        journal_path = launch / "launch.json"
        journal = json.loads(journal_path.read_text())
        journal["record_sha256"] = digest(raw_path)
        journal_path.write_text(json.dumps(journal))
        selection_path = taskroot / "selection.json"
        selection = json.loads(selection_path.read_text())
        selection["launches"][0]["sha256"] = digest(journal_path)
        selection_path.write_text(json.dumps(selection))
    elif fault == "captured-result":
        captured_path = launch / "captured/results" / f"{key}.json"
        captured = json.loads(captured_path.read_text())
        captured["result"]["metrics"]["score"] = 0.999
        captured_path.write_text(json.dumps(captured))
        record["result"]["metrics"]["score"] = 0.999
        record_path.write_text(json.dumps(record))
    else:
        (shard / "logs" / f"{key}.txt").write_text("rewritten selected log\n")
        (launch / "captured/logs" / f"{key}.txt").write_text("rewritten selected log\n")
    reseal(shard)
    summary = collected(prepared, root, tmp_path / "invalid-worker-binding")
    assert not summary["passed"] and summary["errors"]


def test_partial_github_rerun_rejects_old_plan_before_any_measurement(planned, tmp_path, monkeypatch):
    prepared, manifest, suite = planned
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setattr(hosted, "environment", lambda *args, **kwargs: pytest.fail("No measurement may start"))
    with pytest.raises(ValueError, match="attempt.*re-run all jobs"):
        hosted.run(argparse.Namespace(plan=prepared, shard=0, output=tmp_path / "never-started", runtime_image=None))
    with pytest.raises(ValueError, match="attempt.*re-run all jobs"):
        hosted.collect_shards(
            argparse.Namespace(plan=prepared, shards=tmp_path / "absent", output=tmp_path / "never-collected")
        )


@pytest.mark.parametrize("fault", ["threads", "memory-cap"])
def test_invalid_runtime_policy_is_rejected_before_any_fit(planned, tmp_path, monkeypatch, fault):
    prepared, manifest, suite = planned
    runtime = environment(0)
    if fault == "threads":
        runtime["identity"]["environment"]["OMP_NUM_THREADS"] = "4"
    else:
        runtime["identity"]["resources"]["memory_per_task"] = "16g"
    monkeypatch.setattr(hosted, "environment", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(hosted, "wheel_image", lambda *args: "sha256:" + "f" * 64)
    monkeypatch.setattr(hosted, "container", lambda *args, **kwargs: pytest.fail("Invalid runtime launched a fit"))
    with pytest.raises(ValueError, match="thread|policy"):
        hosted.run(
            argparse.Namespace(
                plan=prepared,
                shard=0,
                output=tmp_path / "preflight-rejected",
                runtime_image=None,
                runtime_sha256=None,
                runtime_id=None,
            )
        )


@pytest.mark.parametrize(
    "scenario,expected_launches",
    [
        ("125-then-success", 2),
        ("125-exhausted", 3),
        ("137-oom", 1),
        ("timeout", 1),
        ("measurement-error", 1),
        ("failed-check", 1),
        ("125-with-record", 1),
        ("malformed-record", 1),
        ("cancelled", 1),
        ("missing-runtime-command", 1),
    ],
)
def test_launch_retry_requires_125_without_any_worker_record(
    planned, tmp_path, monkeypatch, scenario, expected_launches
):
    prepared, manifest, suite = planned
    task = next(task for task in manifest["tasks"] if task["role"] == "candidate")
    calls = []

    def docker(command, **options):
        if command[1] == "rm":
            return subprocess.CompletedProcess(command, 0, stdout="")
        assert command[:2] == ["docker", "run"]
        calls.append(command)
        assert len(calls) <= expected_launches, "A terminal measurement or failure was retried"
        mount = next(value for value in command if value.endswith(",dst=/results"))
        scratch = Path(mount.removeprefix("type=bind,src=").removesuffix(",dst=/results"))
        if scenario == "timeout":
            raise subprocess.TimeoutExpired(command, timeout=1850, output="original deadline log")
        code = 0
        if scenario == "125-exhausted" or scenario == "125-then-success" and len(calls) == 1:
            code = 125
        elif scenario == "137-oom":
            code = 137
        elif scenario == "cancelled":
            code = 143
        elif scenario == "missing-runtime-command":
            code = 127
        elif scenario == "malformed-record":
            (scratch / "record.json").write_text("{invalid worker result")
        elif scenario == "125-with-record":
            write_json(scratch / "record.json", fresh(manifest, task))
            code = 125
        else:
            record = fresh(manifest, task, passed=scenario != "failed-check")
            if scenario == "measurement-error":
                record.update(result=None, error="original worker exception")
            write_json(scratch / "record.json", record)
        return subprocess.CompletedProcess(command, code, stdout=f"original launch {len(calls)}: status {code}\n")

    monkeypatch.setattr(lab_run.subprocess, "run", docker)
    execution = environment(0)
    execution["role_images"] = {"candidate": "sha256:" + "f" * 64}
    output = tmp_path / "launch-report"
    record = hosted.launch(
        task,
        output=output,
        suite=prepared / "source",
        manifest=manifest,
        execution=execution,
        execution_sha256="c" * 64,
        shard=0,
    )
    assert len(calls) == expected_launches
    taskroot = output / "tasks" / task_key(task)
    selection = json.loads((taskroot / "selection.json").read_text())
    assert selection["selected"] == expected_launches
    assert len(selection["launches"]) == expected_launches
    for attempt in range(1, expected_launches + 1):
        scratch = taskroot / f"launch-{attempt}"
        assert (scratch / "launch.json").is_file()
        assert (scratch / "captured/results" / f"{task_key(task)}.json").is_file()
        assert (scratch / "captured/logs" / f"{task_key(task)}.txt").is_file()
    if scenario == "125-then-success":
        assert record["error"] is None and record["result"]["checks"]["held-out signal"]
        first_log = taskroot / "launch-1/captured/logs" / f"{task_key(task)}.txt"
        assert "original launch 1: status 125" in first_log.read_text()
    elif scenario == "failed-check":
        assert record["error"] is None and record["result"]["checks"]["held-out signal"] is False
    else:
        assert record["error"]
