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

import evidence_ci
import hosted
import pytest
import runtime
from evidence_ci import digest, plan, registry, suite_identity, task_key, write_json


def wheel(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("relflow/__init__.py", content)
    return path


def source(root, count=24):
    family = root / "experiments/family"
    family.mkdir(parents=True)
    for index in range(count):
        identifier = f"P{901 + index}"
        (family / f"proof_{index}.py").write_text(
            f"# %% [markdown]\n# ---\n# title: Signal {index}\n# ---\n"
            "# %%\nfrom reporting import experiment, report\n"
            f"@experiment({identifier!r}, seed={701 + index})\n"
            "def run(seed, steps, accelerator):\n    return {}, {'signal': True}\n"
            "if __name__ == '__main__':\n    report(run)\n"
        )
    (root / "experiments/reporting.py").write_text("# declared shared reporter fixture\n")
    return root


def hints(root, tasks):
    value = {
        "schema": 1,
        "source": {
            "repository": "relflow/relflow",
            "run_id": 123,
            "run_attempt": 1,
            "head_sha": "a" * 40,
            "executor_sha": "b" * 40,
            "artifact": {"id": 456, "name": "relflow-evidence-attempt-1", "digest": "sha256:" + "c" * 64},
        },
        "tasks": tasks,
    }
    (root / "timings.json").write_text(json.dumps(value))
    return value


@pytest.fixture
def planned(tmp_path, monkeypatch, request):
    selected = getattr(request, "param", 24)
    suite = source(tmp_path / "suite", count=24 if selected == "migration" else selected)
    if selected == "migration":
        original = Path(__file__).resolve().parents[1]
        entries = registry(original, measurements=False)
        for identifier in ("P108", "P109", "P110", "P111", "P112"):
            path = entries[identifier]["script"]
            destination = suite / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original / path, destination)
    baseline_ids = set(registry(suite, measurements=False))
    if selected == "migration":
        baseline_ids -= {"P108", "P109", "P110", "P111", "P112"}
        baseline_ids.add("P018")
    monkeypatch.setattr(hosted, "baseline_catalog", lambda revision: baseline_ids)
    monkeypatch.setattr(evidence_ci, "baseline_catalog", lambda revision: baseline_ids)
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
                    runtime.runtime_recipe(inputs / "requirements.txt").encode()
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


def test_twelve_shards_cover_every_original_pair_without_changing_the_protocol(planned):
    output, manifest, suite = planned
    before = deepcopy(manifest)
    shards = hosted.assignments(manifest, suite)
    assert len(shards) == 12
    assert all(shards)
    identifiers = [identifier for shard in shards for identifier in shard]
    assert Counter(identifiers) == Counter(registry(suite).keys())
    assert len(identifiers) == 24
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


def test_unknown_proofs_have_positive_costs_and_spread_across_twelve_shards(tmp_path):
    suite = source(tmp_path / "suite")
    manifest = plan(
        run_id="unmeasured", candidate={}, baseline={}, evidence_sha="suite", executor_sha="executor", root=suite
    )
    frozen = hosted.scheduling(suite)
    assert frozen["timings_sha256"] is None
    assert all(value == 1.0 for roles in frozen["costs"].values() for value in roles.values())
    buckets = hosted.assignments(manifest, suite)
    assert len(buckets) == 12 and all(len(bucket) == 2 for bucket in buckets)


def test_unknown_proofs_use_the_largest_valid_published_hint(tmp_path, monkeypatch):
    suite = source(tmp_path / "suite")
    entries = registry(suite)
    for identifier, value in zip(entries, [10, 2, 0, -1, float("nan"), float("inf"), True, 10**1000], strict=False):
        entries[identifier]["run"] = {"duration_seconds": value}
    monkeypatch.setattr(hosted, "registry", lambda root: entries)
    costs = hosted.scheduling(suite)["costs"]
    assert costs["P901"] == {"baseline": 10.0, "candidate": 10.0}
    assert costs["P902"] == {"baseline": 2.0, "candidate": 2.0}
    assert all(costs[identifier] == {"baseline": 10.0, "candidate": 10.0} for identifier in list(entries)[2:])


def test_missing_role_hints_use_the_registered_median_without_creating_unknown_tasks(tmp_path):
    suite = source(tmp_path / "suite")
    hints(suite, {"P901": {"baseline": 2, "candidate": 6}, "P999": {"baseline": 86400, "candidate": 86400}})
    frozen = hosted.scheduling(suite)
    assert frozen["timings_sha256"] == digest(suite / "timings.json")
    assert frozen["costs"]["P902"] == {"baseline": 4.0, "candidate": 4.0}
    assert set(frozen["costs"]) == set(registry(suite))


def test_measured_expensive_proof_is_balanced_and_scheduled_before_cheap_work(tmp_path):
    suite = source(tmp_path / "suite")
    script = suite / "experiments/family/proof_0.py"
    script.write_text(script.read_text().replace("P901", "P082"))
    manifest = plan(
        run_id="scheduled", candidate={}, baseline={}, evidence_sha="suite", executor_sha="executor", root=suite
    )
    original = deepcopy(manifest)
    costs = {identifier: {"baseline": 4.0, "candidate": 4.0} for identifier in registry(suite)}
    costs["P082"] = {"baseline": 350.0, "candidate": 330.0}
    hints(suite, costs)
    frozen = hosted.scheduling(suite)
    assert registry(suite)["P082"]["run"] is None
    ordered = hosted.schedule(manifest["tasks"], frozen["costs"])
    assert [(task["id"], task["role"]) for task in ordered[:2]] == [("P082", "baseline"), ("P082", "candidate")]
    buckets = hosted.assignments(manifest, suite)
    assert next(bucket for bucket in buckets if "P082" in bucket) == ["P082"]
    after = plan(
        run_id="scheduled", candidate={}, baseline={}, evidence_sha="suite", executor_sha="executor", root=suite
    )
    assert after["tasks"] == original["tasks"]
    assert after["runtime_sha256"] == original["runtime_sha256"]


def test_pair_assignment_uses_both_role_costs_while_task_order_uses_each_role(tmp_path):
    suite = source(tmp_path / "suite")
    costs = {identifier: {"baseline": 1, "candidate": 1} for identifier in registry(suite)}
    costs.update(P901={"baseline": 100, "candidate": 1}, P902={"baseline": 60, "candidate": 60})
    hints(suite, costs)
    manifest = plan(
        run_id="asymmetric", candidate={}, baseline={}, evidence_sha="suite", executor_sha="executor", root=suite
    )
    buckets = hosted.assignments(manifest, suite)
    assert buckets[0] == ["P902"]
    assert buckets[1] == ["P901"]
    selected = [task for task in manifest["tasks"] if task["id"] in {"P901", "P902"}]
    ordered = hosted.schedule(selected, hosted.scheduling(suite)["costs"])
    assert [(task["id"], task["role"]) for task in ordered] == [
        ("P901", "baseline"),
        ("P902", "baseline"),
        ("P902", "candidate"),
        ("P901", "candidate"),
    ]


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), 1e308, 10**1000])
def test_malformed_timing_costs_fail_before_assignment(tmp_path, value):
    suite = source(tmp_path / "suite")
    hints(suite, {"P901": {"baseline": value}})
    with pytest.raises(ValueError, match="Scheduling hints.*finite positive"):
        hosted.scheduling(suite)


@pytest.mark.parametrize("value", ["{", json.dumps({"schema": True, "source": {}, "tasks": {}})])
def test_optional_timing_file_may_be_missing_but_not_malformed(tmp_path, value):
    suite = source(tmp_path / "suite")
    (suite / "timings.json").write_text(value)
    with pytest.raises(ValueError, match="Scheduling hints"):
        hosted.scheduling(suite)


@pytest.mark.parametrize("changed", ["resolved-costs", "source-hints"])
def test_resealed_plan_rejects_changed_frozen_scheduling_inputs(planned, changed):
    prepared, manifest, suite = planned
    if changed == "resolved-costs":
        manifest["scheduling"]["costs"]["P901"]["baseline"] += 1
    else:
        hints(prepared / "source", {"P901": {"baseline": 300, "candidate": 250}})
        manifest["suite_sha256"] = suite_identity(prepared / "source")
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    reseal(prepared)
    with pytest.raises(ValueError, match="scheduling hints.*frozen"):
        hosted.load_plan(prepared)


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

    monkeypatch.setattr(runtime.subprocess, "run", docker)
    record = runtime.container(
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
    assert command[command.index("sha256:exact-installed-image") + 1 :] == [
        "python",
        "/executor/worker.py",
        "/results/task.json",
    ]
    mounts = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--mount"]
    assert any(f"src={Path(runtime.__file__).resolve().parent},dst=/executor,readonly" in mount for mount in mounts)
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
        **(
            {
                "observations": {
                    "schema": 1,
                    "sha256": "a" * 64,
                    "manifest_sha256": "b" * 64,
                    "bytes": 1,
                    "capture_sha256": manifest["observations"]["capture_sha256"],
                }
            }
            if manifest.get("observations")
            else {}
        ),
        "result": {
            "mode": "full",
            "seed": task["seed"],
            "accelerator": "cpu",
            "outcome": "met" if passed else "not_met",
            "checks": {
                "held-out signal": passed,
                **(
                    {}
                    if manifest.get("policy") == "introduced-passing-checks-v1"
                    else {"existing baseline limitation": False}
                ),
            },
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
            "recipe_sha256": hashlib.sha256(runtime.image_definition("2.12.0").encode()).hexdigest(),
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
                "observation_capture": "py-return-v1",
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

    monkeypatch.setattr(runtime.subprocess, "run", docker)
    monkeypatch.setattr(hosted, "environment", lambda *args, **kwargs: environment(state["index"]))
    monkeypatch.setattr(hosted, "wheel_image", lambda execution, wheel: "sha256:" + digest(wheel))
    root = tmp_path / "shards"

    def execute(*, candidate_passes=True):
        state["candidate_passes"] = candidate_passes
        for index in range(12):
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


def test_full_collection_preserves_twelve_actual_hosts_and_all_original_measurements(shards, tmp_path):
    prepared, manifest, suite, root, execute, state = shards
    execute()
    output = tmp_path / "collected"
    summary = collected(prepared, root, output)
    assert summary["passed"] and summary["complete"]
    assert summary["fresh_tasks"] == 48 and summary["reused_baseline_tasks"] == 0
    assert len(state["launches"]) == 48
    execution = json.loads((output / "execution.json").read_text())
    assert "host" not in execution["identity"]
    assert len({item["identity"]["identity"]["host"]["node"] for item in execution["shards"]}) == 12
    for receipt in root.glob("*/shard.json"):
        shard = receipt.parent
        index = json.loads((shard / "shard.json").read_text())["index"]
        for path in (shard / "results").rglob("*.json"):
            relative = path.relative_to(shard)
            assert (output / relative).read_bytes() == path.read_bytes()
            assert (output / "shards" / f"shard-{index}" / relative).read_bytes() == path.read_bytes()
    assert {task_key(task) for task in state["launches"]} == {task_key(task) for task in manifest["tasks"]}


@pytest.mark.parametrize("planned", ["migration"], indirect=True)
def test_introduced_proofs_collect_all_twelve_shards_with_distinct_roles(shards, tmp_path):
    prepared, manifest, _, root, execute, state = shards
    assert manifest["policy"] == "introduced-passing-checks-v1"
    execute()
    summary = collected(prepared, root, tmp_path / "migration-collected")
    assert summary["passed"] and summary["complete"]
    assert summary["fresh_tasks"] == 53 and len(state["launches"]) == 53
    assert manifest["retired"] == ["P018"]
    assert summary["introduced_unmet_checks"] == []


@pytest.mark.parametrize("planned", ["migration"], indirect=True)
def test_introduced_collector_fails_on_each_unmet_new_group_proof(shards, tmp_path):
    prepared, manifest, _, root, execute, state = shards
    execute(candidate_passes=False)
    summary = collected(prepared, root, tmp_path / "migration-failed")
    assert summary["complete"] and not summary["passed"]
    assert len(state["launches"]) == len(manifest["tasks"])
    assert {item["id"] for item in summary["introduced_unmet_checks"]} == {
        "P108",
        "P109",
        "P110",
        "P111",
        "P112",
    }


def test_completed_failed_learning_checks_are_never_retried_and_only_collector_fails(shards, tmp_path):
    prepared, manifest, suite, root, execute, state = shards
    execute(candidate_passes=False)
    assert len(state["launches"]) == len(manifest["tasks"])
    assert all((root / f"downloaded-{index}/checksums.json").is_file() for index in range(12))
    summary = collected(prepared, root, tmp_path / "failed-learning")
    assert summary["complete"] and not summary["passed"]
    assert len(summary["regressions"]) == 24


@pytest.mark.parametrize("planned", [12], indirect=True)
def test_same_host_baseline_and_candidate_can_overlap_with_two_active_containers(shards, tmp_path, monkeypatch):
    prepared, manifest, suite, root, execute, state = shards
    original = runtime.subprocess.run
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

    monkeypatch.setattr(runtime.subprocess, "run", docker)
    execute()
    assert not active and peak == 2
    assert len(overlaps) == 12
    for overlapping in overlaps:
        assert len({identifier for identifier, role in overlapping}) == 1
        assert {role for identifier, role in overlapping} == {"baseline", "candidate"}
    summary = collected(prepared, root, tmp_path / "overlapping-roles")
    assert summary["passed"] and summary["complete"]
    assert summary["fresh_tasks"] == len(manifest["tasks"]) == 24


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
        receipt["index"] = 12
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

    for index in range(12 if fault == "common-base-image" else 1):
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

    monkeypatch.setattr(runtime.subprocess, "run", docker)
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
