"""Cached baseline slots preserve originals and fail closed on any changed input."""

import json
import shutil
import zipfile
from copy import deepcopy

import pytest
from evidence_ci import (
    collect,
    digest,
    evaluate,
    finish,
    integrity,
    plan,
    suite_identity,
    task_key,
    wheel_payload,
    write_json,
)
from lab_run import cached, retain
from record import validate


def wheel(path, *, payload=b"sdk", year=2020):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        item = zipfile.ZipInfo("relflow/__init__.py", date_time=(year, 1, 1, 0, 0, 0))
        archive.writestr(item, payload)
    return path


def source(root):
    script = root / "experiments/family/signal.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("# %% [markdown]\n# ---\n# evidence-id: P901\n# title: Signal\n# ---\n# %%\nreport(seed=123)\n")
    (root / "experiments/reporting.py").write_text("# trusted reporting fixture\n")
    return script


def setup_report(
    output, suite, *, revision="old", seed_records=True, candidate_payload=b"sdk", baseline_payload=b"sdk-old"
):
    output.mkdir(parents=True)
    shutil.copytree(suite, output / "source")
    wheels = {
        role: wheel(
            output / "artifacts" / role / "relflow.whl",
            year=2020 if role == "baseline" else 2021,
            payload=baseline_payload if role == "baseline" else candidate_payload,
        )
        for role in ("baseline", "candidate")
    }
    lock = output / "artifacts/requirements.txt"
    lock.write_text("torch==2.5.0\n")
    manifest = plan(
        run_id=output.name,
        candidate={"revision": revision},
        baseline={"revision": "previous"},
        evidence_sha=revision,
        executor_sha="trusted-executor",
        root=suite,
    )
    for role, path in wheels.items():
        manifest[role].update(wheel_sha256=digest(path), payload_sha256=wheel_payload(path))
    manifest.update(requirements_sha256=digest(lock), suite_sha256=suite_identity(suite))
    execution = {
        "identity": {
            "image_id": "sha256:actual-image",
            "host": {"cpu": "actual-cpu"},
            "resources": {"workers": 8},
            "environment": {"threads": 1},
        }
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "execution.json", execution)
    if seed_records:
        for task in manifest["tasks"]:
            collect(output, fresh(manifest, task))
    return manifest, execution


def fresh(manifest, task):
    return {
        "task": task,
        "error": None,
        "log": "original execution log\n",
        "duration_seconds": 10,
        "result": {
            "mode": "full",
            "seed": task["seed"],
            "accelerator": task["accelerator"],
            "outcome": "met",
            "checks": {"signal": True, "control": False},
            "provenance": {
                "evidence_revision": manifest["evidence_sha"],
                "relflow_revision": manifest[task["role"]]["revision"],
                "wheel_sha256": manifest[task["role"]]["wheel_sha256"],
                "script_sha256": task["script_sha256"],
            },
        },
    }


@pytest.fixture
def reuse_pair(tmp_path):
    suite = tmp_path / "suite"
    source(suite)
    original = tmp_path / "original"
    manifest, execution = setup_report(original, suite)
    assert finish(original, root=suite)["passed"]
    cache = tmp_path / "cache"
    retain(cache, original, manifest, execution)
    current = tmp_path / "current"
    now, _ = setup_report(
        current, suite, revision="new", seed_records=False, baseline_payload=b"sdk", candidate_payload=b"new-sdk"
    )
    task = now["tasks"][0]
    wrapper = cached(cache, current, now, task, execution, suite)
    assert wrapper is not None
    collect(current, wrapper)
    collect(current, fresh(now, now["tasks"][1]))
    return suite, original, current, cache, now, execution, wrapper


def test_reuse_preserves_original_bytes_and_provenance_and_all_slots(reuse_pair):
    suite, original, current, cache, manifest, execution, wrapper = reuse_pair
    original_key = wrapper["reuse"]["task"]
    assert original_key.startswith("candidate/")
    retained = current / wrapper["reuse"]["report"]
    assert (retained / "results" / (original_key + ".json")).read_bytes() == (
        original / "results" / (original_key + ".json")
    ).read_bytes()
    raw = json.loads((current / "results" / (task_key(wrapper["task"]) + ".json")).read_text())
    assert set(raw) == {"task", "reuse"} and "provenance" not in raw
    summary = finish(current, root=suite)
    assert summary["passed"] and summary["expected_tasks"] == summary["received_tasks"] == 2
    assert summary["reused_baseline_tasks"] == summary["fresh_tasks"] == 1
    selected, records, _ = validate(current, current / "artifacts", root=suite)
    assert selected["candidate"]["revision"] == "new"
    assert (
        next(value for value in records if value["task"]["role"] == "candidate")["result"]["provenance"][
            "relflow_revision"
        ]
        == "new"
    )
    integrity(current)


@pytest.mark.parametrize(
    "field", ["payload", "requirements", "suite", "executor", "cpu", "image", "threads", "seed", "budget", "policy"]
)
def test_any_changed_execution_input_is_a_miss(reuse_pair, field):
    suite, _, current, cache, manifest, execution, _ = reuse_pair
    manifest, execution = deepcopy(manifest), deepcopy(execution)
    task = manifest["tasks"][0]
    if field == "payload":
        manifest["baseline"]["payload_sha256"] = "changed"
    elif field in {"requirements", "suite"}:
        manifest[field + "_sha256"] = "changed"
    elif field == "executor":
        manifest["executor_sha"] = "new-policy-source"
    elif field == "cpu":
        execution["identity"]["host"]["cpu"] = "changed"
    elif field == "image":
        execution["identity"]["image_id"] = "changed"
    elif field == "threads":
        execution["identity"]["environment"]["threads"] = 2
    elif field == "seed":
        task["seed"] += 1
    elif field == "budget":
        task["steps"] = 1
    else:
        manifest["policy"] = "changed"
    assert cached(cache, current, manifest, task, execution, suite) is None


def test_payload_identity_ignores_transport_timestamps_only(tmp_path):
    first = wheel(tmp_path / "first.whl", year=2020)
    second = wheel(tmp_path / "second.whl", year=2024)
    assert digest(first) != digest(second)
    assert wheel_payload(first) == wheel_payload(second)
    wheel(second, payload=b"changed")
    assert wheel_payload(first) != wheel_payload(second)


def test_suite_identity_ignores_only_selected_metadata(tmp_path):
    from experiments.source import frontmatter, metadata

    script = source(tmp_path)
    before = suite_identity(tmp_path)
    text = script.read_text()
    script.write_text(
        frontmatter(
            text, dict(metadata(text), **{"evidence-result": {"metrics": {"x": 5}}, "evidence-report": {"run": "new"}})
        )
    )
    assert suite_identity(tmp_path) == before
    script.write_text(script.read_text().replace("title: Signal", "title: Different contract"))
    assert suite_identity(tmp_path) != before
    script.write_text(text)
    (tmp_path / "experiments/reporting.py").write_text("# different runtime helper\n")
    assert suite_identity(tmp_path) != before


@pytest.mark.parametrize("fault", ["log", "record", "wheel", "source", "candidate-ref", "chain", "failed", "missing"])
def test_invalid_source_or_reference_never_satisfies_gate(reuse_pair, fault):
    suite, _, current, _, manifest, _, wrapper = reuse_pair
    retained = current / wrapper["reuse"]["report"]
    records = [json.loads(path.read_text()) for path in (current / "results").rglob("*.json")]
    if fault == "candidate-ref":
        candidate = next(value for value in records if value["task"]["role"] == "candidate")
        candidate.clear()
        candidate.update(task=manifest["tasks"][1], reuse=wrapper["reuse"])
    elif fault == "chain":
        wrapper["reuse"]["task"] = "baseline/P901/cpu/seed-123/attempt-1"
        source_record = retained / "results" / (wrapper["reuse"]["task"] + ".json")
        source_record.write_text(json.dumps({"task": manifest["tasks"][0], "reuse": wrapper["reuse"]}))
    elif fault in {"failed", "missing"}:
        paths = sorted((retained / "results").rglob("*.json"))
        if fault == "missing":
            paths[0].unlink()
        else:
            value = json.loads(paths[0].read_text())
            value["error"] = "original execution failed"
            paths[0].write_text(json.dumps(value))
        checks = {
            path.relative_to(retained).as_posix(): digest(path)
            for path in retained.rglob("*")
            if path.is_file() and path.name != "checksums.json"
        }
        (retained / "checksums.json").write_text(json.dumps(checks))
    else:
        path = {
            "log": next((retained / "logs").rglob("*.txt")),
            "record": next((retained / "results").rglob("*.json")),
            "wheel": next((retained / "artifacts").rglob("*.whl")),
            "source": retained / "source/experiments/family/signal.py",
        }[fault]
        path.write_bytes(b"tampered")
    assert not evaluate(manifest, records, root=suite, reports=current)["passed"]


def test_cached_passing_checks_still_block_candidate_regression(reuse_pair):
    suite, _, current, _, manifest, _, _ = reuse_pair
    records = [json.loads(path.read_text()) for path in (current / "results").rglob("*.json")]
    next(value for value in records if value["task"]["role"] == "candidate")["result"]["checks"]["signal"] = False
    summary = evaluate(manifest, records, root=suite, reports=current)
    assert not summary["passed"]
    assert summary["regressions"] == [{"id": "P901", "seed": 123, "check": "signal"}]


def test_failed_or_incomplete_run_cannot_seed_cache(tmp_path):
    suite = tmp_path / "suite"
    source(suite)
    for fault in ("failed", "missing"):
        reports = tmp_path / fault
        manifest, execution = setup_report(reports, suite)
        path = next((reports / "results").rglob("*.json"))
        if fault == "missing":
            path.unlink()
        else:
            record = json.loads(path.read_text())
            record["error"] = "training failed"
            path.write_text(json.dumps(record))
        assert not finish(reports, root=suite)["passed"]
        cache = tmp_path / (fault + "-cache")
        retain(cache, reports, manifest, execution)
        assert not cache.exists()


def test_fresh_candidate_from_reused_run_seeds_next_baseline_with_bounded_proof(reuse_pair, tmp_path):
    suite, _, current, cache, manifest, execution, _ = reuse_pair
    assert finish(current, root=suite)["passed"]
    retain(cache, current, manifest, execution)
    next_run = tmp_path / "next"
    next_manifest, _ = setup_report(
        next_run,
        suite,
        revision="next",
        seed_records=False,
        baseline_payload=b"new-sdk",
        candidate_payload=b"third-sdk",
    )
    wrapper = cached(cache, next_run, next_manifest, next_manifest["tasks"][0], execution, suite)
    assert wrapper is not None and wrapper["reuse"]["task"].startswith("candidate/")
    collect(next_run, wrapper)
    collect(next_run, fresh(next_manifest, next_manifest["tasks"][1]))
    assert finish(next_run, root=suite)["passed"]
    retain(cache, next_run, next_manifest, execution)
    deeper = tmp_path / "deeper"
    deeper_manifest, _ = setup_report(
        deeper,
        suite,
        revision="deeper",
        seed_records=False,
        baseline_payload=b"third-sdk",
        candidate_payload=b"fourth-sdk",
    )
    assert cached(cache, deeper, deeper_manifest, deeper_manifest["tasks"][0], execution, suite) is None


def test_missing_slot_and_untrusted_provenance_bypass_cannot_pass(reuse_pair):
    suite, _, current, _, manifest, _, _ = reuse_pair
    records = [json.loads(path.read_text()) for path in (current / "results").rglob("*.json")]
    assert not evaluate(manifest, records[:1], root=suite, reports=current)["passed"]
    candidate = next(value for value in records if value["task"]["role"] == "candidate")
    candidate["result"]["provenance"]["wheel_sha256"] = "forged"
    candidate["reused_from"] = {"report": "pretend validated"}
    assert not evaluate(manifest, records, root=suite, reports=current)["passed"]


def test_runner_only_schedules_missing_baselines_and_keeps_task_isolation(tmp_path, monkeypatch):
    import argparse
    from types import SimpleNamespace

    import lab_run

    suite = tmp_path / "suite"
    source(suite)
    candidate = wheel(tmp_path / "candidate/relflow.whl", payload=b"sdk")
    baseline = wheel(tmp_path / "baseline/relflow.whl", payload=b"prior-sdk")
    lock = tmp_path / "requirements.txt"
    lock.write_text("torch==2.5.0\n")
    runtime = {"identity": {"image_id": "sha256:actual-image"}, "image_id": "sha256:actual-image"}
    monkeypatch.setattr(lab_run, "environment", lambda *args: runtime)
    monkeypatch.setattr(lab_run.Path, "home", lambda: tmp_path / "lab-home")
    monkeypatch.setattr(lab_run.subprocess, "check_output", lambda *args, **kwargs: "")
    calls = []
    exit_code = [0]

    def docker(command, **kwargs):
        if command[:2] == ["docker", "run"]:
            calls.append(command)
            mount = next(value for value in command if value.endswith(",dst=/results"))
            scratch = lab_run.Path(mount.removeprefix("type=bind,src=").removesuffix(",dst=/results"))
            task = json.loads((scratch / "task.json").read_text())
            reports = next(parent for parent in scratch.parents if (parent / "manifest.json").exists())
            manifest = json.loads((reports / "manifest.json").read_text())
            write_json(scratch / "record.json", fresh(manifest, task))
        return SimpleNamespace(
            returncode=exit_code[0] if command[:2] == ["docker", "run"] else 0, stdout="container log"
        )

    monkeypatch.setattr(lab_run.subprocess, "run", docker)

    def run(name, *, force=False):
        lab_run.run(
            argparse.Namespace(
                suite_root=suite,
                output=tmp_path / name,
                candidate_wheel=candidate,
                baseline_wheel=baseline,
                candidate_sha=name,
                baseline_sha="previous",
                evidence_sha=name,
                executor_sha="trusted",
                requirements=lock,
                ids=None,
                steps=None,
                workers=8,
                force_fresh=force,
            )
        )
        return json.loads((tmp_path / name / "summary.json").read_text())

    assert run("first")["fresh_tasks"] == 2
    wheel(baseline, payload=b"sdk", year=2025)
    assert run("second")["reused_baseline_tasks"] == 1
    assert len(calls) == 3
    assert run("forced", force=True)["fresh_tasks"] == 2
    assert len(calls) == 5
    for command in calls:
        assert command[command.index("--network") + 1] == "none"
        assert "--read-only" in command and "--cap-drop" in command
        assert command[command.index("--cpus") + 1] == "2"
        assert command[command.index("--memory") + 1] == "4g"
        assert not any("cache/v1" in value or "docker.sock" in value for value in command)
        assert command[command.index("--mount") + 1].endswith(",dst=/executor,readonly")
    source_count = len(list((tmp_path / "lab-home/relflow-evidence/cache/v1/reports").iterdir()))
    exit_code[0] = 7
    with pytest.raises(SystemExit):
        run("nonzero", force=True)
    failed = json.loads((tmp_path / "nonzero/summary.json").read_text())
    assert not failed["passed"] and any("Container exited with status 7" in error for error in failed["errors"])
    assert len(list((tmp_path / "lab-home/relflow-evidence/cache/v1/reports").iterdir())) == source_count
    exit_code[0] = 0

    def unwritable(*args):
        raise OSError("cache storage unavailable")

    monkeypatch.setattr(lab_run, "retain", unwritable)
    assert run("uncached", force=True)["passed"]


def test_probe_and_runner_share_actual_image_and_environment_policy(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import lab_run

    lock = tmp_path / "requirements.txt"
    lock.write_text("torch==2.5.0\n")
    builds = []

    def docker(command, **kwargs):
        assert command[:2] == ["docker", "build"]
        context = lab_run.Path(command[-1])
        builds.append((context / "Dockerfile").read_text())
        assert (context / "requirements.txt").read_bytes() == lock.read_bytes()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(lab_run.subprocess, "run", docker)
    monkeypatch.setattr(
        lab_run.subprocess,
        "check_output",
        lambda command, **kwargs: "sha256:actual-image\n" if command[1] == "image" else "actual-docker-version\n",
    )
    measured = lab_run.environment(lock, 8)
    assert measured["identity"]["image_id"] == "sha256:actual-image"
    assert measured["identity"]["docker"] == "actual-docker-version"
    assert measured["identity"]["host"]["cpu_count"]
    assert measured["identity"]["resources"]["workers"] == 8
    assert measured["identity"]["resources"]["network"] == "none"
    assert builds and "python:3.12-slim@sha256:b18b" in builds[0]
    assert "pip install --no-cache-dir torch==2.5.0+cpu" in builds[0]


def test_retention_bounds_storage_and_removes_only_orphan_indices(tmp_path):
    import os

    from lab_run import prune

    cache = tmp_path / "cache"
    for index in range(3):
        name = str(index) * 64
        report = cache / "reports" / name
        report.mkdir(parents=True)
        (report / "immutable.json").write_text("execution")
        os.utime(report, (index + 1, index + 1))
        write_json(cache / "keys" / (name + ".json"), {"report": name, "task": "candidate/P901"})
    prune(cache, max_reports=2, max_bytes=100)
    assert {path.name for path in (cache / "reports").iterdir()} == {"1" * 64, "2" * 64}
    assert {path.stem for path in (cache / "keys").iterdir()} == {"1" * 64, "2" * 64}
    prune(cache, max_reports=2, max_bytes=9)
    assert {path.name for path in (cache / "reports").iterdir()} == {"2" * 64}


def test_suite_identity_includes_binary_fixtures_and_modes(tmp_path):
    source(tmp_path)
    fixture = tmp_path / "experiments/data.bin"
    fixture.write_bytes(b"\x00input-a")
    before = suite_identity(tmp_path)
    fixture.write_bytes(b"\x00input-b")
    assert suite_identity(tmp_path) != before
    before = suite_identity(tmp_path)
    fixture.chmod(0o755)
    assert suite_identity(tmp_path) != before
    before = suite_identity(tmp_path)
    (tmp_path / "README.md").write_text("supplementary suite input")
    assert suite_identity(tmp_path) != before
