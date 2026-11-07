"""Historical baseline references preserve originals and reject changed execution inputs."""

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
    resolved,
    suite_identity,
    task_key,
    wheel_payload,
    write_json,
)
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
    script.write_text(
        "# %% [markdown]\n# ---\n# title: Signal\n# ---\n# %%\n"
        "from reporting import experiment, report\n"
        "@experiment('P901', seed=123)\n"
        "def run(seed, steps, accelerator):\n    return {}, {'signal': True}\n"
        "if __name__ == '__main__':\n    report(run)\n"
    )
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


def seal(reports):
    inventory = {
        path.relative_to(reports).as_posix(): digest(path)
        for path in sorted(reports.rglob("*"))
        if path.is_file() and path != reports / "checksums.json"
    }
    (reports / "checksums.json").write_text(json.dumps(inventory, sort_keys=True))
    return digest(reports / "checksums.json")


def archive(original, current, task):
    """Construct an old archive layout without reviving a live cache writer."""
    name = "reused/" + digest(original / "checksums.json")
    shutil.copytree(original, current / name)
    return {"task": task, "reuse": {"report": name, "task": task_key(dict(task, role="candidate"))}}


@pytest.fixture
def reuse_pair(tmp_path):
    suite = tmp_path / "suite"
    source(suite)
    original = tmp_path / "original"
    _, execution = setup_report(original, suite)
    assert finish(original, root=suite)["passed"]
    current = tmp_path / "current"
    now, _ = setup_report(
        current, suite, revision="new", seed_records=False, baseline_payload=b"sdk", candidate_payload=b"new-sdk"
    )
    task = now["tasks"][0]
    wrapper = archive(original, current, task)
    collect(current, wrapper)
    collect(current, fresh(now, now["tasks"][1]))
    return suite, original, current, now, execution, wrapper


def test_reuse_preserves_original_bytes_and_provenance_and_all_slots(reuse_pair):
    suite, original, current, manifest, _, wrapper = reuse_pair
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
def test_archived_reference_rejects_any_changed_execution_input(reuse_pair, field):
    suite, _, current, manifest, execution, wrapper = reuse_pair
    manifest, execution, wrapper = deepcopy((manifest, execution, wrapper))
    task = manifest["tasks"][0]
    if field == "payload":
        manifest["baseline"]["payload_sha256"] = "changed"
    elif field == "requirements":
        manifest["requirements_sha256"] = "changed"
    elif field == "suite":
        task["inputs_sha256"] = "changed"
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
    wrapper["task"] = task
    (current / "execution.json").write_text(json.dumps(execution))
    with pytest.raises(ValueError, match="execution identity differs"):
        resolved(manifest, [wrapper], current, root=suite)


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
    suite, _, current, manifest, _, wrapper = reuse_pair
    retained = current / wrapper["reuse"]["report"]
    records = [json.loads(path.read_text()) for path in (current / "results").rglob("*.json")]
    if fault == "candidate-ref":
        candidate = next(value for value in records if value["task"]["role"] == "candidate")
        candidate.clear()
        candidate.update(task=manifest["tasks"][1], reuse=wrapper["reuse"])
    elif fault == "chain":
        reference = next(value for value in records if "reuse" in value)
        reference["reuse"]["task"] = "baseline/P901/cpu/seed-123/attempt-1"
        source_record = retained / "results" / (reference["reuse"]["task"] + ".json")
        source_record.write_text(json.dumps({"task": manifest["tasks"][0], "reuse": reference["reuse"]}))
    elif fault in {"failed", "missing"}:
        paths = sorted((retained / "results").rglob("*.json"))
        if fault == "missing":
            paths[0].unlink()
        else:
            value = json.loads(paths[0].read_text())
            value["error"] = "original execution failed"
            paths[0].write_text(json.dumps(value))
        # Authenticate the altered inventory so deeper original-run validation
        # must reject the missing or failed measurement, even with a green summary.
        name = "reused/" + seal(retained)
        retained.rename(current / name)
        next(value for value in records if "reuse" in value)["reuse"]["report"] = name
    else:
        path = {
            "log": next((retained / "logs").rglob("*.txt")),
            "record": next((retained / "results").rglob("*.json")),
            "wheel": next((retained / "artifacts").rglob("*.whl")),
            "source": retained / "source/experiments/family/signal.py",
        }[fault]
        path.write_bytes(b"tampered")
    summary = evaluate(manifest, records, root=suite, reports=current)
    assert not summary["passed"]
    if fault in {"failed", "missing"}:
        assert any("complete passing original run" in error for error in summary["errors"])


def test_archived_passing_checks_still_block_candidate_regression(reuse_pair):
    suite, _, current, manifest, _, _ = reuse_pair
    records = [json.loads(path.read_text()) for path in (current / "results").rglob("*.json")]
    next(value for value in records if value["task"]["role"] == "candidate")["result"]["checks"]["signal"] = False
    summary = evaluate(manifest, records, root=suite, reports=current)
    assert not summary["passed"]
    assert summary["regressions"] == [{"id": "P901", "seed": 123, "check": "signal"}]


@pytest.mark.parametrize("fault", ["failed", "missing"])
def test_failed_or_incomplete_archive_cannot_supply_baseline(tmp_path, fault):
    suite = tmp_path / "suite"
    source(suite)
    original = tmp_path / "original"
    setup_report(original, suite)
    path = next((original / "results").rglob("*.json"))
    if fault == "missing":
        path.unlink()
    else:
        record = json.loads(path.read_text())
        record["error"] = "training failed"
        path.write_text(json.dumps(record))
    assert not finish(original, root=suite)["passed"]
    current = tmp_path / "current"
    manifest, _ = setup_report(
        current, suite, revision="new", seed_records=False, baseline_payload=b"sdk", candidate_payload=b"new-sdk"
    )
    wrapper = archive(original, current, manifest["tasks"][0])
    with pytest.raises(ValueError, match="incomplete or failed original run"):
        resolved(manifest, [wrapper], current, root=suite)
    collect(current, wrapper)
    collect(current, fresh(manifest, manifest["tasks"][1]))
    assert not finish(current, root=suite)["passed"]
    with pytest.raises(ValueError, match="Publication requires a complete full-catalog run"):
        validate(current, current / "artifacts", root=suite)


def test_archived_reference_chains_are_bounded(reuse_pair, tmp_path):
    suite, _, current, _, _, _ = reuse_pair
    assert finish(current, root=suite)["passed"]
    next_run = tmp_path / "next"
    next_manifest, _ = setup_report(
        next_run,
        suite,
        revision="next",
        seed_records=False,
        baseline_payload=b"new-sdk",
        candidate_payload=b"third-sdk",
    )
    wrapper = archive(current, next_run, next_manifest["tasks"][0])
    assert wrapper["reuse"]["task"].startswith("candidate/")
    collect(next_run, wrapper)
    collect(next_run, fresh(next_manifest, next_manifest["tasks"][1]))
    assert finish(next_run, root=suite)["passed"]
    deeper = tmp_path / "deeper"
    deeper_manifest, _ = setup_report(
        deeper,
        suite,
        revision="deeper",
        seed_records=False,
        baseline_payload=b"third-sdk",
        candidate_payload=b"fourth-sdk",
    )
    wrapper = archive(next_run, deeper, deeper_manifest["tasks"][0])
    with pytest.raises(ValueError, match="complete passing original run"):
        resolved(deeper_manifest, [wrapper], deeper, root=suite)
    collect(deeper, wrapper)
    collect(deeper, fresh(deeper_manifest, deeper_manifest["tasks"][1]))
    assert not finish(deeper, root=suite)["passed"]


def test_missing_slot_and_untrusted_provenance_bypass_cannot_pass(reuse_pair):
    suite, _, current, manifest, _, _ = reuse_pair
    records = [json.loads(path.read_text()) for path in (current / "results").rglob("*.json")]
    assert not evaluate(manifest, records[:1], root=suite, reports=current)["passed"]
    candidate = next(value for value in records if value["task"]["role"] == "candidate")
    candidate["result"]["provenance"]["wheel_sha256"] = "forged"
    candidate["reused_from"] = {"report": "pretend validated"}
    assert not evaluate(manifest, records, root=suite, reports=current)["passed"]


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


def test_execution_keys_ignore_publication_files_but_bind_runtime_dependencies(tmp_path):
    from evidence_ci import runtime_inputs

    source(tmp_path)
    original = runtime_inputs(tmp_path)
    for name in ("README.md", "results.json", "tests/test_example.py", "publication.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("presentation only\n")
    assert runtime_inputs(tmp_path) == original
    (tmp_path / "experiments/reporting.py").write_text("# different shared runtime\n")
    assert runtime_inputs(tmp_path) != original


def test_one_experiment_change_preserves_other_baseline_execution_keys(tmp_path):
    from evidence_ci import execution_key

    suite = tmp_path / "suite"
    first = source(suite)
    other = first.with_name("other.py")
    other.write_text(first.read_text().replace("P901", "P902"))
    before, execution = setup_report(tmp_path / "before", suite)
    first.write_text(first.read_text() + "check_threshold = 0.25\n")
    after, _ = setup_report(tmp_path / "after", suite)
    for identifier in ("P901", "P902"):
        old = next(task for task in before["tasks"] if task["id"] == identifier and task["role"] == "baseline")
        new = next(task for task in after["tasks"] if task["id"] == identifier and task["role"] == "baseline")
        assert (execution_key(before, old, execution) == execution_key(after, new, execution)) is (identifier == "P902")


def test_runtime_keys_bind_every_byte_exposed_to_the_worker(tmp_path):
    from evidence_ci import runtime_inputs
    from experiments.source import frontmatter, metadata

    script = source(tmp_path)
    original = runtime_inputs(tmp_path)
    text = script.read_text()
    script.write_text(frontmatter(text, dict(metadata(text), **{"evidence-result": {"metrics": {"x": 5}}})))
    assert runtime_inputs(tmp_path) != original
