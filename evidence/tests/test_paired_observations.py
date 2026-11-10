"""Paired figure inputs must match every original task and immutable record."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from evidence.diagnostics.paired import prepare
from evidence.evidence_ci import task_key
from evidence.experiments.source import registry
from evidence.observations import package, read, unpack

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def paired(tmp_path):
    reports, observations = tmp_path / "reports", tmp_path / "observations"
    entry = registry(ROOT / "evidence", measurements=False)["P001"]
    source = reports / "source" / entry["script"]
    source.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "evidence" / entry["script"], source)
    (reports / "results").mkdir()
    manifest = dict(
        mode="full",
        full_catalog=True,
        tasks=[],
        run_id="gh-42-2",
        evidence_sha="a" * 40,
        executor_sha="b" * 40,
        observations={"capture_sha256": "c" * 64},
        candidate={"revision": "a" * 40, "wheel_sha256": "d" * 64},
        baseline={"revision": "e" * 40, "wheel_sha256": "f" * 64},
    )
    for role in ("baseline", "candidate"):
        task = dict(
            id="P001",
            role=role,
            script_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            seed=123,
            accelerator="cpu",
            attempt=1,
        )
        manifest["tasks"].append(task)
        raw = tmp_path / role
        raw.mkdir()
        record = {"checks": {"Fixture criterion": False}, "outcome": "not_met", "seed": 123}
        (raw / "record.json").write_text(json.dumps(record))
        (raw / "frames.json").write_text("[]")
        shutil.copyfile(source, raw / "proof.py")
        header = dict(
            id="P001",
            task=task,
            source_sha256=task["script_sha256"],
            wheel_sha256=manifest[role]["wheel_sha256"],
            capture_sha256="c" * 64,
            runtime={"run_id": "gh-42-2"},
            artifacts={path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in raw.iterdir()},
        )
        (raw / "manifest.json").write_text(json.dumps(header))
        destination = observations / role / "bundle.zip"
        destination.parent.mkdir(parents=True)
        receipt = package(raw, destination)
        record_path = reports / "results" / (task_key(task) + ".json")
        record_path.parent.mkdir(parents=True)
        record_path.write_text(json.dumps(dict(task=task, result=record, observations=receipt)))
    (reports / "manifest.json").write_text(json.dumps(manifest))
    return reports, observations, tmp_path / "selected"


def test_paired_selection_uses_candidate_only_and_retains_unmet_record(paired):
    reports, observations, selected = paired
    ci = prepare(reports, observations, selected)
    assert ci["run_id"] == "gh-42-2"
    assert [path.name for path in selected.iterdir()] == ["P001"]
    assert json.loads((selected / "P001/record.json").read_text())["outcome"] == "not_met"


@pytest.mark.parametrize("fault", ["record", "wheel", "run", "missing", "duplicate", "source"])
def test_paired_selection_rejects_mismatched_observations(paired, fault):
    reports, observations, selected = paired
    manifest = json.loads((reports / "manifest.json").read_text())
    task = manifest["tasks"][1]
    path = reports / "results" / (task_key(task) + ".json")
    if fault == "record":
        value = json.loads(path.read_text())
        value["result"]["checks"]["Fixture criterion"] = True
        path.write_text(json.dumps(value))
    elif fault in {"wheel", "source"}:
        if fault == "wheel":
            manifest["candidate"]["wheel_sha256"] = "0" * 64
        else:
            task["script_sha256"] = "0" * 64
        (reports / "manifest.json").write_text(json.dumps(manifest))
    elif fault == "run":
        manifest["run_id"] = "gh-42-3"
        (reports / "manifest.json").write_text(json.dumps(manifest))
    elif fault == "missing":
        (observations / "baseline/bundle.zip").unlink()
    else:
        target = observations / "duplicate/bundle.zip"
        target.parent.mkdir()
        shutil.copyfile(observations / "baseline/bundle.zip", target)
    with pytest.raises(ValueError):
        prepare(reports, observations, selected)


def test_archive_rejects_changed_bytes_before_installing(paired, tmp_path):
    _, observations, _ = paired
    archive = observations / "candidate/bundle.zip"
    receipt = {"sha256": "0" * 64, "bytes": archive.stat().st_size}
    with pytest.raises(ValueError, match="execution receipt"):
        unpack(archive, tmp_path / "invalid", receipt)
    assert not (tmp_path / "invalid").exists()
    value = read(archive)
    assert value["manifest"]["runtime"]["run_id"] == "gh-42-2"


def test_migration_selects_only_replacement_candidates_and_requires_retired_observations(paired, tmp_path):
    from evidence.migration import BASELINE, verify_sources

    reports, observations, selected = paired
    source = reports / "source"
    shutil.copytree(ROOT / "evidence/migrations", source / "migrations")
    migration = verify_sources(source, BASELINE)["proofs"]
    full_catalog = registry(ROOT / "evidence", measurements=False)
    manifest_path = reports / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(policy="cluster-group-transition-v1", migration=migration)
    manifest["baseline"]["revision"] = BASELINE
    for entry in migration:
        replacement = entry["proposed_replacement"]
        tasks = [(entry["id"], "baseline", entry["archive"])]
        if replacement:
            path = full_catalog[replacement]["script"]
            target = source / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "evidence" / path, target)
            tasks.append((replacement, "candidate", path))
        for identifier, role, script in tasks:
            proof = source / script
            task = dict(
                id=identifier,
                role=role,
                script_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),
                seed=0,
                accelerator="cpu",
                attempt=1,
            )
            manifest["tasks"].append(task)
            raw = tmp_path / f"raw-{identifier}-{role}"
            raw.mkdir()
            record = {"checks": {"criterion": True}, "outcome": "met", "seed": 0}
            (raw / "record.json").write_text(json.dumps(record))
            (raw / "frames.json").write_text("[]")
            shutil.copyfile(proof, raw / "proof.py")
            header = dict(
                id=identifier,
                task=task,
                source_sha256=task["script_sha256"],
                wheel_sha256=manifest[role]["wheel_sha256"],
                capture_sha256="c" * 64,
                runtime={"run_id": "gh-42-2"},
                artifacts={
                    path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in raw.iterdir()
                },
            )
            (raw / "manifest.json").write_text(json.dumps(header))
            destination = observations / f"{identifier}-{role}" / "bundle.zip"
            destination.parent.mkdir()
            receipt = package(raw, destination)
            record_path = reports / "results" / (task_key(task) + ".json")
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(dict(task=task, result=record, observations=receipt)))
    manifest_path.write_text(json.dumps(manifest))
    ci = prepare(reports, observations, selected)
    assert ci["candidate_only"] == ["P108", "P109", "P110", "P111", "P112"]
    assert {path.name for path in selected.iterdir()} == {"P001", *ci["candidate_only"]}
    shutil.rmtree(selected)
    (observations / "P065-baseline/bundle.zip").unlink()
    with pytest.raises(ValueError, match="exact measured task union"):
        prepare(reports, observations, selected)
