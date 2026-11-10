"""Archived Cluster proof sources remain identifiable migration inputs."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from evidence.experiments.source import declaration, registry
from evidence.migration import BASELINE, compare_checks, transition, verify_sources

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "migrations/cluster_to_group"


def test_cluster_migration_archives_preserve_declared_sources():
    manifest = json.loads((MIGRATION / "manifest.json").read_text())
    assert manifest["baseline_revision"] == "c0cae6df38010e232c0f12c966eaa6a06fae4b94"
    assert {entry["id"] for entry in manifest["proofs"]} == {"P018", "P019", "P020", "P065", "P078", "P095"}
    assert len(manifest["proofs"]) == 6
    for entry in manifest["proofs"]:
        path = ROOT / entry["archive"]
        assert path.parent == MIGRATION and path.is_file() and not path.is_symlink()
        source = path.read_bytes()
        assert hashlib.sha256(source).hexdigest() == entry["sha256"]
        declared = declaration(source.decode())
        assert (declared.id, declared.seed) == (entry["id"], entry["seed"])
        assert entry["source_path"].startswith("evidence/experiments/")


def test_reviewed_migration_inventory_rejects_wrong_baseline():
    with pytest.raises(ValueError, match="Migration baseline"):
        verify_sources(ROOT, "0" * 40)


@pytest.mark.parametrize("fault", ["manifest", "proof", "proof_link", "directory_link"])
def test_reviewed_migration_inventory_rejects_tampering(tmp_path, fault):
    root = tmp_path / "evidence"
    shutil.copytree(MIGRATION, root / "migrations/cluster_to_group")
    directory = root / "migrations/cluster_to_group"
    if fault == "manifest":
        with (directory / "manifest.json").open("a") as handle:
            handle.write(" ")
    elif fault == "proof":
        with (directory / "P018.py").open("a") as handle:
            handle.write("\n# changed\n")
    elif fault == "proof_link":
        proof = directory / "P018.py"
        proof.unlink()
        proof.symlink_to(directory / "P019.py")
    else:
        saved = root / "saved"
        directory.rename(saved)
        directory.symlink_to(saved, target_is_directory=True)
    with pytest.raises(ValueError):
        verify_sources(root, BASELINE)


def test_reviewed_migration_inventory_accepts_original_sources():
    assert len(verify_sources(ROOT, BASELINE)["proofs"]) == 6


def test_transition_requires_all_new_core_ids_and_no_reused_cluster_ids():
    catalog = {identifier: {"lane": "core"} for identifier in ("P108", "P109", "P110", "P111", "P112")}
    entries = transition(ROOT, BASELINE, catalog)
    assert {entry["id"] for entry in entries if entry["proposed_replacement"] is None} == {"P065"}
    with pytest.raises(ValueError, match="P108"):
        transition(ROOT, BASELINE, {key: value for key, value in catalog.items() if key != "P108"})
    with pytest.raises(ValueError, match="Retired Cluster"):
        transition(ROOT, BASELINE, {**catalog, "P019": {"lane": "core"}})
    with pytest.raises(ValueError, match="P108"):
        transition(ROOT, BASELINE, {**catalog, "P108": {"lane": "research"}})


def test_registered_group_replacements_use_distinct_new_ids():
    entries = transition(ROOT, BASELINE, registry(ROOT, measurements=False))
    assert {entry["proposed_replacement"] for entry in entries if entry["proposed_replacement"]} == {
        "P108",
        "P109",
        "P110",
        "P111",
        "P112",
    }


def test_transition_checks_keep_unmet_baselines_and_new_failures_separate():
    entries = verify_sources(ROOT, BASELINE)["proofs"]
    baseline = {entry["id"]: {"outcome": "met", "checks": {"metric": True}} for entry in entries}
    candidate = {
        entry["proposed_replacement"]: {"outcome": "met", "checks": {"metric": True}}
        for entry in entries
        if entry["proposed_replacement"]
    }
    baseline["P018"] = {"outcome": "not_met", "checks": {"same label": False}}
    candidate["P108"] = {"outcome": "not_met", "checks": {"same label": False}}
    actual = compare_checks(entries, baseline, candidate)
    assert actual == {
        "baseline_unmet": [{"id": "P018", "check": "same label"}],
        "candidate_unmet": [{"id": "P108", "check": "same label"}],
    }
    with pytest.raises(ValueError, match="every retired and replacement"):
        compare_checks(entries, baseline, {key: value for key, value in candidate.items() if key != "P108"})
    with pytest.raises(ValueError, match="inconsistent outcome"):
        compare_checks(entries, baseline, {**candidate, "P108": {"outcome": "met", "checks": {"x": False}}})
    with pytest.raises(ValueError, match="invalid checks"):
        compare_checks(entries, baseline, {**candidate, "P108": {"outcome": "met", "checks": {"x": 1}}})
