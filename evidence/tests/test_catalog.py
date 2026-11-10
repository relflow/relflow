"""Catalog changes preserve the required gate and never execute baseline code."""

import subprocess
from copy import deepcopy

import evidence_ci
import pytest
from evidence_ci import evaluate, plan, registry


def test_baseline_catalog_reads_git_tree_as_data(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    source = repo / "evidence/experiments/example/proof.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "# %% [markdown]\n# ---\n# title: Signal\n# ---\n"
        "raise RuntimeError('must not execute')\n"
        "@experiment('P001', seed=0)\n"
        "def proof(seed, steps, accelerator):\n    pass\n"
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "proof"],
        check=True,
    )
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(evidence_ci, "ROOT", repo / "evidence")
    assert evidence_ci.baseline_catalog(revision) == {"P001"}
    source.write_text("changed outside the commit")
    assert evidence_ci.baseline_catalog(revision) == {"P001"}
    with pytest.raises(ValueError, match="full Git revision"):
        evidence_ci.baseline_catalog("HEAD")
    with pytest.raises(ValueError, match="Cannot read baseline catalog"):
        evidence_ci.baseline_catalog("0" * 40)


@pytest.fixture
def introduced(monkeypatch):
    entries = registry(measurements=False)
    monkeypatch.setattr(
        evidence_ci, "registry", lambda *args, **kwargs: {key: entries[key] for key in ("P001", "P108")}
    )
    baseline = {"P001", "P018"}
    monkeypatch.setattr(evidence_ci, "baseline_catalog", lambda revision: baseline)
    manifest = plan(
        run_id="test",
        candidate={"revision": "candidate", "wheel_sha256": "cw"},
        baseline={"revision": "baseline", "wheel_sha256": "bw"},
        evidence_sha="suite",
        executor_sha="executor",
        baseline_ids=baseline,
    )
    records = []
    for task in manifest["tasks"]:
        result = {
            "mode": "full",
            "seed": task["seed"],
            "accelerator": "cpu",
            "outcome": "met",
            "checks": {"signal": True},
            "provenance": {
                "evidence_revision": "suite",
                "relflow_revision": manifest[task["role"]]["revision"],
                "wheel_sha256": manifest[task["role"]]["wheel_sha256"],
                "script_sha256": task["script_sha256"],
            },
        }
        records.append({"task": task, "result": result, "error": None})
    return manifest, records


def test_introduced_proof_is_required_and_existing_proof_is_paired(introduced):
    manifest, records = introduced
    assert manifest["introduced"] == ["P108"]
    assert manifest["retired"] == ["P018"]
    assert {(task["id"], task["role"]) for task in manifest["tasks"]} == {
        ("P001", "baseline"),
        ("P001", "candidate"),
        ("P108", "candidate"),
    }
    assert evaluate(manifest, records)["passed"]
    records[-1]["result"]["checks"]["signal"] = False
    result = evaluate(manifest, records)
    assert not result["passed"]
    assert result["introduced_unmet_checks"][0]["id"] == "P108"


@pytest.mark.parametrize("fault", ["introduced", "retired", "remove-task", "missing", "empty", "duplicate"])
def test_catalog_gate_rejects_incomplete_or_forged_results(introduced, fault):
    manifest, records = deepcopy(introduced)
    if fault == "introduced":
        manifest["introduced"] = ["P001", "P108"]
    elif fault == "retired":
        manifest["retired"] = []
    elif fault == "remove-task":
        manifest["tasks"] = manifest["tasks"][1:]
    elif fault == "missing":
        records.pop()
    elif fault == "empty":
        records[-1]["result"]["checks"] = {}
    else:
        records.append(deepcopy(records[-1]))
    assert not evaluate(manifest, records)["passed"]


def test_proof_returns_to_pairing_when_present_in_baseline(introduced):
    manifest, _ = introduced
    paired = plan(
        run_id=manifest["run_id"],
        candidate=manifest["candidate"],
        baseline=manifest["baseline"],
        evidence_sha=manifest["evidence_sha"],
        executor_sha=manifest["executor_sha"],
        baseline_ids={"P001", "P108"},
    )
    assert paired["policy"] == "paired-passing-checks-v1"
    assert {(task["id"], task["role"]) for task in paired["tasks"]} == {
        (identifier, role) for identifier in ("P001", "P108") for role in ("baseline", "candidate")
    }
