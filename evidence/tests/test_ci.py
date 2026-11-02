"""Failure paths and controls for the merge gate, without training a model."""

from copy import deepcopy
from pathlib import Path

import evidence_ci
import pytest
from evidence_ci import ROOT, default_seed, evaluate, plan, registry, write_json


@pytest.fixture
def paired(monkeypatch):
    entries = registry()
    monkeypatch.setattr(evidence_ci, "registry", lambda *args: {"P001": entries["P001"]})
    manifest = plan(
        run_id="test",
        candidate={"revision": "candidate", "wheel_sha256": "candidate-wheel"},
        baseline={"revision": "base", "wheel_sha256": "baseline-wheel"},
        evidence_sha="suite",
        executor_sha="executor",
        identifiers=["P001"],
    )
    manifest["full_catalog"] = True
    records = [
        {
            "task": task,
            "error": None,
            "result": {
                "mode": "full",
                "seed": task["seed"],
                "accelerator": "cpu",
                "outcome": "met",
                "checks": {"learned signal": True, "noise control": True},
            },
        }
        for task in manifest["tasks"]
    ]
    for record in records:
        task = record["task"]
        record["result"]["provenance"] = {
            "evidence_revision": "suite",
            "relflow_revision": manifest[task["role"]]["revision"],
            "wheel_sha256": manifest[task["role"]]["wheel_sha256"],
            "script_sha256": task["script_sha256"],
        }
    return manifest, records


def test_baseline_passing_check_regression_blocks(paired):
    manifest, records = paired
    assert evaluate(manifest, records)["passed"]
    records[1]["result"]["checks"]["learned signal"] = False
    result = evaluate(manifest, records)
    assert not result["passed"]
    assert result["regressions"] == [{"id": "P001", "seed": records[1]["task"]["seed"], "check": "learned signal"}]


def test_existing_capability_miss_is_visible_and_improvement_allowed(paired):
    manifest, records = paired
    for record in records:
        record["result"]["checks"]["learned signal"] = False
    result = evaluate(manifest, records)
    assert result["passed"] and result["baseline_unmet_checks"]
    records[1]["result"]["checks"]["learned signal"] = True
    assert evaluate(manifest, records)["passed"]


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "unexpected",
        "error",
        "baseline-error",
        "empty-checks",
        "missing-check",
        "invalid-check",
        "wrong-seed",
        "smoke",
        "partial",
        "executor",
        "schema",
    ],
)
def test_incomplete_or_invalid_execution_never_passes(paired, fault):
    manifest, records = deepcopy(paired)
    if fault == "missing":
        records.pop()
    elif fault == "duplicate":
        records.append(deepcopy(records[0]))
    elif fault == "unexpected":
        records[1]["task"]["id"] = "P999"
    elif fault in {"error", "baseline-error"}:
        records[0 if fault == "baseline-error" else 1]["error"] = "timeout"
    elif fault == "empty-checks":
        records[1]["result"]["checks"] = {}
    elif fault == "missing-check":
        records[1]["result"]["checks"].pop("noise control")
    elif fault == "invalid-check":
        records[1]["result"]["checks"]["noise control"] = "true"
    elif fault == "wrong-seed":
        records[1]["result"]["seed"] += 1
    elif fault == "smoke":
        manifest["mode"] = "smoke"
    elif fault == "executor":
        manifest["executor_sha"] = "another-executor"
    elif fault == "schema":
        manifest["schema"] = 1
    else:
        manifest["full_catalog"] = False
    assert not evaluate(manifest, records)["passed"]


def test_full_catalog_plans_each_declared_seed_for_both_wheels():
    manifest = plan(run_id="full", candidate={}, baseline={}, evidence_sha="suite", executor_sha="executor")
    assert manifest["full_catalog"]
    assert len(manifest["tasks"]) == 2 * len(registry())
    for task in manifest["tasks"]:
        assert task["seed"] == default_seed(ROOT / task["script"])


def test_records_cannot_be_overwritten(tmp_path: Path):
    output = tmp_path / "result.json"
    write_json(output, {"score": 0.8})
    with pytest.raises(FileExistsError):
        write_json(output, {"score": 0.9})


def test_records_reject_nonfinite_metrics(tmp_path: Path):
    with pytest.raises(ValueError):
        write_json(tmp_path / "result.json", {"score": float("nan")})
