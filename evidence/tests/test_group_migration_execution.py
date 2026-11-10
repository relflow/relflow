"""The one-time Group migration preserves task identity and fails closed."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from evidence_ci import evaluate, plan, runtime_inputs, script_paths
from experiments.source import registry
from hosted import assignments
from migration import BASELINE
from record import validate
from runtime import isolate

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "c" * 40
EXECUTOR = "f" * 40
SUITE = "e" * 40


@pytest.fixture
def execution():
    manifest = plan(
        run_id="group-migration-test",
        candidate={"revision": CANDIDATE, "wheel_sha256": "1" * 64},
        baseline={"revision": BASELINE, "wheel_sha256": "2" * 64},
        evidence_sha=SUITE,
        executor_sha=EXECUTOR,
        root=ROOT,
    )
    records = []
    for task in manifest["tasks"]:
        package = manifest[task["role"]]
        result = {
            "seed": task["seed"],
            "accelerator": task["accelerator"],
            "mode": "full",
            "outcome": "met",
            "checks": {"example": True},
            "provenance": {
                "evidence_revision": SUITE,
                "relflow_revision": package["revision"],
                "wheel_sha256": package["wheel_sha256"],
                "script_sha256": task["script_sha256"],
            },
        }
        records.append({"task": deepcopy(task), "result": result, "error": None})
    return manifest, records


def test_full_transition_plans_distinct_roles_and_same_shard(execution):
    manifest, records = execution
    assert manifest["policy"] == "cluster-group-transition-v1"
    tasks = {(task["role"], task["id"]) for task in manifest["tasks"]}
    for item in manifest["migration"]:
        assert ("baseline", item["id"]) in tasks
        assert ("candidate", item["id"]) not in tasks
        replacement = item["proposed_replacement"]
        if replacement:
            assert ("candidate", replacement) in tasks
            assert ("baseline", replacement) not in tasks
    buckets = assignments(manifest, ROOT)
    for item in manifest["migration"]:
        replacement = item["proposed_replacement"]
        if replacement:
            assert any(item["id"] in bucket and replacement in bucket for bucket in buckets)
    summary = evaluate(manifest, records, root=ROOT)
    assert summary["complete"] and summary["passed"]
    assert summary["migration_candidate_unmet_checks"] == []


def test_transition_keeps_new_failures_and_baseline_limitations_distinct(execution):
    manifest, records = execution
    old = next(record for record in records if record["task"]["id"] == "P018")
    new = next(record for record in records if record["task"]["id"] == "P108")
    old["result"].update(outcome="not_met", checks={"old criterion": False})
    new["result"].update(outcome="not_met", checks={"new criterion": False})
    summary = evaluate(manifest, records, root=ROOT)
    assert not summary["passed"] and not summary["regressions"]
    assert summary["migration_candidate_unmet_checks"] == [
        {
            "id": "P108",
            "check": "new criterion",
            "seed": next(t["seed"] for t in manifest["tasks"] if t["id"] == "P108"),
        }
    ]
    assert {
        "id": "P018",
        "check": "old criterion",
        "seed": 0,
        "candidate_passed": None,
        "comparison": "not directly comparable",
    } in summary["baseline_unmet_checks"]


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "changed_task", "wrong_wheel", "changed_plan", "wrong_policy"]
)
def test_transition_rejects_missing_or_tampered_results(execution, fault):
    manifest, records = execution
    if fault == "missing":
        records = [record for record in records if record["task"]["id"] != "P065"]
    elif fault == "duplicate":
        records.append(deepcopy(records[0]))
    elif fault == "changed_task":
        records[0]["task"]["steps"] = 1
    elif fault == "wrong_wheel":
        records[0]["result"]["provenance"]["wheel_sha256"] = "0" * 64
    elif fault == "changed_plan":
        manifest["tasks"] = [task for task in manifest["tasks"] if task["id"] != "P065"]
        records = [record for record in records if record["task"]["id"] != "P065"]
    else:
        manifest["policy"] = "paired-passing-checks-v1"
    assert not evaluate(manifest, records, root=ROOT)["passed"]


def test_later_baselines_return_to_ordinary_pairing():
    manifest = plan(
        run_id="later",
        candidate={},
        baseline={"revision": "b" * 40},
        evidence_sha=SUITE,
        executor_sha=EXECUTOR,
        root=ROOT,
    )
    assert manifest["policy"] == "paired-passing-checks-v1"
    assert "migration" not in manifest
    tasks = {(task["role"], task["id"]) for task in manifest["tasks"]}
    assert ("baseline", "P111") in tasks and ("candidate", "P111") in tasks
    assert ("baseline", "P078") not in tasks
    with pytest.raises(ValueError, match="Retired Cluster"):
        script_paths(ROOT, {"P019": {"script": "experiments/group/plain_input_is_dormant.py"}})


def test_retired_task_isolation_contains_only_its_reviewed_proof(execution, tmp_path):
    manifest, _ = execution
    task = next(task for task in manifest["tasks"] if task["role"] == "baseline" and task["id"] == "P018")
    scripts = script_paths(ROOT, registry(ROOT, measurements=False))
    view = isolate(ROOT, tmp_path / "view", task, runtime_inputs(ROOT), scripts)
    assert (view / "migrations/cluster_to_group/P018.py").is_file()
    assert not (view / "migrations/cluster_to_group/P019.py").exists()
    assert not (view / "experiments/group/hidden_regression_regimes.py").exists()
    assert (view / "experiments/reporting.py").is_file()


def test_publication_validates_retired_baseline_sources(execution, tmp_path):
    manifest, records = execution
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    for index, record in enumerate(records):
        (result_dir / f"{index}.json").write_text(json.dumps(record))
    _, retained, summary = validate(tmp_path, root=ROOT)
    assert summary["passed"] and len(retained) == len(manifest["tasks"])
