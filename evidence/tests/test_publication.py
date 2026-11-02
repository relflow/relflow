"""Publication must not disguise incomplete execution with historical results."""

import json
import shutil

import pytest
from evidence_ci import ROOT, plan, task_key, write_json
from experiments.source import registry
from record import select


@pytest.fixture
def reports(tmp_path):
    root = tmp_path / "reports"
    manifest = plan(
        run_id="publication-test",
        candidate={"revision": "candidate", "wheel_sha256": "candidate-wheel"},
        baseline={"revision": "base", "wheel_sha256": "base-wheel"},
        evidence_sha="tested-suite",
        executor_sha="tested-executor",
    )
    write_json(root / "manifest.json", manifest)
    for task in manifest["tasks"]:
        result = {
            "mode": "full",
            "seed": task["seed"],
            "accelerator": task["accelerator"],
            "outcome": "met",
            "checks": {"held-out signal": True},
            "provenance": {
                "evidence_revision": manifest["evidence_sha"],
                "relflow_revision": manifest[task["role"]]["revision"],
                "wheel_sha256": manifest[task["role"]]["wheel_sha256"],
                "script_sha256": task["script_sha256"],
            },
        }
        write_json(root / "results" / f"{task_key(task)}.json", {"task": task, "error": None, "result": result})
    # An old aggregate must not override what the individual records prove.
    write_json(root / "summary.json", {"complete": True, "passed": True, "errors": []})
    return root


@pytest.fixture
def publication(tmp_path):
    root = tmp_path / "publication"
    for entry in registry().values():
        source = ROOT / entry["script"]
        target = root / entry["script"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return root


def test_selection_preserves_candidate_and_github_provenance(reports, publication, tmp_path):
    github_run = tmp_path / "github-run.json"
    github_run.write_text(json.dumps({"run_id": 123, "run_attempt": 2, "head_sha": "pr-head"}))
    select(reports, root=publication, github_run=github_run, release="evidence-test")
    results = registry(publication)
    selection = results["P001"]["header"]["evidence-report"]
    assert selection["release"] == "https://github.com/relflow/relflow/releases/tag/evidence-test"
    assert selection["evidence_revision"] == "tested-suite"
    assert selection["github_run"]["head_sha"] == "pr-head"
    assert len(results) == 79
    assert all(entry["run"]["provenance"]["relflow_revision"] == "candidate" for entry in results.values())


@pytest.mark.parametrize("fault", ["missing", "error", "provenance", "smoke", "partial"])
def test_selection_rejects_invalid_records_even_with_successful_summary(reports, publication, fault):
    path = next((reports / "results/candidate").rglob("*.json"))
    if fault == "missing":
        path.unlink()
    elif fault in {"error", "provenance"}:
        record = json.loads(path.read_text())
        if fault == "error":
            record["error"] = "timeout"
        else:
            record["result"]["provenance"]["wheel_sha256"] = "wrong-wheel"
        path.write_text(json.dumps(record))
    else:
        path = reports / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["mode" if fault == "smoke" else "full_catalog"] = "smoke" if fault == "smoke" else False
        path.write_text(json.dumps(manifest))
    before = {path: path.read_bytes() for path in publication.rglob("*.py")}
    with pytest.raises(ValueError, match="complete full-catalog"):
        select(reports, root=publication)
    assert all(path.read_bytes() == content for path, content in before.items())
