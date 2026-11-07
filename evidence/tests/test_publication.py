"""Publication must not disguise incomplete execution with historical results."""

import json
import shutil

import pytest
from evidence_ci import ROOT, plan, task_key, write_json
from experiments.source import fingerprint, frontmatter, registry
from record import readme, select, validate


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
            "metrics": {
                "P021": {"hash_auc": 0.981234, "category_oov_auc": 0.511234},
                "P025": {"median": {"intact_nrmse": 0.2234, "broken_nrmse": 1.23456}},
                "P038": {"state_accuracy": 0.98765, "prefilled_accuracy": 0.49321},
            }.get(task["id"], {"score": 0.5}),
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
    for helper in ("source.py", "reporting.py", "run.py"):
        shutil.copy2(ROOT / "experiments" / helper, root / "experiments" / helper)
    shutil.copy2(ROOT / "results.json", root / "results.json")
    return root


def test_selection_preserves_candidate_and_github_provenance(reports, publication, tmp_path):
    github_run = tmp_path / "github-run.json"
    github_run.write_text(json.dumps({"run_id": 123, "run_attempt": 2, "head_sha": "pr-head"}))
    before = {path: path.read_bytes() for path in publication.rglob("*.py")}
    select(
        reports, root=publication, github_run=github_run, release="evidence-test", output=publication / "results.json"
    )
    results = registry(publication)
    selection = results["P001"]["report"]
    assert selection["release"] == "https://github.com/relflow/relflow/releases/tag/evidence-test"
    assert selection["evidence_revision"] == "tested-suite"
    assert selection["github_run"]["head_sha"] == "pr-head"
    assert set(results) == set(registry(ROOT))
    assert all(entry["run"]["provenance"]["relflow_revision"] == "candidate" for entry in results.values())
    for path, source in before.items():
        selected = path.read_bytes()
        assert selected == source
        assert fingerprint(selected.decode()) == fingerprint(source.decode())


@pytest.mark.parametrize("fault", ["missing", "error", "provenance", "regression", "smoke", "partial"])
def test_selection_rejects_invalid_records_even_with_successful_summary(reports, publication, fault):
    path = next((reports / "results/candidate").rglob("*.json"))
    if fault == "missing":
        path.unlink()
    elif fault in {"error", "provenance", "regression"}:
        record = json.loads(path.read_text())
        if fault == "error":
            record["error"] = "timeout"
        elif fault == "provenance":
            record["result"]["provenance"]["wheel_sha256"] = "wrong-wheel"
        else:
            record["result"]["checks"]["held-out signal"] = False
            record["result"]["outcome"] = "not_met"
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


@pytest.mark.parametrize("fault", ["code", "header", "renamed", "removed", "added"])
def test_selection_rejects_catalog_and_source_drift_before_writing(reports, publication, fault):
    source = publication / registry(publication)["P001"]["script"]
    if fault == "code":
        source.write_text(source.read_text() + "\nCHANGED_EXPERIMENT = True\n")
    elif fault == "header":
        entry = registry(publication)["P001"]
        source.write_text(frontmatter(source.read_text(), dict(entry["header"], title="Changed protocol title")))
    elif fault == "renamed":
        source.rename(source.with_name("renamed_experiment.py"))
    elif fault == "removed":
        source.unlink()
    else:
        original = source.read_text()
        added = original.replace('@experiment("P001",', '@experiment("P999",', 1)
        assert added != original, "the added experiment fixture must declare its own ID"
        source.with_name("new_experiment.py").write_text(added)
        assert "P999" in registry(publication), "the fixture must add a valid catalog entry"
    before = {path: path.read_bytes() for path in publication.rglob("*.py")}
    with pytest.raises(ValueError, match="source|catalog"):
        select(reports, root=publication)
    assert all(path.read_bytes() == content for path, content in before.items())


def test_selection_rejects_altered_manifest_paths_even_when_records_agree(reports, publication):
    manifest_path = reports / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for task in manifest["tasks"]:
        if task["id"] == "P001":
            task["script"] = "../outside.py"
            record_path = reports / "results" / f"{task_key(task)}.json"
            record = json.loads(record_path.read_text())
            record["task"] = task
            record_path.write_text(json.dumps(record))
    manifest_path.write_text(json.dumps(manifest))
    before = {path: path.read_bytes() for path in publication.rglob("*.py")}
    with pytest.raises(ValueError, match="source path"):
        select(reports, root=publication)
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("measured", [float("nan"), float("inf"), -float("inf")])
def test_publication_rejects_nonfinite_measurements(reports, publication, measured):
    path = next((reports / "results/candidate").rglob("*.json"))
    record = json.loads(path.read_text())
    record["result"]["metrics"]["invalid"] = measured
    path.write_text(json.dumps(record))
    before = {path: path.read_bytes() for path in publication.rglob("*.py")}
    with pytest.raises(ValueError, match="finite JSON measurements"):
        select(reports, root=publication)
    assert all(path.read_bytes() == content for path, content in before.items())


def test_validation_uses_the_supplied_catalog(reports, publication):
    source = publication / registry(publication)["P079"]["script"]
    source.unlink()
    validate(reports)
    with pytest.raises(ValueError, match="full-catalog"):
        validate(reports, root=publication)


@pytest.fixture
def readme_target(tmp_path):
    target = tmp_path / "README.md"
    target.write_bytes(
        b"# Keep this heading\r\n\r\nKeep **0.9991 AUC** outside the table.\r\n\r\n"
        b"<!-- evidence-results:start -->\n\nOld table\n\n<!-- evidence-results:end -->\r\n\r\nKeep the footer.\r\n"
    )
    return target


def test_selection_refreshes_readme_from_candidate_metrics_with_exact_boundaries(reports, publication, readme_target):
    before = readme_target.read_bytes()
    select(reports, root=publication, readme_target=readme_target)
    selected = readme_target.read_bytes()
    assert selected.split(b"<!-- evidence-results:start -->")[0] == before.split(b"<!-- evidence-results:start -->")[0]
    assert selected.split(b"<!-- evidence-results:end -->")[1] == before.split(b"<!-- evidence-results:end -->")[1]
    content = selected.decode()
    assert "**0.9812 AUC**" in content
    assert "0.5112 with unknown `Category` labels" in content
    assert "**0.223 normalized RMSE**" in content
    assert "1.235 after corrupting the lookup IDs" in content
    assert "**98.77% accuracy**" in content
    assert "49.32% after filling nulls with zero" in content
    assert content.count("| [") == 3
    assert "https://www.relflow.ai/en/latest/evidence/identity/unseen-identity-equality.html" in content


def test_readme_reads_current_selection_instead_of_an_old_report(reports, publication, readme_target):
    selected = publication / "results.json"
    select(reports, root=publication, output=selected)
    document = json.loads(selected.read_text())
    document["results"]["P021"]["metrics"]["hash_auc"] = 0.912345
    selected.write_text(json.dumps(document))
    readme(publication, readme_target)
    assert "**0.9123 AUC**" in readme_target.read_text()
    assert "**0.9812 AUC**" not in readme_target.read_text()


@pytest.mark.parametrize(
    "content",
    [
        "No markers",
        "<!-- evidence-results:start -->\n<!-- evidence-results:start -->\n<!-- evidence-results:end -->",
        "<!-- evidence-results:end -->\n<!-- evidence-results:start -->",
        "Inline <!-- evidence-results:start -->\n<!-- evidence-results:end -->",
        "<!-- evidence-results:start -->\n<!-- evidence-results:end -->\n<!-- evidence-results:end -->",
    ],
)
def test_invalid_readme_boundaries_leave_every_selection_unchanged(reports, publication, readme_target, content):
    readme_target.write_text(content)
    before = {path: path.read_bytes() for path in publication.rglob("*.py")}
    with pytest.raises(ValueError, match="standalone evidence-results"):
        select(reports, root=publication, readme_target=readme_target)
    assert readme_target.read_text() == content
    assert all(path.read_bytes() == source for path, source in before.items())


@pytest.mark.parametrize("measured", [None, True, "0.9", float("nan"), float("inf"), -0.1, 1.1])
def test_readme_rejects_unusable_metrics_without_writing(publication, readme_target, measured):
    selected = publication / "results.json"
    document = json.loads(selected.read_text())
    document["results"]["P021"]["metrics"]["hash_auc"] = measured
    selected.write_text(json.dumps(document))
    before = readme_target.read_bytes()
    with pytest.raises(ValueError, match="metric|measurements"):
        readme(publication, readme_target)
    assert readme_target.read_bytes() == before


def test_missing_candidate_readme_metric_does_not_partially_select(reports, publication, readme_target):
    path = next((reports / "results/candidate/P038").rglob("*.json"))
    record = json.loads(path.read_text())
    del record["result"]["metrics"]["state_accuracy"]
    path.write_text(json.dumps(record))
    before = {path: path.read_bytes() for path in publication.rglob("*.py")}
    readme_before = readme_target.read_bytes()
    with pytest.raises(ValueError, match="P038.*missing README metric state_accuracy"):
        select(reports, root=publication, readme_target=readme_target)
    assert readme_target.read_bytes() == readme_before
    assert all(path.read_bytes() == content for path, content in before.items())
