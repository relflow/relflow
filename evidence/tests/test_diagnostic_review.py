"""Static previews bind figures to captured runs without checking generated assets into Git."""

import gzip
import hashlib
import io
import json
import shutil
from copy import deepcopy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from evidence.diagnostics import review as reviewing
from evidence.diagnostics.review import load, prepare
from evidence.experiments.source import fingerprint, registry

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def project(tmp_path):
    entry = registry(ROOT / "evidence", measurements=False)["P001"]
    source = tmp_path / "evidence" / entry["script"]
    source.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "evidence" / entry["script"], source)
    assets = tmp_path / "docs/assets/evidence-plots"
    assets.mkdir(parents=True)
    run = {
        "mode": "full",
        "outcome": "not_met",
        "checks": {"Unmet unit fixture": False},
        "metrics": {},
        "provenance": {"script_sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    data = {
        "id": "P001",
        "focus": "Unit fixture",
        "comparison": "Captured values",
        "limit": "Fixture only",
        "record": run,
        "capture_run": "1",
        "manifest": "P001-manifest.json.gz",
        "interactive": {},
        "figures": [{"name": "P001", "caption": "Captured test fixture"}],
    }
    (assets / "P001.json").write_text(json.dumps(data))
    (assets / "P001-record.json").write_text(json.dumps(run))
    (assets / "P001-proof.py").write_bytes(source.read_bytes())
    (assets / "P001-manifest.json.gz").write_bytes(gzip.compress(b"{}", mtime=0))
    for theme in ["light", "dark"]:
        (assets / f"P001-{theme}.svg").write_text("<svg/>")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in assets.iterdir()}
    review = {
        "schema": 1,
        "revision": "a" * 40,
        "expected": ["P001"],
        "errors": {},
        "entries": {
            "P001": {
                "data": "P001.json",
                "figures": data["figures"],
                "code_sha256": fingerprint(source.read_text()),
                "record_sha256": hashes["P001-record.json"],
                "assets": hashes,
            }
        },
    }
    (assets / "review.json").write_text(json.dumps(review))
    return tmp_path


def test_every_experiment_owns_its_diagnostic_plan():
    from evidence.experiments.source import view

    catalog = registry(ROOT / "evidence", measurements=False)
    for entry in catalog.values():
        plan = view((ROOT / "evidence" / entry["script"]).read_text())
        assert all(plan[name] for name in ("focus", "comparison", "limit"))


def test_new_diagnostic_uses_captured_frontmatter_without_a_catalog_entry(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from evidence.diagnostics import build as building

    directory = tmp_path / "bundles" / "new"
    directory.mkdir(parents=True)
    source = (
        "# %% [markdown]\n# ---\n# evidence-plot:\n#   mode: summary\n"
        "#   focus: A new experiment owns this question.\n"
        "#   comparison: Its own comparison.\n#   limit: Its own limit.\n# ---\n"
        "raise RuntimeError('experiment module must never execute while rendering')\n"
    )
    manifest = {
        "id": "P999",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "code_sha256": fingerprint(source),
        "artifacts": {},
        "runtime": {"github_run_id": "fixture"},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "record.json").write_text(
        json.dumps({"outcome": "met", "provenance": {"evidence_revision": "fixture"}})
    )
    (directory / "proof.py").write_text(source)
    (directory / "frames.json").write_text("[]")

    def render(bundle, plan, output):
        for theme in ("light", "dark"):
            (output / f"P999-{theme}.svg").write_text("<svg/>")
        return dict(plan, id="P999", figures=[{"name": "P999", "caption": plan["focus"]}])

    monkeypatch.setitem(sys.modules, "evidence.diagnostics.plots", SimpleNamespace(render=render))
    output = tmp_path / "output"
    result = building.build(tmp_path / "bundles", output, ci={"expected": ["P999"]})
    assert result == {"rendered": 1, "failed": 0, "missing": [], "errors": {}}
    assert json.loads((output / "P999.json").read_text())["focus"] == "A new experiment owns this question."


def test_diagnostic_component_is_isolated_from_model_imports_and_training():
    import ast

    from evidence.diagnostics.data import Bundle

    bundle = object.__new__(Bundle)
    bundle.source = (
        "raise RuntimeError('module and model imports must not execute')\n"
        "def plot(bundle, spec):\n    return spec['value']\n"
    )
    bundle.syntax = ast.parse(bundle.source)
    bundle.directory = Path("/unused")
    bundle.components = {}
    callback = bundle.component("plot")
    assert callback(bundle, {"value": 17}) == 17
    assert bundle.component("plot") is callback


def test_custom_components_cannot_depend_on_training_module_globals():
    from evidence.experiments.source import component

    with pytest.raises(ValueError, match="must own its imports and helpers.*training_model"):
        component("training_model = object()\ndef plot(bundle, spec):\n    return training_model\n", "plot")


def test_every_custom_renderer_loads_independently_without_importing_its_experiment():
    from evidence.diagnostics.data import Bundle
    from evidence.experiments.source import view

    for entry in registry(ROOT / "evidence", measurements=False).values():
        source = (ROOT / "evidence" / entry["script"]).read_text()
        plan = view(source)
        bundle = object.__new__(Bundle)
        bundle.source, bundle.directory, bundle.components = source, Path("/unused"), {}
        for selection in [plan, *plan.get("supplements", {}).values()]:
            if "renderer" in selection:
                assert callable(bundle.component(selection["renderer"]))


@pytest.mark.parametrize("change", ["figure", "data", "missing", "unsafe"])
def test_static_review_rejects_changed_or_incomplete_assets(project, change):
    assets = project / "docs/assets/evidence-plots"
    review = json.loads((assets / "review.json").read_text())
    if change == "figure":
        (assets / "P001-light.svg").write_text("<svg>changed</svg>")
    elif change == "data":
        (assets / "P001.json").write_text(json.dumps({"id": "P001", "record": {"seed": 99}}))
    elif change == "missing":
        review["expected"].append("P002")
    else:
        review["entries"]["P001"]["assets"]["../outside.json"] = "a" * 64
    (assets / "review.json").write_text(json.dumps(review))
    with pytest.raises(ValueError):
        load(project)


@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize(
    "asset",
    [
        "evidence-ci-1-attempt-1/proof-visualizations.zip",
        "evidence-v0.5.0/proof-visualizations.zip",
        "evidence-v0.5.0/relflow-0.5.0-evidence-plots.zip",
    ],
)
def test_preview_fetch_verifies_archive_and_assets_before_installing(project, monkeypatch, corrupt, asset):
    assets = project / "docs/assets/evidence-plots"
    stream = io.BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as bundle:
        for path in assets.iterdir():
            bundle.writestr(path.name, path.read_bytes())
    payload = stream.getvalue()
    spec = {
        "schema": 1,
        "url": f"https://github.com/relflow/relflow/releases/download/{asset}",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if corrupt:
        payload = payload[:-1] + bytes([payload[-1] ^ 1])
    calls = []

    def open_bundle(query, timeout):
        calls.append(query.full_url)
        assert timeout == 60
        return io.BytesIO(payload)

    monkeypatch.setattr(reviewing.request, "urlopen", open_bundle)
    shutil.rmtree(assets)
    if corrupt:
        with pytest.raises(ValueError, match="pinned checksum"):
            prepare(project, spec)
        assert not assets.exists()
    else:
        assert prepare(project, spec) == assets
        assert load(project) is not None
        assert prepare(project, spec) == assets
        assert len(calls) == 1


def test_paired_bundle_rejects_another_run_or_candidate_record(project, tmp_path):
    from evidence.visuals import verify

    assets = project / "docs/assets/evidence-plots"
    data = json.loads((assets / "P001.json").read_text())
    data["scope"] = "paired Evidence CI candidate"
    (assets / "P001.json").write_text(json.dumps(data))
    review = json.loads((assets / "review.json").read_text())
    review["ci"] = dict(
        run_id="gh-42-2",
        candidate_revision="a" * 40,
        baseline_revision="b" * 40,
        evidence_revision="a" * 40,
        executor_sha="c" * 40,
        capture_sha256="d" * 64,
    )
    entry = review["entries"]["P001"]
    entry["source_sha256"] = hashlib.sha256((assets / "P001-proof.py").read_bytes()).hexdigest()
    entry["assets"]["P001.json"] = hashlib.sha256((assets / "P001.json").read_bytes()).hexdigest()
    (assets / "review.json").write_text(json.dumps(review))
    archive = tmp_path / "plots.zip"
    with ZipFile(archive, "w") as bundle:
        for path in assets.iterdir():
            bundle.write(path, path.name)
    manifest = dict(
        run_id="gh-42-2",
        candidate={"revision": "a" * 40},
        baseline={"revision": "b" * 40},
        evidence_sha="a" * 40,
        executor_sha="c" * 40,
        observations={"capture_sha256": "d" * 64},
    )
    records = [
        dict(task={"id": "P001", "role": "candidate", "script_sha256": entry["source_sha256"]}, result=data["record"])
    ]
    assert verify(archive, manifest, records)["ci"]["run_id"] == "gh-42-2"
    altered = deepcopy(records)
    altered[0]["result"]["checks"] = {"Fabricated pass": True}
    with pytest.raises(ValueError, match="canonical candidate"):
        verify(archive, manifest, altered)
    manifest["run_id"] = "gh-42-3"
    with pytest.raises(ValueError, match="different paired execution"):
        verify(archive, manifest, records)


def test_introduced_plots_authenticate_candidate_only_scope(project, tmp_path):
    from evidence.visuals import verify

    assets = project / "docs/assets/evidence-plots"
    data_path = assets / "P001.json"
    review_path = assets / "review.json"
    data = json.loads(data_path.read_text())
    data["scope"] = "candidate-only Evidence CI proof"
    data_path.write_text(json.dumps(data))
    review = json.loads(review_path.read_text())
    review["ci"] = dict(
        run_id="gh-42-2",
        candidate_revision="a" * 40,
        baseline_revision="b" * 40,
        evidence_revision="a" * 40,
        executor_sha="c" * 40,
        capture_sha256="d" * 64,
        candidate_only=["P001"],
    )
    entry = review["entries"]["P001"]
    entry["source_sha256"] = hashlib.sha256((assets / "P001-proof.py").read_bytes()).hexdigest()
    entry["assets"]["P001.json"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    review_path.write_text(json.dumps(review))
    archive = tmp_path / "plots.zip"
    with ZipFile(archive, "w") as bundle:
        for path in assets.iterdir():
            bundle.write(path, path.name)
    manifest = dict(
        run_id="gh-42-2",
        candidate={"revision": "a" * 40},
        baseline={"revision": "b" * 40},
        evidence_sha="a" * 40,
        executor_sha="c" * 40,
        observations={"capture_sha256": "d" * 64},
        introduced=["P001"],
    )
    records = [
        dict(task={"id": "P001", "role": "candidate", "script_sha256": entry["source_sha256"]}, result=data["record"])
    ]
    assert verify(archive, manifest, records)["ci"]["candidate_only"] == ["P001"]
    manifest["introduced"] = []
    with pytest.raises(ValueError, match="introduced-proof scope"):
        verify(archive, manifest, records)
