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
from evidence.diagnostics.review import fragment, load, prepare, selection
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
    run = deepcopy(json.loads((ROOT / "evidence/results.json").read_text())["results"]["P001"])
    run.update(outcome="not_met", checks={"Unmet unit fixture": False})
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


def test_authored_views_cover_both_catalog_lanes():
    specs = json.loads((ROOT / "evidence/diagnostics/views.json").read_text())
    catalog = registry(ROOT / "evidence", measurements=False, lane="all")
    assert set(specs) == set(catalog)
    assert all(spec["focus"] and spec["comparison"] and spec["limit"] for spec in specs.values())
    assert {key for key, spec in specs.items() if spec["mode"] == "research"} == {
        key for key, entry in catalog.items() if entry["lane"] == "research"
    }


def test_figures_require_the_identical_selected_record_and_keep_unmet_checks(project):
    review = load(project)
    selected = selection(project)
    results = json.loads(selected.read_text())["results"]
    run = results["P001"]
    assert run["outcome"] == "not_met" and not all(run["checks"].values())
    markup = fragment(project, "P001", run, review)
    assert "unpaired visualization run" in markup
    assert "P001-record.json" in markup
    altered = deepcopy(run)
    altered["seed"] += 1
    assert fragment(project, "P001", altered, review) == ""
    assert fragment(project, "P001", None, review) == ""


def test_illustrations_do_not_gain_a_learning_claim(project):
    assets = project / "docs/assets/evidence-plots"
    data = json.loads((assets / "P001.json").read_text())
    data["record"] = None
    (assets / "P001.json").write_text(json.dumps(data))
    review = json.loads((assets / "review.json").read_text())
    review["entries"]["P001"]["assets"]["P001.json"] = hashlib.sha256((assets / "P001.json").read_bytes()).hexdigest()
    (assets / "review.json").write_text(json.dumps(review))
    markup = fragment(project, "P001", None, load(project))
    assert "Generator illustration" in markup and "No full learning result" in markup
    assert "-record.json" not in markup


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


def test_selection_rejects_changed_computation(project):
    catalog = registry(project / "evidence", measurements=False)
    source = project / "evidence" / catalog["P001"]["script"]
    source.write_text(source.read_text().replace("def run(", "def changed_run("))
    with pytest.raises(ValueError, match="computation changed"):
        selection(project)


@pytest.mark.parametrize("corrupt", [False, True])
def test_preview_fetch_verifies_archive_and_assets_before_installing(project, monkeypatch, corrupt):
    assets = project / "docs/assets/evidence-plots"
    stream = io.BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as bundle:
        for path in assets.iterdir():
            bundle.writestr(path.name, path.read_bytes())
    payload = stream.getvalue()
    spec = {
        "schema": 1,
        "url": "https://github.com/relflow/relflow/releases/download/docs-preview-unit/bundle.zip",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    (project / "evidence/diagnostics").mkdir()
    (project / "evidence/diagnostics/preview.json").write_text(json.dumps(spec))
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
            prepare(project)
        assert not assets.exists()
    else:
        assert prepare(project) == assets
        assert load(project) is not None
        assert prepare(project) == assets
        assert len(calls) == 1
