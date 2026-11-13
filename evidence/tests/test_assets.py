"""Published documentation consumes frozen, checked assets without executing models."""

import hashlib
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from evidence import assets
from evidence.experiments.source import metadata, registry, snapshot


@pytest.fixture
def released(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion="0.4.3"\n')
    family = tmp_path / "evidence/experiments/example"
    family.mkdir(parents=True)
    (family / "signal.py").write_text(
        "# %% [markdown]\n# ---\n# evidence-id: P001\n# ---\nraise RuntimeError('do not execute')\n"
    )
    bootstrap = {"schema": 1, "version": "0.4.2", "revision": "b" * 40, "report": {}, "results": {}}
    (tmp_path / "evidence/results.json").write_text(json.dumps(bootstrap))
    revision = "a" * 40
    tag = "evidence-v0.4.3"
    selected = {
        "schema": 1,
        "version": "0.4.3",
        "revision": revision,
        "report": {"passed": True, "release": f"https://github.com/relflow/relflow/releases/tag/{tag}"},
        "results": {"P001": {"mode": "full", "outcome": "met", "seed": 7, "metrics": {"score": 0.9}}},
    }
    content = json.dumps(selected).encode()
    url = f"https://github.com/relflow/relflow/releases/download/{tag}/publication.json"
    package = {"tag_name": "v0.4.3", "target_commitish": revision, "draft": False}
    report = {
        "tag_name": tag,
        "target_commitish": revision,
        "draft": False,
        "assets": [
            {
                "name": "publication.json",
                "browser_download_url": url,
                "size": len(content),
                "digest": "sha256:" + hashlib.sha256(content).hexdigest(),
            }
        ],
    }
    responses = {
        "https://api.github.com/repos/relflow/relflow/releases/tags/v0.4.3": package,
        "https://api.github.com/repos/relflow/relflow/releases/tags/" + tag: report,
        url: content,
    }
    calls = []

    def download(location, **kwargs):
        calls.append(location)
        value = responses[location]
        return value if isinstance(value, bytes) or value is None else json.dumps(value).encode()

    monkeypatch.setattr(assets, "download", download)
    return tmp_path, responses, selected, report, calls, url


def test_release_asset_is_digest_checked_cached_and_keeps_sources_unchanged(released):
    root, _, _, _, calls, url = released
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    result = assets.fetch(root, required=True)
    assert result is not None and result.is_relative_to(root / ".cache")
    assert assets.fetch(root, required=True) == result
    assert calls.count(url) == 1
    assert all(path.read_bytes() == content for path, content in before.items())
    result.write_text("corrupt cache")
    assets.fetch(root, required=True)
    assert calls.count(url) == 2
    assert json.loads(result.read_text())["results"]["P001"]["seed"] == 7


@pytest.mark.parametrize(
    "fault", ["digest", "size", "origin", "version", "revision", "report", "smoke", "partial", "nan"]
)
def test_invalid_release_material_never_replaces_the_snapshot(released, fault):
    root, responses, selected, report, _, url = released
    asset = report["assets"][0]
    if fault == "digest":
        asset["digest"] = "sha256:" + "0" * 64
    elif fault == "size":
        asset["size"] += 1
    elif fault == "origin":
        asset["browser_download_url"] = "https://example.invalid/publication.json"
    else:
        if fault == "version":
            selected["version"] = "0.4.4"
        elif fault == "revision":
            selected["revision"] = "c" * 40
        elif fault == "report":
            selected["report"]["passed"] = False
        elif fault == "smoke":
            selected["results"]["P001"]["mode"] = "smoke"
        elif fault == "partial":
            selected["results"]["P002"] = selected["results"]["P001"]
        else:
            selected["results"]["P001"]["metrics"]["score"] = float("nan")
        content = json.dumps(selected).encode()
        responses[url] = content
        asset["digest"] = "sha256:" + hashlib.sha256(content).hexdigest()
        asset["size"] = len(content)
    before = (root / "evidence/results.json").read_bytes()
    with pytest.raises(ValueError, match="Evidence|publication"):
        assets.fetch(root, required=True)
    assert (root / "evidence/results.json").read_bytes() == before
    assert not list((root / ".cache").rglob("*.json"))


def test_unreleased_preview_uses_latest_released_measurements(released):
    root, responses, _, _, _, _ = released
    (root / "pyproject.toml").write_text('[project]\nversion="0.4.4"\n')
    responses["https://api.github.com/repos/relflow/relflow/releases/tags/v0.4.4"] = None
    responses["https://api.github.com/repos/relflow/relflow/releases/latest"] = responses[
        "https://api.github.com/repos/relflow/relflow/releases/tags/v0.4.3"
    ]
    selected = assets.fetch(root)
    assert selected is not None
    assert json.loads(selected.read_text())["version"] == "0.4.3"


def test_existing_bootstrap_version_needs_no_remote_report(released):
    root, _, _, _, calls, _ = released
    assert assets.fetch(root, "0.4.2") is None
    assert not calls


def test_unreleased_first_version_uses_bootstrap_without_changing_it(released):
    root, responses, _, _, _, _ = released
    responses["https://api.github.com/repos/relflow/relflow/releases/tags/v0.4.3"] = None
    responses["https://api.github.com/repos/relflow/relflow/releases/latest"] = {"tag_name": "v0.4.2"}
    assert assets.fetch(root) is None


def test_http_failure_is_not_mistaken_for_an_unreleased_version(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise HTTPError("https://api.github.com/", 403, "rate limited", {}, None)

    monkeypatch.setattr(assets, "urlopen", unavailable)
    with pytest.raises(HTTPError):
        assets.download("https://api.github.com/", missing=True)


def test_download_is_bounded_before_parsing(monkeypatch):
    monkeypatch.setattr(assets, "LIMIT", 8)
    monkeypatch.setattr(assets, "urlopen", lambda *args, **kwargs: io.BytesIO(b"x" * 9))
    with pytest.raises(ValueError, match="exceeds 8 bytes"):
        assets.download("https://github.com/")


def test_bootstrap_preserves_historical_measurements_outside_canonical_sources():
    root = Path(__file__).resolve().parents[1]
    selected = snapshot(root / "results.json")
    entries = registry(root)
    assert selected["version"] == "0.4.2"
    assert len(selected["results"]) == 79
    # The bootstrap predates the Cluster-to-Group migration. Its retired
    # measurements retain their original identities outside the live catalog.
    assert selected["results"].keys() - entries.keys() == {"P018", "P019", "P020", "P065", "P078"}
    for identifier, entry in entries.items():
        assert not {"evidence-result", "evidence-report"} & metadata((root / entry["script"]).read_text()).keys()
        if identifier in selected["results"]:
            assert entry["run"] == selected["results"][identifier]
            assert entry["report"] == selected["report"]
