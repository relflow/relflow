"""Published documentation consumes frozen, checked assets without executing models."""

import hashlib
import io
import json
import tarfile
from pathlib import Path
from urllib.error import HTTPError

import pytest

from evidence import assets
from evidence.experiments.source import metadata, registry
from evidence.releases import filenames


@pytest.fixture
def released(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion="0.6.0"\n')
    family = tmp_path / "evidence/experiments"
    family.mkdir(parents=True)
    (family / "P001.py").write_text(
        "# %% [markdown]\n# ---\n# evidence-id: P001\n# ---\nraise RuntimeError('do not execute')\n"
    )
    revision = "a" * 40
    tag = "evidence-v0.6.0"
    selected = {
        "schema": 1,
        "version": "0.6.0",
        "revision": revision,
        "report": {"passed": True, "release": f"https://github.com/relflow/relflow/releases/tag/{tag}"},
        "results": {
            "P001": {
                "mode": "full",
                "outcome": "met",
                "seed": 7,
                "metrics": {"score": 0.9},
                "provenance": {"script_sha256": hashlib.sha256((family / "P001.py").read_bytes()).hexdigest()},
            }
        },
    }
    bundle = io.BytesIO()
    with tarfile.open(fileobj=bundle, mode="w:gz") as archive:
        content = (family / "P001.py").read_bytes()
        member = tarfile.TarInfo("source/experiments/P001.py")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    reports = bundle.getvalue()
    selected["report"]["reports_sha256"] = hashlib.sha256(reports).hexdigest()
    content = json.dumps(selected).encode()
    url = f"https://github.com/relflow/relflow/releases/download/{tag}/{filenames('0.6.0')['publication']}"
    package = {"tag_name": "v0.6.0", "target_commitish": revision, "draft": False}
    report = {
        "tag_name": tag,
        "target_commitish": revision,
        "draft": False,
        "assets": [
            {
                "name": filenames("0.6.0")["publication"],
                "browser_download_url": url,
                "size": len(content),
                "digest": "sha256:" + hashlib.sha256(content).hexdigest(),
            }
        ],
    }
    responses = {
        "https://api.github.com/repos/relflow/relflow/releases/tags/v0.6.0": package,
        "https://api.github.com/repos/relflow/relflow/releases/latest": package,
        "https://api.github.com/repos/relflow/relflow/releases/tags/" + tag: report,
        url: content,
    }
    reports_url = url.removesuffix(filenames("0.6.0")["publication"]) + filenames("0.6.0")["reports"]
    report["assets"].append(
        {
            "name": filenames("0.6.0")["reports"],
            "browser_download_url": reports_url,
            "size": len(reports),
            "digest": "sha256:" + hashlib.sha256(reports).hexdigest(),
        }
    )
    responses[reports_url] = reports
    calls = []

    def urlopen(request, **kwargs):
        location = request.full_url
        calls.append(location)
        value = responses[location]
        if value is None:
            raise HTTPError(location, 404, "not found", {}, None)
        return io.BytesIO(value if isinstance(value, bytes) else json.dumps(value).encode())

    monkeypatch.setattr(assets, "urlopen", urlopen)
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


def test_release_accepts_publication_larger_than_sixteen_mib(released):
    root, responses, _, report, _, url = released
    content = responses[url] + b" " * (16 * 1024 * 1024)
    responses[url] = content
    report["assets"][0].update(size=len(content), digest="sha256:" + hashlib.sha256(content).hexdigest())

    path = assets.fetch(root, required=True)

    assert path is not None and path.stat().st_size == len(content)


def test_release_rejects_oversized_publication_before_download(released):
    root, _, _, report, calls, url = released
    report["assets"][0]["size"] = 32 * 1024 * 1024 + 1

    with pytest.raises(ValueError, match="unexpected URL or size"):
        assets.fetch(root, required=True)

    assert url not in calls


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
            selected["version"] = "0.6.1"
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
    before = (root / "evidence/experiments/P001.py").read_bytes()
    with pytest.raises(ValueError, match="Evidence|publication"):
        assets.fetch(root, required=True)
    assert (root / "evidence/experiments/P001.py").read_bytes() == before
    assert not list((root / ".cache").rglob("*.json"))


@pytest.mark.parametrize("checkout", ["0.6.0", "0.6.1", "0.7.0"])
def test_preview_uses_latest_formal_release_independent_of_checkout_version(released, checkout):
    root, responses, _, _, _, _ = released
    (root / "pyproject.toml").write_text(f'[project]\nversion="{checkout}"\n')
    responses["https://api.github.com/repos/relflow/relflow/releases/tags/v0.6.1"] = None
    responses["https://api.github.com/repos/relflow/relflow/releases/latest"] = responses[
        "https://api.github.com/repos/relflow/relflow/releases/tags/v0.6.0"
    ]
    selected = assets.fetch(root)
    assert selected is not None
    assert json.loads(selected.read_text())["version"] == "0.6.0"


def test_unreleased_first_version_keeps_measurements_pending(released):
    root, responses, _, _, _, _ = released
    responses["https://api.github.com/repos/relflow/relflow/releases/latest"] = None
    assert assets.fetch(root) is None
    assert not (root / "evidence/results.json").exists()
    assert not (root / ".cache").exists()


def test_pre_06_release_does_not_supply_current_measurements(released):
    root, responses, _, _, calls, _ = released
    responses["https://api.github.com/repos/relflow/relflow/releases/latest"] = {"tag_name": "v0.5.4"}
    assert assets.fetch(root) is None
    assert calls == ["https://api.github.com/repos/relflow/relflow/releases/latest"]
    with pytest.raises(ValueError, match="requires relflow >=0.6.0"):
        assets.fetch(root, required=True)
    with pytest.raises(ValueError, match="requires relflow >=0.6.0"):
        assets.fetch(root, "0.5.4", required=True)
    assert not (root / ".cache").exists()


def test_http_failure_is_not_mistaken_for_an_unreleased_version(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise HTTPError("https://api.github.com/", 403, "rate limited", {}, None)

    monkeypatch.setattr(assets, "urlopen", unavailable)
    with pytest.raises(HTTPError):
        assets.download("https://api.github.com/", missing=True)


def test_download_is_bounded_before_parsing(monkeypatch):
    monkeypatch.setattr(assets, "urlopen", lambda *args, **kwargs: io.BytesIO(b"x" * 9))
    with pytest.raises(ValueError, match="exceeds 8 bytes"):
        assets.download("https://github.com/", limit=8)


@pytest.mark.parametrize(("status", "headers", "delay"), [(502, {}, 1), (403, {"Retry-After": "2"}, 2)])
def test_download_retries_transient_responses(monkeypatch, status, headers, delay):
    sleeps = []
    monkeypatch.setattr(assets.time, "sleep", sleeps.append)

    def urlopen(request, **kwargs):
        if not sleeps:
            raise HTTPError(request.full_url, status, "temporary failure", headers, None)
        return io.BytesIO(b"publication")

    monkeypatch.setattr(assets, "urlopen", urlopen)
    assert assets.download("https://api.github.com/repos/relflow/relflow/releases/latest") == b"publication"
    assert sleeps == [delay]


def test_download_fails_after_bounded_attempts(monkeypatch):
    sleeps = []
    monkeypatch.setattr(assets.time, "sleep", sleeps.append)

    def unavailable(request, **kwargs):
        raise HTTPError(request.full_url, 503, "still unavailable", {}, None)

    monkeypatch.setattr(assets, "urlopen", unavailable)
    with pytest.raises(HTTPError, match="503"):
        assets.download("https://api.github.com/repos/relflow/relflow/releases/latest", missing=True)
    assert sleeps == [1, 4]


def test_download_authenticates_only_release_metadata_without_forwarding_credentials(monkeypatch):
    requests = []
    monkeypatch.setenv("RELFLOW_EVIDENCE_GITHUB_TOKEN", "fixture-token")

    def urlopen(request, **kwargs):
        requests.append(request)
        return io.BytesIO(b"publication")

    monkeypatch.setattr(assets, "urlopen", urlopen)
    assets.download("https://api.github.com/repos/relflow/relflow/releases/latest")
    assets.download("https://github.com/relflow/relflow/releases/download/evidence-v0.5.4/publication.json")
    assets.download("https://www.relflow.ai/latest/evidence/selected.json")
    assert requests[0].unredirected_hdrs["Authorization"] == "Bearer fixture-token"
    assert all("Authorization" not in request.headers for request in requests)
    assert all(not request.has_header("Authorization") for request in requests[1:])


def test_canonical_sources_have_no_checked_in_measurements():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "results.json").exists()
    for entry in registry(root).values():
        assert entry["run"] is None
        assert not {"evidence-result", "evidence-report"} & metadata((root / entry["script"]).read_text()).keys()


def test_publication_requires_matching_plot_asset_before_downloading(tmp_path):
    selected = {"version": "0.6.0", "report": {"plots": {"sha256": "a" * 64, "size": 10, "url": "wrong"}}}
    with pytest.raises(ValueError, match="retained GitHub release asset"):
        assets.plots(tmp_path, selected, {"assets": []})


@pytest.mark.parametrize("corrupt", [False, True])
def test_released_plots_bind_to_frozen_source_instead_of_newer_checkout(tmp_path, monkeypatch, corrupt):
    from evidence.diagnostics import review as reviewing
    from evidence.experiments.source import fingerprint

    directory = tmp_path / "docs/assets/evidence-plots"
    directory.mkdir(parents=True)
    source = "# A frozen proof\nvalue = 1\n"
    record = {"provenance": {"script_sha256": hashlib.sha256(source.encode()).hexdigest()}, "metrics": {"score": 0.8}}
    (directory / "P001-proof.py").write_text(source if not corrupt else source + "value = 2\n")
    (directory / "P001.json").write_text(json.dumps({"record": record}))
    (directory / "P999.json").write_text(json.dumps({"record": None}))
    review = {
        "ci": {"run_id": "gh-123-1"},
        "entries": {
            "P001": {
                "data": "P001.json",
                "assets": {"P001-proof.py": record["provenance"]["script_sha256"]},
                "code_sha256": fingerprint(source),
            },
            "P999": {"data": "P999.json", "assets": {}, "code_sha256": "unmeasured illustration"},
        },
    }
    name = filenames("0.6.0")["plots"]
    spec = {
        "url": f"https://github.com/relflow/relflow/releases/download/evidence-v0.6.0/{name}",
        "sha256": "a" * 64,
        "size": 10,
    }
    selected = {"version": "0.6.0", "report": {"run_id": "gh-123-1", "plots": spec}, "results": {"P001": record}}
    release = {
        "assets": [
            {
                "name": name,
                "browser_download_url": spec["url"],
                "digest": "sha256:" + spec["sha256"],
                "size": spec["size"],
            }
        ]
    }
    monkeypatch.setattr(reviewing, "prepare", lambda *args: directory)
    monkeypatch.setattr(reviewing, "load", lambda *args: review)
    if corrupt:
        with pytest.raises(ValueError, match="frozen measurement"):
            assets.plots(tmp_path, selected, release)
    else:
        assets.plots(tmp_path, selected, release)


@pytest.mark.parametrize("fault", ["asset", "bytes", "record", "missing", "unsafe", "duplicate", "oversized"])
def test_report_sources_reject_unbound_changed_and_unsafe_material(released, fault):
    root, responses, selected, release, _, _ = released
    destination = root / "cache"
    destination.mkdir()
    asset = release["assets"][1]
    url = asset["browser_download_url"]
    if fault == "asset":
        asset["digest"] = "sha256:" + "0" * 64
    elif fault == "bytes":
        responses[url] = b"corrupt report"
    elif fault == "record":
        selected["results"]["P001"]["provenance"]["script_sha256"] = "0" * 64
    else:
        source = (root / "evidence/experiments/P001.py").read_bytes()
        name = "source/experiments/P001.py"
        if fault == "missing":
            name = "ignored.txt"
        elif fault == "unsafe":
            name = "source/experiments/../outside.py"
        elif fault == "oversized":
            source = b"x" * (1024 * 1024 + 1)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for _ in range(2 if fault == "duplicate" else 1):
                member = tarfile.TarInfo(name)
                member.size = len(source)
                archive.addfile(member, io.BytesIO(source))
        content = buffer.getvalue()
        responses[url] = content
        digest = hashlib.sha256(content).hexdigest()
        asset.update(digest="sha256:" + digest, size=len(content))
        selected["report"]["reports_sha256"] = digest
    with pytest.raises(ValueError, match="source|Source"):
        assets.sources(selected, release, destination)
    assert not (destination / "source").exists()
    assert not (destination / "outside.py").exists()


@pytest.mark.parametrize("duplicate", [False, True])
def test_frozen_sources_keep_historical_paths_and_reject_duplicate_ids(released, duplicate):
    root, responses, selected, release, _, _ = released
    source = (root / "evidence/experiments/P001.py").read_bytes()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for collection in ["research", "experiments"] if duplicate else ["research"]:
            member = tarfile.TarInfo(f"source/{collection}/example/signal.py")
            member.size = len(source)
            archive.addfile(member, io.BytesIO(source))
    content = buffer.getvalue()
    asset = release["assets"][1]
    responses[asset["browser_download_url"]] = content
    digest = hashlib.sha256(content).hexdigest()
    asset.update(digest="sha256:" + digest, size=len(content))
    selected["report"]["reports_sha256"] = digest
    destination = root / "cache"
    destination.mkdir()
    if duplicate:
        with pytest.raises(ValueError, match="Duplicate publication source identifier"):
            assets.sources(selected, release, destination)
        assert not (destination / "source").exists()
    else:
        assets.sources(selected, release, destination)
        assert (destination / "source/research/example/signal.py").read_bytes() == source


def test_cached_source_tampering_is_repaired_from_the_verified_ci_report(released):
    root, _, _, _, calls, _ = released
    path = assets.fetch(root, required=True)
    source = path.parent / "source/experiments/P001.py"
    expected = source.read_bytes()
    source.write_text("changed source")
    assets.fetch(root, required=True)
    assert source.read_bytes() == expected
    assert sum(url.endswith("evidence-reports.tar.gz") for url in calls) == 1
