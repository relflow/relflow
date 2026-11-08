"""Fetch a released publication as bounded static data, without model dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import tomllib
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from evidence.experiments.source import registry, snapshot

REPOSITORY = "relflow/relflow"
VERSION = re.compile(r"[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?")
SHA = re.compile(r"[0-9a-f]{40}")
LIMIT = 8 * 1024 * 1024


def download(url: str, *, missing: bool = False) -> bytes | None:
    """Bound public asset reads and distinguish an unreleased version from an outage."""
    try:
        with urlopen(Request(url, headers={"User-Agent": "relflow-evidence-publication"}), timeout=30) as response:
            content = response.read(LIMIT + 1)
    except HTTPError as error:
        if missing and error.code == 404:
            return None
        raise
    if len(content) > LIMIT:
        raise ValueError(f"Evidence publication response exceeds {LIMIT} bytes: {url}")
    return content


def fetch(root: Path, version: str | None = None, *, required: bool = False) -> Path | None:
    """Resolve only the evidence frozen for a published package version."""
    version = version or tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    if not VERSION.fullmatch(version):
        raise ValueError(f"Evidence publication requires a normalized package version, got {version!r}")
    baseline = root / "evidence/results.json"
    if baseline.is_file() and snapshot(baseline)["version"] == version and not required:
        return None
    endpoint = f"https://api.github.com/repos/{REPOSITORY}/releases/tags/"
    package = download(endpoint + "v" + version, missing=not required)
    if package is None:
        package = download(f"https://api.github.com/repos/{REPOSITORY}/releases/latest", missing=True)
        if package is None:
            print(f"relflow {version} is unreleased; using the recorded bootstrap evidence snapshot")
            return None
        latest = json.loads(package).get("tag_name", "")
        if not latest.startswith("v") or not VERSION.fullmatch(latest[1:]):
            raise ValueError("Latest package release does not identify a normalized version")
        version = latest[1:]
        if baseline.is_file() and snapshot(baseline)["version"] == version:
            return None
    released = json.loads(package)
    revision = released.get("target_commitish", "")
    if released.get("tag_name") != "v" + version or released.get("draft") is not False or not SHA.fullmatch(revision):
        raise ValueError(f"relflow {version} has no published release bound to a full source revision")
    tag = "evidence-v" + version
    retained = download(endpoint + tag)
    assert retained is not None
    report = json.loads(retained)
    if report.get("draft") is not False or report.get("tag_name") != tag or report.get("target_commitish") != revision:
        raise ValueError(f"Evidence {tag} does not identify the published package source {revision}")
    assets = [asset for asset in report.get("assets", []) if asset.get("name") == "publication.json"]
    if len(assets) != 1 or not re.fullmatch(r"sha256:[0-9a-f]{64}", assets[0].get("digest") or ""):
        raise ValueError(f"Evidence {tag} requires one publication.json asset with a GitHub SHA256 digest")
    asset = assets[0]
    url = f"https://github.com/{REPOSITORY}/releases/download/{tag}/publication.json"
    if (
        asset.get("browser_download_url") != url
        or type(asset.get("size")) is not int
        or not 0 <= asset["size"] <= LIMIT
    ):
        raise ValueError(f"Evidence {tag} publication asset has an unexpected URL or size")
    destination = root / ".cache/evidence" / (tag + ".json")
    expected = asset["digest"].removeprefix("sha256:")
    content = destination.read_bytes() if destination.is_file() else None
    if content is None or hashlib.sha256(content).hexdigest() != expected:
        content = download(url)
    if content is None or len(content) != asset["size"] or hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Evidence {tag} publication bytes differ from the retained asset")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        pending = Path(handle.name)
        handle.write(content)
    try:
        selected = snapshot(pending)
        if (
            selected["version"] != version
            or selected["revision"] != revision
            or selected["report"].get("passed") is not True
            or selected["report"].get("release") != f"https://github.com/{REPOSITORY}/releases/tag/{tag}"
            or not selected["results"]
            or any(
                result.get("mode") != "full" or result.get("outcome") == "error"
                for result in selected["results"].values()
            )
        ):
            raise ValueError(f"Evidence {tag} publication does not contain its verified full-run selection")
        if required and selected["results"].keys() != registry(root / "evidence").keys():
            raise ValueError(f"Evidence {tag} publication does not cover the release's complete experiment catalog")
        plots(root, selected, report)
        pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)
    print(f"Using frozen {tag} publication for {revision}")
    return destination


def plots(root: Path, selected: dict, release: dict) -> None:
    """Install only the plots bound to this exact publication and GitHub asset."""
    spec = selected["report"].get("plots")
    target = root / "docs/assets/evidence-plots/review.json"
    target.unlink(missing_ok=True)
    if spec is None:
        return
    matches = [asset for asset in release.get("assets", []) if asset.get("name") == "proof-visualizations.zip"]
    if len(matches) != 1 or any(
        [
            matches[0].get("digest") != "sha256:" + spec.get("sha256", ""),
            matches[0].get("size") != spec.get("size"),
            matches[0].get("browser_download_url") != spec.get("url"),
        ]
    ):
        raise ValueError("Publication plots differ from the retained GitHub release asset")
    from evidence.diagnostics.review import load, prepare

    prepare(root, spec)
    review = load(root)
    if review is None or review.get("ci", {}).get("run_id") != selected["report"]["run_id"]:
        raise ValueError("Publication plots identify a different paired Evidence run")
    results = {}
    catalog = registry(root / "evidence", measurements=False, lane="all")
    from evidence.experiments.source import fingerprint

    for identifier, entry in review["entries"].items():
        if (
            identifier not in catalog
            or fingerprint((root / "evidence" / catalog[identifier]["script"]).read_text()) != entry["code_sha256"]
        ):
            raise ValueError(f"{identifier}: publication plot computation differs from this checkout")
        data = json.loads((target.parent / entry["data"]).read_text())
        if data.get("record") is not None:
            results[identifier] = data["record"]
    if results != selected["results"]:
        raise ValueError("Publication plots differ from the exact selected candidate records")


def ci(root: Path, revision: str) -> Path:
    """Select the newest exact-head attempt; pending or failed attempts never fall back."""
    if not SHA.fullmatch(revision):
        raise ValueError("CI documentation requires a full checkout revision")
    endpoint = f"https://api.github.com/repos/{REPOSITORY}/"
    payload = download(endpoint + f"actions/workflows/ci.yml/runs?head_sha={revision}&per_page=100")
    runs = [
        run
        for run in json.loads(payload)["workflow_runs"]
        if run.get("head_sha") == revision
        and run.get("head_repository", {}).get("full_name") == REPOSITORY
        and run.get("event") in {"pull_request", "push", "workflow_dispatch"}
    ]
    run = max(runs, key=lambda row: row["id"]) if runs else None
    selected = {
        "schema": 1,
        "version": "ci-pending",
        "revision": revision,
        "report": {"status": "Evidence CI has not published plots for this commit yet."},
        "results": {},
    }
    (root / "docs/assets/evidence-plots/review.json").unlink(missing_ok=True)
    if run:
        tag = f"evidence-ci-{run['id']}-attempt-{run['run_attempt']}"
        payload = download(endpoint + "releases/tags/" + tag, missing=True)
        if payload is not None:
            release = json.loads(payload)
            if (
                release.get("draft") is not False
                or release.get("target_commitish") != revision
                or release.get("tag_name") != tag
            ):
                raise ValueError("CI plots release does not identify this exact source revision")
            matches = [asset for asset in release["assets"] if asset["name"] == "publication.json"]
            if len(matches) != 1:
                raise ValueError("CI plots release requires its single publication selection")
            asset = matches[0]
            url = f"https://github.com/{REPOSITORY}/releases/download/{tag}/publication.json"
            if asset.get("browser_download_url") != url or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", asset.get("digest") or ""
            ):
                raise ValueError("CI publication requires a checksum-bound repository asset")
            content = download(url)
            if len(content) != asset["size"] or "sha256:" + hashlib.sha256(content).hexdigest() != asset["digest"]:
                raise ValueError("CI publication differs from its GitHub asset digest")
            selected = json.loads(content)
            if (
                selected.get("schema") != 1
                or selected.get("version") != "ci-preview"
                or selected.get("revision") != revision
                or selected.get("report", {}).get("run_id") != f"gh-{run['id']}-{run['run_attempt']}"
            ):
                raise ValueError("CI publication belongs to another commit or attempt")
            plots(root, selected, release)
        else:
            selected["report"]["status"] = (
                f"Evidence CI run {run['id']}, attempt {run['run_attempt']}: "
                + (run.get("conclusion") or run.get("status", "pending"))
                + ". Plots are not available yet."
            )
    destination = root / ".cache/evidence/ci-preview.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(selected, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version")
    parser.add_argument("--required", action="store_true")
    arguments = parser.parse_args()
    fetch(Path(__file__).resolve().parents[1], arguments.version, required=arguments.required)
