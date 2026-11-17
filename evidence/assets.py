"""Fetch a released publication as bounded static data, without model dependencies."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from evidence.experiments.source import describe, registry, snapshot
from evidence.releases import VERSION, inventory, supported

REPOSITORY = "relflow/relflow"
SHA = re.compile(r"[0-9a-f]{40}")
LIMIT = 8 * 1024 * 1024


def download(url: str, *, missing: bool = False, limit: int = LIMIT) -> bytes | None:
    """Bound public asset reads and distinguish an unreleased version from an outage."""
    try:
        with urlopen(Request(url, headers={"User-Agent": "relflow-evidence-publication"}), timeout=30) as response:
            content = response.read(limit + 1)
    except HTTPError as error:
        if missing and error.code == 404:
            return None
        raise
    if len(content) > limit:
        raise ValueError(f"Evidence publication response exceeds {limit} bytes: {url}")
    return content


def fetch(root: Path, version: str | None = None, *, required: bool = False) -> Path | None:
    """Resolve only the evidence frozen for a published package version."""
    if version is not None and not VERSION.fullmatch(version):
        raise ValueError(f"Evidence publication requires a normalized package version, got {version!r}")
    if version is not None and not supported(version):
        raise ValueError(f"Evidence publication requires relflow >=0.5.0, got {version}")
    endpoint = f"https://api.github.com/repos/{REPOSITORY}/releases/tags/"
    address = endpoint + "v" + version if version else f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
    package = download(address, missing=not required)
    if package is None:
        print("No matching formal release is available; Evidence measurements remain pending")
        return None
    if version is None:
        latest = json.loads(package).get("tag_name", "")
        if not latest.startswith("v") or not VERSION.fullmatch(latest[1:]):
            raise ValueError("Latest package release does not identify a normalized version")
        version = latest[1:]
    if not supported(version):
        if required:
            raise ValueError(f"Evidence publication requires relflow >=0.5.0, got {version}")
        print("No relflow >=0.5.0 release is available; Evidence measurements remain pending")
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
    files = inventory(version, (asset["name"] for asset in report.get("assets", [])))
    assets = [asset for asset in report.get("assets", []) if asset.get("name") == files["publication"]]
    if len(assets) != 1 or not re.fullmatch(r"sha256:[0-9a-f]{64}", assets[0].get("digest") or ""):
        raise ValueError(f"Evidence {tag} requires one {files['publication']} asset with a GitHub SHA256 digest")
    asset = assets[0]
    url = f"https://github.com/{REPOSITORY}/releases/download/{tag}/{files['publication']}"
    if (
        asset.get("browser_download_url") != url
        or type(asset.get("size")) is not int
        or not 0 <= asset["size"] <= LIMIT
    ):
        raise ValueError(f"Evidence {tag} publication asset has an unexpected URL or size")
    destination = root / ".cache/evidence" / tag / "publication.json"
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
        sources(selected, report, destination.parent)
        plots(root, selected, report)
        pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)
    print(f"Using frozen {tag} publication for {revision}")
    return destination


def sources(selected: dict, release: dict, destination: Path) -> None:
    """Recover executed declarations from the digest-bound CI report, never from Git blobs."""
    files = inventory(selected["version"], (asset["name"] for asset in release.get("assets", [])))
    matches = [asset for asset in release.get("assets", []) if asset.get("name") == files["reports"]]
    expected = selected["report"].get("reports_sha256", "")
    url = f"https://github.com/{REPOSITORY}/releases/download/evidence-v{selected['version']}/{files['reports']}"
    limit = 64 * 1024 * 1024
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected)
        or len(matches) != 1
        or matches[0].get("digest") != "sha256:" + expected
        or matches[0].get("browser_download_url") != url
        or type(matches[0].get("size")) is not int
        or not 0 < matches[0]["size"] <= limit
    ):
        raise ValueError("Publication sources require the matching digest-bound CI reports asset")
    cached = destination / files["reports"]
    content = cached.read_bytes() if cached.is_file() else None
    if content is None or hashlib.sha256(content).hexdigest() != expected:
        content = download(url, limit=limit)
    if content is None or len(content) != matches[0]["size"] or hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("Publication sources differ from the retained CI reports asset")
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        staging = Path(temporary) / "source"
        staging.mkdir()
        total = 0
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as reports:
            for member in reports:
                if not member.name.startswith("source/"):
                    continue
                if (
                    not re.fullmatch(r"source/(?:experiments|research)/[a-z0-9_]+/[a-z0-9_]+\.py", member.name)
                    or not member.isfile()
                ):
                    # Shared runner modules and non-source report files are not documentation inputs.
                    if ".." in member.name.split("/") or member.issym() or member.islnk():
                        raise ValueError(f"Unsafe publication source: {member.name}")
                    continue
                total += member.size
                if member.size > 1024 * 1024 or total > 16 * 1024 * 1024:
                    raise ValueError("Publication source declarations exceed the extraction bound")
                target = staging / member.name.removeprefix("source/")
                if target.exists():
                    raise ValueError(f"Duplicate publication source: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = reports.extractfile(member)
                assert stream is not None
                target.write_bytes(stream.read())
        entries = {}
        for source in sorted(staging.glob("*/*/*.py")):
            identifier = describe(source.read_text())["evidence-id"]
            if identifier in entries:
                raise ValueError(f"Duplicate publication source identifier: {identifier}")
            entries[identifier] = source
        for identifier, record in selected["results"].items():
            source = entries.get(identifier)
            if source is None or hashlib.sha256(source.read_bytes()).hexdigest() != record.get("provenance", {}).get(
                "script_sha256"
            ):
                raise ValueError(f"{identifier}: publication source differs from its frozen measurement")
        shutil.rmtree(destination / "source", ignore_errors=True)
        staging.replace(destination / "source")
    cached.write_bytes(content)


def plots(root: Path, selected: dict, release: dict) -> None:
    """Install only the plots bound to this exact publication and GitHub asset."""
    spec = selected["report"].get("plots")
    target = root / "docs/assets/evidence-plots/review.json"
    target.unlink(missing_ok=True)
    if spec is None:
        return
    files = inventory(selected["version"], (asset["name"] for asset in release.get("assets", [])))
    matches = [asset for asset in release.get("assets", []) if asset.get("name") == files["plots"]]
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
    from evidence.experiments.source import fingerprint

    for identifier, entry in review["entries"].items():
        data = json.loads((target.parent / entry["data"]).read_text())
        if data.get("record") is None:
            continue
        source = target.parent / f"{identifier}-proof.py"
        if (
            not source.is_file()
            or source.name not in entry["assets"]
            or fingerprint(source.read_text()) != entry["code_sha256"]
            or hashlib.sha256(source.read_bytes()).hexdigest()
            != data["record"].get("provenance", {}).get("script_sha256")
        ):
            raise ValueError(f"{identifier}: publication plot source differs from its frozen measurement")
        results[identifier] = data["record"]
    if results != selected["results"]:
        raise ValueError("Publication plots differ from the exact selected candidate records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version")
    parser.add_argument("--required", action="store_true")
    arguments = parser.parse_args()
    fetch(Path(__file__).resolve().parents[1], arguments.version, required=arguments.required)
