"""Build and validate the static Starlight site for one Read the Docs version."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from collections.abc import Mapping
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

from evidence.assets import fetch
from evidence.experiments.source import snapshot

ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    """Collect actual HTML links without interpreting code samples or scripts."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.canonical: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        identifier = attributes.get("id") or (attributes.get("name") if tag == "a" else None)
        if identifier:
            self.ids.add(identifier)
        if tag == "link" and attributes.get("rel") == "canonical":
            self.canonical.append(attributes.get("href") or "")
            return
        for name in ("href", "src", "poster", "action"):
            value = attributes.get(name)
            if value:
                self.links.append(value)


def validate(directory: Path, site_url: str, revision: str) -> None:
    """Check per-page canonical URLs, local files/fragments, and source revisions."""
    origin = urlsplit(site_url)
    errors = []
    pages = {}
    for path in sorted(directory.rglob("*.html")):
        page = Page()
        page.feed(path.read_text())
        pages[path.resolve()] = page
    for path, page in pages.items():
        relative = path.relative_to(directory.resolve()).as_posix()
        address = urljoin(site_url, relative.removesuffix("index.html"))
        expected = address.removesuffix(".html") + "/" if address.endswith(".html") else address
        if page.canonical != [expected]:
            errors.append(f"{relative}: canonical URL must be {expected}; received {page.canonical}")
        for target in page.links:
            url = urlsplit(urljoin(address, target))
            if (url.scheme, url.netloc) != (origin.scheme, origin.netloc):
                continue
            if not url.path.startswith(origin.path):
                errors.append(f"{relative}: local link escapes documentation version: {target}")
                continue
            destination = (directory / unquote(url.path.removeprefix(origin.path))).resolve()
            if not destination.is_relative_to(directory.resolve()):
                errors.append(f"{relative}: local link escapes static output: {target}")
                continue
            if destination.is_dir():
                destination /= "index.html"
            if not destination.is_file():
                errors.append(f"{relative}: local link has no built file: {target}")
            elif url.fragment and destination in pages and unquote(url.fragment) not in pages[destination].ids:
                errors.append(f"{relative}: local fragment has no matching element: {target}")
        for target in page.links:
            marker = "https://github.com/relflow/relflow/blob/"
            if target.startswith(marker) and "/docs/starlight/" in target:
                if not target.startswith(f"{marker}{revision}/docs/starlight/"):
                    errors.append(f"{relative}: documentation source link differs from checkout: {target}")
    if not (directory / "index.html").is_file():
        errors.append("Static documentation build has no index.html")
    if errors:
        raise ValueError("Read the Docs rendered site failed validation:\n" + "\n".join(errors))


def canonical(site_url: str) -> str:
    """Normalize the absolute version URL shared by building and static validation."""
    message = "READTHEDOCS_CANONICAL_URL must be an absolute HTTPS version URL without credentials, query, or fragment"
    try:
        location = urlsplit(site_url)
        port = location.port
    except ValueError as error:
        raise ValueError(message) from error
    if (
        location.scheme != "https"
        or not location.hostname
        or port == 0
        or location.username is not None
        or location.password is not None
        or location.query
        or location.fragment
        or any(character.isspace() for character in site_url)
    ):
        raise ValueError(message)
    return site_url.rstrip("/") + "/"


def publish(root: Path = ROOT, *, environ: Mapping[str, str] | None = None) -> Path:
    """Copy validated HTML with URLs and source links bound to this checkout."""
    root = root.resolve()
    environment = dict(os.environ if environ is None else environ)
    site_url = canonical(environment.get("READTHEDOCS_CANONICAL_URL", ""))
    output = environment.get("READTHEDOCS_OUTPUT", "")
    if not output:
        raise ValueError("READTHEDOCS_OUTPUT must name the build output directory supplied by Read the Docs")
    requested = Path(os.path.abspath(root / output / "html"))
    destination = requested.resolve()
    for path in (requested, destination):
        if path.is_relative_to(root) and not path.is_relative_to(root / "_readthedocs"):
            raise ValueError(
                "READTHEDOCS_OUTPUT/html must be outside the checkout or under its _readthedocs/ directory"
            )
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"Read the Docs output must be empty: {destination}; choose a fresh build output directory")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    expected = environment.get("READTHEDOCS_GIT_COMMIT_HASH", revision)
    if expected != revision:
        raise ValueError(f"Read the Docs revision {expected!r} does not match checkout {revision!r}")
    environment.update(RELFLOW_DOCS_URL=site_url, RELFLOW_DOCS_REVISION=revision, ASTRO_TELEMETRY_DISABLED="1")

    if (root / "evidence/experiments").is_dir():
        version = (
            environment.get("READTHEDOCS_GIT_IDENTIFIER")
            if environment.get("READTHEDOCS_VERSION_TYPE") == "tag"
            else None
        )
        # Hosted versions always select formal releases, never inherited CI overrides.
        environment.pop("RELFLOW_EVIDENCE_PUBLICATION", None)
        publication = fetch(root, version.removeprefix("v") if version else None, required=bool(version))
        if publication is not None:
            if version and snapshot(publication)["revision"] != revision:
                raise ValueError("Released evidence does not match the documentation tag's source revision")
            environment["RELFLOW_EVIDENCE_PUBLICATION"] = str(publication)

    site = root / "docs/starlight"
    for command in (["npm", "run", "check"], ["npm", "test"], ["npm", "run", "build"]):
        subprocess.run(command, cwd=site, env=environment, check=True)
    validate(site / "dist", site_url, revision)
    shutil.copytree(site / "dist", destination, dirs_exist_ok=True)
    print(f"Published validated documentation for {site_url} to {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Validate existing static output without building or fetching"
    )
    parser.add_argument(
        "--directory", type=Path, default=ROOT / "docs/starlight/dist", help="Static output to validate"
    )
    arguments = parser.parse_args()
    if arguments.check:
        address = canonical(os.environ.get("RELFLOW_DOCS_URL", "https://www.relflow.ai/"))
        revision = (
            os.environ.get("RELFLOW_DOCS_REVISION")
            or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        )
        validate(arguments.directory, address, revision)
        print(f"Validated static documentation at {arguments.directory}")
    else:
        publish()


if __name__ == "__main__":
    main()
