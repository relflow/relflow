"""Build and validate the Quarto site for one Read the Docs version."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from docs.check import ROOT, rendered_checks


def publish(root: Path = ROOT, *, environ: Mapping[str, str] | None = None, quarto: str | None = None) -> Path:
    """Publish only validated HTML, with URLs and source links bound to this checkout."""
    root = root.resolve()
    environment = dict(os.environ if environ is None else environ)
    site_url = environment.get("READTHEDOCS_CANONICAL_URL", "")
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
    site_url = site_url.rstrip("/") + "/"
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

    subprocess.run(["make", "check"], cwd=root, env=environment, check=True)
    with tempfile.NamedTemporaryFile(
        mode="w", prefix="_quarto-readthedocs-", suffix=".yml", dir=root / "docs", delete=False
    ) as handle:
        yaml.safe_dump(
            {"website": {"site-url": site_url, "repo-branch": revision}, "canonical-url": True},
            handle,
        )
        profile = Path(handle.name)
    environment["QUARTO_PROFILE"] = profile.stem.removeprefix("_quarto-")
    try:
        command = ["make", "render"]
        if quarto:
            command.append(f"QUARTO={shlex.quote(quarto)}")
        subprocess.run(command, cwd=root, env=environment, check=True)
    finally:
        profile.unlink(missing_ok=True)
    errors = rendered_checks(root, site_url=site_url)
    if errors:
        raise ValueError("Read the Docs rendered site failed validation:\n" + "\n".join(errors))
    shutil.copytree(root / "docs/site", destination, dirs_exist_ok=True)
    print(f"Published validated documentation for {site_url} to {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarto", help="Use an installed Quarto executable instead of the pinned make default")
    arguments = parser.parse_args()
    publish(quarto=arguments.quarto)


if __name__ == "__main__":
    main()
