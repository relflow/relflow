"""Discover branch documentation and assemble one complete Pages site."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote


def discover(repository: Path) -> dict[str, list[dict[str, str]]]:
    refs = subprocess.check_output(
        ["git", "for-each-ref", "--format=%(refname)%00%(objectname)%00%(symref)", "refs/remotes/origin/"],
        cwd=repository,
        text=True,
    )
    branches = []
    for line in refs.splitlines():
        ref, commit, symbolic = line.split("\0")
        if symbolic:
            continue
        name = ref.removeprefix("refs/remotes/origin/")
        if not any(
            subprocess.run(
                ["git", "cat-file", "-e", f"{commit}:{config}"],
                cwd=repository,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
            for config in ("docs/_quarto.yml", "mkdocs.yml")
        ):
            print(f"Skipping {name}: no Quarto or MkDocs configuration.", file=sys.stderr)
            continue
        destination = "" if name == "main" else f"branches/{name}"
        branches.append(
            {
                "name": name,
                "ref": commit,
                "artifact": f"docs-{len(branches)}",
                "destination": destination,
                "url": f"https://relflow.github.io/relflow/{quote(destination, safe='/')}".rstrip("/") + "/",
            }
        )
    if not any(branch["name"] == "main" for branch in branches):
        raise ValueError("The main branch must contain documentation to publish the site root.")
    return {"include": sorted(branches, key=lambda branch: (branch["name"] != "main", branch["name"]))}


def assemble(matrix: dict[str, list[dict[str, str]]], artifacts: Path, site: Path) -> None:
    # Check every build before touching the publication directory.
    for branch in matrix["include"]:
        source = artifacts / branch["artifact"]
        if not (source / "index.html").is_file():
            raise ValueError(f"Missing rendered index.html for branch {branch['name']} in {source}.")

    site.mkdir(parents=True, exist_ok=False)
    for branch in sorted(matrix["include"], key=lambda branch: len(branch["destination"])):
        source = artifacts / branch["artifact"]
        destination = site / branch["destination"]
        if destination != site and destination.exists():
            raise ValueError(f"Branch {branch['name']} would overwrite existing documentation at {destination}.")
        shutil.copytree(source, destination, dirs_exist_ok=destination == site)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    discovery = commands.add_parser("discover")
    discovery.add_argument("repository", type=Path, nargs="?", default=Path.cwd())
    assembly = commands.add_parser("assemble")
    assembly.add_argument("artifacts", type=Path)
    assembly.add_argument("site", type=Path)
    args = parser.parse_args()
    if args.command == "discover":
        print(json.dumps(discover(args.repository), separators=(",", ":")))
    else:
        assemble(json.loads(os.environ["DOCS_MATRIX"]), args.artifacts, args.site)


if __name__ == "__main__":
    main()
