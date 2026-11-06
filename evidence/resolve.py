"""Resolve the locked runtime graph against the explicit Linux CPU Torch backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

UV_VERSION = "0.13.7"
PLATFORM = "x86_64-manylinux_2_28"
PYTHON_VERSION = "3.12"


def pins(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9.!+_-]+)(?:\s*;[^\r\n]+)?", line)
        if match is None:
            raise ValueError(f"Runtime requirements must contain exact registry pins: {line!r}")
        name = re.sub(r"[-_.]+", "-", match[1]).lower()
        result.setdefault(name, set()).add(match[2])
    if not result:
        raise ValueError(f"Runtime requirements are empty: {path}")
    return result


def check(constraints: Path, requirements: Path) -> dict[str, str]:
    locked, selected = pins(constraints), pins(requirements)
    if any(len(versions) != 1 for versions in selected.values()):
        raise ValueError("The Linux CPU runtime must select exactly one version per distribution")
    resolved = {name: next(iter(versions)) for name, versions in selected.items()}
    if not resolved.get("torch", "").endswith("+cpu"):
        raise ValueError("The resolved Torch wheel must explicitly use the +cpu backend")
    for name, version in resolved.items():
        original = version.removesuffix("+cpu") if name == "torch" else version
        if original not in locked.get(name, set()):
            raise ValueError(f"CPU runtime {name}=={version} differs from the candidate's locked versions")
    return resolved


def execute(command: list[str]) -> None:
    for attempt in range(3):
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            return
        if attempt == 2:
            raise subprocess.CalledProcessError(result.returncode, command)
        time.sleep(5)


def resolve(project: Path, output: Path) -> dict:
    project, output = project.resolve(), output.resolve()
    version = subprocess.check_output(["uv", "--version"], text=True).strip()
    if version.split()[:2] != ["uv", UV_VERSION]:
        raise ValueError(f"Evidence runtime resolution requires uv {UV_VERSION}; found {version}")
    names = ("runtime-constraints.txt", "requirements.txt", "resolution.json", "uv-version.txt")
    if any((output / name).exists() for name in names):
        raise ValueError("Use fresh runtime resolution inputs; never overwrite an earlier attempt")
    output.mkdir(parents=True, exist_ok=True)
    constraints, requirements = output / names[0], output / names[1]
    execute(
        [
            "uv",
            "export",
            "--project",
            str(project),
            "--locked",
            "--no-dev",
            "--no-emit-project",
            "--no-hashes",
            "--no-header",
            "--no-annotate",
            "--format",
            "requirements-txt",
            "--output-file",
            str(constraints),
        ]
    )
    execute(
        [
            "uv",
            "pip",
            "compile",
            str(project / "pyproject.toml"),
            "--constraints",
            str(constraints),
            "--torch-backend",
            "cpu",
            "--python-version",
            PYTHON_VERSION,
            "--python-platform",
            PLATFORM,
            "--only-binary",
            ":all:",
            "--no-header",
            "--no-annotate",
            "--output-file",
            str(requirements),
        ]
    )
    resolved = check(constraints, requirements)
    receipt = {
        "schema": 1,
        "uv_version": UV_VERSION,
        "python_version": PYTHON_VERSION,
        "platform": PLATFORM,
        "torch_backend": "cpu",
        "packages": resolved,
        "uv_lock_sha256": hashlib.sha256((project / "uv.lock").read_bytes()).hexdigest(),
        "constraints_sha256": hashlib.sha256(constraints.read_bytes()).hexdigest(),
        "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
    }
    (output / "uv-version.txt").write_text(version + "\n")
    (output / "resolution.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(resolve(args.project, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
