"""Read experiment frontmatter and identify source without executing it."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any

import yaml


def fingerprint(source: str) -> str:
    """Hash Python syntax while ignoring comments, formatting, and line positions."""
    syntax = ast.dump(ast.parse(source), include_attributes=False)
    return hashlib.sha256(syntax.encode()).hexdigest()


def metadata(source: str) -> dict[str, Any]:
    """Read the commented YAML header in a percent script's first Markdown cell."""
    lines = source.lstrip().splitlines()
    if not lines or lines[0].strip() != "# %% [markdown]":
        raise ValueError("Evidence script must begin with a '# %% [markdown]' cell containing its YAML header.")
    header: list[str] = []
    opened = False
    for number, line in enumerate(lines[1:], start=2):
        if not line.startswith("#"):
            raise ValueError(f"Evidence script YAML header line {number} must be a Python comment beginning with '#'.")
        value = line.removeprefix("#").removeprefix(" ")
        if not opened:
            if not value.strip():
                continue
            if value.strip() != "---":
                raise ValueError("Evidence script YAML header must begin with a commented '# ---' delimiter.")
            opened = True
        elif value.rstrip() == "---":
            try:
                document = yaml.safe_load("\n".join(header))
            except yaml.YAMLError as error:
                raise ValueError(f"Evidence script YAML header is invalid: {error}") from error
            if not isinstance(document, dict) or not all(isinstance(key, str) for key in document):
                raise ValueError("Evidence script YAML header must be a mapping with string keys.")
            return document
        else:
            header.append(value)
    raise ValueError("Evidence script YAML header must end with a commented '# ---' delimiter.")


def frontmatter(source: str, header: dict[str, Any]) -> str:
    """Replace the commented header while preserving every executable byte."""
    metadata(source)
    lines = source.splitlines(keepends=True)
    start, end = [
        index for index, line in enumerate(lines) if line.removeprefix("#").removeprefix(" ").rstrip() == "---"
    ][:2]
    encoded = yaml.safe_dump(header, sort_keys=False, allow_unicode=True, width=110).rstrip()
    comments = "".join("#" + (" " + line if line else "") + "\n" for line in encoded.splitlines())
    return "".join(lines[: start + 1]) + comments + "".join(lines[end:])


def registry(root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Discover canonical experiments directly from their frontmatter."""
    root = root or Path(__file__).resolve().parents[1]
    entries = {}
    for script in sorted((root / "experiments").glob("*/*.py")):
        if script.is_symlink() or script.parent.is_symlink():
            raise ValueError("Canonical experiment sources must be regular files")
        header = metadata(script.read_text())
        identifier = header["evidence-id"]
        if not isinstance(identifier, str) or not re.fullmatch(r"P\d+", identifier) or identifier in entries:
            raise ValueError(f"Invalid or duplicate evidence ID: {identifier}")
        entries[identifier] = {
            "script": script.relative_to(root).as_posix(),
            "page": f"docs/evidence/{script.parent.name}/{script.stem.replace('_', '-')}.py",
            "header": header,
            "run": header.get("evidence-result"),
        }
    return dict(sorted(entries.items()))


__all__ = ["fingerprint", "frontmatter", "metadata", "registry"]
