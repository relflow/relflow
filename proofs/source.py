"""Read proof documentation and identify executable source without running it."""

from __future__ import annotations

import ast
import hashlib
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
        raise ValueError("Proof script must begin with a '# %% [markdown]' cell containing its YAML header.")
    header: list[str] = []
    opened = False
    for number, line in enumerate(lines[1:], start=2):
        if not line.startswith("#"):
            raise ValueError(f"Proof script YAML header line {number} must be a Python comment beginning with '#'.")
        value = line.removeprefix("#").removeprefix(" ")
        if not opened:
            if not value.strip():
                continue
            if value.strip() != "---":
                raise ValueError("Proof script YAML header must begin with a commented '# ---' delimiter.")
            opened = True
        elif value.strip() == "---":
            try:
                document = yaml.safe_load("\n".join(header))
            except yaml.YAMLError as error:
                raise ValueError(f"Proof script YAML header is invalid: {error}") from error
            if not isinstance(document, dict) or not all(isinstance(key, str) for key in document):
                raise ValueError("Proof script YAML header must be a mapping with string keys.")
            return document
        else:
            header.append(value)
    raise ValueError("Proof script YAML header must end with a commented '# ---' delimiter.")


__all__ = ["fingerprint", "metadata"]
