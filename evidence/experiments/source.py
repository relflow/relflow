"""Read experiment frontmatter and identify source without executing it."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal, NamedTuple

import yaml


class Experiment(NamedTuple):
    """Immutable execution declaration shared by static discovery and reporting."""

    id: str
    seed: int


def check_experiment(identifier: object, seed: object) -> Experiment:
    if not isinstance(identifier, str) or re.fullmatch(r"P\d+", identifier) is None:
        raise ValueError(f"Experiment ID must be a stable P identifier, got {identifier!r}")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError(f"Experiment {identifier}: seed must be an integer in 0..2^32-1, got {seed!r}")
    return Experiment(identifier, seed)


def declaration(source: str) -> Experiment | None:
    """Read one literal decorator without importing or executing the experiment.

    An absent declaration identifies the archived frontmatter-based format.
    Malformed declarations must never fall back to that historical reader.
    """
    syntax = ast.parse(source)
    declared = [
        (node, decorator)
        for node in ast.walk(syntax)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for decorator in node.decorator_list
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "experiment"
        )
        or (isinstance(decorator, ast.Name) and decorator.id == "experiment")
    ]
    if not declared:
        return None
    if len(declared) != 1:
        raise ValueError("Each experiment script must declare exactly one @experiment")
    function, decorator = declared[0]
    if not isinstance(function, ast.FunctionDef) or function not in syntax.body:
        raise ValueError("@experiment must decorate a synchronous top-level function")
    arguments = function.args
    if (
        [argument.arg for argument in arguments.args] != ["seed", "steps", "accelerator"]
        or arguments.posonlyargs
        or arguments.kwonlyargs
        or arguments.vararg
        or arguments.kwarg
        or arguments.defaults
    ):
        raise ValueError("@experiment requires the signature (seed, steps, accelerator)")
    if (
        not isinstance(decorator, ast.Call)
        or len(decorator.args) != 1
        or isinstance(decorator.args[0], ast.Starred)
        or len(decorator.keywords) != 1
        or decorator.keywords[0].arg != "seed"
    ):
        raise ValueError("Declare @experiment('P001', seed=123) with one ID and one seed")
    try:
        identifier = ast.literal_eval(decorator.args[0])
        seed = ast.literal_eval(decorator.keywords[0].value)
    except (ValueError, TypeError) as error:
        raise ValueError("@experiment ID and seed must be literals for static discovery") from error
    return check_experiment(identifier, seed)


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


def describe(source: str) -> dict[str, Any]:
    """Bind presentation metadata to the statically declared experiment identity."""
    header = metadata(source)
    declared = declaration(source)
    identifier = declared.id if declared is not None else header.get("evidence-id")
    if declared is not None and header.get("evidence-id", identifier) != identifier:
        raise ValueError(f"Frontmatter evidence ID differs from @experiment {identifier}")
    if not isinstance(identifier, str) or re.fullmatch(r"P\d+", identifier) is None:
        raise ValueError(f"Invalid evidence ID: {identifier}")
    return {**header, "evidence-id": identifier}


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


def snapshot(path: Path) -> dict[str, Any]:
    """Read a finite result selection without importing any experiment or model."""
    document = json.loads(path.read_text())
    if (
        not isinstance(document, dict)
        or set(document) != {"schema", "version", "revision", "report", "results"}
        or type(document["schema"]) is not int
        or document["schema"] != 1
        or not isinstance(document["version"], str)
        or not isinstance(document["revision"], str)
        or not isinstance(document["report"], dict)
        or not isinstance(document["results"], dict)
        or any(
            not re.fullmatch(r"P\d+", key) or not isinstance(value, dict) for key, value in document["results"].items()
        )
    ):
        raise ValueError(
            f"Evidence publication {path} must contain a versioned result selection keyed by P identifiers"
        )
    try:
        json.dumps(document, allow_nan=False)
    except ValueError as error:
        raise ValueError(f"Evidence publication {path} contains nonfinite measurements") from error
    return document


def registry(
    root: Path | None = None,
    *,
    publication: Path | None = None,
    measurements: bool = True,
    lane: Literal["core", "research", "all"] = "core",
) -> dict[str, dict[str, Any]]:
    """Discover static experiment declarations; hosted execution selects the core lane."""
    if lane not in {"core", "research", "all"}:
        raise ValueError(f"Evidence lane must be core, research, or all; got {lane!r}")
    root = root or Path(__file__).resolve().parents[1]
    selected = publication or root / "results.json"
    recorded = (
        snapshot(selected)
        if measurements and (publication is not None or selected.is_file())
        else {"report": {}, "results": {}}
    )
    entries = {}
    seen = set()
    pages = set()
    scripts = [
        (script, kind)
        for directory, kind in (("experiments", "core"), ("research", "research"))
        for script in sorted((root / directory).glob("*/*.py"))
    ]
    for script, kind in scripts:
        if script.is_symlink() or script.parent.is_symlink() or script.parent.parent.is_symlink():
            raise ValueError("Canonical experiment sources must be regular files")
        header = describe(script.read_text())
        identifier = header["evidence-id"]
        if "evidence-lane" in header:
            raise ValueError(
                f"{identifier}: the experiment directory owns its lane; remove evidence-lane from frontmatter"
            )
        if identifier in seen:
            raise ValueError(f"Invalid or duplicate evidence ID: {identifier}")
        seen.add(identifier)
        page = f"docs/evidence/{script.parent.name}/{script.stem.replace('_', '-')}.py"
        if page in pages:
            raise ValueError(f"{identifier}: another experiment already uses page {page}")
        pages.add(page)
        requirements = header.get("evidence-requires", [])
        if not isinstance(requirements, list) or any(
            not isinstance(item, str) or not item.strip() for item in requirements
        ):
            raise ValueError(f"{identifier}: evidence-requires must be a list of nonempty resource descriptions")
        if kind == "research" and not requirements:
            raise ValueError(f"{identifier}: research experiments must declare evidence-requires")
        if lane != "all" and kind != lane:
            continue
        entries[identifier] = {
            "script": script.relative_to(root).as_posix(),
            "page": page,
            "lane": kind,
            "header": header,
            "run": recorded["results"].get(identifier),
            "report": recorded["report"],
        }
    return dict(sorted(entries.items()))


__all__ = [
    "Experiment",
    "check_experiment",
    "declaration",
    "describe",
    "fingerprint",
    "frontmatter",
    "metadata",
    "registry",
    "snapshot",
]
