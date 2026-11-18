"""Read experiment frontmatter and identify source without executing it."""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import re
import symtable
import tomllib
from pathlib import Path
from typing import Any, NamedTuple

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


def environment(source: str) -> dict[str, Any] | None:
    """Read PEP 723 dependencies without importing an experiment or resolving packages."""
    lines = source.splitlines()
    starts = [index for index, line in enumerate(lines) if line == "# /// script"]
    if not starts:
        return None  # Archived suites predate inline dependency declarations.
    if len(starts) != 1:
        raise ValueError("Evidence script must declare exactly one PEP 723 script block")
    content = []
    for line in lines[starts[0] + 1 :]:
        if line == "# ///":
            break
        if line != "#" and not line.startswith("# "):
            raise ValueError("Evidence script PEP 723 metadata must contain commented TOML and end with '# ///'")
        content.append(line[2:] if line != "#" else "")
    else:
        raise ValueError("Evidence script PEP 723 metadata must end with '# ///'")
    try:
        document = tomllib.loads("\n".join(content))
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Evidence script PEP 723 metadata is invalid: {error}") from error
    dependencies = document.get("dependencies")
    if not isinstance(dependencies, list) or any(
        not isinstance(item, str) or not item.strip() for item in dependencies
    ):
        raise ValueError("Evidence script PEP 723 dependencies must be a list of nonempty requirement strings")
    python = document.get("requires-python")
    if python is not None and (not isinstance(python, str) or not python.strip()):
        raise ValueError("Evidence script PEP 723 requires-python must be a nonempty version constraint")
    if "tool" in document and not isinstance(document["tool"], dict):
        raise ValueError("Evidence script PEP 723 tool settings must be a table")
    return document


def tags(source: str) -> dict[str, Any] | None:
    """Read experiment-owned tags while preserving untagged historical suites."""
    declared = environment(source)
    if declared is None:
        return None
    tool = declared.get("tool", {})
    relflow = tool.get("relflow", {})
    if not isinstance(relflow, dict):
        raise ValueError("Evidence tool.relflow settings must be a table")
    classified = relflow.get("evidence")
    if classified is None:
        return None
    if not isinstance(classified, dict) or set(classified) != {"capability", "types", "conditions"}:
        raise ValueError("Evidence tags must declare capability, types, and conditions in [tool.relflow.evidence]")
    if not isinstance(classified["capability"], str) or not classified["capability"].strip():
        raise ValueError("Evidence capability must be a nonempty name")
    for name in ("types", "conditions"):
        values = classified[name]
        if (
            not isinstance(values, list)
            or (name == "conditions" and not values)
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"Evidence {name} must be a list of unique nonempty tag names")
    return classified


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
                document = yaml.load("\n".join(header), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
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


def component(source: str, name: str) -> ast.FunctionDef:
    """Read one isolated diagnostic component without evaluating the experiment module."""
    definitions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(definitions) != 1:
        raise ValueError(f"Diagnostic renderer {name!r} must name one top-level function")
    definition = definitions[0]
    args = definition.args
    if (
        [arg.arg for arg in args.args] != ["bundle", "spec"]
        or args.posonlyargs
        or args.kwonlyargs
        or args.vararg
        or args.kwarg
        or args.defaults
        or definition.decorator_list
    ):
        raise ValueError(f"Diagnostic renderer {name!r} requires the signature (bundle, spec) without decorators")
    scope = symtable.symtable(ast.unparse(definition), "<diagnostic component>", "exec").get_children()[0]
    scopes = [scope]
    unresolved = set()
    while scopes:
        current = scopes.pop()
        unresolved.update(
            symbol.get_name() for symbol in current.get_symbols() if symbol.is_global() and symbol.is_referenced()
        )
        scopes.extend(current.get_children())
    unresolved -= {*vars(builtins), name}
    if unresolved:
        raise ValueError(
            f"Diagnostic renderer {name!r} must own its imports and helpers; "
            f"unresolved module names: {', '.join(sorted(unresolved))}"
        )
    return definition


def view(source: str) -> dict[str, Any]:
    """Read an experiment's diagnostic plan without importing its model code."""
    plan = metadata(source).get("evidence-plot")
    if not isinstance(plan, dict):
        raise ValueError("Experiment frontmatter must declare an evidence-plot mapping")
    for name in ("focus", "comparison", "limit"):
        if not isinstance(plan.get(name), str) or not plan[name].strip():
            raise ValueError(f"evidence-plot.{name} must be a nonempty string")
    supplements = plan.get("supplements", {})
    if not isinstance(supplements, dict) or any(
        not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9-]*", name) is None or not isinstance(selection, dict)
        for name, selection in supplements.items()
    ):
        raise ValueError("evidence-plot.supplements must map safe figure names to renderer selections")
    for selection in [plan, *supplements.values()]:
        selected = selection.keys() & {"mode", "renderer"}
        if len(selected) != 1:
            raise ValueError("Each evidence-plot figure must select exactly one mode or renderer")
        name = selection[next(iter(selected))]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each evidence-plot figure needs a nonempty mode or renderer name")
        if "renderer" in selection:
            component(source, name)
    return plan


def registry(
    root: Path | None = None,
    *,
    publication: Path | None = None,
    measurements: bool = True,
) -> dict[str, dict[str, Any]]:
    """Discover the complete experiment catalog without importing model code."""
    root = root or Path(__file__).resolve().parents[1]
    selected = publication or root / "results.json"
    recorded = (
        snapshot(selected)
        if measurements and (publication is not None or selected.is_file())
        else {"report": {}, "results": {}}
    )
    entries = {}
    seen = set()
    directory = root / "experiments"
    nested = list(directory.glob("*/*.py"))
    if nested:
        raise ValueError(f"Experiment {nested[0]} must live directly in experiments/ as its P identifier")
    for script in sorted(directory.glob("P*.py")):
        if script.is_symlink() or script.parent.is_symlink() or script.parent.parent.is_symlink():
            raise ValueError("Canonical experiment sources must be regular files")
        source = script.read_text()
        header = describe(source)
        tags(source)
        identifier = header["evidence-id"]
        if identifier in seen:
            raise ValueError(f"Invalid or duplicate evidence ID: {identifier}")
        seen.add(identifier)
        if script.name != f"{identifier}.py":
            raise ValueError(f"Experiment {identifier}: filename must be {identifier}.py, got {script.name}")
        requirements = header.get("evidence-requires", [])
        if not isinstance(requirements, list) or any(
            not isinstance(item, str) or not item.strip() for item in requirements
        ):
            raise ValueError(f"{identifier}: evidence-requires must be a list of nonempty resource descriptions")
        entries[identifier] = {
            "script": script.relative_to(root).as_posix(),
            "page": f"docs/evidence/{identifier}.py",
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
    "environment",
    "fingerprint",
    "frontmatter",
    "metadata",
    "registry",
    "snapshot",
    "tags",
]
