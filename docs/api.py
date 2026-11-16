"""Extract API reference data from Python source without importing the SDK."""

from __future__ import annotations

import ast
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import griffe


def target(obj: griffe.Object | griffe.Alias) -> griffe.Object:
    """Resolve imports, retaining ordinary type aliases as declarations."""
    return obj.final_target if obj.is_alias else obj


def resolve(package: griffe.Module, path: str) -> griffe.Object | None:
    if not path.startswith(f"{package.path}."):
        return None
    try:
        return target(package[path.removeprefix(f"{package.path}.")])
    except (KeyError, griffe.AliasResolutionError):
        return None


def bases(package: griffe.Module, obj: griffe.Class) -> Iterable[griffe.Class]:
    """Follow local bases, including the RequestBase = Leaf type alias."""
    for expression in obj.bases:
        base = resolve(package, expression.canonical_path)
        seen: set[str] = set()
        while base is not None and base.is_attribute and base.path not in seen:
            seen.add(base.path)
            base = resolve(package, getattr(base.value, "canonical_path", ""))
        if base is not None and base.is_class:
            yield base


def members(package: griffe.Module, obj: griffe.Class) -> dict[str, griffe.Object]:
    """Use Python's local C3 order without loading external base classes."""

    def linearize(node: griffe.Class) -> list[griffe.Class]:
        parents = list(bases(package, node))
        queues = [linearize(parent) for parent in parents] + [parents.copy()]
        order = [node]
        while queues := [queue for queue in queues if queue]:
            candidate = next((queue[0] for queue in queues if all(queue[0] not in other[1:] for other in queues)), None)
            if candidate is None:
                raise ValueError(f"{node.path}: inconsistent source inheritance order")
            order.append(candidate)
            for queue in queues:
                if queue[0] is candidate:
                    queue.pop(0)
        return order

    result: dict[str, griffe.Object] = {}
    for node in reversed(linearize(obj)):
        for name, member in node.members.items():
            if not member.is_alias:
                result[name] = member
    return result


def location(obj: griffe.Object, root: Path) -> dict[str, Any]:
    return {"file": obj.filepath.relative_to(root).as_posix(), "line": obj.lineno or 1}


def documentation(obj: griffe.Object) -> dict[str, Any]:
    """Parse source-authored Google-style sections and preserve the remaining prose."""
    result: dict[str, Any] = {"text": "", "parameters": {}, "returns": [], "raises": []}
    if not obj.docstring:
        return result
    text = []
    for section in obj.docstring.parse(griffe.Parser.google, warnings=False):
        kind = section.kind.value
        if kind in {"parameters", "other parameters", "attributes"}:
            result["parameters"].update({item.name.lstrip("*"): item.description for item in section.value})
        elif kind in {"returns", "yields", "receives", "raises"}:
            result["raises" if kind == "raises" else "returns"].extend(
                {"type": str(item.annotation or ""), "description": item.description} for item in section.value
            )
        elif kind == "examples":
            for example_kind, value in section.value:
                text.append(f"```pycon\n{value}\n```" if example_kind == "examples" else value)
        elif isinstance(section.value, str):
            text.append((f"### {section.title}\n\n" if section.title else "") + section.value)
        else:
            # Admonitions carry a description rather than a plain string.
            description = getattr(section.value, "description", "")
            if description:
                text.append(description)
    result["text"] = "\n\n".join(text)
    return result


def default(value: str | None) -> dict[str, Any]:
    """Read Field/dataclass defaults as expressions; never call a factory."""
    result = {"value": value, "factory": False, "description": ""}
    if value is None:
        return result
    expression = ast.parse(value, mode="eval").body
    if isinstance(expression, ast.Call) and ast.unparse(expression.func).split(".")[-1] == "cast":
        return default(ast.unparse(expression.args[-1]))
    if not isinstance(expression, ast.Call) or ast.unparse(expression.func).split(".")[-1] not in {"Field", "field"}:
        return result
    result["value"] = ast.unparse(expression.args[0]) if expression.args else None
    for option in expression.keywords:
        if option.arg == "default":
            result["value"] = ast.unparse(option.value)
        elif option.arg == "default_factory":
            result.update(value=f"{ast.unparse(option.value)}()", factory=True)
        elif option.arg == "description" and isinstance(option.value, ast.Constant):
            result["description"] = option.value.value
    if result["value"] == "...":
        result["value"] = None
    return result


def field(member: griffe.Attribute, root: Path, *, owner: griffe.Class) -> dict[str, Any]:
    value = default(str(member.value) if member.value is not None else None)
    return {
        "name": member.name,
        "type": str(member.annotation or ""),
        "default": value["value"],
        "factory": value["factory"],
        "description": documentation(member)["text"] or value["description"],
        "property": "property" in member.labels,
        "inherited": member.parent is not owner,
        "source": location(member, root),
    }


def function(obj: griffe.Function, name: str, root: Path, *, constructor: bool = False) -> dict[str, Any]:
    docs = documentation(obj)
    parameters = [parameter for parameter in obj.parameters if parameter.name not in {"self", "cls"}]
    pieces = []
    keyword_marker = False
    for index, parameter in enumerate(parameters):
        kind = parameter.kind.value
        if kind == "keyword-only" and not keyword_marker:
            pieces.append("*")
            keyword_marker = True
        prefix = "**" if kind == "variadic keyword" else "*" if kind == "variadic positional" else ""
        if prefix == "*":
            keyword_marker = True
        piece = prefix + parameter.name
        if parameter.annotation is not None:
            piece += f": {parameter.annotation}"
        if parameter.default is not None and not prefix:
            piece += f" = {parameter.default}"
        pieces.append(piece)
        if kind == "positional-only" and (
            index + 1 == len(parameters) or parameters[index + 1].kind.value != "positional-only"
        ):
            pieces.append("/")
    returns = f" -> {obj.returns}" if not constructor and obj.returns is not None else ""
    code = f"{name}({', '.join(pieces)}){returns}"
    if len(code) > 100:
        code = f"{name}(\n" + "".join(f"    {piece},\n" for piece in pieces) + f"){returns}"
    return {
        "name": name,
        "code": code,
        "text": docs["text"],
        "parameters": [
            {
                "name": parameter.name,
                "type": str(parameter.annotation or ""),
                "default": str(parameter.default) if parameter.default is not None else None,
                "kind": parameter.kind.value,
                "description": docs["parameters"].get(parameter.name, ""),
            }
            for parameter in parameters
        ],
        "returns": docs["returns"]
        or ([{"type": str(obj.returns), "description": ""}] if not constructor and obj.returns is not None else []),
        "raises": docs["raises"],
        "source": location(obj, root),
    }


def reference(package: griffe.Module, name: str, root: Path) -> dict[str, Any]:
    path = name.replace("rf.", "", 1) if name.startswith("rf.") else name.replace("model.", "Model.", 1)
    exported = package[path]
    result: dict[str, Any] = {
        "name": name,
        "kind": "",
        "text": "",
        "definition": "",
        "signatures": [],
        "overloads": [],
        "fields": [],
        "methods": [],
        "external": "",
    }
    try:
        obj = target(exported)
    except griffe.AliasResolutionError:
        if not exported.is_alias or exported.target_path.startswith("relflow."):
            raise
        # The local re-export is authoritative; dependency APIs remain upstream.
        result.update(
            kind="re-export",
            external=exported.target_path,
            source={"file": exported.parent.filepath.relative_to(root).as_posix(), "line": exported.alias_lineno or 1},
            definition=f"from {exported.target_path.rsplit('.', 1)[0]} import {exported.target_path.rsplit('.', 1)[1]}",
        )
        return result

    result.update(kind=obj.kind.value, source=location(obj, root), canonicalPath=obj.path)
    documented = obj
    if obj.is_class and not obj.docstring:
        queue = list(bases(package, obj))
        while queue:
            parent = queue.pop(0)
            if parent.docstring:
                documented = parent
                break
            queue.extend(bases(package, parent))
    result.update(text=documentation(documented)["text"], documentationSource=location(documented, root))
    if obj.is_function:
        signature = function(obj, name, root)
        result.update(signatures=[signature])
        result["overloads"] = [function(overload, name, root) for overload in obj.overloads or []]
    elif obj.is_class:
        available = members(package, obj)
        constructor = available.get("__init__")
        if constructor and constructor.is_function:
            result["signatures"] = [function(constructor, name, root, constructor=True)]
            result["overloads"] = [
                function(overload, name, root, constructor=True) for overload in constructor.overloads or []
            ]
        else:
            suffix = f"({', '.join(str(base) for base in obj.bases)})" if obj.bases else ""
            result["definition"] = f"class {obj.name}{suffix}: ..."
        for key, member in available.items():
            if key.startswith("_") or key in {"model_config", "model_fields"}:
                continue
            if member.is_attribute:
                result["fields"].append(field(member, root, owner=obj))
            elif member.is_function and member.docstring and not key.startswith("check_"):
                result["methods"].append(function(member, f"{name}.{key}", root))
    elif obj.is_attribute or obj.is_type_alias:
        annotation = getattr(obj, "annotation", None)
        value = str(getattr(obj, "value", ""))
        result["definition"] = f"{obj.name}{f': {annotation}' if annotation else ''} = {value}"
        aliased = resolve(package, getattr(getattr(obj, "value", None), "canonical_path", ""))
        if aliased is not None:
            result["aliasOf"] = aliased.path
    elif obj.is_module:
        result["definition"] = f"import {obj.path}"
    return result


def extract(root: Path, names: list[str]) -> list[dict[str, Any]]:
    package = griffe.load(
        "relflow",
        search_paths=[root / "src"],
        allow_inspection=False,
        resolve_aliases=True,
        resolve_external=False,
        resolve_implicit=True,
    )
    return [reference(package, name, root) for name in names]


if __name__ == "__main__":
    print(json.dumps(extract(Path(__file__).resolve().parent.parent, json.load(sys.stdin))))
