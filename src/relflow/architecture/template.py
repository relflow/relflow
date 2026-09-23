"""Read named YAML templates into the existing model construction inputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import pydantic
import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from relflow import presets
from relflow.structs.enums import AttentionInput
from relflow.structs.experiment import TreeFieldInput
from relflow.structs.reduction import ReductionConfig
from relflow.structs.structure import Branch
from relflow.structs.tree import Mask, MaskInput, Rate
from relflow.tensorfields.base import TENSORFIELDS

if TYPE_CHECKING:
    from relflow.architecture.root import OptimizerConfig, SchedulerConfig

__all__ = ["Options", "read"]


class Options(TypedDict, total=False):
    """Root architecture and runtime overrides; fields and preset belong to the file."""

    d_model: int
    n_layers: int
    n_heads: int
    batch_size: int
    query: str | None
    description: str | None
    embed: bool
    attention: AttentionInput
    reduction: ReductionConfig | None
    dropout: Rate | None
    mask: MaskInput
    optimizer: OptimizerConfig | None
    scheduler: SchedulerConfig | None


class Loader(yaml.SafeLoader):
    """Reject duplicate keys before a YAML mapping can discard schema intent."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict:
        self.flatten_mapping(node)
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as error:
                raise ConstructorError(
                    "while reading a model template",
                    node.start_mark,
                    "mapping keys must be hashable",
                    key_node.start_mark,
                ) from error
            if duplicate:
                raise ConstructorError(
                    "while reading a model template", node.start_mark, f"duplicate key {key!r}", key_node.start_mark
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def fields(value: Any, *, location: str = "fields", ancestors: tuple[int, ...] = ()) -> dict[str, TreeFieldInput]:
    """Bind named template children through their live extension request validators."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping from names to field definitions")
    if id(value) in ancestors:
        raise ValueError(f"{location} contains a recursive YAML alias; fields must form a finite tree")
    ancestors = (*ancestors, id(value))
    result: dict[str, TreeFieldInput] = {}
    for name, definition in value.items():
        if not isinstance(name, str):
            raise ValueError(f"{location} field names must be strings; got {name!r}")
        child = f"{location}.{name}"
        if not isinstance(definition, Mapping):
            raise ValueError(f"{child} must be a mapping with a registered 'type'")
        if "name" in definition:
            raise ValueError(f"{child}: names come from the fields mapping; remove the 'name' option")
        payload = dict(definition)
        kind = payload.get("type")
        if kind == "branch":
            request = Branch
            payload["fields"] = list(
                fields(payload.get("fields"), location=f"{child}.fields", ancestors=ancestors).values()
            )
        elif isinstance(kind, str) and kind in TENSORFIELDS:
            request = TENSORFIELDS[kind].Request
        else:
            raise ValueError(f"{child}: unknown field type {kind!r}; use 'branch' or a registered tensorfield type")
        payload["name"] = name
        try:
            if "mask" in payload:
                payload["mask"] = Mask.restore(payload["mask"])
            result[name] = request.model_validate(payload)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{child}: {error}") from error
    return result


def read(pathname: str | Path, overrides: Mapping[str, Any]) -> tuple[dict[str, Any], presets.Preset | None]:
    """Parse a template and retain explicit options for ordinary preset resolution."""
    path = Path(pathname)
    unknown = overrides.keys() - Options.__annotations__.keys()
    if unknown:
        raise TypeError(f"model template {path}: unsupported Python override(s): {sorted(unknown)}")

    with path.open(encoding="utf-8") as source:
        try:
            payload = yaml.load(source, Loader=Loader)
        except yaml.YAMLError as error:
            raise ValueError(f"model template {path}: {error}") from error

    try:
        if not isinstance(payload, Mapping):
            raise ValueError("expected one mapping with model options and a 'fields' mapping")
        if any(not isinstance(key, str) for key in payload):
            raise ValueError("model option names must be strings")
        allowed = (Options.__annotations__.keys() - {"optimizer", "scheduler"}) | {"preset", "fields"}
        unknown = payload.keys() - allowed
        if unknown:
            raise ValueError(f"unknown model option(s): {sorted(unknown)}; declare child fields under 'fields'")
        options = dict(payload)
        profile = None
        if "preset" in options:
            name = options.pop("preset")
            profiles = {
                profile.name: profile for profile in (presets.XS, presets.SM, presets.MD, presets.LG, presets.XL)
            }
            if not isinstance(name, str) or name not in profiles:
                raise ValueError(f"preset must be one of {', '.join(profiles)}; got {name!r}")
            profile = profiles[name]
        options.update(overrides)
        options["fields"] = fields(options.get("fields"))
        if "mask" in options:
            options["mask"] = Mask.restore(options["mask"])
        if "reduction" in options:
            options["reduction"] = pydantic.TypeAdapter(ReductionConfig | None).validate_python(options["reduction"])
        return options, profile
    except (TypeError, ValueError) as error:
        raise ValueError(f"model template {path}: {error}") from error
