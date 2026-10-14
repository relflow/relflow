"""Immutable model policies and schema resolution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, cast

import pydantic

from relflow.structs.enums import AttentionInput, AttentionMode
from relflow.structs.experiment import Schema, TreeFieldInput, build
from relflow.structs.reduction import ReductionConfig
from relflow.structs.structure import OMITTED, Branch
from relflow.structs.tree import MaskInput, Rate

__all__ = ["BranchDefaults", "LeafDefaults", "Preset", "resolve"]


class BranchDefaults(pydantic.BaseModel):
    """Architecture defaults shared by the root and non-root branch profiles."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    n_layers: Annotated[int, pydantic.Field(gt=0)]
    n_heads: Annotated[int, pydantic.Field(gt=0, multiple_of=2)]
    ffnx: Annotated[pydantic.StrictInt, pydantic.Field(gt=0)] = 8
    attention: AttentionMode | None = AttentionMode.mha
    dropout: Rate | None = 0.0
    reduction: ReductionConfig | None


class LeafDefaults(pydantic.BaseModel):
    """Defaults for the common extension decoder contract."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    n_heads: Annotated[int, pydantic.Field(gt=0, multiple_of=2)]
    ffnx: Annotated[pydantic.StrictInt, pydantic.Field(gt=0)] = 8
    pooling: Literal["query", "mean"] = "query"
    n_linear: Annotated[int, pydantic.Field(gt=0)] = 1
    dropout: Rate | None = 0.0
    decoder_position: bool | None = None


class Preset(pydantic.BaseModel):
    """Persisted construction policy; a resolved schema remains authoritative."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    name: Annotated[str, pydantic.Field(min_length=1)]
    revision: Annotated[int, pydantic.Field(gt=0)] = 1
    d_model: Annotated[int, pydantic.Field(gt=0)]
    root: BranchDefaults
    branch: BranchDefaults
    leaf: LeafDefaults

    def defaults(self, definition: TreeFieldInput) -> Mapping[str, Any]:
        """Select the architecture defaults for one unbound branch or leaf."""
        scope = self.branch if isinstance(definition, Branch) else self.leaf
        return {option: getattr(scope, option) for option in type(scope).model_fields}


def resolve(
    preset: Preset,
    /,
    *,
    d_model: int = cast(int, OMITTED),
    n_layers: int = cast(int, OMITTED),
    n_heads: int = cast(int, OMITTED),
    ffnx: int = cast(int, OMITTED),
    fields: Mapping[str, TreeFieldInput] | None = None,
    query: str | None = None,
    description: str | None = None,
    embed: bool = False,
    attention: AttentionInput = cast(AttentionInput, OMITTED),
    reduction: ReductionConfig | None = cast(ReductionConfig | None, OMITTED),
    dropout: Rate | None = cast(Rate | None, OMITTED),
    mask: MaskInput = False,
    **children: TreeFieldInput,
) -> Schema:
    """Resolve local overrides into a fresh tree without changing its definitions."""
    options: dict[str, Any] = {name: getattr(preset.root, name) for name in type(preset.root).model_fields}
    options["d_model"] = preset.d_model
    overrides = dict(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffnx=ffnx,
        attention=attention,
        reduction=reduction,
        dropout=dropout,
    )
    options.update({name: value for name, value in overrides.items() if value is not OMITTED})
    width = options.pop("d_model")
    options.update(query=query, description=description, embed=embed, mask=mask)
    return build(
        Schema,
        d_model=width,
        fields=fields,
        children=children,
        options=options,
        defaults=preset.defaults,
    )
