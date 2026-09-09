"""Frozen schema configurations for branch reduction."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

import pydantic

StrictPositiveInt: TypeAlias = Annotated[pydantic.StrictInt, pydantic.Field(gt=0)]
ReductionDropout: TypeAlias = Annotated[float, pydantic.Field(strict=True, ge=0.0, lt=1.0)]


class Mean(pydantic.BaseModel):
    """Arithmetic-mean branch reduction."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    type: Literal["mean"] = "mean"


class Attention(pydantic.BaseModel):
    """Learned-query attention branch reduction."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    type: Literal["attention"] = "attention"
    n_outputs: StrictPositiveInt = 1
    n_heads: StrictPositiveInt | None = None
    n_layers: StrictPositiveInt = 1
    dropout: ReductionDropout | None = None
    position: pydantic.StrictBool = True


ReductionConfig: TypeAlias = Annotated[
    Mean | Attention,
    pydantic.Field(discriminator="type"),
]


__all__ = [
    "Attention",
    "Mean",
    "ReductionConfig",
]
