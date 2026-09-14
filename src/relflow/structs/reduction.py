"""Frozen schema configurations for branch reduction."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

import pydantic

StrictPositiveInt: TypeAlias = Annotated[pydantic.StrictInt, pydantic.Field(gt=0)]
ReductionDropout: TypeAlias = Annotated[float, pydantic.Field(strict=True, ge=0.0, lt=1.0)]


class Mean(pydantic.BaseModel):
    """Reduce a branch to one vector by averaging its present coordinates.

    Padding and other absent coordinates do not contribute to the mean; an
    empty selection yields zero. This frozen configuration adds no learned
    reduction parameters and is passed as ``Branch(reduction=rf.Mean())``.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    type: Literal["mean"] = "mean"


class Attention(pydantic.BaseModel):
    """Reduce a branch to a fixed number of learned context vectors.

    ``n_outputs`` sets the number of vectors passed to the parent, and
    ``n_layers`` sets reduction depth. ``n_heads`` and ``dropout`` inherit the
    branch configuration when omitted; explicit values override it here.
    ``position`` enables rotary position information in reduction attention.
    Absent coordinates are excluded from attention. This frozen configuration
    is passed as ``Branch(reduction=rf.Attention(...))``.
    """

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
