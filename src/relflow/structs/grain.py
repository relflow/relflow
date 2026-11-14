"""Frozen schema configurations for what one branch sequence token stands for."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

import pydantic


class Coordinates(pydantic.BaseModel):
    """Keep one sequence token for every leaf coordinate of every item.

    This is the standard grain. It adds no parameters and is the default of
    ``Branch(grain=...)``.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    type: Literal["coordinates"] = "coordinates"


class Steps(pydantic.BaseModel):
    """Pool every row of a branch into one sequence token.

    ``pooling`` chooses how the coordinates of a row combine: ``"query"`` uses
    a learned query and ``"mean"`` averages them. The branch sequence then
    holds one token per row, so a branch of ``length`` rows with
    ``reduction=None`` is ``length`` tokens wide however many leaves it has.
    Every leaf decoder under the branch reads the token of its own row.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    type: Literal["steps"] = "steps"
    pooling: Literal["query", "mean"] = "query"


GrainConfig: TypeAlias = Annotated[
    Coordinates | Steps,
    pydantic.Field(discriminator="type"),
]


__all__ = [
    "Coordinates",
    "GrainConfig",
    "Steps",
]
