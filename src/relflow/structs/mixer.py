"""Frozen schema configurations for the branch sequence mixer."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

import pydantic

from relflow.structs.reduction import StrictPositiveInt


class Transformer(pydantic.BaseModel):
    """Mix the sequence tokens of a branch with rotary self-attention layers.

    This is the standard mixer and the default of ``Branch(mixer=...)``.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    type: Literal["transformer"] = "transformer"


class Conv(pydantic.BaseModel):
    """Mix the sequence tokens of a branch with ``n_blocks`` dilated convolution blocks.

    A relflow version that does not build this mixer rejects it when the model
    is built.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    type: Literal["conv"] = "conv"
    n_blocks: StrictPositiveInt = 6


MixerConfig: TypeAlias = Annotated[
    Transformer | Conv,
    pydantic.Field(discriminator="type"),
]


__all__ = [
    "Conv",
    "MixerConfig",
    "Transformer",
]
