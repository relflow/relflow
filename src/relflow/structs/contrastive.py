"""Frozen schema configurations for the branch contrastive objective."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

import pydantic

__all__ = ["Contrastive", "Crops", "Hierarchical"]

StrictPositiveInt: TypeAlias = Annotated[pydantic.StrictInt, pydantic.Field(gt=0)]


class Crops(pydantic.BaseModel):
    """Two windows of a record, each ``width`` rows long.

    ``overlap`` is the fraction of rows the two windows share. A record shorter
    than ``width`` is masked instead of padded with data, so every window has
    the same shape.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    type: Literal["crops"] = "crops"
    width: StrictPositiveInt = 128
    overlap: Annotated[float, pydantic.Field(gt=0.0, lt=1.0)] = 0.5


class Hierarchical(pydantic.BaseModel):
    """Contrast two views of the same records at matching rows.

    Each row of one view is pulled toward the same row of the other view and
    pushed away from the rows of other records and from other rows of the same
    record. The comparison repeats at coarser time scales, each one pooling
    pairs of rows by their maximum.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    type: Literal["hierarchical"] = "hierarchical"


class Contrastive(pydantic.BaseModel):
    """Train a branch to represent its rows by contrasting two crops of each record.

    The objective runs in training only. The full record still reaches the
    parent branch and the decoders; two crops of ``views.width`` rows are
    encoded in addition, so the branch costs about three times its memory and
    compute. Records shorter than a crop are masked, not padded with data.
    Crop offsets replay from the data seed, epoch, batch ordinal and branch
    address. Under distributed training, records are compared only within the
    batch of each device. ``weight`` scales the loss against the other
    objectives.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    views: Crops = pydantic.Field(default_factory=Crops)
    loss: Hierarchical = pydantic.Field(default_factory=Hierarchical)
    weight: Annotated[float, pydantic.Field(gt=0.0)] = 1.0
