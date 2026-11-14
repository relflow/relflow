"""Grain stages of a branch, picked once from a table by grain setting."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, override

import torch

from relflow.architecture.pool import LearnedQueryCrossAttention, MeanPool
from relflow.structs.grain import Coordinates, Steps
from relflow.structs.packages import Parcel

if TYPE_CHECKING:
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch


class Grain(Protocol):
    """Decide what one sequence token of a branch stands for."""

    def __call__(self, parcels: list[Parcel]) -> list[Parcel]:
        """Return the parcels whose tokens form the branch sequence."""
        ...


class GrainBuilder[S](Protocol):
    """Build the grain stage of one branch from its grain setting."""

    def __call__(self, grain: S, *, branch: Branch, schema: Schema) -> Grain: ...


def keep(parcels: list[Parcel]) -> list[Parcel]:
    """Keep one token for every leaf coordinate.

    Args:
        parcels: Contextualized child parcels.

    Returns:
        The same parcels.
    """
    return parcels


def coordinates(grain: Coordinates, *, branch: Branch, schema: Schema) -> Grain:
    """Build the coordinate grain, which has no weights and so is a plain function.

    Args:
        grain: The coordinates setting.
        branch: Branch that owns the sequence.
        schema: Schema that owns the branch.

    Returns:
        The stage that keeps every coordinate token.
    """
    return keep


def extents(parcels: list[Parcel]) -> tuple[int, list[torch.Tensor]]:
    """Return the item count of a branch instance and the tokens each parcel holds per item.

    Args:
        parcels: Parcels of one branch instance grid.

    Returns:
        The item count and, for every parcel, an int64 tensor of per-item token counts.

    Raises:
        ValueError: If a parcel holds a token count that is not a multiple of the items.
    """
    first = parcels[0]
    shape = tuple(first.payload.shape[:-2])
    declared = next((parcel.widths for parcel in parcels if parcel.widths is not None), None)
    # Leaves hold one token per item, so the narrowest undeclared parcel sets the item count.
    items = declared.shape[-1] if declared is not None else min(parcel.payload.shape[-2] for parcel in parcels)
    widths: list[torch.Tensor] = []
    for parcel in parcels:
        count = parcel.payload.shape[-2]
        if parcel.widths is not None:
            widths.append(parcel.widths)
        elif items and count % items:
            raise ValueError(
                f"parcel from '{parcel.origin}' holds {count} tokens per instance of '{parcel.destination}', "
                f"which is not a multiple of its {items} items"
            )
        else:
            widths.append(
                torch.full(
                    (*shape, items), count // items if items else 0, dtype=torch.int64, device=first.payload.device
                )
            )
    return items, widths


def rows(parcels: list[Parcel]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Number the rows of a branch instance, one position per row.

    Args:
        parcels: Parcels holding the same rows, with any number of tokens per row.

    Returns:
        Row positions, instance spans and row slots, read from the first parcel with
        its tokens per row divided out.
    """
    items, _ = extents(parcels)
    first = parcels[0]
    device = first.payload.device
    shape = tuple(first.payload.shape[:-2])
    per_row = first.payload.shape[-2] // items if items else 1
    positions = torch.arange(items, device=device).expand(*shape, items)
    spans = (
        torch.full(shape, items, device=device, dtype=torch.int64) if first.spans is None else first.spans // per_row
    )
    slots = (
        torch.ones((*shape, items), device=device, dtype=torch.bool)
        if first.slots is None
        else first.slots.reshape(*shape, items, per_row).any(dim=-1)
    )
    return positions, spans, slots


pools: Mapping[str, Callable[..., torch.nn.Module]] = MappingProxyType(
    {
        "query": lambda *, branch, schema: LearnedQueryCrossAttention(
            n_context=1,
            d_model=schema.d_model,
            nhead=branch.n_heads,
            dropout=float(branch.dropout or 0.0),
            position=False,
            ffnx=branch.ffnx,
        ),
        "mean": lambda *, branch, schema: MeanPool(n_context=1),
    }
)


class StepStage(torch.nn.Module):
    """Pool the tokens of every row of a branch into one token.

    The parcels are already contextualized, so this stage only pools: it never
    runs coordinate attention again.
    """

    def __init__(self, grain: Steps, *, branch: Branch, schema: Schema) -> None:
        super().__init__()
        self.origin = branch.address
        self.pool = pools[grain.pooling](branch=branch, schema=schema)

    @override
    def forward(self, parcels: list[Parcel]) -> list[Parcel]:
        """Pool each row to one token.

        Args:
            parcels: Contextualized child parcels of one branch instance grid.

        Returns:
            One parcel of shape ``(N, *dims, L, d_model)`` whose row is present when any of
            its tokens is present and its slot exists.
        """
        items, _ = extents(parcels)
        first = parcels[0]
        shape = tuple(first.payload.shape[:-2])
        channels = first.payload.shape[-1]
        payloads, presence = [], []
        for parcel in parcels:
            per_row = parcel.payload.shape[-2] // items if items else 0
            payloads.append(parcel.payload.reshape(*shape, items, per_row, channels))
            presence.append(parcel.present.reshape(*shape, items, per_row))
        tokens = torch.cat(payloads, dim=-2)
        present = torch.cat(presence, dim=-1)
        count = math.prod(shape) * items
        pooled = self.pool(tokens.reshape(count, -1, channels), present=present.reshape(count, -1))
        positions, spans, slots = rows(parcels)
        return [
            Parcel(
                payload=pooled.reshape(*shape, items, channels),
                present=present.any(dim=-1) & slots,
                origin=self.origin,
                destination=self.origin,
                positions=positions,
                spans=spans,
                slots=slots,
                batch_size=first.payload.shape[0],
            )
        ]

    if TYPE_CHECKING:
        __call__ = forward


def steps(grain: Steps, *, branch: Branch, schema: Schema) -> Grain:
    """Build the stage that pools every row of a branch to one token.

    Args:
        grain: The steps setting.
        branch: Branch that owns the sequence.
        schema: Schema that owns the branch.

    Returns:
        The pooling stage.
    """
    return StepStage(grain, branch=branch, schema=schema)


grains: Mapping[type, GrainBuilder[Any]] = MappingProxyType({Coordinates: coordinates, Steps: steps})

__all__ = ["Grain", "GrainBuilder", "StepStage", "coordinates", "extents", "grains", "keep", "pools", "rows", "steps"]
