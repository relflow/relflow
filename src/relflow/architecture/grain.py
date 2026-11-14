"""Grain stages of a branch, picked once from a table by grain setting."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NoReturn, Protocol

from relflow.structs.grain import Coordinates, Steps

if TYPE_CHECKING:
    from relflow.structs.experiment import Schema
    from relflow.structs.packages import Parcel
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


def pending(grain: Steps, *, branch: Branch, schema: Schema) -> NoReturn:
    """Reject a grain this relflow version does not build.

    Args:
        grain: The unsupported setting.
        branch: Branch that owns the sequence.
        schema: Schema that owns the branch.

    Raises:
        ValueError: Always, naming the branch and the supported grain.
    """
    message = (
        f"branch '{branch.address}' sets grain={grain!r}, which this relflow version does not build yet; "
        "use rf.Coordinates()"
    )
    raise ValueError(message)


grains: Mapping[type, GrainBuilder[Any]] = MappingProxyType({Coordinates: coordinates, Steps: pending})

__all__ = ["Grain", "GrainBuilder", "coordinates", "grains", "keep", "pending"]
