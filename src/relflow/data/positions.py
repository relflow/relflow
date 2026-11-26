"""Rotary positions read from the Index twig of a branch."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from relflow.data.query import query

if TYPE_CHECKING:
    from relflow.data.ragged import Layout
    from relflow.structs.twigs import Index

__all__ = ["place"]

# Seconds in one Index granularity unit and in one Arrow timestamp tick.
GRANULARITY = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0, "days": 86400.0, "weeks": 604800.0}
TICKS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


def ticks(values: pa.Array | pa.ChunkedArray, index: Index) -> tuple[np.ndarray, float]:
    """Lower index values to int64 ticks and the ticks one rotary position spans."""

    datatype = values.type
    temporal = (
        (pa.int64(), TICKS[cast("pa.TimestampType", datatype).unit])
        if pa.types.is_timestamp(datatype)
        else (pa.int32(), 86400.0)
        if pa.types.is_date32(datatype)
        else (pa.int64(), 1e-3)
        if pa.types.is_date64(datatype)
        else None
    )
    if temporal is not None:
        # Arrow stores both naive and aware timestamps as ticks on one timeline.
        storage, seconds = temporal
        clock = np.asarray(pc.cast(values, storage).to_numpy(zero_copy_only=False)).astype(np.int64)
        return clock, GRANULARITY[index.granularity or "days"] / seconds
    if not pa.types.is_integer(datatype):
        raise TypeError(
            f"Index '{index.address}' reads Arrow {datatype}; an index takes integers, dates, or timestamps, "
            "so cast the member in an rf.Preprocessor"
        )
    if index.granularity is not None:
        raise ValueError(
            f"Index '{index.address}' reads Arrow {datatype}, whose values count their own units; "
            f"remove granularity={index.granularity!r} or read dates or timestamps"
        )
    try:
        integers = pc.cast(values, pa.int64())
    except pa.ArrowInvalid as error:
        raise ValueError(
            f"Index '{index.address}' holds {datatype} values beyond the int64 range; "
            "shift them toward zero in an rf.Preprocessor"
        ) from error
    return np.asarray(integers.to_numpy(zero_copy_only=False)), 1.0


def place(layout: Layout, index: Index) -> np.ndarray:
    """Place every retained record of a branch at its index, in its dense geometry.

    Each collection counts from its smallest index, so positions stay small in
    float32 while rotary attention, which reads only differences, sees the same
    spacing as the source. Slots without a record hold zero.

    Raises:
        ValueError: If a retained record has no index, or a collection's index span
            exceeds int64 ticks.
        TypeError: If the index is not an integer, date, or timestamp member.
    """

    shape = layout.shape
    placed = np.zeros(math.prod(shape), dtype=np.float32)
    if not len(layout.records):
        return placed.reshape(shape)

    (expression,) = index.queries()
    selected = query(layout.records, expression, address=str(index.address))
    if selected.present.type != pa.bool_():
        raise ValueError(
            f"Index '{index.address}' query {expression!r} must select one value per record, "
            f"got Arrow {selected.values.type}"
        )
    # Arrow accepts Python scalars; the stubs require an explicit Scalar.
    present = cast(pa.Array, pc.fill_null(selected.present, False))  # pyrefly: ignore[bad-specialization]
    missing = ~np.asarray(pc.and_(present, pc.is_valid(selected.values)).to_numpy(zero_copy_only=False), dtype=bool)
    if missing.any():
        raise ValueError(
            f"Index '{index.address}' has {int(missing.sum())} null or absent value(s) among {len(missing)} "
            "retained records; every record needs an index, so fill or drop those records in an rf.Preprocessor"
        )

    values, step = ticks(selected.values, index)
    instances = layout.placement // shape[-1]
    origins = np.full(math.prod(shape[:-1]), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(origins, instances, values)
    elapsed = values - origins[instances]
    if (elapsed < 0).any():
        raise ValueError(
            f"Index '{index.address}' spans more than the int64 range within one collection; "
            "shift its values toward a common origin in an rf.Preprocessor"
        )
    placed[layout.placement] = elapsed / step
    return placed.reshape(shape)
