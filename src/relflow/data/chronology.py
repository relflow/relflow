"""Row positions in time for branches that name a ``time_index`` child."""

import datetime
import math
from collections.abc import Callable
from types import MappingProxyType
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

SECONDS = MappingProxyType({"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9})

type Values = pa.Array | pa.ChunkedArray
type Family = tuple[Callable[[pa.DataType], bool], pa.DataType, Callable[[pa.DataType], float | None]]


def unit(kind: pa.DataType) -> float:
    """Seconds per tick of a timestamp or duration type."""

    # Only the timestamp and duration family predicate routes types here.
    return SECONDS[cast(pa.TimestampType | pa.DurationType, kind).unit]


# Each family names the numeric type its ticks lower to and the seconds one tick
# lasts. Arrow stores timestamps as UTC ticks, so time zones need no handling.
FAMILIES: tuple[Family, ...] = (
    (
        lambda kind: pa.types.is_timestamp(kind) or pa.types.is_duration(kind),
        pa.int64(),
        unit,
    ),
    (pa.types.is_date32, pa.int32(), lambda kind: 86400.0),
    (pa.types.is_date64, pa.int64(), lambda kind: 1e-3),
    (lambda kind: pa.types.is_integer(kind) or pa.types.is_floating(kind), pa.float64(), lambda kind: None),
)


def ticks(values: Values, *, address: str) -> tuple[np.ndarray, float | None]:
    """Read a time column as raw numbers and the seconds one tick lasts.

    Args:
        values: Arrow time values; nulls read as zero and the caller masks them out.
        address: str of the time leaf, for error messages.

    Returns:
        Raw ticks (int64 for temporal types, float64 for numeric) and seconds per
        tick, which is ``None`` for numeric columns.

    Raises:
        TypeError: If the Arrow type is not numeric, timestamp, date or duration.
    """

    family = next((family for family in FAMILIES if family[0](values.type)), None)
    if family is None:
        raise TypeError(
            f"time leaf '{address}' has Arrow type {values.type}; use a numeric, timestamp, date or duration column"
        )
    _, target, seconds = family
    # The compute stubs also admit scalars; an array in gives an array out.
    lowered = cast(Values, pc.fill_null(pc.cast(values, target), pa.scalar(0, type=target)))
    raw = lowered.to_numpy(zero_copy_only=False)
    seconds_per_tick = seconds(values.type)
    return (raw if seconds_per_tick is None else raw.astype(np.int64)), seconds_per_tick


def divisor(*, kind: pa.DataType, seconds: float | None, step: float | datetime.timedelta | None, where: str) -> float:
    """Return how many ticks one ``time_step`` lasts."""

    if isinstance(step, datetime.timedelta) and seconds is not None:
        return step.total_seconds() / seconds
    if isinstance(step, datetime.timedelta):
        raise ValueError(
            f"{where} has numeric Arrow type {kind} but time_step={step!r}; set time_step=<number> or leave it unset"
        )
    if seconds is not None:
        raise ValueError(
            f"{where} has Arrow type {kind}, which counts elapsed time; "
            "set time_step=timedelta(...) to the spacing of one position"
        )
    return 1.0 if step is None else float(step)


def positions(
    values: Values,
    *,
    valued: np.ndarray,
    placement: np.ndarray,
    shape: tuple[int, ...],
    step: float | datetime.timedelta | None,
    where: str,
) -> np.ndarray:
    """Place the rows of every branch instance in time, relative to its first row.

    Positions divide elapsed time by ``step`` and start at zero for each
    instance, so float32 devices keep them exact. Absent slots hold zero.

    Args:
        values: One time value per retained record.
        valued: Which records hold a non-null time.
        placement: Flat dense position of each record in ``shape``.
        shape: Dense branch geometry; the last axis is the row axis.
        step: ``time_step`` of the branch.
        where: Branch and leaf description for error messages.

    Returns:
        A float64 array of shape ``shape``.

    Raises:
        ValueError: If times decrease within an instance or ``step`` does not suit
            the column type.
    """

    size = math.prod(shape)
    if not size:
        return np.zeros(shape)

    raw, seconds = ticks(values, address=where)
    scale = divisor(kind=values.type, seconds=seconds, step=step, where=where)
    dense = np.zeros(size, dtype=raw.dtype)
    live = np.zeros(size, dtype=bool)
    dense[placement[valued]] = raw[valued]
    live[placement[valued]] = True
    dense, live = dense.reshape(-1, shape[-1]), live.reshape(-1, shape[-1])

    # Subtract in tick dtype so nanosecond epochs keep their precision.
    first = live.argmax(axis=-1)[:, None]
    relative = np.where(live, dense - np.take_along_axis(dense, first, axis=-1), 0)
    lowest = np.iinfo(relative.dtype).min if relative.dtype.kind == "i" else -np.inf
    running = np.maximum.accumulate(np.where(live, relative, lowest), axis=-1)
    decreases = live[:, 1:] & (relative[:, 1:] < running[:, :-1])
    if decreases.any():
        instance = int(np.flatnonzero(decreases.any(axis=-1))[0])
        row = int(np.flatnonzero(decreases[instance])[0]) + 1
        raise ValueError(
            f"{where} decreases in record {instance} at row {row}; "
            "sort the rows by time within each record in an rf.Preprocessor"
        )
    return (relative / scale).reshape(shape)


__all__ = ["positions", "ticks"]
