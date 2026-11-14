"""Pure NumPy arithmetic over the retained rows of branch records.

``placement`` is the sorted dense position of each retained record and
``extent`` the row axis size, so ``placement // extent`` is the parent of a
record and the records of one parent are contiguous.
"""

import datetime
from collections.abc import Callable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

__all__ = ["advance", "last", "ranks", "runs", "trailing"]

type Step = float | datetime.timedelta | None


def ranks(placement: np.ndarray, *, extent: int) -> tuple[np.ndarray, np.ndarray]:
    """Number each retained record within its parent.

    Args:
        placement: Sorted dense positions of the retained records.
        extent: Size of the row axis.

    Returns:
        The rank of each record within its parent and the retained count of that parent.
    """
    if not len(placement):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    _, first, counts = np.unique(placement // extent, return_index=True, return_counts=True)
    rank = np.arange(len(placement), dtype=np.int64) - np.repeat(first, counts)
    return rank, np.repeat(counts, counts)


def last(placement: np.ndarray, *, extent: int, count: int) -> np.ndarray:
    """Mark the final ``count`` retained records of each parent.

    Args:
        placement: Sorted dense positions of the retained records.
        extent: Size of the row axis.
        count: Number of trailing records to mark.

    Returns:
        A Boolean per record.
    """
    rank, total = ranks(placement, extent=extent)
    return rank >= total - count


def runs(placement: np.ndarray, *, extent: int, starts: np.ndarray, length: int) -> np.ndarray:
    """Mark records within ``length`` records after a start of the same parent, start included.

    Args:
        placement: Sorted dense positions of the retained records.
        extent: Size of the row axis.
        starts: Boolean per record that begins a run.
        length: Records one run covers.

    Returns:
        A Boolean per record.
    """
    rank, _ = ranks(placement, extent=extent)
    index = np.arange(len(placement))
    window = np.maximum(index - length + 1, index - rank)
    counted = np.concatenate(([0], np.cumsum(starts)))
    return counted[index + 1] - counted[window] > 0


def trailing(placement: np.ndarray, *, extent: int, observed: np.ndarray) -> np.ndarray:
    """Mark the records after the last observed record of each parent.

    Args:
        placement: Sorted dense positions of the retained records.
        extent: Size of the row axis.
        observed: Boolean per record that holds an observation.

    Returns:
        A Boolean per record; a parent without an observed record has none marked.
    """
    if not len(placement):
        return np.empty(0, dtype=bool)
    rank, total = ranks(placement, extent=extent)
    first = np.flatnonzero(rank == 0)
    counts = total[first]
    latest = np.repeat(np.maximum.reduceat(np.where(observed, rank, -1), first), counts)
    return (latest >= 0) & (rank > latest)


def shift_numbers(last: pa.Array, *, steps: np.ndarray, step: Step) -> pa.Array:
    """Add whole or fractional steps to numbers; a result the type cannot hold raises."""
    if isinstance(step, datetime.timedelta):
        message = f"time_step={step!r} cannot advance numeric Arrow type {last.type}; set time_step=<number>"
        raise ValueError(message)
    moved = pc.add(pc.cast(last, pa.float64()), pa.array(steps * (1.0 if step is None else float(step))))
    return pc.cast(moved, last.type, safe=True)


def shift_times(last: pa.Array, *, steps: np.ndarray, step: Step) -> pa.Array:
    """Add multiples of a timedelta to timestamps or durations."""
    if not isinstance(step, datetime.timedelta):
        message = f"time_step={step!r} cannot advance Arrow type {last.type}; set time_step=timedelta(...)"
        raise ValueError(message)
    micros = step // datetime.timedelta(microseconds=1)
    moved = pc.add(last, pa.array(steps * micros, type=pa.duration("us")))
    return pc.cast(moved, last.type, safe=True)


def shift_dates(last: pa.Array, *, steps: np.ndarray, step: Step) -> pa.Array:
    """Add whole days to dates through second-resolution timestamps."""
    moved = shift_times(pc.cast(last, pa.timestamp("s")), steps=steps, step=step)
    days = pc.cast(moved, last.type)
    # The cast truncates a time of day instead of failing, so compare the round trip.
    if not pc.all(pc.equal(pc.cast(days, pa.timestamp("s")), moved), min_count=0).as_py():
        message = f"time_step={step!r} does not move {last.type} by whole days; use a multiple of timedelta(days=1)"
        raise pa.ArrowInvalid(message)
    return days


shifts: tuple[tuple[Callable[[pa.DataType], bool], Callable[..., pa.Array]], ...] = (
    (lambda kind: pa.types.is_timestamp(kind) or pa.types.is_duration(kind), shift_times),
    (lambda kind: pa.types.is_date32(kind) or pa.types.is_date64(kind), shift_dates),
    (lambda kind: pa.types.is_integer(kind) or pa.types.is_floating(kind), shift_numbers),
)


def advance(last: pa.Array, *, steps: np.ndarray, step: Step) -> pa.Array:
    """Move each value forward by ``steps[i]`` multiples of ``step``, in the Arrow type of ``last``.

    Args:
        last: The value each slot follows.
        steps: How many steps ahead of ``last[i]`` each slot lies.
        step: ``time_step`` of the branch; a number for numeric columns, a timedelta otherwise.

    Returns:
        Values of the same Arrow type as ``last``.

    Raises:
        TypeError: The Arrow type is not numeric, timestamp, duration or date.
        ValueError: ``step`` does not suit the column type.
        pyarrow.ArrowInvalid: The moved value is not representable in the column type.
    """
    shift = next((shift for fits, shift in shifts if fits(last.type)), None)
    if shift is None:
        message = f"cannot advance Arrow type {last.type}; use a numeric, timestamp, duration or date time column"
        raise TypeError(message)
    return shift(last, steps=steps, step=step)
