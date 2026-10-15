"""Project batch geometry into stable Arrow prediction axes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch

from relflow.data.arrow import Geometry
from relflow.structs.packages import Prediction
from relflow.tensorfields.output import fixed

if TYPE_CHECKING:
    from relflow.architecture.runtime import OutputEntry


@dataclass(frozen=True, slots=True)
class Axis:
    """One output limit and its concrete batch extent and parent counts."""

    limit: int | None
    extent: int
    lengths: np.ndarray | None = None
    valid: np.ndarray | None = None
    positions: np.ndarray | None = None
    slots: np.ndarray | None = None
    wrap: bool = True


@dataclass(frozen=True, slots=True)
class Projection:
    """Gather coordinates once, then wrap their fixed or variable Arrow axes."""

    size: int
    indices: np.ndarray
    axes: tuple[tuple[int | None, np.ndarray, np.ndarray | None], ...]

    def write(self, values: pa.Array) -> pa.Array:
        if len(values) != self.size:
            raise ValueError(f"output projection requires {self.size} coordinates, got {len(values)}")
        values = pc.take(values, pa.array(self.indices, mask=self.indices < 0, type=pa.int64()))
        for limit, offsets, valid in reversed(self.axes):
            if limit is not None:
                values = fixed(values, limit)
            else:
                if offsets[-1] > np.iinfo(np.int32).max:
                    raise OverflowError("prediction lists exceed the Arrow int32 offset limit")
                values = pa.ListArray.from_arrays(
                    pa.array(offsets, type=pa.int32()),
                    values,
                    mask=None if valid is None else pa.array(~valid),
                )
        return values


def numpy(values: torch.Tensor) -> np.ndarray:
    """Stage structural metadata on the Arrow writer's CPU boundary."""

    return values.detach().cpu().numpy()


def projection(entry: OutputEntry, prediction: Prediction, rows: int) -> Projection:
    """Resolve public coordinates without specializing their Arrow types."""

    axes: list[Axis] = []
    geometry = prediction.geometry
    for limit, address in zip(entry.axes, entry.branches):
        if geometry is None:
            if limit is None:
                raise ValueError(
                    f"prediction at {entry.address} requires batch geometry for unbounded branch {address}"
                )
            axes.append(Axis(limit, limit))
            continue
        branch = geometry[address]
        if not isinstance(branch, Geometry):
            raise TypeError(f"prediction geometry for {address} must be Geometry, got {type(branch).__name__}")
        axes.append(
            Axis(limit, branch.slots.shape[-1], numpy(branch.lengths).reshape(-1), numpy(branch.valid).reshape(-1))
        )

    tail = len(entry.axes) > len(entry.branches)
    if tail or entry.passthrough:
        limit = entry.axes[-1] if tail else 1
        if entry.passthrough and prediction.positions is not None:
            if prediction.spans is None or prediction.slots is None:
                raise ValueError(f"pass-through prediction at {entry.address} requires structural spans and slots")
            extent = prediction.positions.shape[-1]
            axes.append(
                Axis(
                    limit,
                    extent,
                    numpy(prediction.spans).reshape(-1),
                    valid=numpy(cast(Geometry, geometry[entry.address]).valid).reshape(-1)
                    if geometry is not None
                    else None,
                    positions=numpy(prediction.positions).reshape(-1, extent)
                    if extent
                    else np.empty((prediction.spans.numel(), 0), dtype=np.int64),
                    slots=numpy(prediction.slots).reshape(-1, extent)
                    if extent
                    else np.empty((prediction.spans.numel(), 0), dtype=bool),
                    wrap=tail,
                )
            )
        elif limit is not None:
            axes.append(Axis(limit, limit, wrap=tail))
        else:
            raise ValueError(f"pass-through prediction at {entry.address} requires structural coordinate metadata")

    size = rows * math.prod(axis.extent for axis in axes)
    parents = np.arange(rows, dtype=np.int64)
    wrappers: list[tuple[int | None, np.ndarray, np.ndarray | None]] = []
    for axis in axes:
        retained = parents >= 0
        if axis.limit is not None:
            counts = np.full(len(parents), axis.limit, dtype=np.int64)
        else:
            assert axis.lengths is not None
            counts = np.zeros(len(parents), dtype=np.int64)
            counts[retained] = axis.lengths[parents[retained]]
        offsets = np.empty(len(parents) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        validity = None
        if axis.limit is None and axis.valid is not None:
            validity = np.zeros(len(parents), dtype=bool)
            validity[retained] = axis.valid[parents[retained]]
        if axis.wrap:
            wrappers.append((axis.limit, offsets, validity))

        total = int(offsets[-1])
        slots = np.arange(total, dtype=np.int64) - np.repeat(offsets[:-1], counts)
        parent = np.repeat(parents, counts)
        indices = np.full(total, -1, dtype=np.int64)
        if axis.positions is None:
            selected = (parent >= 0) & (slots < axis.extent)
            indices[selected] = parent[selected] * axis.extent + slots[selected]
        elif axis.extent and total:
            assert axis.slots is not None
            source_rows, source_slots = np.nonzero(axis.slots)
            positions = axis.positions[source_rows, source_slots]
            stride = max(int(positions.max(initial=0)), int(slots.max(initial=0))) + 1
            keys = source_rows * stride + positions
            order = np.argsort(keys)
            keys = keys[order]
            desired = parent * stride + slots
            locations = np.searchsorted(keys, desired)
            selected = (parent >= 0) & (locations < len(keys))
            candidates = np.flatnonzero(selected)
            candidates = candidates[keys[locations[candidates]] == desired[candidates]]
            original = order[locations[candidates]]
            indices[candidates] = source_rows[original] * axis.extent + source_slots[original]
        parents = indices

    return Projection(size, parents, tuple(wrappers))


__all__ = ["projection"]
