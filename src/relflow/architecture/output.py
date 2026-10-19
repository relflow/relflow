"""Project dense model coordinates into retained Arrow collection geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from relflow.data.arrow import Geometry
from relflow.structs.packages import Prediction
from relflow.structs.tree import Address

if TYPE_CHECKING:
    from relflow.architecture.runtime import OutputEntry


LIST_MAX = np.iinfo(np.int32).max


@dataclass(frozen=True, slots=True)
class Axis:
    """One dense axis and its retained per-parent coordinates."""

    extent: int
    lengths: np.ndarray
    valid: np.ndarray
    slots: np.ndarray | None = None
    positions: np.ndarray | None = None
    structural: bool = True


@dataclass(frozen=True, slots=True)
class Layout:
    """Arrow list boundaries, collection validity, and retained item validity."""

    offsets: np.ndarray
    valid: np.ndarray
    slots: np.ndarray


@dataclass(frozen=True, slots=True)
class Projection:
    """Retained coordinates and their shared structural and representation axes."""

    size: int
    indices: np.ndarray
    axes: tuple[Layout, ...]
    rows: np.ndarray
    parents: tuple[np.ndarray, ...]
    outputs: np.ndarray


def numpy(values: torch.Tensor) -> np.ndarray:
    """Stage structural metadata on the Arrow writer's CPU boundary."""

    return values.detach().cpu().numpy()


def structure(
    branch: Geometry,
    address: Address,
    shape: tuple[int, ...],
    limit: int | None,
    parent_slots: np.ndarray | None,
) -> Axis:
    """Validate shared geometry once at the public Arrow writer boundary."""

    for name in ("lengths", "present", "valid", "slots"):
        tensor = getattr(branch, name)
        dtype = torch.int64 if name == "lengths" else torch.bool
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype:
            raise TypeError(f"prediction geometry for {address} {name} must be a {dtype} tensor")
        actual = tuple(tensor.shape)
        if (name != "slots" and actual != shape) or (
            name == "slots" and (len(actual) != len(shape) + 1 or actual[:-1] != shape)
        ):
            expected = f"{shape} plus its collection axis" if name == "slots" else str(shape)
            raise ValueError(f"prediction geometry for {address} {name} must have shape {expected}, got {actual}")
    extent = branch.slots.shape[-1]
    if limit is not None and extent > limit:
        raise ValueError(f"prediction geometry for {address} extent must be at most {limit}, got {extent}")
    lengths = numpy(branch.lengths).reshape(-1)
    present = numpy(branch.present).reshape(-1)
    valid = numpy(branch.valid).reshape(-1)
    slots = numpy(branch.slots).reshape(len(lengths), extent)
    if np.any((lengths < 0) | (lengths > extent)):
        raise ValueError(f"prediction geometry for {address} lengths must be within its extent {extent}")
    if np.any(valid & ~present) or np.any(lengths[~valid] != 0):
        raise ValueError(f"prediction geometry for {address} validity must agree with presence and collection lengths")
    if np.any(slots & (np.arange(extent) >= lengths[:, None])):
        raise ValueError(f"prediction geometry for {address} slots must lie within collection lengths")
    if parent_slots is not None and np.any(present & ~parent_slots.reshape(-1)):
        raise ValueError(f"prediction geometry for {address} cannot exist outside its parent slots")
    return Axis(extent, lengths, valid, slots)


def tokens(prediction: Prediction, shape: tuple[int, ...], valid: np.ndarray) -> Axis:
    """Validate the encoder's intrinsic positions without compacting logical holes."""

    address = prediction.address
    for name in ("positions", "spans", "slots"):
        tensor = getattr(prediction, name)
        dtype = torch.bool if name == "slots" else torch.int64
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype:
            raise TypeError(f"pass-through prediction at {address} {name} must be a {dtype} tensor")
    assert prediction.positions is not None and prediction.spans is not None and prediction.slots is not None
    if tuple(prediction.spans.shape) != shape:
        raise ValueError(f"pass-through prediction at {address} spans must have shape {shape}")
    if tuple(prediction.positions.shape[:-1]) != shape or prediction.positions.ndim != len(shape) + 1:
        raise ValueError(f"pass-through prediction at {address} positions must start with parent shape {shape}")
    if prediction.slots.shape != prediction.positions.shape:
        raise ValueError(f"pass-through prediction at {address} slots must match positions shape")
    lengths = numpy(prediction.spans).reshape(-1)
    if np.any(lengths < 0):
        raise ValueError(f"pass-through prediction at {address} spans must be non-negative")
    if np.any(lengths > LIST_MAX):
        raise OverflowError(f"pass-through prediction at {address} spans exceed the Arrow int32 offset limit")
    extent = prediction.positions.shape[-1]
    positions = numpy(prediction.positions).reshape(len(lengths), extent)
    slots = numpy(prediction.slots).reshape(len(lengths), extent)
    selected_rows, selected_slots = np.nonzero(slots)
    selected = positions[selected_rows, selected_slots]
    if np.any((selected < 0) | (selected >= lengths[selected_rows])):
        raise ValueError(f"pass-through prediction at {address} selected positions must lie within their spans")
    keys = np.sort(selected_rows * (int(lengths.max(initial=0)) + 1) + selected)
    if np.any(keys[1:] == keys[:-1]):
        raise ValueError(f"pass-through prediction at {address} selected positions must be unique within each parent")
    return Axis(extent, lengths, valid, slots, positions, structural=False)


def projection(entry: OutputEntry, prediction: Prediction, rows: int) -> Projection:
    """Resolve structural coordinates without interpreting tensorfield values."""

    axes: list[Axis] = []
    geometry = prediction.geometry
    parents = rows
    parent_shape = (rows, 1)
    for limit, address in zip(entry.axes, entry.branches):
        if geometry is None:
            raise ValueError(f"prediction at {entry.address} requires batch geometry for branch {address}")
        branch = geometry[address]
        if not isinstance(branch, Geometry):
            raise TypeError(f"prediction geometry for {address} must be Geometry, got {type(branch).__name__}")
        axis = structure(branch, address, parent_shape, limit, None if not axes else axes[-1].slots)
        axes.append(axis)
        parents *= axis.extent
        parent_shape = (*parent_shape, axis.extent)

    tail = len(entry.axes) > len(entry.branches)
    if tail or entry.passthrough:
        width = entry.axes[-1] if tail else 1
        if entry.passthrough:
            if prediction.positions is None or prediction.spans is None or prediction.slots is None:
                raise ValueError(f"pass-through prediction at {entry.address} requires structural spans and slots")
            own = None if geometry is None else geometry.get(entry.address)
            valid = np.ones(parents, dtype=bool)
            if isinstance(own, Geometry):
                expected = (rows,) if entry.address == Address() else parent_shape
                if tuple(own.valid.shape) != expected or own.valid.dtype != torch.bool:
                    raise ValueError(
                        f"prediction geometry for {entry.address} validity must be boolean with shape {expected}"
                    )
                valid = numpy(own.valid).reshape(-1)
            axes.append(tokens(prediction, (rows,) if entry.address == Address() else parent_shape, valid))
        elif width is not None:
            axes.append(
                Axis(
                    width,
                    np.full(parents, width, dtype=np.int64),
                    np.ones(parents, dtype=bool),
                    structural=False,
                )
            )
        else:
            raise ValueError(f"pass-through prediction at {entry.address} requires structural coordinate metadata")

    size = rows * math.prod(axis.extent for axis in axes)
    indices = np.arange(rows, dtype=np.int64)
    row_ids = indices.copy()
    coordinates: list[np.ndarray] = []
    outputs = np.zeros(rows, dtype=np.int32)
    layouts: list[Layout] = []
    for axis in axes:
        retained = indices >= 0
        valid = np.zeros(len(indices), dtype=bool)
        valid[retained] = axis.valid[indices[retained]]
        counts = np.zeros(len(indices), dtype=np.int64)
        counts[valid] = axis.lengths[indices[valid]]
        offsets = np.empty(len(indices) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        total = int(offsets[-1])
        if total > LIST_MAX:
            raise OverflowError(f"prediction at {entry.address} collections exceed the Arrow int32 offset limit")
        ordinal = np.arange(total, dtype=np.int64) - np.repeat(offsets[:-1], counts)
        parent = np.repeat(indices, counts)
        selected_indices = np.full(total, -1, dtype=np.int64)
        if axis.positions is None:
            selected = (parent >= 0) & (ordinal < axis.extent)
            if axis.slots is not None:
                candidates = np.flatnonzero(selected)
                selected[candidates] = axis.slots[parent[candidates], ordinal[candidates]]
            selected_indices[selected] = parent[selected] * axis.extent + ordinal[selected]
        elif axis.extent and total:
            assert axis.slots is not None
            source_rows, source_slots = np.nonzero(axis.slots)
            positions = axis.positions[source_rows, source_slots]
            stride = max(int(positions.max(initial=0)), int(ordinal.max(initial=0))) + 1
            keys = source_rows * stride + positions
            order = np.argsort(keys)
            keys = keys[order]
            desired = parent * stride + ordinal
            locations = np.searchsorted(keys, desired)
            selected = (parent >= 0) & (locations < len(keys))
            candidates = np.flatnonzero(selected)
            candidates = candidates[keys[locations[candidates]] == desired[candidates]]
            original = order[locations[candidates]]
            selected_indices[candidates] = source_rows[original] * axis.extent + source_slots[original]

        layouts.append(Layout(offsets, valid, selected_indices >= 0))
        row_ids = np.repeat(row_ids, counts)
        coordinates = [np.repeat(coordinate, counts) for coordinate in coordinates]
        if axis.structural:
            coordinates.append(ordinal)
            outputs = np.repeat(outputs, counts)
        else:
            outputs = ordinal.astype(np.int32)
        indices = selected_indices

    return Projection(size, indices, tuple(layouts), row_ids, tuple(coordinates), outputs)


__all__ = ["Layout", "Projection", "projection"]
