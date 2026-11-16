"""Compile prediction contracts and write model coordinates into Arrow records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch

from relflow.data.arrow import Geometry, Retain, retention
from relflow.structs.enums import TensorKey, Tokens
from relflow.structs.packages import Prediction
from relflow.structs.tree import Address, Node
from relflow.tensorfields.base import TENSORFIELDS, Write
from relflow.tensorfields.output import STATE, embedding, inferred, state, struct

if TYPE_CHECKING:
    from relflow.architecture.root import Model

__all__ = ["Layout", "OutputEntry", "OutputPlan", "Projection", "metadata", "plan", "projection", "write"]

RESERVED = frozenset({TensorKey.state.name, TensorKey.inferred.name})
LIST_MAX = np.iinfo(np.int32).max


@dataclass(frozen=True, slots=True)
class OutputEntry:
    """Frozen output contract for one expected forward address."""

    address: Address
    axes: tuple[int | None, ...]
    branches: tuple[Address, ...]
    passthrough: bool
    decoded: pa.StructType | None
    writer: Write | None
    embed: bool
    coordinate: pa.StructType | None
    path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OutputNode:
    """Selected decoded children and their shared Arrow record geometry."""

    address: Address
    name: str | None
    datatype: pa.DataType
    children: tuple[OutputNode, ...] = ()


@dataclass(frozen=True, slots=True)
class OutputPlan:
    """Ordered prediction contract compiled for one model generation."""

    generation: int
    retain: Retain
    expected: frozenset[Address]
    entries: tuple[OutputEntry, ...]
    tree: OutputNode | None
    representations: pa.ListType | None


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


def project(source: pa.Table, names: Retain) -> pa.Array:
    """Project a validated column selection into one struct per source row."""

    selected = tuple(source.column_names) if names == "*" else names

    missing = [name for name in selected if name not in source.column_names]
    if missing:
        formatted = ", ".join(repr(name) for name in missing)
        raise KeyError(f"retained column(s) are absent after preprocessing: {formatted}")
    if not selected:
        return pa.nulls(source.num_rows)

    fields = [source.schema.field(name) for name in selected]
    arrays = [source[name].combine_chunks() for name in selected]
    return pa.StructArray.from_arrays(arrays, fields=fields)


def axes(module: Model, address: Address) -> tuple[int | None, ...]:
    """Return repeated model axes represented inside one output row."""

    if address in module.schema.requests:
        return tuple(module.schema.requests[address].shape[1:])
    if address in module.schema.branches:
        lengths = tuple(
            node.length for node in module.schema.branches[address].path if getattr(node, "type", None) == "branch"
        )
        structural = lengths[1:-1]
        outputs = module.schema.branch_outputs[address]
        return (*structural, outputs) if outputs is None or outputs > 1 else structural
    raise KeyError(f"prediction address {str(address)!r} is absent from the model schema")


def metadata(module: Model) -> dict[bytes, bytes]:
    """Record the model's RelFlow package version on every output table."""

    return {b"relflow.model_version": module.version.encode()}


def check_output_type(datatype: pa.DataType, address: Address) -> None:
    """Require portable list element names at the extension declaration boundary."""

    if pa.types.is_list(datatype) or pa.types.is_large_list(datatype) or pa.types.is_fixed_size_list(datatype):
        if datatype.value_field.name != "element":
            raise ValueError(
                f"extension output at {address} requires list children named 'element'; "
                f"got {datatype.value_field.name!r}; declare pa.field('element', value_type)"
            )
        check_output_type(datatype.value_type, address)
    elif pa.types.is_struct(datatype):
        for field in datatype:
            check_output_type(field.type, address)
    elif pa.types.is_map(datatype):
        check_output_type(datatype.key_type, address)
        check_output_type(datatype.item_type, address)


def output_tree(node: Node, entries: Mapping[Address, OutputEntry]) -> OutputNode | None:
    """Select rich leaf results while retaining their schema-owned nesting."""

    address = Address(str(node.address))
    entry = entries.get(address)
    if entry is not None and entry.coordinate is not None:
        return OutputNode(address, node.name, entry.coordinate)
    children = tuple(child for item in node.children if (child := output_tree(item, entries)) is not None)
    if not children:
        return None
    record = pa.struct(
        [pa.field(cast(str, child.name), child.datatype, nullable=bool(child.children)) for child in children]
    )
    datatype = record if address == Address() else pa.list_(pa.field("element", record))
    return OutputNode(address, node.name, datatype, children)


def plan(module: Model, retain: Retain = (), *, refresh: bool = False) -> OutputPlan:
    """Compile one exact, nested Arrow output contract per model generation."""

    generation = int(getattr(module, "_contract_generation", 0))
    retained = retention(retain)
    key = (generation, retained)
    cached = module.output_plans.get(key)
    if not refresh and isinstance(cached, OutputPlan):
        return cached

    embeds = set(module.schema.embed)
    decodes = set(module.schema.decodes)
    expected = frozenset(Address(str(address)) for address in (*decodes, *embeds))
    entries: list[OutputEntry] = []
    for node in (module.schema.fields, *module.schema.fields.descendants):
        address = Address(str(node.address))
        if address not in expected:
            continue
        decoded: pa.StructType | None = None
        writer: Write | None = None
        if address in decodes:
            extension = TENSORFIELDS[module.schema.requests[address].type]
            declared = extension.output(module=module, address=address)
            if declared is not None and not isinstance(declared, pa.StructType):
                raise TypeError(f"extension output for {address} must return a pyarrow StructType or None")
            if declared is not None:
                conflicts = RESERVED.intersection(field.name for field in declared)
                if conflicts:
                    raise ValueError(
                        f"extension output for {address} uses reserved field(s): {', '.join(sorted(conflicts))}"
                    )
                check_output_type(declared, address)
                decoded, writer = declared, extension.write

        coordinate_type = (
            None
            if decoded is None
            else pa.struct(
                [
                    pa.field(TensorKey.state.name, STATE, nullable=False),
                    *decoded,
                    pa.field(TensorKey.inferred.name, pa.bool_(), nullable=False),
                ]
            )
        )
        entries.append(
            OutputEntry(
                address=address,
                axes=axes(module, address),
                branches=tuple(
                    owner.address
                    for owner in node.path
                    if getattr(owner, "type", None) == "branch"
                    and owner.address != Address()
                    and (address in module.schema.requests or owner.address != address)
                ),
                passthrough=address in module.schema.branches and module.schema.branches[address].reduction is None,
                decoded=decoded,
                writer=writer,
                embed=address in embeds,
                coordinate=coordinate_type,
                path=tuple(
                    owner.name
                    for owner in node.path
                    if owner.name is not None and owner.address and owner.address != Address()
                ),
            )
        )

    absent = expected - {entry.address for entry in entries}
    if absent:
        raise ValueError(
            f"configured prediction address(es) are absent from schema traversal: {', '.join(sorted(absent))}"
        )
    tree = output_tree(module.schema.fields, {entry.address: entry for entry in entries})
    representations = None
    if embeds:
        representations = pa.list_(
            pa.field(
                "element",
                pa.struct(
                    [
                        pa.field("path", pa.list_(pa.field("element", pa.string(), nullable=False)), nullable=False),
                        pa.field("parents", pa.list_(pa.field("element", pa.int32(), nullable=False)), nullable=False),
                        pa.field("output_index", pa.int32(), nullable=False),
                        pa.field(
                            "embedding",
                            pa.list_(pa.field("element", pa.float32()), module.schema.d_model),
                            nullable=False,
                        ),
                    ]
                ),
                nullable=False,
            )
        )
    if tree is None and representations is None:
        raise ValueError("model has no public prediction outputs; configure a reconstructing field or embed=True")
    compiled = OutputPlan(generation, retained, expected, tuple(entries), tree, representations)
    module.output_plans[key] = compiled
    return compiled


def coordinate(module: Model, entry: OutputEntry, prediction: Prediction, expected: int) -> pa.StructArray | None:
    """Write one extension's flat rich results without interpreting its values."""

    if entry.decoded is None:
        return None
    address = entry.address
    if TensorKey.state not in prediction.payload or TensorKey.inferred not in prediction.payload:
        raise ValueError(f"decoded prediction for {address} is missing shared state or inferred tensors")
    state_tensor = cast(torch.Tensor, prediction.payload[TensorKey.state])
    inferred_tensor = cast(torch.Tensor, prediction.payload[TensorKey.inferred])
    if state_tensor.ndim == 0 or state_tensor.shape[-1] != len(Tokens):
        raise ValueError(f"prediction state at {address} must end with {len(Tokens)} logits")
    if state_tensor.numel() != expected * len(Tokens) or inferred_tensor.numel() != expected:
        raise ValueError(f"prediction at {address} contains the wrong coordinate count; expected {expected}")
    if entry.writer is None:
        raise RuntimeError(f"compiled output for {address} has no extension writer")
    written = entry.writer(module=module, prediction=prediction, datatype=entry.decoded)
    if not isinstance(written, pa.StructArray):
        raise TypeError(f"extension write for {address} must return a pyarrow StructArray")
    if not written.type.equals(entry.decoded, check_metadata=True):
        raise TypeError(f"extension write for {address} returned {written.type}; expected {entry.decoded}")
    values = struct(
        {
            TensorKey.state.name: state(state_tensor),
            **{field.name: written.field(index) for index, field in enumerate(entry.decoded)},
            TensorKey.inferred.name: inferred(inferred_tensor),
        },
        cast(pa.StructType, entry.coordinate),
    )
    if len(values) != expected:
        raise ValueError(f"prediction at {address} wrote {len(values)} coordinates; expected {expected}")
    return values


def records(node: OutputNode, leaves: Mapping[Address, pa.Array], layouts: Mapping[Address, Layout]) -> pa.Array:
    """Merge sibling results into one record collection for each schema branch."""

    if not node.children:
        return leaves[node.address]
    arrays = [records(child, leaves, layouts) for child in node.children]
    root = node.address == Address()
    datatype = cast(pa.StructType, node.datatype if root else cast(pa.ListType, node.datatype).value_type)
    layout = None if root else layouts[node.address]
    values = pa.StructArray.from_arrays(
        arrays,
        fields=list(datatype),
        mask=None if layout is None else pa.array(~layout.slots),
    )
    if layout is None:
        return values
    return pa.ListArray.from_arrays(
        pa.array(layout.offsets, type=pa.int32()),
        values,
        type=node.datatype,
        mask=pa.array(~layout.valid),
    )


def representation(
    entry: OutputEntry, prediction: Prediction, projected: Projection, datatype: pa.StructType
) -> pa.StructArray:
    """Write selected embedding coordinates with schema paths and local ordinals."""

    tensor = cast(torch.Tensor, prediction.payload[TensorKey.embedding])
    vector = cast(pa.FixedSizeListType, datatype.field("embedding").type)
    if tensor.ndim == 0 or tensor.shape[-1] != vector.list_size:
        raise ValueError(f"prediction embedding at {entry.address} must end with model width {vector.list_size}")
    values = embedding(tensor, datatype=vector)
    if len(values) != projected.size:
        raise ValueError(
            f"prediction embedding at {entry.address} contains {len(values)} coordinates; expected {projected.size}"
        )
    retained = projected.indices >= 0
    count = int(retained.sum())
    depth = len(projected.parents)
    if count * max(len(entry.path), depth) > np.iinfo(np.int32).max:
        raise OverflowError(f"representation coordinates at {entry.address} exceed the Arrow int32 offset limit")
    path_type = datatype.field("path").type
    paths = pa.ListArray.from_arrays(
        pa.array(np.arange(count + 1, dtype=np.int64) * len(entry.path), type=pa.int32()),
        pc.take(
            pa.array(entry.path, type=pa.string()),
            pa.array(np.tile(np.arange(len(entry.path)), count), type=pa.int64()),
        ),
        type=path_type,
    )
    parents = (
        np.column_stack([axis[retained] for axis in projected.parents]).reshape(-1)
        if depth
        else np.empty(0, dtype=np.int32)
    )
    parent_values = pa.ListArray.from_arrays(
        pa.array(np.arange(count + 1, dtype=np.int64) * depth, type=pa.int32()),
        pa.array(parents, type=pa.int32()),
        type=datatype.field("parents").type,
    )
    return struct(
        {
            "path": paths,
            "parents": parent_values,
            "output_index": pa.array(projected.outputs[retained], type=pa.int32()),
            "embedding": pc.take(values, pa.array(projected.indices[retained], type=pa.int64())),
        },
        datatype,
    )


def envelope(module: Model, predictions: list[Prediction], rows: int, compiled: OutputPlan) -> dict[str, pa.Array]:
    """Assemble the natural prediction tree and separately indexed representations."""

    indexed: dict[Address, Prediction] = {}
    for prediction in predictions:
        address = Address(str(prediction.address))
        if address in indexed:
            raise ValueError(f"forward returned duplicate prediction address {address}")
        indexed[address] = prediction
    missing = compiled.expected - indexed.keys()
    if missing:
        raise ValueError(f"forward omitted configured prediction address(es): {', '.join(sorted(missing))}")
    unplanned = indexed.keys() - compiled.expected
    if unplanned:
        raise ValueError(f"forward returned unplanned prediction address(es): {', '.join(sorted(unplanned))}")

    leaves: dict[Address, pa.Array] = {}
    layouts: dict[Address, Layout] = {}
    embeddings: list[pa.Array] = []
    root_rows: list[np.ndarray] = []
    for entry in compiled.entries:
        prediction = indexed[entry.address]
        projected = projection(entry, prediction, rows)
        decoded = coordinate(module, entry, prediction, projected.size)
        if decoded is not None:
            # Record masks own structural holes; keep their nonnullable child buffers populated.
            leaves[entry.address] = pc.take(decoded, pa.array(np.maximum(projected.indices, 0), type=pa.int64()))
            for address, layout in zip(entry.branches, projected.axes, strict=True):
                prior = layouts.get(address)
                if prior is not None and any(
                    not np.array_equal(getattr(prior, name), getattr(layout, name))
                    for name in ("offsets", "valid", "slots")
                ):
                    raise ValueError(f"prediction siblings disagree on structural geometry for branch {address}")
                layouts[address] = layout
        if entry.embed != (TensorKey.embedding in prediction.payload):
            raise ValueError(
                f"prediction embedding at {entry.address} does not match its configured embed={entry.embed}"
            )
        if entry.embed:
            datatype = cast(pa.ListType, compiled.representations).value_type
            embeddings.append(representation(entry, prediction, projected, datatype))
            root_rows.append(projected.rows[projected.indices >= 0])

    columns: dict[str, pa.Array] = {}
    if compiled.tree is not None:
        columns["predictions"] = records(compiled.tree, leaves, layouts)
    if compiled.representations is not None:
        if sum(map(len, embeddings)) > np.iinfo(np.int32).max:
            raise OverflowError("prediction representations exceed the Arrow int32 offset limit")
        row_ids = np.concatenate(root_rows)
        values = pa.concat_arrays(embeddings)
        values = pc.take(values, pa.array(np.argsort(row_ids, kind="stable"), type=pa.int64()))
        offsets = np.concatenate(([0], np.bincount(row_ids, minlength=rows).cumsum()))
        columns["representations"] = pa.ListArray.from_arrays(
            pa.array(offsets, type=pa.int32()), values, type=compiled.representations
        )
    return columns


def vacant(compiled: OutputPlan) -> dict[str, pa.Array]:
    """Build selected columns with the planned types for a zero-row source."""

    columns: dict[str, pa.Array] = {}
    if compiled.tree is not None:
        columns["predictions"] = pa.array([], type=compiled.tree.datatype)
    if compiled.representations is not None:
        columns["representations"] = pa.array([], type=compiled.representations)
    return columns


def write(
    module: Model,
    predictions: list[Prediction],
    *,
    source: pa.Table,
    retain: Retain = (),
    compiled: OutputPlan | None = None,
) -> pa.Table:
    """Convert tensor predictions to one canonical Arrow output table."""

    if not isinstance(source, pa.Table):
        raise TypeError(f"source must be a pyarrow.Table, got {type(source).__name__}")
    active = plan(module, retain, refresh=module.training) if compiled is None else compiled
    if active.retain != retain:
        raise ValueError(f"compiled output plan retain {active.retain!r} does not match write retain {retain!r}")
    columns: dict[str, pa.Array] = {}
    if active.retain:
        columns["inputs"] = project(source, active.retain)
    columns.update(
        vacant(active)
        if not source.num_rows and not predictions
        else envelope(module, predictions, source.num_rows, active)
    )
    data = pa.table(columns).replace_schema_metadata(metadata(module))
    if data.num_rows != source.num_rows:
        raise ValueError("prediction output is not aligned with the canonical source")
    return data
