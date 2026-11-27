"""Coalesce schema-shaped inputs into canonical Arrow-backed fields."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from functools import cache
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from relflow.data.arrow import variants
from relflow.data.positions import place
from relflow.data.query import (
    Literal,
    Member,
    MissingFieldError,
    Query,
    compile,
    gather,
    listed,
    member,
    origin,
    parts,
    query,
)
from relflow.data.query import Projection as SourceProjection
from relflow.logging.config import logger
from relflow.structs.enums import Overflow, Strata, Tokens
from relflow.structs.tree import Address, Leaf, selections
from relflow.structs.twigs import Twig, indexed

if TYPE_CHECKING:
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch
    from relflow.structs.tasks import Selection
    from relflow.structs.tree import Field


@cache
def absent(address: Address, expression: str, datatype: pa.DataType) -> None:
    """Warn once per branch and record type when a learning split lacks its collection."""

    logger.bind(component="data", address=str(address)).warning(
        f"branch '{address}' reads collection {expression!r}, which is absent from the processed records "
        f"{datatype}; every slot counts as an absent collection. Rename the branch, set query=, or add the "
        "column in a preprocessor."
    )


def boolean(values: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """Lower an Arrow validity vector to a dense Boolean array."""

    if not values.null_count:
        return values.to_numpy(zero_copy_only=False).astype(bool, copy=False)
    # Arrow accepts Python scalars; the stubs require an explicit Scalar.
    filled = cast(pa.Array | pa.ChunkedArray, pc.fill_null(values, False))  # pyrefly: ignore[bad-specialization]
    return filled.to_numpy(zero_copy_only=False).astype(bool, copy=False)


def dictionary(datatype: pa.DataType) -> bool:
    """Return whether an Arrow type contains dictionary-encoded storage."""

    if pa.types.is_dictionary(datatype):
        return True
    if isinstance(datatype, pa.ExtensionType):
        return dictionary(datatype.storage_type)
    if pa.types.is_list(datatype) or pa.types.is_large_list(datatype) or pa.types.is_fixed_size_list(datatype):
        return dictionary(datatype.value_type)
    if pa.types.is_struct(datatype) or pa.types.is_union(datatype):
        return any(dictionary(field.type) for field in datatype)
    if pa.types.is_map(datatype):
        return dictionary(datatype.key_type) or dictionary(datatype.item_type)
    return False


def compact(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Compact ordinary chunks without combining independent dictionaries."""

    if not isinstance(values, pa.ChunkedArray):
        return values
    if values.num_chunks == 1:
        return values.chunk(0)
    if values.num_chunks and not dictionary(values.type):
        return values.combine_chunks()
    return values


def retain(values: pa.Array | pa.ChunkedArray, selected: np.ndarray) -> pa.Array | pa.ChunkedArray:
    """Filter Arrow values without invoking kernels on an invalid zero-chunk container."""

    if selected.all():
        return compact(values)
    if not selected.any():
        return compact(values.slice(0, 0))
    return compact(pc.filter(values, pa.array(selected, type=pa.bool_())))


def valid(values: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """Return logical validity, including the selected children of Arrow unions."""

    if isinstance(values, pa.ChunkedArray):
        if not values.chunks:
            return np.empty(0, dtype=bool)
        return np.concatenate([valid(chunk) for chunk in values.chunks])
    if isinstance(values, pa.ExtensionArray):
        return valid(values.storage)
    if pa.types.is_dictionary(values.type):
        values = cast(pa.DictionaryArray, values)
        present = boolean(pc.is_valid(values.indices))
        # Arrow accepts Python scalars; the stubs require an explicit Scalar.
        filled = cast(pa.Array, pc.fill_null(values.indices, 0))  # pyrefly: ignore[bad-specialization]
        positions = filled.to_numpy(zero_copy_only=False)
        result = present.copy()
        result[present] &= valid(values.dictionary)[positions[present]]
        return result
    if not pa.types.is_union(values.type) and not values.null_count:
        return np.ones(len(values), dtype=bool)
    if not pa.types.is_union(values.type):
        return boolean(pc.is_valid(values))

    codes, offsets = variants(values)
    values = cast(pa.UnionArray, values)
    result = np.zeros(len(values), dtype=bool)
    for index, code in enumerate(values.type.type_codes):
        selected = codes == code
        child = valid(values.field(index))
        result[selected] = child[offsets[selected]] if offsets is not None else child[selected]
    return result


@dataclass(frozen=True, slots=True)
class Collection:
    """Candidate records and the retained window of each parent in one Arrow chunk."""

    records: pa.Array
    first: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    base: int
    valid: np.ndarray | None


@dataclass(frozen=True, slots=True)
class Layout:
    """Structurally present records and their positions in dense model geometry."""

    records: pa.Array | pa.ChunkedArray
    placement: np.ndarray
    shape: tuple[int, ...]
    lengths: np.ndarray
    present: np.ndarray
    valid: np.ndarray
    collections: tuple[Collection, ...] = ()

    def retaining(self, kept: np.ndarray) -> Layout:
        """Project selected owners while preserving their dense slots and original neighbors."""
        if kept.all():
            return self
        collections = []
        offset = 0
        for collection in self.collections:
            valid = collection.valid
            count = int(collection.counts.sum() if valid is None else valid.sum())
            selection = kept[offset : offset + count]
            offset += count
            if selection.any():
                if valid is not None:
                    slots = valid.copy()
                    slots[valid] = selection
                    selection = slots
                collections.append(replace(collection, valid=selection))
        return replace(
            self,
            records=retain(self.records, kept),
            placement=self.placement[kept],
            collections=tuple(collections),
        )


@dataclass(frozen=True, slots=True)
class RaggedField:
    """Arrow-backed retained values and their dense model geometry."""

    values: pa.Array | pa.ChunkedArray
    state: pa.Array | pa.ChunkedArray
    placement: pa.Array | pa.ChunkedArray
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.values, (pa.Array, pa.ChunkedArray)):
            raise TypeError(f"RaggedField.values must be an Arrow array, got {type(self.values).__name__}")
        if not isinstance(self.state, (pa.Array, pa.ChunkedArray)) or self.state.type != pa.int8():
            datatype = getattr(self.state, "type", type(self.state).__name__)
            raise TypeError(f"RaggedField.state must be an int8 Arrow array, got {datatype}")
        if not isinstance(self.placement, (pa.Array, pa.ChunkedArray)) or self.placement.type != pa.int64():
            datatype = getattr(self.placement, "type", type(self.placement).__name__)
            raise TypeError(f"RaggedField.placement must be an int64 Arrow array, got {datatype}")
        if self.state.null_count or self.placement.null_count:
            raise ValueError("RaggedField state and placement cannot contain nulls")
        if not valid(self.values).all():
            raise ValueError("RaggedField values cannot contain nulls")
        if not self.shape or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in self.shape
        ):
            raise ValueError("RaggedField.shape must contain non-negative integer dimensions")
        if len(self.state) != math.prod(self.shape):
            raise ValueError(
                f"RaggedField state length must equal shape product {math.prod(self.shape)}, got {len(self.state)}"
            )
        if len(self.values) != len(self.placement):
            raise ValueError(
                f"RaggedField values/placement length mismatch: {len(self.values)} values, "
                f"{len(self.placement)} placements"
            )
        states = self.state.to_numpy(zero_copy_only=False)
        if len(states) and (states.min() < Tokens.valued.value or states.max() > Tokens.other.value):
            raise ValueError("RaggedField state contains an unknown token")

        positions = self.placement.to_numpy(zero_copy_only=False)
        if len(positions) and (np.any(positions < 0) or np.any(positions >= len(self.state))):
            raise ValueError("RaggedField placement contains an out-of-range dense position")
        if len(positions) > 1 and np.any(positions[1:] <= positions[:-1]):
            raise ValueError("RaggedField placement must be strictly increasing")
        valued = states == Tokens.valued.value
        if len(positions) != np.count_nonzero(valued) or not valued[positions].all():
            raise ValueError("RaggedField placement must contain every valued state position exactly once")

    @property
    def batch_size(self) -> int:
        return self.shape[0]

    @property
    def dense(self) -> np.ndarray:
        """Return model state as one dense int64 NumPy view."""

        values = self.state.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        return values.reshape(self.shape)

    def place(
        self,
        encoded: np.ndarray,
        *,
        fill: Any,
        value_shape: tuple[int, ...] = (),
    ) -> np.ndarray:
        """Scatter encoded retained values into this field's dense geometry."""
        if any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in value_shape):
            raise ValueError("value_shape entries must be non-negative integers")

        array = np.asarray(encoded)
        expected = (len(self.values), *value_shape)
        if array.shape != expected:
            raise ValueError(f"encoded values must have shape {expected}, got {array.shape}")

        placed = np.full((*self.shape, *value_shape), fill, dtype=array.dtype)
        flat_shape = (len(self.state), *value_shape)
        placement = self.placement.to_numpy(zero_copy_only=False)
        placed.reshape(flat_shape)[placement] = array
        return placed


@dataclass(frozen=True, slots=True)
class Decision:
    """One policy selection in its owner's fixed coordinate geometry."""

    rule: Selection
    selected: np.ndarray
    shape: tuple[int, ...]
    fields: frozenset[Address] | None = None


@dataclass(frozen=True, slots=True)
class Projection:
    """Pristine Arrow values plus dense Boolean model-input and objective routing."""

    pristine: RaggedField
    visible: np.ndarray
    present: np.ndarray
    trainable: np.ndarray
    inferred: np.ndarray
    vacant: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.vacant, bool):
            raise TypeError(f"Projection.vacant must be a boolean, got {type(self.vacant).__name__}")
        if self.vacant and (len(self.pristine.values) or not pa.types.is_null(self.pristine.values.type)):
            raise ValueError("a vacant Projection must carry one zero-length Arrow NullArray")

        size = math.prod(self.pristine.shape)
        for name in ("visible", "present", "trainable", "inferred"):
            values = getattr(self, name)
            if not isinstance(values, np.ndarray) or values.dtype != np.bool_:
                datatype = getattr(values, "dtype", type(values).__name__)
                raise TypeError(f"Projection.{name} must be a Boolean NumPy array, got {datatype}")
            if values.shape != (size,):
                raise ValueError(f"Projection.{name} shape must be ({size},), got {values.shape}")

    def split(self) -> tuple[RaggedField, RaggedField]:
        """Split pristine values into input and target fields."""
        values = self.pristine.values
        state = self.pristine.state.to_numpy(zero_copy_only=False).astype(np.int8, copy=True)
        placement = self.pristine.placement.to_numpy(zero_copy_only=False)

        input_state = state.copy()
        input_state[~self.present] = Tokens.padded.value
        input_state[self.present & ~self.visible] = Tokens.masked.value
        input_values = self.visible[placement]
        input_placement = placement[input_values]

        target_state = np.full(len(state), Tokens.padded.value, dtype=np.int8)
        target_state[self.trainable] = state[self.trainable]
        target_values = self.trainable[placement]
        target_placement = placement[target_values]

        if self.vacant and (input_values.any() or target_values.any()):
            raise ValueError("vacant projection values cannot satisfy visible input or target positions")

        return (
            RaggedField(
                values=retain(values, input_values),
                state=pa.array(input_state, type=pa.int8()),
                placement=pa.array(input_placement, type=pa.int64()),
                shape=self.pristine.shape,
            ),
            RaggedField(
                values=retain(values, target_values),
                state=pa.array(target_state, type=pa.int8()),
                placement=pa.array(target_placement, type=pa.int64()),
                shape=self.pristine.shape,
            ),
        )


@dataclass(frozen=True, slots=True)
class Sources:
    """Arrow query inputs and owner coordinates for one tensorfield.

    Inputs follow the request's query order. Each provides Arrow values and
    structural presence for the retained owner records. ``select`` applies
    the field's mask decisions to one value and presence per owner record.
    """

    inputs: tuple[SourceProjection[pa.Array | pa.ChunkedArray], ...]
    placement: np.ndarray
    shape: tuple[int, ...]
    decisions: tuple[Decision, ...]
    strata: Strata
    address: Address
    vacant: bool = False
    collections: tuple[Collection, ...] = ()

    @property
    def rows(self) -> int:
        """Number of structurally retained owner records."""
        return len(self.placement)

    def needs_targets(self, schema: Schema) -> bool:
        """Build answer tensors for a fixed objective or a field selected for this batch."""
        return self.address in schema.roles.fixed_targets or any(d.fields is not None for d in self.decisions)

    def single(self) -> Projection:
        """Route a single source, or an absent prediction target."""
        if self.vacant:
            return route(self.placement, vacancy(self.shape), self.decisions, strata=self.strata, vacant=True)
        if len(self.inputs) != 1:
            raise ValueError(f"field '{self.address}' expects one source, got {len(self.inputs)}")
        source = self.inputs[0]
        if not isinstance(source, SourceProjection):
            raise TypeError(f"field '{self.address}' expects a query projection, got {type(source).__name__}")
        return self.select(source.values, source.present)

    def select(self, values: pa.Array | pa.ChunkedArray, present: pa.Array | pa.ChunkedArray) -> Projection:
        """Apply the field's geometry and masks to selected Arrow values."""
        value = selected(values, present, rows=self.rows, address=self.address)
        field = pristine(self.placement, self.shape, value)
        return route(self.placement, field, self.decisions, strata=self.strata, vacant=False)


@dataclass(frozen=True, slots=True)
class Coalesced:
    """Policy-resolved fields, their shared pristine branch layouts, and indexed branch positions."""

    fields: dict[Address, Sources]
    layouts: dict[Address, Layout]
    positions: dict[Address, np.ndarray]


def head(expression: Query) -> str:
    """Return the record member a query reads first."""

    first = expression.steps[0]
    if isinstance(first, Member):
        return first.name
    if isinstance(first, Literal) and isinstance(first.value, str):
        return first.value
    raise ValueError(f"query {expression.text!r} must begin with a field name")


def referenced(branch: Branch) -> set[str]:
    """Return the record members that a branch, its masks, and its children read."""

    names = {head(compile(branch.query))} if branch.query is not None else set()
    for rule in selections(branch):
        if rule.query is not None:
            names.add(head(compile(rule.query)))
    if branch.query is not None:
        return names
    for child in branch.fields:
        if getattr(child, "type", None) == "branch":
            branch_child = cast("Branch", child)
            names.add(head(compile(branch_child.query)) if branch_child.query is not None else cast(str, child.name))
        else:
            for expression in cast(Leaf | Twig, child).queries():
                depth, path = origin(expression, address=str(child.address))
                if not depth:
                    names.add(head(path))
        for rule in () if isinstance(child, Twig) else selections(child):
            if rule.query is not None and getattr(child, "type", None) != "branch":
                names.add(head(compile(rule.query)))
    for child in branch.descendants:
        if not isinstance(child, Leaf):
            continue
        owners = [node for node in child.path if node.type == "branch"]
        for expression in child.queries():
            depth, path = origin(expression, address=str(child.address))
            if depth and depth < len(owners) and owners[-1 - depth] is branch:
                names.add(head(path))
    return names


def root(values: pa.Table, branch: Branch, *, size: int | None = None) -> Layout:
    """Build the singleton root layout, combining only the columns the schema reads."""

    rows = values.num_rows if values.num_columns else (0 if size is None else size)
    names = referenced(branch)
    selected = values.select([index for index, name in enumerate(values.column_names) if name in names])
    records: pa.Array | pa.ChunkedArray
    if not selected.num_columns:
        records = pa.StructArray.from_buffers(pa.struct([]), rows, [None])
    elif not rows:
        # Arrow accepts nested zero-chunk types; its stubs omit that overload.
        records = pa.chunked_array([], type=pa.struct(list(selected.schema)))  # pyrefly: ignore[no-matching-overload]
    else:
        records = compact(selected.to_struct_array())
    if branch.query is None:
        return Layout(
            records=records,
            placement=np.arange(rows, dtype=np.int64),
            shape=(rows, 1),
            lengths=np.ones(rows, dtype=np.int64),
            present=np.ones(rows, dtype=bool),
            valid=np.ones(rows, dtype=bool),
        )

    selected = query(records, branch.query, address=str(branch.address))
    if selected.present.type != pa.bool_():
        raise ValueError(f"root query {branch.query!r} must select one object per observation")

    present = boolean(selected.present)
    retained = present & valid(selected.values)
    origin = np.flatnonzero(retained).astype(np.int64, copy=False)
    return Layout(
        records=retain(selected.values, retained),
        placement=origin,
        shape=(rows, 1),
        lengths=np.ones(rows, dtype=np.int64),
        present=np.ones(rows, dtype=bool),
        valid=np.ones(rows, dtype=bool),
    )


def collection_source(
    records: pa.Array | pa.ChunkedArray, expression: str, *, kept: np.ndarray | None, address: Address
) -> SourceProjection[pa.Array | pa.ChunkedArray]:
    """Read the child collection only for retained parents, preserving independent dictionaries."""
    if kept is None or kept.all():
        return query(records, expression, address=str(address))

    chunks = records.chunks if isinstance(records, pa.ChunkedArray) else [records]
    offset = 0
    selected = []
    for chunk in chunks:
        indices = np.flatnonzero(kept[offset : offset + len(chunk)])
        offset += len(chunk)
        if len(indices):
            selected.append(gather(chunk, expression, pa.array(indices, type=pa.int64()), address=str(address)))
    if len(selected) == 1:
        return selected[0]
    return SourceProjection(
        values=pa.chunked_array([item.values for item in selected]),
        present=pa.chunked_array([item.present for item in selected]),
    )


def retention(layout: Layout, node: Field, *, strata: Strata, seed: int, epoch: int, ordinal: int) -> np.ndarray | None:
    """Draw one training ablation decision per structurally present owner of this node."""
    if strata != Strata.train or not node.ablate or not len(layout.placement):
        return None
    payload = f"relflow-ablate-v1:{seed}:{epoch}:{ordinal}:{node.address}".encode()
    random = np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))
    return random.random(len(layout.placement)) >= node.ablate


def descend(
    layout: Layout, branch: Branch, address: Address, *, strata: Strata, seed: int, epoch: int, ordinal: int
) -> Layout:
    """Retain collection slots before resolving their shared execution extent."""

    parent_size = math.prod(layout.shape)
    retained_lengths = np.zeros(parent_size, dtype=np.int64)
    collection_present = np.zeros(parent_size, dtype=bool)
    collection_valid = np.zeros(parent_size, dtype=bool)
    owners = layout.placement
    kept = retention(layout, branch, strata=strata, seed=seed, epoch=epoch, ordinal=ordinal)
    if kept is not None:
        owners = owners[kept]

    expression = branch.query or member(cast(str, branch.name))
    selected = None
    if len(owners):
        try:
            selected = collection_source(layout.records, expression, kept=kept, address=branch.address)
        except MissingFieldError:
            if strata != Strata.predict:
                absent(branch.address, expression, layout.records.type)
    if selected is None or pa.types.is_null(selected.values.type):
        if selected is not None:
            collection_present[owners] = boolean(selected.present)
        empty = pa.array([], type=pa.struct([]))
        return Layout(
            records=empty,
            placement=np.empty(0, dtype=np.int64),
            shape=(*layout.shape, 0),
            lengths=retained_lengths.reshape(layout.shape),
            present=collection_present.reshape(layout.shape),
            valid=collection_valid.reshape(layout.shape),
        )
    if selected.present.type != pa.bool_():
        raise ValueError(f"branch query {expression!r} must select one list per parent")
    if not listed(selected.values.type):
        raise ValueError(f"branch query {expression!r} expected a list, got {selected.values.type}")

    present = boolean(selected.present)
    source = compact(selected.values)
    chunks = source.chunks if isinstance(source, pa.ChunkedArray) else [source]
    retained: list[tuple[np.ndarray, np.ndarray, pa.Array, np.ndarray, np.ndarray]] = []
    randomized = branch.overflow == Overflow.random and strata == Strata.train
    ranges = np.zeros(parent_size, dtype=np.int64) if randomized else None
    tail = branch.overflow == Overflow.tail or (branch.overflow == Overflow.random and not randomized)
    offset = 0
    for values in chunks:
        parent_placement = owners[offset : offset + len(values)]
        available = present[offset : offset + len(values)]
        offset += len(values)

        list_offsets, children, valid_list = parts(values)
        lengths = np.diff(list_offsets)
        collection_present[parent_placement] = available
        available = available & valid_list
        collection_valid[parent_placement] = available
        if (
            branch.length is not None
            and branch.overflow == Overflow.error
            and np.any(available & (lengths > branch.length))
        ):
            axis = len(layout.shape)
            raise ValueError(f"branch overflow at dimension {axis} for {address}: capacity is {branch.length}")

        counts = np.where(available, lengths if branch.length is None else np.minimum(lengths, branch.length), 0)
        retained_lengths[parent_placement] = counts
        starts = list_offsets[:-1] + (lengths - counts if tail else 0)
        if ranges is not None:
            ranges[parent_placement] = np.where(available, lengths - counts, 0)
        retained.append((list_offsets, starts, children, counts, parent_placement))

    shifts = None
    if ranges is not None:
        payload = f"relflow-overflow-v1:{seed}:{epoch}:{ordinal}:{branch.address}".encode()
        random = np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))
        # One canonical parent vector keeps draws independent of Arrow chunking.
        shifts = random.integers(0, ranges + 1, dtype=np.int64)

    extent = int(retained_lengths.max(initial=0))
    shape = (*layout.shape, extent)
    record_chunks: list[pa.Array] = []
    placement_chunks: list[np.ndarray] = []
    collections: list[Collection] = []
    for list_offsets, starts, children, counts, parent_placement in retained:
        if shifts is not None:
            starts = starts + shifts[parent_placement]
        total = int(counts.sum())
        if total == 0:
            record_chunks.append(children.slice(0, 0))
            placement_chunks.append(np.empty(0, dtype=np.int64))
            continue

        parents = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        boundaries = np.repeat(np.cumsum(counts) - counts, counts)
        slots = np.arange(total, dtype=np.int64) - boundaries
        indices = starts[parents] + slots
        if int(indices[-1]) - int(indices[0]) + 1 == total:
            records = children.slice(int(indices[0]), total)
        else:
            records = pc.take(children, pa.array(indices, type=pa.int64()))
        valid_record = valid(records)
        placement = parent_placement[parents] * extent + slots
        if valid_record.any():
            start = int(list_offsets[0])
            stop = int(list_offsets[-1])
            collections.append(
                Collection(
                    records=children.slice(start, stop - start),
                    first=list_offsets[:-1],
                    starts=starts,
                    counts=counts,
                    base=start,
                    valid=None if valid_record.all() else valid_record,
                )
            )
        if valid_record.all():
            record_chunks.append(records)
            placement_chunks.append(placement)
        else:
            record_chunks.append(pc.filter(records, pa.array(valid_record)))
            placement_chunks.append(placement[valid_record])

    records = (
        record_chunks[0] if len(record_chunks) == 1 else pa.chunked_array(record_chunks, type=source.type.value_type)
    )

    return Layout(
        records=records,
        placement=np.concatenate(placement_chunks) if placement_chunks else np.empty(0, dtype=np.int64),
        shape=shape,
        lengths=retained_lengths.reshape(layout.shape),
        present=collection_present.reshape(layout.shape),
        valid=collection_valid.reshape(layout.shape),
        collections=tuple(collections),
    )


def vacancy(shape: tuple[int, ...]) -> RaggedField:
    """Represent a source-less leaf without inventing an extension datatype."""

    return RaggedField(
        values=pa.nulls(0),
        state=pa.array(np.full(math.prod(shape), Tokens.padded.value, dtype=np.int8)),
        placement=pa.array([], type=pa.int64()),
        shape=shape,
    )


def source(
    layout: Layout,
    expression: str,
    *,
    ancestors: tuple[Layout, ...],
    address: Address,
) -> SourceProjection[pa.Array | pa.ChunkedArray]:
    """Resolve a local source or gather an ancestor source by retained coordinates."""
    depth, path = origin(expression, address=str(address))
    if not depth:
        return query(layout.records, path, address=str(address))
    if depth > len(ancestors):
        raise ValueError(
            f"source query {expression!r} at '{address}' ascends {depth} owner branches, "
            f"but only {len(ancestors)} ancestor branches exist; keep the source within the model root"
        )
    owner = ancestors[-depth]
    selected = query(owner.records, path, address=str(address))
    stride = math.prod(layout.shape[len(owner.shape) :])
    positions = layout.placement // stride if stride else np.empty(0, dtype=np.int64)
    indices = np.searchsorted(owner.placement, positions)
    if np.any(indices >= len(owner.placement)) or not np.array_equal(owner.placement[indices], positions):
        raise RuntimeError(f"source query {expression!r} at '{address}' has no retained ancestor coordinate")
    gather = pa.array(indices, type=pa.int64())
    return SourceProjection(values=pc.take(selected.values, gather), present=pc.take(selected.present, gather))


def selected(
    values: object, present: object, *, rows: int, address: Address
) -> SourceProjection[pa.Array | pa.ChunkedArray]:
    """Validate one selected value and presence per owner record."""
    for name, array in (("values", values), ("presence", present)):
        if not isinstance(array, (pa.Array, pa.ChunkedArray)):
            raise TypeError(f"field '{address}' {name} must be an Arrow array, got {type(array).__name__}")
        if len(array) != rows:
            raise ValueError(f"field '{address}' {name} must retain {rows} owner rows, got {len(array)}")
    values = cast(pa.Array | pa.ChunkedArray, values)
    present = cast(pa.Array | pa.ChunkedArray, present)
    datatype = present.type
    while listed(datatype):
        datatype = datatype.value_type
    if datatype != pa.bool_() or (present.type == pa.bool_() and present.null_count):
        raise ValueError(
            f"field '{address}' presence must contain one non-null Boolean per owner row, "
            f"got {present.type} with {present.null_count} nulls"
        )
    return SourceProjection(values=values, present=present)


def pristine(
    placement: np.ndarray, shape: tuple[int, ...], selected: SourceProjection[pa.Array | pa.ChunkedArray]
) -> RaggedField:
    """Project one untouched leaf while preserving extension-owned Arrow values."""
    present = (
        boolean(selected.present) if selected.present.type == pa.bool_() else boolean(pc.is_valid(selected.present))
    )
    available = valid(selected.values)
    valued = present & available

    state = np.full(math.prod(shape), Tokens.padded.value, dtype=np.int8)
    state[placement[present]] = Tokens.null.value
    state[placement[valued]] = Tokens.valued.value
    placement = placement[valued]
    return RaggedField(
        values=retain(selected.values, valued),
        state=pa.array(state, type=pa.int8()),
        placement=pa.array(placement, type=pa.int64()),
        shape=shape,
    )


def active(rule: Selection, strata: Strata) -> bool:
    """Return whether one normalized policy applies in a stratum."""

    if strata == Strata.train:
        return True
    if rule.action == "hide":
        return False
    return strata != Strata.predict or not rule.sampled


def sample_fields(
    schema: Schema, *, seed: int, epoch: int, ordinal: int, strata: Strata
) -> dict[tuple[Address, Selection], frozenset[Address]]:
    """Draw once per declaration and batch, independently of row sampling and encoder visibility."""
    chosen: dict[tuple[Address, Selection], frozenset[Address]] = {}
    if strata == Strata.predict:
        return chosen
    for (owner, rule), pool in schema.field_pools.items():
        limit = cast(int, rule.fields)
        if limit >= len(pool):
            targets = frozenset(pool)
        else:
            stream = json.dumps(
                (
                    "relflow-fields-v1",
                    seed,
                    strata.value,
                    epoch if strata == Strata.train else 0,
                    ordinal,
                    owner,
                    limit,
                    rule.query,
                    rule.rate,
                    rule.span,
                    rule.ignore,
                ),
                separators=(",", ":"),
            ).encode()
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(stream).digest()[:8], "big"))
            targets = frozenset(pool[i] for i in rng.choice(len(pool), size=limit, replace=False))
        chosen[owner, rule] = targets
    return chosen


def eligible(layout: Layout, rule: Selection, *, address: Address) -> np.ndarray:
    """Resolve one policy query against its owner's retained records."""

    if not len(layout.records):
        return np.empty(0, dtype=bool)
    if rule.query is None:
        return np.ones(len(layout.records), dtype=bool)

    option = rule.action
    selected = query(layout.records, rule.query, address=str(address))
    if len(selected.values) != len(layout.records):
        raise ValueError(
            f"{option} query {rule.query!r} at '{address}' returned {len(selected.values)} values for "
            f"{len(layout.records)} owner records"
        )
    if selected.values.type != pa.bool_():
        raise TypeError(
            f"{option} query {rule.query!r} at '{address}' must return Boolean values, got {selected.values.type}"
        )
    if selected.present.type != pa.bool_():
        raise ValueError(
            f"{option} query {rule.query!r} at '{address}' must return one scalar Boolean per owner record"
        )
    structural = boolean(selected.present)
    if not structural.all():
        missing = int((~structural).sum())
        raise ValueError(
            f"{option} query {rule.query!r} at '{address}' is structurally absent for {missing} owner record(s); "
            "fill missing selector values explicitly"
        )
    if selected.values.null_count:
        raise ValueError(
            f"{option} query {rule.query!r} at '{address}' contains {selected.values.null_count} null value(s); "
            "fill null selector values explicitly"
        )
    return boolean(selected.values)


def scores(
    layout: Layout,
    *,
    address: Address,
    rule: Selection,
    seed: int,
    epoch: int,
    ordinal: int,
    strata: Strata,
) -> np.ndarray:
    """Draw reproducible scores in canonical record order, distinct for every batch ordinal."""

    if not len(layout.records):
        return np.empty(0, dtype=np.uint64)

    effective_epoch = epoch if strata == Strata.train else 0
    selection = json.dumps(
        (rule.query, float(rule.rate).hex() if rule.rate is not None else None),
        separators=(",", ":"),
    )
    if rule.span is not None:
        selection += f":span={rule.span}"
    if rule.action == "hide":
        # Input dropout and reconstruction draw independently even at the same rate.
        selection += ":hide"
    payload = f"relflow-mask-v3:{seed}:{strata}:{effective_epoch}:{ordinal}:{address}:{selection}".encode()
    random = np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))
    return random.integers(0, np.iinfo(np.uint64).max, len(layout.records), dtype=np.uint64)


def select(
    layout: Layout,
    rule: Selection,
    *,
    address: Address,
    seed: int,
    epoch: int,
    ordinal: int,
    strata: Strata,
) -> np.ndarray:
    """Resolve one literal, query-backed, or uniformly sampled policy."""

    chosen = eligible(layout, rule, address=address)
    dense = np.zeros(math.prod(layout.shape), dtype=bool)
    if not len(chosen):
        return dense
    if rule.rate is not None and rule.rate < 1:
        # Each row in a long eligible run has up to span candidate starts.
        # P(at least one start) = 1 - (1 - start_probability) ** span.
        span = rule.span or 1
        probability = rule.rate if span == 1 else -math.expm1(math.log1p(-rule.rate) / span)
        threshold = np.uint64(int(probability * np.iinfo(np.uint64).max))
        draws = scores(layout, address=address, rule=rule, seed=seed, epoch=epoch, ordinal=ordinal, strata=strata)
        starts = chosen & (draws <= threshold)
        if span == 1:
            chosen = starts
        else:
            index = np.arange(len(chosen))
            parents = layout.placement // layout.shape[-1]
            new_run = np.r_[True, (parents[1:] != parents[:-1]) | (np.diff(layout.placement) != 1)]
            # A missing observation or query-ineligible row ends the current run.
            first = np.maximum.accumulate(np.where(~chosen, index + 1, np.where(new_run, index, 0)))
            latest = np.maximum.accumulate(np.where(starts, index, -span))
            chosen = (latest >= first) & (index - latest < span)
    dense[layout.placement] = chosen
    return dense


def resolve(
    layout: Layout,
    node: Any,
    *,
    seed: int,
    epoch: int,
    ordinal: int,
    strata: Strata,
    sampled: dict[tuple[Address, Selection], frozenset[Address]],
) -> tuple[Decision, ...]:
    """Resolve every active policy once in one node's owner geometry."""

    return tuple(
        Decision(
            rule=rule,
            selected=select(
                layout,
                rule,
                address=node.address,
                seed=seed,
                epoch=epoch,
                ordinal=ordinal,
                strata=strata,
            ),
            shape=layout.shape,
            fields=sampled.get((node.address, rule)) if rule.fields is not None else None,
        )
        for rule in selections(node)
        if active(rule, strata)
        and (not rule.ignore or any(child.name not in rule.ignore for child in node.fields))
        and (rule.fields is None or (node.address, rule) in sampled)
    )


def broadcast(decision: Decision, shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast one ancestor decision through fixed descendant geometry."""

    if shape[: len(decision.shape)] != decision.shape:
        raise ValueError(f"selection shape {decision.shape} is not a prefix of descendant shape {shape}")
    return np.repeat(decision.selected, math.prod(shape[len(decision.shape) :]))


def project(
    layout: Layout,
    request: Any,
    decisions: tuple[Decision, ...],
    *,
    strata: Strata,
    ancestors: tuple[Layout, ...],
) -> Sources:
    """Resolve a leaf's source queries and owner geometry."""

    allow_vacancy = strata == Strata.predict and any(
        decision.rule.action != "hide" and decision.rule.query is None and not decision.rule.sampled
        for decision in decisions
    )
    source_less = bool(
        not len(layout.records)
        or allow_vacancy
        and request.query is None
        and pa.types.is_struct(layout.records.type)
        and not layout.records.type.get_all_field_indices(request.name)
    )
    inputs: tuple[SourceProjection[pa.Array | pa.ChunkedArray], ...] = ()
    if not source_less:
        try:
            inputs = tuple(
                source(layout, expression, ancestors=ancestors, address=request.address)
                for expression in request.queries()
            )
        except MissingFieldError:
            if not allow_vacancy:
                raise
            source_less = True
    return Sources(
        inputs=inputs,
        placement=layout.placement,
        shape=layout.shape,
        decisions=decisions,
        strata=strata,
        address=request.address,
        vacant=source_less,
        collections=layout.collections,
    )


def route(
    placement: np.ndarray, field: RaggedField, decisions: tuple[Decision, ...], *, strata: Strata, vacant: bool
) -> Projection:
    """Route pristine values to visible inputs, training answers and predictions."""
    state = field.state.to_numpy(zero_copy_only=False)
    observed = state != Tokens.padded.value
    owner = np.zeros(len(state), dtype=bool)
    owner[placement] = True
    requested = np.zeros(len(state), dtype=bool)
    skipped = np.zeros(len(state), dtype=bool)
    masked = np.zeros(len(state), dtype=bool)

    for decision in decisions:
        selected = broadcast(decision, field.shape) & owner
        if decision.rule.action != "hide":
            requested |= selected
        if decision.rule.skips_input:
            skipped |= selected
        else:
            masked |= selected

    predicting = strata == Strata.predict
    modeled = observed.copy()
    if predicting:
        modeled |= requested
    skipped &= modeled
    masked &= modeled & ~skipped
    trainable = np.zeros(len(state), dtype=bool) if predicting else observed & requested
    inferred = requested if predicting else np.zeros(len(state), dtype=bool)
    present = modeled & ~skipped
    visible = observed & ~masked & ~skipped
    return Projection(
        pristine=field,
        visible=visible,
        present=present,
        trainable=trainable,
        inferred=inferred,
        vacant=vacant,
    )


def coalesce(
    values: pa.Table,
    schema: Schema,
    strata: Strata | str,
    *,
    seed: int = 0,
    epoch: int = 0,
    ordinal: int = 0,
) -> Coalesced:
    """Resolve schema geometry and masks before datatype conversion.

    ``ordinal`` numbers the batch within its epoch so sampled masks differ
    between batches while every batch stays reproducible.
    """
    if not isinstance(values, pa.Table):
        raise TypeError(f"coalesce values must be a pyarrow.Table, got {type(values).__name__}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"coalesce seed must be an integer, got {type(seed).__name__}")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("coalesce epoch must be a non-negative integer")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
        raise ValueError("coalesce ordinal must be a non-negative batch index")

    strata = Strata.normalize(strata)
    sampled = sample_fields(schema, seed=seed, epoch=epoch, ordinal=ordinal, strata=strata)
    addresses = tuple(schema.active_requests)
    descendants: dict[Address, list[Address]] = {}
    for address in addresses:
        for node in schema.requests[address].path:
            if getattr(node, "type", None) == "branch":
                descendants.setdefault(node.address, []).append(address)

    fields: dict[Address, Sources] = {}
    layouts: dict[Address, Layout] = {}
    positions: dict[Address, np.ndarray] = {}

    def visit(
        branch: Branch,
        layout: Layout,
        inherited: tuple[Decision, ...],
        ancestors: tuple[Layout, ...],
    ) -> None:
        layouts[branch.address] = layout
        index = indexed(branch)
        if index is not None:
            positions[branch.address] = place(layout, index)
        owned = resolve(layout, branch, seed=seed, epoch=epoch, ordinal=ordinal, strata=strata, sampled=sampled)
        for child in branch.fields:
            # Twigs shape their branch's geometry above; they are never coalesced as fields.
            if isinstance(child, Twig):
                continue
            scoped = (*inherited, *(decision for decision in owned if child.name not in decision.rule.ignore))
            if getattr(child, "type", None) == "branch":
                child = cast("Branch", child)
                active = descendants.get(child.address, [])
                if active:
                    child_layout = descend(
                        layout, child, active[0], strata=strata, seed=seed, epoch=epoch, ordinal=ordinal
                    )
                    visit(child, child_layout, scoped, (*ancestors, layout))
            elif child.address in schema.active_requests:
                kept = retention(layout, child, strata=strata, seed=seed, epoch=epoch, ordinal=ordinal)
                owners = layout if kept is None else layout.retaining(kept)
                decisions = resolve(
                    owners,
                    child,
                    seed=seed,
                    epoch=epoch,
                    ordinal=ordinal,
                    strata=strata,
                    sampled=sampled,
                )
                inherited_fields = (d for d in scoped if d.fields is None or child.address in d.fields)
                fields[child.address] = project(
                    owners, child, (*inherited_fields, *decisions), strata=strata, ancestors=ancestors
                )

    layout = root(values, schema.fields)
    visit(schema.fields, layout, (), ())

    return Coalesced(fields=fields, layouts=layouts, positions=positions)


__all__ = ["Coalesced", "Projection", "RaggedField"]
