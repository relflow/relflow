"""Forward, loss, encoding, and Arrow output for RelFlow models."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict, cast, overload

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from tensordict import TensorDict

from relflow.architecture.binding import bind
from relflow.architecture.contracts import sanitize
from relflow.architecture.encoder import BranchEncoder, locate, sequence
from relflow.architecture.node import NodeModule
from relflow.architecture.output import Layout, Projection, projection
from relflow.data.arrow import Encoded, Retain, mappings, retention
from relflow.data.datasets.base import EncodedInput
from relflow.data.iterables import encode as encode_batch
from relflow.data.processors import (
    Postprocessor,
    PostprocessorInput,
    Preprocessor,
    PreprocessorInput,
    apply,
)
from relflow.distributed import all_reduce_max
from relflow.logging import logger
from relflow.structs.enums import Component, Metric, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address, Node
from relflow.tensorfields.base import (
    TENSORFIELDS,
    DecoderBase,
    EmbedderBase,
    Extension,
    RequestBase,
    TensorFieldBase,
    Write,
)
from relflow.tensorfields.output import STATE, embedding, inferred, state, struct

if TYPE_CHECKING:
    from relflow.architecture.root import Model

PredictionInput: TypeAlias = pa.Table | pa.RecordBatch | Sequence[Mapping[str, object]]
RESERVED = frozenset({TensorKey.state.name, TensorKey.inferred.name})


class Output(TypedDict):
    """Scalar objective returned by a nonprediction step that was not skipped."""

    loss: torch.Tensor


@dataclass(frozen=True, slots=True)
class DecoderRoute:
    """Schema-owned inputs and output roles for one leaf decoder."""

    address: Address
    heritage: tuple[Address, ...]
    embed: bool
    decode: bool


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Immutable graph routing retained until the model resets its contracts."""

    requests: tuple[Address, ...]
    depthwise: tuple[tuple[Address, ...], ...]
    embeds: frozenset[Address]
    objectives: tuple[Address, ...]
    decoders: tuple[DecoderRoute, ...]


def execution(module: Model) -> ExecutionPlan:
    """Resolve schema predicates and decoder routes once per model generation."""

    if module.execution_plan is not None:
        return module.execution_plan

    requests = tuple(module.schema.active_requests)
    embeds = frozenset(module.schema.embed)
    objectives = tuple(module.schema.objectives)
    decodes = frozenset(module.schema.decodes)
    selected = set(objectives) | decodes | embeds
    decoders = tuple(
        DecoderRoute(
            address=address,
            heritage=tuple(module.schema.requests[address].heritage),
            embed=address in embeds,
            decode=address in decodes,
        )
        for address in requests
        if address in selected
    )
    module.execution_plan = ExecutionPlan(
        requests=requests,
        depthwise=tuple(tuple(depth) for depth in reversed(module.schema.depthwise)),
        embeds=embeds,
        objectives=objectives,
        decoders=decoders,
    )
    return module.execution_plan


def participation(
    module: Model,
    inputs: EncodedInput,
    strata: Strata,
) -> tuple[set[Address], set[Address]]:
    """Resolve local objectives and those selected on any distributed rank."""

    objectives = execution(module).objectives
    if strata == Strata.predict or not objectives:
        return set(), set()

    local = torch.stack([cast(TensorFieldBase, inputs[address]).trainable.any() for address in objectives]).to(
        dtype=torch.uint8
    )
    selected = torch.stack((local, all_reduce_max(local.clone()))).tolist()
    return (
        {address for address, active in zip(objectives, selected[0], strict=True) if active},
        {address for address, active in zip(objectives, selected[1], strict=True) if active},
    )


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


def ingress(source: PredictionInput) -> pa.Table | pa.RecordBatch:
    """Adapt one small Python prediction collection to Arrow exactly once."""

    if isinstance(source, (pa.Table, pa.RecordBatch)):
        return source
    if isinstance(source, (str, bytes)) or not isinstance(source, Sequence):
        raise TypeError(
            "prediction input must be a pyarrow Table, pyarrow RecordBatch, "
            f"or a sequence of mappings; got {type(source).__name__}"
        )
    if not source:
        raise ValueError("an empty Python prediction sequence has no Arrow schema; pass a typed Arrow table")
    try:
        return pa.Table.from_pylist(mappings(source, context="Python prediction observations"))
    except (pa.ArrowException, TypeError, ValueError) as error:
        raise TypeError(f"Python prediction observations are not Arrow-compatible: {error}") from error


def observations(
    module: Model,
    batch: pa.Table | pa.RecordBatch,
    *,
    preprocess: PreprocessorInput,
    strata: Strata,
) -> pa.Table:
    """Produce the canonical Arrow observations before encoding or batching."""

    from relflow.data.datasets.arrow import convert, merge, process

    source = convert(batch)
    processors = Preprocessor.normalize(preprocess)
    if not processors:
        return source

    source = merge(
        process(
            (source,),
            preprocessor=processors,
            strata=strata,
            schema=module.schema,
            encoding_context=module.interprocess_encoding_context,
        )
    )
    if source is None:
        raise ValueError("preprocessor pipeline returned no observations")
    return source


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
    """Identify the public output contract and model's recorded package version."""

    return {b"relflow.prediction_format": b"2", b"relflow.model_version": module.version.encode()}


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


class ModelRuntime:
    """Own runtime behavior that depends on an already-built model graph."""

    @staticmethod
    def forward(
        module: Model,
        inputs: EncodedInput,
        *,
        strata: Strata | str,
        dataloader_idx: int = 0,
        participating: set[Address] | None = None,
    ) -> list[Prediction]:
        """Validate encoded fields and decode the routes active in this loop phase."""
        strata = Strata.normalize(strata)
        sanitize(module, inputs, strata=strata, dataloader_idx=dataloader_idx)
        participating = participation(module, inputs, strata)[1] if participating is None else participating
        plan = execution(module)
        predict = strata == Strata.predict
        decoders = tuple(
            route
            for route in plan.decoders
            if route.embed or (route.decode if predict else route.address in participating)
        )
        return ModelRuntime.compute(module, inputs, plan=plan, decoders=decoders, predict=predict)

    @staticmethod
    def compute(
        module: Model,
        inputs: EncodedInput,
        *,
        plan: ExecutionPlan,
        decoders: tuple[DecoderRoute, ...],
        predict: bool,
    ) -> list[Prediction]:
        """Execute the model graph after host validation and objective selection."""

        processed: dict[Address, list[Parcel]] = defaultdict(list)
        outgoing: dict[Address, Parcel] = {}
        predictions: list[Prediction] = []
        geometry = cast(TensorDict, inputs[Address()]) if Address() in inputs.keys() else None

        for address in plan.requests:
            tensorfield = cast(TensorFieldBase, inputs[address])
            node_module = cast(NodeModule, module.nodes[address])
            embedder: EmbedderBase = node_module.embedder
            embedded = locate(embedder.embed(tensorfield), schema=module.schema, geometry=geometry)
            if embedded.destination is None:
                raise ValueError(f"parcel from '{embedded.origin}' has no destination")
            processed[embedded.destination].append(embedded)
            outgoing[embedded.origin] = embedded

        for depth in plan.depthwise:
            for address in depth:
                if not processed[address]:
                    continue

                node_module = cast(NodeModule, module.nodes[address])
                encoder: BranchEncoder = node_module.encoder
                encoded: Parcel = encoder(processed[address], geometry=geometry)
                if encoded.destination is None:
                    raise ValueError(f"parcel from '{encoded.origin}' has no destination")
                processed[encoded.destination].append(encoded)
                outgoing[encoded.origin] = encoded

                if address in plan.embeds:
                    positions, spans, slots = (
                        sequence(processed[address], order=encoder.child_order)
                        if encoder.reduction is None
                        else (None, None, None)
                    )
                    predictions.append(
                        Prediction(
                            address=encoded.origin,
                            payload=TensorDict(
                                {TensorKey.embedding: encoded.payload},
                                batch_size=encoded.payload.shape[0],
                            ),
                            geometry=geometry,
                            positions=positions,
                            spans=spans,
                            slots=slots,
                            batch_size=encoded.payload.shape[0],
                        )
                    )

        for route in decoders:
            address = route.address
            tensorfield = cast(TensorFieldBase, inputs[address])
            parcels = [outgoing[item] for item in route.heritage if item in outgoing]

            node_module = cast(NodeModule, module.nodes[address])
            decoder: DecoderBase = node_module.decoder
            contexts = [outgoing[item] for item in decoder.context_addresses if item in outgoing]
            decode = decoder if predict else partial(decoder.objective, selection=tensorfield.trainable)
            prediction = decode(
                parcels,
                contexts=contexts,
                batch_size=tensorfield.state.shape[0],
                device=tensorfield.state.device,
                embed=route.embed,
                target_shape=tuple(tensorfield.state.shape[1:]),
                geometry=geometry,
            )
            prediction.geometry = geometry
            prediction.payload[TensorKey.inferred] = (
                tensorfield.inferred if predict else tensorfield.trainable
            ).flatten(start_dim=1)
            predictions.append(prediction)

        return predictions

    @staticmethod
    @overload
    def step(
        module: Model,
        batch: Encoded,
        batch_idx: int,
        dataloader_idx: int = 0,
        *,
        strata: Literal[Strata.predict],
    ) -> pa.Table: ...

    @staticmethod
    @overload
    def step(
        module: Model,
        batch: Encoded | EncodedInput,
        batch_idx: int,
        dataloader_idx: int = 0,
        *,
        strata: Literal[Strata.train],
    ) -> Output | None: ...

    @staticmethod
    @overload
    def step(
        module: Model,
        batch: Encoded | EncodedInput,
        batch_idx: int,
        dataloader_idx: int = 0,
        *,
        strata: Literal[Strata.validate, Strata.test],
    ) -> Output: ...

    @staticmethod
    @overload
    def step(
        module: Model,
        batch: Encoded | EncodedInput,
        batch_idx: int,
        dataloader_idx: int = 0,
        *,
        strata: Strata,
    ) -> Output | pa.Table | None: ...

    @staticmethod
    def step(
        module: Model,
        batch: Encoded | EncodedInput,
        batch_idx: int,
        dataloader_idx: int = 0,
        *,
        strata: Strata,
    ) -> Output | pa.Table | None:
        """Evaluate a loop batch, retaining Arrow sources for public predictions."""
        if isinstance(batch, Encoded) and batch.bindings:
            trainer = getattr(module, "_trainer", None)
            if trainer is not None and trainer.world_size > 1:
                raise RuntimeError(
                    "distributed batches must resolve resource bindings before forward; "
                    "retain Model.on_before_batch_transfer when customizing data-transfer hooks"
                )
            batch = bind(module, batch, strata)
        inputs = batch.tensors if isinstance(batch, Encoded) else batch
        if isinstance(batch, Encoded):
            ModelRuntime.learn(module, batch.observations, strata=strata)
        if strata == Strata.predict and not isinstance(batch, Encoded):
            raise TypeError("prediction batches must retain their Arrow source")
        compiled = plan(module, batch.retain) if isinstance(batch, Encoded) and strata == Strata.predict else None
        local, participating = participation(module, inputs, strata)
        predictions = ModelRuntime.forward(
            module,
            inputs,
            strata=strata,
            dataloader_idx=dataloader_idx,
            participating=participating,
        )

        if strata == Strata.predict:
            assert isinstance(batch, Encoded)
            return ModelRuntime.write(
                module,
                predictions,
                source=batch.source,
                retain=batch.retain,
                compiled=compiled,
            )

        if not participating:
            logger.bind(
                component="runtime",
                strata=strata.value,
                rank=module.global_rank,
                batch=batch_idx,
            ).warning("no reconstruction objective was selected on any rank, skipping batch")
            if strata == Strata.train:
                return None
            return Output(loss=torch.tensor(0.0, device=inputs.device))

        objectives = set(module.schema.objectives)
        losses: list[torch.Tensor] = []
        anchors: list[torch.Tensor] = []
        for prediction in predictions:
            if prediction.address not in module.schema.requests:
                continue
            if prediction.address not in objectives:
                continue
            if strata == Strata.train and prediction.address in participating:
                anchors.extend(
                    value.sum() * 0.0
                    for value in prediction.payload.values(include_nested=True, leaves_only=True)
                    if torch.is_tensor(value) and value.requires_grad
                )
            # Decoder participation is coordinated across ranks in forward.
            # Extension losses and scalar metrics remain local to ranks carrying
            # selected targets; the zero-loss path below anchors every module
            # into backward when only a peer has an objective.
            if prediction.address not in local:
                continue
            if set(prediction.payload.keys()) <= {TensorKey.embedding, TensorKey.inferred}:
                continue

            address = Address(str(prediction.address))
            request: RequestBase = module.schema.requests[address]
            extension: Extension = TENSORFIELDS[request.type]
            loss_fn = extension.loss
            loss = loss_fn(
                module=module, prediction=prediction, batch=cast(TensorFieldBase, inputs[address]), strata=strata
            )
            losses.append(loss * torch.tensor(request.weight))

        if strata == Strata.train:
            if not anchors:
                raise RuntimeError("a distributed reconstruction objective produced no gradient-bearing decoder output")
            anchor = torch.stack(anchors).sum()
        else:
            anchor = torch.zeros((), device=inputs.device)
        if not losses:
            suffix = "anchored zero loss" if strata == Strata.train else "zero loss"
            logger.bind(
                component="runtime",
                strata=strata.value,
                rank=module.global_rank,
                batch=batch_idx,
            ).warning(f"reconstruction targets are present only on peer ranks, returning {suffix}")
            return Output(loss=anchor)

        loss = module.track((Metric.loss, strata), value=torch.stack(losses).sum() + anchor)
        if strata == Strata.train:
            module.track((Metric.loss, "train_epoch"), value=loss, on_step=False, on_epoch=True, logger=False)
        return Output(loss=loss)

    @staticmethod
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
        active = plan(module, retain, refresh=True) if compiled is None else compiled
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

    @staticmethod
    def prepare(
        module: Model,
        batch: pa.Table | pa.RecordBatch,
        *,
        preprocess: PreprocessorInput,
        strata: Strata,
        seed: int = 0,
        epoch: int = 0,
    ) -> Encoded:
        """Normalize Arrow input, preprocess it, and encode one table."""

        source = observations(module, batch, preprocess=preprocess, strata=strata)
        encoded = encode_batch(
            batch=source,
            schema=module.schema,
            strata=strata,
            interprocess_encoding_context=module.interprocess_encoding_context,
            seed=seed,
            epoch=epoch,
        )
        return bind(module, encoded, strata)

    @staticmethod
    def learn(
        module: Model,
        observations: Mapping[Address, TensorDict],
        *,
        strata: Strata | str,
    ) -> None:
        """Apply every model-owned pristine observation exactly once."""

        normalized = Strata.normalize(strata)
        if normalized != Strata.train:
            if observations:
                raise ValueError(f"{normalized} input cannot carry learnable observations")
            return

        learners = {
            address: TENSORFIELDS[request.type]
            for address, request in module.schema.active_requests.items()
            if Component.learn in TENSORFIELDS[request.type].components
        }
        missing = set(learners) - set(observations)
        extra = set(observations) - set(learners)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(map(str, sorted(missing, key=str))))
            if extra:
                details.append("unexpected " + ", ".join(map(str, sorted(extra, key=str))))
            raise ValueError("training observations do not match registered learners: " + "; ".join(details))

        for address in module.schema.active_requests:
            if address not in learners:
                continue
            learners[address].learn(
                module=module,
                observation=observations[address].to(module.device),
                address=address,
                strata=normalized,
            )

    @staticmethod
    def encode(
        module: Model,
        batch: pa.Table | pa.RecordBatch,
        preprocess: PreprocessorInput = (),
        strata: Strata | str = Strata.predict,
        seed: int = 0,
        epoch: int = 0,
    ) -> EncodedInput:
        """Encode one Arrow input unit to tensorfields."""

        normalized = Strata.normalize(strata)
        encoded = ModelRuntime.prepare(
            module,
            batch,
            preprocess=preprocess,
            strata=normalized,
            seed=seed,
            epoch=epoch,
        )
        ModelRuntime.learn(module, encoded.observations, strata=normalized)
        return encoded.tensors

    @staticmethod
    def predict(
        module: Model,
        batch: PredictionInput,
        preprocess: PreprocessorInput = (),
        postprocess: PostprocessorInput = (),
        retain: Retain = (),
        *,
        batch_size: int | None = None,
    ) -> pa.Table:
        """Preprocess once, predict bounded Arrow batches, and postprocess once."""

        postprocessors = Postprocessor.normalize(postprocess)
        source = observations(
            module,
            ingress(batch),
            preprocess=preprocess,
            strata=Strata.predict,
        )
        size = max(source.num_rows, 1) if batch_size is None else batch_size
        written: list[pa.Table] = []
        compiled: OutputPlan | None = None
        for offset in range(0, max(source.num_rows, 1), size):
            encoded = ModelRuntime.prepare(
                module,
                source.slice(offset, size),
                preprocess=(),
                strata=Strata.predict,
            )
            if compiled is None:
                compiled = plan(module, retain, refresh=True)
            raw: list[Prediction] = []
            if encoded.source.num_rows:
                was_training = module.training
                module.eval()
                try:
                    with torch.inference_mode():
                        raw = module(encoded.tensors.to(module.device), strata=Strata.predict)
                finally:
                    if was_training:
                        module.train()

            written.append(ModelRuntime.write(module, raw, source=encoded.source, retain=retain, compiled=compiled))
            del encoded, raw

        combined = pa.concat_tables(written)
        return apply(combined, postprocessors).replace_schema_metadata(combined.schema.metadata)


step = ModelRuntime.step
