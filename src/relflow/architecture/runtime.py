"""Forward, loss, encoding, and Arrow output for RelFlow models."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict, cast, overload

import pyarrow as pa
import torch
from tensordict import TensorDict

from relflow.architecture.binding import bind
from relflow.architecture.contracts import sanitize
from relflow.architecture.encoder import BranchEncoder
from relflow.architecture.node import NodeModule
from relflow.data.arrow import Encoded, mappings
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
from relflow.structs.tree import Address
from relflow.tensorfields.base import (
    TENSORFIELDS,
    DecoderBase,
    EmbedderBase,
    Extension,
    RequestBase,
    TensorFieldBase,
    Write,
)
from relflow.tensorfields.output import STATE, embedding, inferred, shape, state, struct

if TYPE_CHECKING:
    from relflow.architecture.root import Model

Retain = tuple[str, ...] | Literal["*"]
PredictionInput: TypeAlias = pa.Table | pa.RecordBatch | Sequence[Mapping[str, object]]
RESERVED = frozenset({TensorKey.state.name, TensorKey.inferred.name, TensorKey.embedding.name})


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
    axes: tuple[int, ...]
    decoded: pa.StructType | None
    writer: Write | None
    embed: bool
    extension: str | None
    coordinate: pa.StructType | None
    output: pa.DataType | None


@dataclass(frozen=True, slots=True)
class OutputPlan:
    """Ordered prediction contract compiled for one model generation."""

    generation: int
    retain: Retain
    expected: frozenset[Address]
    entries: tuple[OutputEntry, ...]


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


def retention(names: Retain) -> Retain:
    """Validate and normalize one retained-input selection."""

    if names == "*":
        return names
    if not isinstance(names, tuple):
        raise TypeError("retain must be a tuple of column names or '*'")
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError("retain entries must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("retain entries must be unique")
    return names


def project(source: pa.Table, names: Retain) -> pa.Array:
    """Build the canonical retained-input value for each source row."""

    normalized = retention(names)
    selected = tuple(source.column_names) if normalized == "*" else normalized

    missing = [name for name in selected if name not in source.column_names]
    if missing:
        formatted = ", ".join(repr(name) for name in missing)
        raise KeyError(f"retained column(s) are absent after preprocessing: {formatted}")
    if not selected:
        return pa.nulls(source.num_rows)

    fields = [source.schema.field(name) for name in selected]
    arrays = [source[name].combine_chunks() for name in selected]
    return pa.StructArray.from_arrays(arrays, fields=fields)


def axes(module: Model, address: Address) -> tuple[int, ...]:
    """Return repeated model axes represented inside one output row."""

    if address in module.schema.requests:
        return tuple(module.schema.requests[address].shape[1:])
    if address in module.schema.branches:
        lengths = tuple(
            node.length for node in module.schema.branches[address].path if getattr(node, "type", None) == "branch"
        )
        structural = lengths[1:-1]
        outputs = module.schema.branch_outputs[address]
        return (*structural, outputs) if outputs > 1 else structural
    raise KeyError(f"prediction address {str(address)!r} is absent from the model schema")


def plan(module: Model, retain: Retain = (), *, refresh: bool = False) -> OutputPlan:
    """Compile and cache the exact prediction contract for this model generation."""

    generation = int(getattr(module, "_contract_generation", 0))
    retained = retention(retain)
    plans = getattr(module, "output_plans", None)
    if not isinstance(plans, dict):
        plans = {}
        module.output_plans = plans
    key = (generation, retained)
    cached = plans.get(key)
    if not refresh and isinstance(cached, OutputPlan):
        return cached

    embeds = set(module.schema.embed)
    expected = frozenset(Address(str(address)) for address in (*module.schema.decodes, *embeds))
    entries: list[OutputEntry] = []
    visited: set[Address] = set()
    for node in (module.schema.fields, *module.schema.fields.descendants):
        address = Address(str(node.address))
        if address not in expected:
            continue

        visited.add(address)
        decoded: pa.StructType | None = None
        writer: Write | None = None
        extension_name: str | None = None
        if address in module.schema.requests:
            extension = TENSORFIELDS[module.schema.requests[address].type]
            extension_name = extension.name
            declared = extension.output(module=module, address=address)
            if declared is not None and not isinstance(declared, pa.StructType):
                raise TypeError(f"extension output for {address!s} must return a pyarrow StructType or None")
            if declared is not None:
                conflicts = RESERVED.intersection(field.name for field in declared)
                if conflicts:
                    names = ", ".join(sorted(conflicts))
                    raise ValueError(f"extension output for {address!s} uses reserved field(s): {names}")
                decoded = declared
                writer = extension.write

        coordinate_fields: list[pa.Field] = []
        if decoded is not None:
            coordinate_fields.append(pa.field(TensorKey.state.name, STATE, nullable=False))
            coordinate_fields.extend(decoded)
            coordinate_fields.append(pa.field(TensorKey.inferred.name, pa.bool_(), nullable=False))
        if address in embeds:
            coordinate_fields.append(
                pa.field(
                    TensorKey.embedding.name,
                    pa.list_(pa.float32(), module.schema.d_model),
                    nullable=False,
                )
            )
        coordinate_type = pa.struct(coordinate_fields) if coordinate_fields else None
        model_axes = axes(module, address)
        output_type: pa.DataType | None = coordinate_type
        if output_type is not None:
            for size in reversed(model_axes):
                output_type = pa.list_(output_type, size)

        entries.append(
            OutputEntry(
                address=address,
                axes=model_axes,
                decoded=decoded,
                writer=writer,
                embed=address in embeds,
                extension=extension_name,
                coordinate=coordinate_type,
                output=output_type,
            )
        )

    absent = expected - visited
    if absent:
        formatted = ", ".join(str(address) for address in sorted(absent, key=str))
        raise ValueError(f"configured prediction address(es) are absent from schema traversal: {formatted}")

    compiled = OutputPlan(generation=generation, retain=retained, expected=expected, entries=tuple(entries))
    plans[key] = compiled
    return compiled


def coordinate(module: Model, entry: OutputEntry, prediction: Prediction, rows: int) -> pa.Array | None:
    """Assemble and shape one address-level prediction array."""

    address = entry.address
    fields: list[pa.Field] = []
    arrays: dict[str, pa.Array] = {}
    expected = rows * math.prod(entry.axes)

    if address in module.schema.requests:
        if TensorKey.state not in prediction.payload or TensorKey.inferred not in prediction.payload:
            raise ValueError(f"decoded prediction for {address!s} is missing shared state or inferred tensors")
        state_tensor = cast(torch.Tensor, prediction.payload[TensorKey.state])
        inferred_tensor = cast(torch.Tensor, prediction.payload[TensorKey.inferred])
        if state_tensor.ndim == 0 or state_tensor.shape[-1] != len(Tokens):
            raise ValueError(f"prediction state at {address!s} must end with {len(Tokens)} logits")
        if state_tensor.numel() != expected * len(Tokens) or inferred_tensor.numel() != expected:
            raise ValueError(
                f"prediction at {address!s} contains the wrong coordinate count; expected {expected} "
                f"for {rows} rows and model axes {entry.axes}"
            )

        if entry.decoded is not None:
            if entry.writer is None:
                raise RuntimeError(f"compiled output for {address!s} has no extension writer")
            written = entry.writer(module=module, prediction=prediction, datatype=entry.decoded)
            if not isinstance(written, pa.StructArray):
                raise TypeError(f"extension write for {address!s} must return a pyarrow StructArray")
            if written.type != entry.decoded:
                raise TypeError(f"extension write for {address!s} returned {written.type}; expected {entry.decoded}")

            state_values = state(state_tensor)
            fields.append(pa.field(TensorKey.state.name, state_values.type, nullable=False))
            arrays[TensorKey.state.name] = state_values
            for index, field in enumerate(entry.decoded):
                fields.append(field)
                arrays[field.name] = written.field(index)

            inferred_values = inferred(inferred_tensor)
            fields.append(pa.field(TensorKey.inferred.name, inferred_values.type, nullable=False))
            arrays[TensorKey.inferred.name] = inferred_values

    if entry.embed and TensorKey.embedding not in prediction.payload:
        raise ValueError(f"prediction for configured embedding address {address!s} has no embedding tensor")
    if not entry.embed and TensorKey.embedding in prediction.payload:
        raise ValueError(f"prediction for {address!s} returned an unplanned embedding tensor")
    if entry.embed:
        embedding_tensor = cast(torch.Tensor, prediction.payload[TensorKey.embedding])
        if embedding_tensor.ndim == 0 or embedding_tensor.shape[-1] != module.schema.d_model:
            raise ValueError(f"prediction embedding at {address!s} must end with model width {module.schema.d_model}")
        embedding_values = embedding(embedding_tensor)
        fields.append(pa.field(TensorKey.embedding.name, embedding_values.type, nullable=False))
        arrays[TensorKey.embedding.name] = embedding_values

    if not fields:
        return None

    dtype = pa.struct(fields)
    if dtype != entry.coordinate:
        raise TypeError(f"prediction at {address!s} produced coordinate type {dtype}; expected {entry.coordinate}")
    values = struct(arrays, dtype)
    if len(values) != expected:
        raise ValueError(
            f"prediction at {address!s} wrote {len(values)} coordinates; expected {expected} "
            f"for {rows} rows and model axes {entry.axes}"
        )

    shaped = shape(values, entry.axes)
    if shaped.type != entry.output:
        raise TypeError(f"prediction at {address!s} produced output type {shaped.type}; expected {entry.output}")
    if len(shaped) != rows:
        raise ValueError(f"prediction at {address!s} produced {len(shaped)} output rows; expected {rows}")
    return shaped


def envelope(module: Model, predictions: list[Prediction], rows: int, compiled: OutputPlan | None = None) -> pa.Array:
    """Assemble address arrays in schema traversal order."""

    active = plan(module, refresh=True) if compiled is None else compiled
    indexed: dict[Address, Prediction] = {}
    for prediction in predictions:
        address = Address(str(prediction.address))
        if address in indexed:
            raise ValueError(f"forward returned duplicate prediction address {address!s}")
        indexed[address] = prediction

    missing = active.expected - indexed.keys()
    if missing:
        formatted = ", ".join(str(address) for address in sorted(missing, key=str))
        raise ValueError(f"forward omitted configured prediction address(es): {formatted}")

    unplanned = indexed.keys() - active.expected
    if unplanned:
        formatted = ", ".join(str(address) for address in sorted(unplanned, key=str))
        raise ValueError(f"forward returned unplanned prediction address(es): {formatted}")

    fields: list[pa.Field] = []
    arrays: list[pa.Array] = []
    for entry in active.entries:
        values = coordinate(module, entry, indexed[entry.address], rows)
        if values is None:
            continue
        fields.append(pa.field(str(entry.address), values.type, nullable=False))
        arrays.append(values)

    if not arrays:
        return pa.nulls(rows)
    return pa.StructArray.from_arrays(arrays, fields=fields)


def vacant(compiled: OutputPlan) -> pa.Array:
    """Build the planned public prediction type for a zero-row source."""

    entries = [entry for entry in compiled.entries if entry.output is not None]
    if not entries:
        return pa.nulls(0)
    fields = [pa.field(str(entry.address), cast(pa.DataType, entry.output), nullable=False) for entry in entries]
    arrays = [pa.array([], type=entry.output) for entry in entries]
    return pa.StructArray.from_arrays(arrays, fields=fields)


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

        for address in plan.requests:
            tensorfield = cast(TensorFieldBase, inputs[address])
            node_module = cast(NodeModule, module.nodes[address])
            embedder: EmbedderBase = node_module.embedder
            embedded = embedder.embed(tensorfield)
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
                encoded: Parcel = encoder(processed[address])
                if encoded.destination is None:
                    raise ValueError(f"parcel from '{encoded.origin}' has no destination")
                processed[encoded.destination].append(encoded)
                outgoing[encoded.origin] = encoded

                if address in plan.embeds:
                    predictions.append(
                        Prediction(
                            address=encoded.origin,
                            payload=TensorDict(
                                {TensorKey.embedding: encoded.payload},
                                batch_size=encoded.payload.shape[0],
                            ),
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
            prediction = decoder(
                parcels,
                contexts=contexts,
                batch_size=tensorfield.state.shape[0],
                device=tensorfield.state.device,
                embed=route.embed,
            )
            prediction.payload[TensorKey.inferred] = tensorfield.inferred if predict else tensorfield.trainable
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
                    for value in prediction.payload.values()
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
        if active.retain != retention(retain):
            raise ValueError(f"compiled output plan retain {active.retain!r} does not match write retain {retain!r}")
        inputs = project(source, active.retain)
        outputs = (
            vacant(active)
            if not source.num_rows and not predictions
            else envelope(module, predictions, source.num_rows, active)
        )
        data = pa.Table.from_arrays(
            [inputs, outputs],
            names=["inputs", "predictions"],
        )
        if len(inputs) != source.num_rows:
            raise ValueError("retained input output is not aligned with the canonical source")
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

        from relflow.data.datasets.arrow import convert, merge, process

        source = convert(batch)
        processors = Preprocessor.normalize(preprocess)
        if processors:
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
    ) -> pa.Table:
        """Predict one Arrow input unit and return an Arrow table."""

        postprocessors = Postprocessor.normalize(postprocess)
        encoded = ModelRuntime.prepare(
            module,
            ingress(batch),
            preprocess=preprocess,
            strata=Strata.predict,
        )
        source = encoded.source
        inputs = encoded.tensors.to(module.device)
        compiled = plan(module, retain, refresh=True)
        raw: list[Prediction] = []
        if source.num_rows:
            was_training = module.training
            module.eval()
            try:
                with torch.inference_mode():
                    raw = module(inputs, strata=Strata.predict)
            finally:
                if was_training:
                    module.train()

        written = ModelRuntime.write(module, raw, source=source, retain=retain, compiled=compiled)
        return apply(written, postprocessors)


step = ModelRuntime.step
