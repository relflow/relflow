"""Encode observations, execute the model graph, and evaluate loop objectives."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict, cast, overload

import pyarrow as pa
import torch
from tensordict import TensorDict

from relflow.architecture import output
from relflow.architecture.binding import bind
from relflow.architecture.contracts import sanitize
from relflow.architecture.encoder import BranchEncoder, locate, sequence
from relflow.architecture.node import NodeModule
from relflow.architecture.output import OutputPlan
from relflow.data.arrow import Encoded, Retain, mappings
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
from relflow.structs.enums import Component, Metric, Strata, TensorKey
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address
from relflow.tensorfields.base import (
    TENSORFIELDS,
    DecoderBase,
    EmbedderBase,
    Extension,
    RequestBase,
    TensorFieldBase,
)

if TYPE_CHECKING:
    from relflow.architecture.root import Model

__all__ = ["ExecutionPlan", "Output", "PredictionInput", "encode", "forward", "predict", "step", "transfer"]

PredictionInput: TypeAlias = pa.Table | pa.RecordBatch | Sequence[Mapping[str, object]]


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
    sequenced: frozenset[Address]


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
        # Only ancestors of a decoded leaf need their encoded sequence beside the summary.
        sequenced=frozenset(item for route in decoders for item in route.heritage),
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


def rows(module: Model, inputs: EncodedInput) -> int:
    """Count the observations in one forward batch."""

    if inputs.batch_dims:
        return int(inputs.batch_size[0])
    address = next(iter(module.schema.active_requests), None)
    return 0 if address is None else int(cast(TensorFieldBase, inputs[address]).state.shape[0])


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


def transfer(value: TensorDict, device: torch.device) -> TensorDict:
    """Move encoded tensors to ``device``.

    MPS holds no float64, so tensorfields that keep source values wide
    continue there in float32.
    """
    if device.type == "mps":
        value = cast(
            TensorDict,
            value.apply(lambda tensor: tensor.to(dtype=torch.float32) if tensor.dtype == torch.float64 else tensor),
        )
    return value.to(device)


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
        route for route in plan.decoders if route.embed or (route.decode if predict else route.address in participating)
    )
    return compute(module, inputs, plan=plan, decoders=decoders, predict=predict)


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
            conditioning = {"context": processed[encoder.destination]} if encoder.conditioning is not None else {}
            encoded: Parcel = encoder(
                processed[address], geometry=geometry, sequenced=address in plan.sequenced, **conditioning
            )
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
        # A reduced ancestor hands the decoders beneath it its summary and its encoded sequence.
        parcels = []
        for item in route.heritage:
            if item in outgoing:
                parcels.append(outgoing[item])
                kept = outgoing[item].sequence
                if kept is not None:
                    parcels.append(kept)

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
        prediction.payload[TensorKey.inferred] = (tensorfield.inferred if predict else tensorfield.trainable).flatten(
            start_dim=1
        )
        predictions.append(prediction)

    return predictions


def objective(
    module: Model,
    inputs: EncodedInput,
    predictions: list[Prediction],
    *,
    strata: Strata,
    local: set[Address],
    participating: set[Address],
    batch_idx: int,
) -> torch.Tensor:
    """Evaluate local reconstruction losses and retain every participating decoder's backward edges."""

    losses: dict[Address, torch.Tensor] = {}
    anchors: list[torch.Tensor] = []
    for prediction in predictions:
        if prediction.address not in participating:
            continue
        if strata == Strata.train:
            anchors.extend(
                value.sum() * 0.0
                for value in prediction.payload.values(include_nested=True, leaves_only=True)
                if torch.is_tensor(value) and value.requires_grad
            )
        # Extension losses and scalar metrics stay local to ranks with selected
        # targets; decoder anchors retain backward participation on their peers.
        if prediction.address not in local:
            continue
        if set(prediction.payload.keys()) <= {TensorKey.embedding, TensorKey.inferred}:
            continue

        address = Address(str(prediction.address))
        request: RequestBase = module.schema.requests[address]
        extension: Extension = TENSORFIELDS[request.type]
        loss = extension.loss(
            module=module, prediction=prediction, batch=cast(TensorFieldBase, inputs[address]), strata=strata
        )
        losses[address] = loss * torch.tensor(request.weight)

    if strata == Strata.train:
        if not anchors:
            raise RuntimeError("a distributed reconstruction objective produced no gradient-bearing decoder output")
        anchor = torch.stack(anchors).sum()
    else:
        anchor = torch.zeros((), device=inputs.device)
    # Peer ranks reduce this metric, including ranks without local targets,
    # whose contribution carries zero observations.
    weight = rows(module, inputs) if losses else 0
    if not losses:
        suffix = "anchored zero loss" if strata == Strata.train else "zero loss"
        logger.bind(
            component="runtime",
            strata=strata.value,
            rank=module.global_rank,
            batch=batch_idx,
        ).warning(f"reconstruction targets are present only on peer ranks, returning {suffix}")

    total = torch.stack(tuple(losses.values())).sum() + anchor if losses else anchor
    loss = module.track((Metric.loss, strata), value=total, rows=weight, uniform=True)
    if strata == Strata.train:
        module.track(
            (Metric.loss, "train_epoch"),
            value=loss,
            on_step=False,
            on_epoch=True,
            logger=False,
            rows=weight,
            uniform=True,
        )
        if module.jacobian:
            from relflow.architecture.jacobian import combine

            # Every rank uses schema objective order. Missing local targets
            # contribute constant zero rows, separate from decoder anchors.
            zero = anchor.new_zeros(())
            ordered = [
                losses.get(address, zero) for address in execution(module).objectives if address in participating
            ]
            return combine(module, ordered, anchor)
    return loss


@overload
def step(
    module: Model,
    batch: Encoded,
    batch_idx: int,
    dataloader_idx: int = 0,
    *,
    strata: Literal[Strata.predict],
) -> pa.Table: ...


@overload
def step(
    module: Model,
    batch: Encoded | EncodedInput,
    batch_idx: int,
    dataloader_idx: int = 0,
    *,
    strata: Literal[Strata.train],
) -> Output | None: ...


@overload
def step(
    module: Model,
    batch: Encoded | EncodedInput,
    batch_idx: int,
    dataloader_idx: int = 0,
    *,
    strata: Literal[Strata.validate, Strata.test],
) -> Output: ...


@overload
def step(
    module: Model,
    batch: Encoded | EncodedInput,
    batch_idx: int,
    dataloader_idx: int = 0,
    *,
    strata: Strata,
) -> Output | pa.Table | None: ...


def step(
    module: Model,
    batch: Encoded | EncodedInput,
    batch_idx: int,
    dataloader_idx: int = 0,
    *,
    strata: Strata,
) -> Output | pa.Table | None:
    """Evaluate one loop batch, retaining Arrow sources for public predictions."""
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
        learn(module, batch.observations, strata=strata)
    if strata == Strata.predict and not isinstance(batch, Encoded):
        raise TypeError("prediction batches must retain their Arrow source")
    compiled = output.plan(module, batch.retain) if isinstance(batch, Encoded) and strata == Strata.predict else None
    local, participating = participation(module, inputs, strata)
    predictions = forward(
        module,
        inputs,
        strata=strata,
        dataloader_idx=dataloader_idx,
        participating=participating,
    )

    if strata == Strata.predict:
        assert isinstance(batch, Encoded)
        return output.write(
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

    return Output(
        loss=objective(
            module,
            inputs,
            predictions,
            strata=strata,
            local=local,
            participating=participating,
            batch_idx=batch_idx,
        )
    )


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
            observation=transfer(observations[address], module.device),
            address=address,
            strata=normalized,
        )


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
    encoded = prepare(
        module,
        batch,
        preprocess=preprocess,
        strata=normalized,
        seed=seed,
        epoch=epoch,
    )
    learn(module, encoded.observations, strata=normalized)
    return encoded.tensors


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
        encoded = prepare(
            module,
            source.slice(offset, size),
            preprocess=(),
            strata=Strata.predict,
        )
        if compiled is None:
            # An evaluation model's contract is fixed per generation, so serving
            # reuses it across calls; a training model may still grow resources.
            compiled = output.plan(module, retain, refresh=module.training)
        raw: list[Prediction] = []
        if encoded.source.num_rows:
            was_training = module.training
            module.eval()
            try:
                with torch.inference_mode():
                    raw = module(transfer(encoded.tensors, module.device), strata=Strata.predict)
            finally:
                if was_training:
                    module.train()

        written.append(output.write(module, raw, source=encoded.source, retain=retain, compiled=compiled))
        del encoded, raw

    combined = pa.concat_tables(written)
    return apply(combined, postprocessors).replace_schema_metadata(combined.schema.metadata)
