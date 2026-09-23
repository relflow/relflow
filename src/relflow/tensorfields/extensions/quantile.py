"""Continuous scalars represented by their online training percentiles."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from functools import partial
from multiprocessing import Manager
from typing import TYPE_CHECKING, Annotated, Any, Literal, Unpack, cast

import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from tensordict import TensorDict, tensorclass

from relflow.data.ragged import RaggedField
from relflow.distributed import all_gather_object, broadcast_object, is_distributed
from relflow.helpers.digest import Digest
from relflow.structs.enums import Metric, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address, FieldOptions
from relflow.tensorfields.base import (
    Context,
    DecoderBase,
    EmbedderBase,
    Extension,
    RequestBase,
    TensorFieldBase,
    TensorInput,
)
from relflow.tensorfields.output import array, struct
from relflow.tensorfields.shared.counter import Counter, CounterUpdateCallback, tally

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.data.datasets.base import InterprocessEncodingContext
    from relflow.helpers.resize import Resize
    from relflow.structs.experiment import Schema


@dataclass(frozen=True)
class Batch:
    """Pristine Arrow values; workers never commit distribution updates."""

    values: pa.Array | pa.ChunkedArray


class State:
    """Publish one complete serialized snapshot atomically, including its mass."""

    def __init__(self, compression: int):
        self.master: Any = [Digest(compression).serialize()]
        self.manager: Any = None
        self.cached: bytes | None = None
        self.digest: Digest | None = None

    def __getstate__(self) -> dict[str, Any]:
        return {"master": self.master, "manager": None, "cached": None, "digest": None}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def snapshot(self) -> Digest:
        payload = self.master[0]
        if self.digest is None or payload != self.cached:
            self.digest = Digest.deserialize(payload)
            self.cached = payload
        return self.digest

    def publish(self, payload: bytes) -> None:
        self.master[0] = payload
        self.cached = None
        self.digest = None

    def batch(self, values: pa.Array | pa.ChunkedArray) -> tuple[object, object]:
        return self.snapshot(), Batch(values)

    def share(self) -> None:
        if self.manager is None:
            payload = self.master[0]
            self.manager = Manager()
            self.master = self.manager.list([payload])

    def freeze(self) -> None:
        if self.manager is not None:
            manager = self.manager
            self.master = [self.master[0]]
            self.manager = None
            manager.shutdown()


class Distribution(torch.nn.Module):
    def __init__(self, compression: int):
        super().__init__()
        self.compression = compression
        self.state = State(compression)

    def get_extra_state(self) -> bytes:
        return self.state.master[0]

    def set_extra_state(self, state: bytes) -> None:
        if not isinstance(state, bytes):
            raise TypeError("Quantile distribution checkpoint must contain serialized digest bytes")
        digest = Digest.deserialize(state)
        if digest.compression != self.compression:
            resized = Digest(self.compression)
            resized.merge(digest)
            state = resized.serialize()
        self.state.publish(state)


class Synchronize(Callback):
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if is_distributed():
            for _, resource in sorted(pl_module.named_modules()):
                if isinstance(resource, Distribution):
                    resource.set_extra_state(broadcast_object(resource.get_extra_state(), src=0))

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        for resource in pl_module.modules():
            if isinstance(resource, Distribution):
                resource.state.freeze()


quantile = Extension(name="quantile", types=(int | float,))
quantile.callback(CounterUpdateCallback)
quantile.callback(Synchronize)


@quantile.register
class Request(RequestBase):
    """A finite scalar normalized to its estimated training percentile.

    The digest learns pristine consumed training values, including hidden targets.
    Evaluation freezes the distribution. Predictions use its inverse transform
    and remain in source units; values outside learned support clip to its tails.
    """

    type: Literal["quantile"] = "quantile"
    compression: Annotated[int, pydantic.Field(ge=10)] = 100
    n_bands: Annotated[int, pydantic.Field(gt=0)] = 8
    objective: Literal["mae", "mse", "huber"] = "mae"

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            compression: int = 100,
            n_bands: int = 8,
            objective: Literal["mae", "mse", "huber"] = "mae",
            type: Literal["quantile"] = "quantile",
            **options: Unpack[FieldOptions],
        ) -> None: ...

    @classmethod
    def normalization(
        cls, source: Model | InterprocessEncodingContext, address: Address | str, /
    ) -> dict[str, int | float | None]:
        """Inspect a fitted distribution through a model or its encoding context."""
        from relflow.architecture.root import Model

        address = Address(str(address))
        if isinstance(source, Model):
            if address not in source.nodes:
                raise KeyError(f"no field at address {str(address)!r}")
            embedder = getattr(source.nodes[address], "embedder", None)
            if not isinstance(embedder, Embedder):
                raise TypeError(f"address '{address}' is not a Quantile field")
            state = embedder.distribution.state
        elif isinstance(source, Mapping):
            if address not in source:
                raise KeyError(f"no encoding context at address {str(address)!r}")
            state = source[address]
            if not isinstance(state, State):
                raise TypeError(f"encoding context at '{address}' is not a Quantile distribution")
        else:
            raise TypeError("Quantile.normalization source must be a Model or encoding context")
        digest = state.snapshot()
        values = digest.quantile(pa.array([0.0, 0.5, 1.0])).to_pylist()
        return dict(
            count=digest.count, compression=digest.compression, minimum=values[0], median=values[1], maximum=values[2]
        )


def numeric(values: pa.Array | pa.ChunkedArray, address: Address) -> pa.Array:
    """Preserve float64 distinctions and reject unordered or nonfinite scalars."""
    if not (pa.types.is_integer(values.type) or pa.types.is_floating(values.type) or pa.types.is_null(values.type)):
        raise ValueError(f"Quantile field at '{address}' expects finite scalar numbers, got {values.type}")
    converted = pc.cast(values, pa.float64(), safe=True)
    converted = converted.combine_chunks() if isinstance(converted, pa.ChunkedArray) else converted
    if pc.any(pc.invert(pc.is_finite(converted))).as_py():
        raise ValueError(f"Quantile field at '{address}' requires finite numbers; use null for missing values")
    return converted


def values(raw: torch.Tensor) -> pa.Array:
    """Restore source doubles on the host from device-portable integer bits."""
    return array(raw.detach().cpu().contiguous().view(torch.float64), pa.float64())


def percentiles(digest: Digest, raw: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    encoded = digest.cdf(values(raw)).to_numpy(zero_copy_only=False).copy()
    ranks = torch.from_numpy(encoded).reshape(raw.shape).to(device=raw.device, dtype=torch.float32)
    return torch.where(state.eq(Tokens.valued), ranks, 0.5)


@quantile.register
def observe(
    field: RaggedField, *, address: Address, schema: Schema, state: object | None, learn: bool
) -> TensorDict | None:
    if not learn:
        return None
    return TensorDict({TensorKey.state: tally(torch.from_numpy(field.dense.copy()), len(Tokens))}, batch_size=[])


@quantile.register
@tensorclass
class TensorField(TensorFieldBase[torch.Tensor]):
    content: torch.Tensor
    raw: torch.Tensor
    state: torch.Tensor
    present: torch.Tensor
    trainable: torch.Tensor
    inferred: torch.Tensor
    targets: TensorDict

    if TYPE_CHECKING:
        batch_size: InitVar[int | torch.Size | list[int] | tuple[int, ...] | None] = None
        device: InitVar[torch.device | str | int | None] = None
        names: InitVar[list[str | None] | None] = None

    @classmethod
    def new(
        cls,
        input: RaggedField,
        target: RaggedField,
        present: torch.Tensor,
        trainable: torch.Tensor,
        inferred: torch.Tensor,
        address: Address,
        schema: Schema,
        strata: Strata,
        context: Context,
    ) -> TensorField:
        request = cast(Request, schema.requests[address])
        digest = context.state if isinstance(context.state, Digest) else Digest(request.compression)

        def raw(field: RaggedField) -> torch.Tensor:
            values = numeric(field.values, address).to_numpy(zero_copy_only=False)
            # Float64 bits retain source precision without putting unsupported
            # float64 arithmetic on devices such as MPS. Batch axes stay explicit.
            return torch.from_numpy(field.place(values, fill=0.0).copy()).view(torch.int64)

        state = torch.from_numpy(input.dense.copy())
        content = raw(input)
        targets = TensorDict({}, batch_size=input.shape)
        if address in schema.objectives:
            target_state = torch.from_numpy(target.dense.copy())
            target_raw = raw(target)
            targets = TensorDict(
                {
                    TensorKey.state: target_state,
                    TensorKey.content: percentiles(digest, target_raw, target_state),
                    "raw": target_raw,
                },
                batch_size=input.shape,
            )
        return cls(
            content=percentiles(digest, content, state),
            raw=content,
            state=state,
            present=present,
            trainable=trainable,
            inferred=inferred,
            targets=targets,
            batch_size=input.batch_size,
        )


@quantile.register
def bind(
    module: Model,
    field: TensorFieldBase,
    observation: TensorDict | None,
    binding: object,
    *,
    address: Address,
    strata: Strata,
    resize: Resize,
) -> None:
    """Resolve prefetched raw values against one synchronized batch snapshot."""
    if not isinstance(binding, Batch) or not isinstance(field, TensorField):
        raise TypeError(f"Quantile field at '{address}' requires its raw Arrow batch binding")
    embedder = cast(Embedder, module.nodes[address].embedder)
    resource = embedder.distribution
    digest = resource.state.snapshot()
    values = numeric(binding.values, address)
    if strata == Strata.train:
        delta = Digest(resource.compression)
        delta.update(values)
        digest = Digest.deserialize(digest.serialize())
        for payload in all_gather_object(delta.serialize()):
            digest.merge(Digest.deserialize(payload))
        resize.publish(partial(resource.state.publish, digest.serialize()))
    field.content = percentiles(digest, field.raw, field.state)
    if TensorKey.content in field.targets:
        field.targets[TensorKey.content] = percentiles(
            digest, cast(torch.Tensor, field.targets["raw"]), cast(torch.Tensor, field.targets[TensorKey.state])
        )


@quantile.register
class Embedder(EmbedderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema, address)
        request = cast(Request, schema.requests[address])
        self.distribution = Distribution(request.compression)
        self.counter = Counter(address, len(Tokens))
        self.embeddings = torch.nn.Embedding(len(Tokens), schema.d_model)
        self.linear = torch.nn.Linear(2 * request.n_bands + 1, schema.d_model)
        self.register_buffer("frequencies", torch.arange(1, request.n_bands + 1).float() * math.pi)
        self.frequencies: torch.Tensor

    @property
    def context(self) -> State:
        return self.distribution.state

    def forward(self, inputs: TensorInput) -> Parcel:
        state = inputs.state.reshape(-1)
        ranks = cast(torch.Tensor, inputs.content).reshape(-1, 1)
        weighted = ranks * self.frequencies
        features = torch.cat((ranks, weighted.sin(), weighted.cos()), dim=-1)
        payload = self.embeddings(state) + torch.nn.functional.gelu(self.linear(features))
        lane = torch.where(state.eq(Tokens.valued), ranks[:, 0], payload[:, -1])
        payload = torch.cat((payload[:, :-1], (lane + payload[:, -1] * 0.0).unsqueeze(-1)), dim=-1)
        return Parcel(
            payload=payload,
            present=torch.ones(len(state), dtype=torch.bool, device=state.device),
            origin=self.address,
            destination=self.destination,
            batch_size=len(state),
        )


@quantile.register
def learn(module: Model, observation: TensorDict, *, address: Address, strata: Strata) -> None:
    if strata != Strata.train:
        raise ValueError(f"Quantile learner at '{address}' requires train strata, got {strata}")
    cast(Embedder, module.nodes[address].embedder).counter.learn(cast(torch.Tensor, observation[TensorKey.state]))


@quantile.register
class Decoder(DecoderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema, address)
        self.classification = torch.nn.Linear(schema.d_model, len(Tokens))
        self.regression = torch.nn.Linear(schema.d_model, 1)
        self.distribution = State(cast(Request, schema.requests[address]).compression)

    def attach(self, context: object | None) -> None:
        if not isinstance(context, State):
            raise TypeError(f"Quantile decoder at '{self.address}' requires its distribution context")
        self.distribution = context

    def decode(self, pooled: torch.Tensor) -> TensorDict:
        ranks = self.regression(pooled).sigmoid()
        raw = self.distribution.snapshot().quantile(array(ranks, pa.float64())).to_numpy(zero_copy_only=False).copy()
        # Source units are output metadata; retain their float64 precision on CPU.
        content = torch.from_numpy(raw).reshape(ranks.shape)
        return TensorDict(
            {TensorKey.state: self.classification(pooled), TensorKey.content: content, "percentile": ranks}
        )


@quantile.register
def loss(module: Model, prediction: Prediction, batch: TensorFieldBase, strata: Strata) -> torch.Tensor:
    address = prediction.address
    request = cast(Request, module.schema.requests[address])
    embedder = cast(Embedder, module.nodes[address].embedder)
    logits = cast(torch.Tensor, prediction.payload[TensorKey.state]).reshape(-1, len(Tokens))
    ranks = cast(torch.Tensor, prediction.payload["percentile"]).reshape(-1)
    selected = batch.trainable.reshape(-1)
    states = cast(torch.Tensor, batch.targets[TensorKey.state]).reshape(-1)
    result = logits.sum() * 0.0 + ranks.sum() * 0.0
    if selected.any():
        result = result + module.track(
            (address, strata, Metric.loss, TensorKey.state),
            torch.nn.functional.cross_entropy(logits, states, weight=embedder.counter.weight, reduction="none")
            .masked_select(selected)
            .mean(),
        )
    valued = selected & states.eq(Tokens.valued)
    if valued.any():
        target = cast(torch.Tensor, batch.targets[TensorKey.content]).reshape(-1)
        match request.objective:
            case "mae":
                residual = (ranks - target).abs()
            case "mse":
                residual = (ranks - target).square()
            case "huber":
                residual = torch.nn.functional.huber_loss(ranks, target, reduction="none")
        result = result + module.track((address, strata, Metric.loss, TensorKey.content), residual[valued].mean())
        raw = cast(torch.Tensor, prediction.payload[TensorKey.content]).detach().cpu().reshape(-1)
        raw_target = (
            cast(torch.Tensor, batch.targets["raw"]).detach().cpu().contiguous().view(torch.float64).reshape(-1)
        )
        valid = valued.cpu() & torch.isfinite(raw)
        if valid.any():
            error = (raw - raw_target)[valid]
            module.track((address, strata, Metric.mae, TensorKey.content), error.abs().mean().to(ranks))
            module.track((address, strata, Metric.rmse, TensorKey.content), error.square().mean().sqrt().to(ranks))
    return result


@quantile.register
def output(module: Model, address: Address) -> pa.StructType:
    return pa.struct([pa.field(TensorKey.content.name, pa.float64(), nullable=True)])


@quantile.register
def write(module: Model, prediction: Prediction, datatype: pa.StructType) -> pa.StructArray:
    content = array(cast(torch.Tensor, prediction.payload[TensorKey.content]), pa.float64())
    content = pc.if_else(pc.is_nan(content), pa.scalar(None, type=pa.float64()), content)
    return struct({TensorKey.content.name: content}, datatype)


__all__ = [
    "Batch",
    "State",
    "Distribution",
    "Synchronize",
    "Request",
    "TensorField",
    "Embedder",
    "Decoder",
    "quantile",
    "observe",
    "bind",
    "learn",
    "loss",
    "output",
    "write",
]
