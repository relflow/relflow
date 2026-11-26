"""Frozen pretrained encoders, field projections and vector reconstruction."""

from __future__ import annotations

import enum
import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, cast

import torch
from beartype import beartype
from tensordict import TensorDict

from relflow.structs.enums import Metric, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address
from relflow.tensorfields.base import DecoderBase, EmbedderBase, TensorFieldBase, TensorInput
from relflow.tensorfields.shared.counter import Counter, tally
from relflow.tensorfields.shared.jitter import Jitter
from relflow.tensorfields.shared.reconstruction import Errors, Scores, errors

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from relflow.architecture.root import Model
    from relflow.data.ragged import RaggedField
    from relflow.structs.experiment import Schema


class Objective(enum.StrEnum):
    l1 = "l1"
    l2 = "l2"

    def loss(self, diff: torch.Tensor) -> torch.Tensor:
        """Mean error over every coordinate of the selected embeddings."""
        if self == Objective.l1:
            diff = diff.absolute()
        else:
            diff = diff.square()

        return diff.mean()


class Options(Protocol):
    model: str
    objective: Objective
    jitter: Jitter


@dataclass
class Backbone:
    _models: ClassVar[dict[str, "Backbone"]] = {}

    key: str
    model: PreTrainedModel
    hidden_size: int
    device: torch.device | None = None
    revision: str | None = None
    versions: tuple | None = None

    def identity(self) -> str:
        parameters = tuple(self.model.parameters()) if callable(getattr(self.model, "parameters", None)) else ()
        buffers = tuple(self.model.buffers()) if callable(getattr(self.model, "buffers", None)) else ()
        config = self.model.config
        configuration = config.to_dict() if callable(getattr(config, "to_dict", None)) else vars(config)
        serialized = json.dumps(
            (configuration, getattr(config, "_attn_implementation", None)), sort_keys=True, default=str
        )
        versions = (serialized, *((id(p), p._version, p.dtype, tuple(p.shape)) for p in (*parameters, *buffers)))
        if self.revision is None or self.versions != versions:
            digest = hashlib.sha256()
            digest.update(serialized.encode())
            digest.update(self.key.encode())
            state = (
                (*self.model.named_parameters(), *self.model.named_buffers())
                if callable(getattr(self.model, "named_parameters", None))
                else ()
            )
            for name, value in state:
                digest.update(name.encode())
                digest.update(str((value.dtype, tuple(value.shape))).encode())
                digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
            if not state:
                digest.update(repr((type(self.model).__module__, type(self.model).__qualname__)).encode())
            self.revision = digest.hexdigest()
            self.versions = versions
        return self.revision

    @classmethod
    def get_model(
        cls,
        key: str,
    ) -> "Backbone":
        if key not in cls._models:
            from transformers import AutoModel

            # Version counters must remain available after inference-only warm-up.
            with torch.inference_mode(False):
                model = cast("PreTrainedModel", AutoModel.from_pretrained(key))
            model.eval()
            model.requires_grad_(False)
            hidden_size = getattr(model.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError(f"encoder '{key}' does not expose `config.hidden_size`")
            cls._models[key] = cls(key=key, model=model, hidden_size=int(hidden_size))

        return cls._models[key]

    def module_for(self, device: torch.device) -> PreTrainedModel:
        target = torch.device(device)
        if self.device != target:
            # Device transfers can replace buffers. Cache identity needs their version counters.
            with torch.inference_mode(False):
                self.model.to(target)
            self.device = target

        self.model.eval()
        return self.model

    @torch.compiler.disable
    def variant(self, device: torch.device, processing: str) -> str:
        dtype = torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type) else None
        hardware = torch.cuda.get_device_capability(device) if device.type == "cuda" else None
        payload = (
            self.identity(),
            processing,
            str(dtype),
            device.type,
            hardware,
            torch.get_float32_matmul_precision(),
            torch.__version__,
        )
        return hashlib.sha256(repr(payload).encode()).hexdigest()


def observe(
    field: RaggedField,
    *,
    address: Address,
    schema: Schema,
    state: object | None,
    learn: bool,
) -> TensorDict | None:
    """Count pristine field states without opening hidden content."""

    if not learn:
        return None
    return TensorDict(
        {TensorKey.state: tally(torch.from_numpy(field.dense.copy()), len(Tokens))},
        batch_size=[],
    )


class Embedder(EmbedderBase):
    backbone: ClassVar[type[Backbone]] = Backbone

    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request = cast(Options, schema.requests[address])

        self.origin: Address = address
        self._cached_model = self.backbone.get_model(request.model)
        self.hidden_size = self._cached_model.hidden_size
        self.jitter: Jitter = request.jitter
        self.input_norm: torch.nn.Module = torch.nn.Identity()

        self.embeddings = torch.nn.Embedding(
            num_embeddings=len(Tokens),
            embedding_dim=schema.d_model,
        )
        self.counter = Counter(address=address, size=len(Tokens))
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.hidden_size, out_features=schema.d_model),
            torch.nn.GELU(approximate="tanh"),
        )

    def encode(self, content: TensorDict, state: torch.Tensor) -> torch.Tensor:
        """Return pristine vectors on valued coordinates in the field geometry."""
        raise NotImplementedError

    @beartype
    def forward(self, inputs: TensorInput) -> Parcel:
        N, *dims = inputs.state.shape
        D = math.prod((N, *dims))

        state = inputs.state.reshape(D)
        valued = state.eq(Tokens.valued.value).unsqueeze(-1)
        encoded = self.encode(content=cast(TensorDict, inputs.content), state=inputs.state)
        encoded = self.input_norm(encoded.reshape(D, self.hidden_size))
        if self.training:
            eligible = valued & torch.isfinite(encoded)
            encoded = self.jitter.apply(encoded, eligible)
        projected = self.linear(encoded) * valued
        embeddings = self.embeddings(state)

        return Parcel(
            payload=(embeddings + projected).reshape(N, *dims, -1),
            present=torch.ones(N, dtype=torch.bool, device=embeddings.device),
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )


def learn(
    module: Model,
    observation: TensorDict,
    *,
    address: Address,
    strata: Strata,
) -> None:
    """Apply pristine field-state counts to the model resource."""

    if strata != Strata.train:
        raise ValueError(f"field learner at '{address}' requires train strata, got {strata}")
    embedder: Embedder = cast(Embedder, module.nodes[address].embedder)
    embedder.counter.learn(cast(torch.Tensor, observation[TensorKey.state]))


class Decoder(DecoderBase):
    coordinatewise = True
    backbone: ClassVar[type[Backbone]] = Backbone

    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request = cast(Options, schema.requests[address])
        hidden_size = self.backbone.get_model(request.model).hidden_size
        self.classification = torch.nn.Linear(
            in_features=schema.d_model,
            out_features=len(Tokens),
        )
        self.linear = torch.nn.Linear(
            in_features=schema.d_model,
            out_features=hidden_size,
        )
        self.metrics = Scores(address, Errors)

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict:
        return TensorDict(
            source={
                TensorKey.state: self.classification(pooled),
                TensorKey.content: self.linear(pooled),
            }
        )


def loss(
    module: Model,
    prediction: Prediction,
    batch: TensorFieldBase,
    strata: Strata,
) -> torch.Tensor:
    address: Address = prediction.address
    request = cast(Options, module.schema.requests[address])
    embedder: Embedder = cast(Embedder, module.nodes[address].embedder)
    decoder: Decoder = cast(Decoder, module.nodes[address].decoder)

    trainable = batch.trainable.reshape(-1)
    state_targets = cast(torch.Tensor, batch.targets[TensorKey.state]).reshape(-1)
    state_inputs = cast(torch.Tensor, prediction.payload[TensorKey.state]).reshape(-1, len(Tokens))

    loss: torch.Tensor = module.track(
        (address, strata, Metric.loss, TensorKey.state),
        value=(
            torch.nn.functional.cross_entropy(
                input=state_inputs,
                target=state_targets,
                weight=embedder.counter.weight,
                reduction="none",
            )
            .masked_select(trainable)
            .mean()
        ),
    )

    module.track(
        (address, strata, Metric.accuracy, TensorKey.state),
        value=state_inputs.argmax(dim=1).eq(state_targets).masked_select(trainable).float().mean(),
    )

    valued = trainable & state_targets.eq(Tokens.valued.value)
    inputs = cast(torch.Tensor, prediction.payload[TensorKey.content]).reshape(-1, embedder.hidden_size)
    predicted = inputs[valued]
    observed = inputs.new_zeros((0, embedder.hidden_size))
    if predicted.numel():
        targets = batch.targets.get(TensorKey.embedding, None)
        if targets is None:
            targets = embedder.encode(
                content=cast(TensorDict, batch.targets[TensorKey.content]),
                state=cast(torch.Tensor, batch.targets[TensorKey.state]),
            )
            # Worker-shipped targets have locked, consolidated storage.
            if not batch.targets.is_locked:
                batch.targets[TensorKey.embedding] = targets
        observed = cast(torch.Tensor, targets).reshape(-1, embedder.hidden_size)[valued]
        loss += module.track(
            (address, strata, Metric.loss, TensorKey.content),
            value=request.objective.loss(predicted.subtract(observed)),
        )

    decoder.metrics.record(module, strata, *errors(predicted, observed))

    return loss
