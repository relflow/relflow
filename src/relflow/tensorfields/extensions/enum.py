"""Scalar classification over an explicitly declared, fixed set of labels."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Unpack, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from tensordict import TensorDict, tensorclass

from relflow.data.ragged import RaggedField
from relflow.helpers.state import compatible
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
from relflow.tensorfields.output import array, struct, variable
from relflow.tensorfields.shared.counter import Counter, CounterUpdateCallback, tally
from relflow.tensorfields.shared.reconstruction import Metrics, Reconstruction, Scores

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.data.datasets.base import InterprocessEncodingContext
    from relflow.helpers.resize import Resize
    from relflow.structs.experiment import Schema

Value = bool | int | float | str | bytes

enum = Extension(name="enum", types=(bool, int, float, str, bytes))
enum.callback(CounterUpdateCallback, Metrics)


@enum.register
class Request(RequestBase):
    """One label from a required, nonempty tuple of unique scalar values.

    Declaration order defines fixed class IDs. Unknown values are rejected in
    every stage. Null is a value state, not a declared class. Input augmentation,
    reconstruction, and string-label predictions follow Category's conventions.
    """

    type: Literal["enum"] = "enum"
    values: tuple[Value, ...]
    p_unavailable: float = pydantic.Field(default=0.01, ge=0.0, le=1.0)
    topk: list[int] | None = None

    def __init__(
        self,
        *,
        values: tuple[Value, ...],
        p_unavailable: float = 0.01,
        topk: list[int] | None = None,
        type: Literal["enum"] = "enum",
        **options: Unpack[FieldOptions],
    ) -> None:
        # Serialized schemas use JSON arrays; public Python calls require tuples.
        if not isinstance(values, tuple):
            raise TypeError("Enum values must be a tuple of unique scalar labels")
        if any(isinstance(value, Mapping) for value in values):
            raise ValueError("Enum values must be scalar labels, not serialized objects")
        super().__init__(values=values, p_unavailable=p_unavailable, topk=topk, type=type, **options)

    @pydantic.field_validator("values", mode="before")
    @classmethod
    def check_values(cls, values: Any) -> tuple[Value, ...]:
        if not isinstance(values, (tuple, list)) or not values:
            raise ValueError("Enum values must be a nonempty tuple of unique scalar labels")
        # A tag preserves binary labels through JSON without conflating strings.
        values = tuple(
            bytes.fromhex(value["bytes"])
            if isinstance(value, dict) and set(value) == {"bytes"} and isinstance(value["bytes"], str)
            else value
            for value in values
        )
        kind = type(values[0])
        if kind not in (bool, int, float, str, bytes) or any(type(value) is not kind for value in values):
            raise ValueError("Enum values must all have one scalar type: bool, int, float, str, or bytes")
        if kind is float and any(not math.isfinite(cast(float, value)) for value in values):
            raise ValueError("Enum floating-point values must be finite")
        if len(set(values)) != len(values):
            raise ValueError("Enum values must be unique; remove duplicate labels")
        try:
            pa.array(values)
        except (pa.ArrowException, OverflowError) as error:
            raise ValueError("Enum values must fit one Arrow scalar type") from error
        return values

    @pydantic.field_serializer("values", when_used="json")
    def serialize_values(self, values: tuple[Value, ...]) -> list[Any]:
        return [{"bytes": value.hex()} if isinstance(value, bytes) else value for value in values]

    @pydantic.field_validator("topk")
    @classmethod
    def check_topk(cls, values: list[int] | None) -> list[int]:
        if values is None:
            return []
        if any(value <= 1 for value in values):
            raise ValueError("Enum topk values must be integers greater than one")
        return sorted(set(values))

    @classmethod
    def vocabulary(cls, source: Model | InterprocessEncodingContext, address: Address | str, /) -> tuple[Value, ...]:
        """Return declared labels in class order through a model or encoding context."""
        from relflow.architecture.root import Model

        address = Address(str(address))
        if isinstance(source, Model):
            if address not in source.nodes:
                raise KeyError(f"no field at address {str(address)!r}")
            embedder = getattr(source.nodes[address], "embedder", None)
            if not isinstance(embedder, Embedder):
                raise TypeError(f"address '{address}' is not an Enum field")
            return embedder.classes.values
        if isinstance(source, Mapping):
            if address not in source:
                raise KeyError(f"no encoding context at address {str(address)!r}")
            state = source[address]
            if not isinstance(state, State):
                raise TypeError(f"encoding context at '{address}' is not an Enum declaration")
            return state.values
        raise TypeError("Enum.vocabulary source must be a Model or encoding context")

    @classmethod
    def counts(cls, model: Model, address: Address | str, /) -> dict[Value, int]:
        """Return pristine training exposure for every declared class, excluding the prior."""
        from relflow.architecture.root import Model

        if not isinstance(model, Model):
            raise TypeError("Enum.counts source must be a Model")
        labels = cls.vocabulary(model, address)
        embedder = cast(Embedder, model.nodes[Address(str(address))].embedder)
        counts = cast(Counter, embedder.counters[TensorKey.content.name]).counts.detach().cpu().sub(1).clamp_min(0)
        return dict(zip(labels, map(int, counts.tolist()), strict=True))


@dataclass(frozen=True)
class State:
    """Immutable class order shared directly with encoding workers."""

    values: tuple[Value, ...]
    address: Address
    tokens: pa.Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", self.canonical(pa.array(cast(tuple[Any, ...], self.values))))

    @staticmethod
    def canonical(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
        """Make Arrow lookup agree with Python equality for signed floating zero."""
        if pa.types.is_floating(values.type):
            zero = pa.scalar(0.0, type=values.type)
            return pc.if_else(pc.equal(values, zero), zero, values)
        return values

    @property
    def signature(self) -> tuple[str, tuple[Value, ...]]:
        return type(self.values[0]).__name__, self.values

    def indices(self, values: pa.Array | pa.ChunkedArray) -> np.ndarray:
        if not len(values):
            return np.empty(0, dtype=np.int64)
        if not enum.matches(type(self.values[0]), values.type):
            raise ValueError(
                f"Enum field at '{self.address}' expects {type(self.values[0]).__name__} labels, got {values.type}"
            )
        try:
            values = self.canonical(pc.cast(values, self.tokens.type, safe=True))
            indices = pc.index_in(values, value_set=self.tokens)
        except pa.ArrowException as error:
            raise ValueError(
                f"Enum field at '{self.address}' contains values outside its declared scalar type"
            ) from error
        if indices.null_count:
            unknown = pc.filter(values, pc.is_null(indices))[0].as_py()
            raise ValueError(
                f"Enum field at '{self.address}' contains undeclared value {unknown!r}; "
                "declare it in values or normalize the input"
            )
        return pc.cast(indices, pa.int64()).to_numpy(zero_copy_only=False)

    def batch(self, values: pa.Array | pa.ChunkedArray) -> tuple[State, State]:
        self.indices(values)
        return self, self


class Classes(torch.nn.Module):
    """Protect parameter rows from being restored under a different class order."""

    classes: State

    def get_extra_state(self) -> tuple[str, tuple[Value, ...]]:
        return self.classes.signature

    def set_extra_state(self, state: tuple[str, tuple[Value, ...]]) -> None:
        if state != self.classes.signature:
            raise ValueError(f"Enum checkpoint at '{self.classes.address}' has a different values declaration")

    def rebuild_state(self, previous: torch.nn.Module) -> dict[str, Any]:
        if not isinstance(previous, type(self)) or previous.classes.signature != self.classes.signature:
            return {}
        resources = dict(previous.named_modules())
        for name, resource in self.named_modules():
            if isinstance(resource, Counter) and name in resources:
                resource.rebuild_state(resources[name])
        return compatible(self.state_dict(), previous.state_dict())


@enum.register
def observe(
    field: RaggedField, *, address: Address, schema: Schema, state: object | None, learn: bool
) -> TensorDict | None:
    if not learn:
        return None
    if not isinstance(state, State):
        raise TypeError(f"Enum field at '{address}' requires its declaration context")
    return TensorDict(
        {
            TensorKey.state: tally(torch.from_numpy(field.dense.copy()), len(Tokens)),
            TensorKey.content: tally(torch.from_numpy(state.indices(field.values).copy()), len(state.values)),
        },
        batch_size=[],
    )


@enum.register
@tensorclass
class TensorField(TensorFieldBase[torch.Tensor]):
    state: torch.Tensor
    content: torch.Tensor
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
        vocabulary = context.state if isinstance(context.state, State) else State(request.values, address)

        def encode(field: RaggedField) -> torch.Tensor:
            return torch.from_numpy(field.place(vocabulary.indices(field.values), fill=0))

        content = encode(input)
        targets = TensorDict(
            {TensorKey.state: torch.from_numpy(target.dense.copy()), TensorKey.content: encode(target)}
            if address in schema.objectives
            else {},
            batch_size=input.shape,
        )
        state = torch.from_numpy(input.dense.copy())
        if strata == Strata.train and request.p_unavailable:
            unavailable = torch.rand_like(state, dtype=torch.float).lt(request.p_unavailable) & state.eq(Tokens.valued)
            content = content.masked_fill(unavailable, -1)
        return cls(
            state=state,
            content=content,
            present=present,
            trainable=trainable,
            inferred=inferred,
            targets=targets,
            batch_size=input.batch_size,
        )


@enum.register
class Embedder(Classes, EmbedderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)
        self.classes = State(cast(Request, schema.requests[address]).values, address)
        self.embeddings = torch.nn.ModuleDict(
            {
                TensorKey.state.name: torch.nn.Embedding(len(Tokens), schema.d_model),
                TensorKey.content.name: torch.nn.Embedding(len(self.classes.values), schema.d_model),
            }
        )
        self.counters = torch.nn.ModuleDict(
            {
                TensorKey.state.name: Counter(address, len(Tokens)),
                TensorKey.content.name: Counter(address, len(self.classes.values)),
            }
        )

    @property
    def context(self) -> State:
        return self.classes

    def forward(self, inputs: TensorInput) -> Parcel:
        state = inputs.state.reshape(-1)
        content = cast(torch.Tensor, inputs.content).reshape(-1)
        known = state.eq(Tokens.valued) & content.ge(0)
        embedding = self.embeddings[TensorKey.content.name](content.masked_fill(~known, 0)) * known.unsqueeze(-1)
        return Parcel(
            payload=self.embeddings[TensorKey.state.name](state) + embedding,
            present=torch.ones(len(state), dtype=torch.bool, device=state.device),
            origin=self.address,
            destination=self.destination,
            batch_size=len(state),
        )


@enum.register
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
    classes = cast(Embedder, module.nodes[address].embedder).classes
    if not isinstance(binding, State) or binding.address != address or binding.signature != classes.signature:
        raise ValueError(f"Enum batch at '{address}' has a different values declaration; encode it again")


@enum.register
def learn(module: Model, observation: TensorDict, *, address: Address, strata: Strata) -> None:
    if strata != Strata.train:
        raise ValueError(f"Enum learner at '{address}' requires train strata, got {strata}")
    embedder = cast(Embedder, module.nodes[address].embedder)
    for key in (TensorKey.state, TensorKey.content):
        cast(Counter, embedder.counters[key.name]).learn(cast(torch.Tensor, observation[key]))


@enum.register
class Decoder(Classes, DecoderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)
        request = cast(Request, schema.requests[address])
        self.classes = State(request.values, address)
        self.linears = torch.nn.ModuleDict(
            {
                TensorKey.state.name: torch.nn.Linear(schema.d_model, len(Tokens)),
                TensorKey.content.name: torch.nn.Linear(schema.d_model, len(request.values)),
            }
        )
        self.metrics = Scores(address, partial(Reconstruction, cast(list[int], request.topk)))

    def decode(self, pooled: torch.Tensor) -> TensorDict:
        return TensorDict({key: linear(pooled) for key, linear in self.linears.items()})


@enum.register
def loss(module: Model, prediction: Prediction, batch: TensorFieldBase, strata: Strata) -> torch.Tensor:
    address = prediction.address
    selected = batch.trainable.reshape(-1)
    states = cast(torch.Tensor, batch.targets[TensorKey.state]).reshape(-1)
    logits = cast(torch.Tensor, prediction.payload[TensorKey.state]).reshape(-1, len(Tokens))
    state_loss = torch.nn.functional.cross_entropy(logits, states, reduction="none")[selected]
    result = module.track(
        (address, strata, Metric.loss, TensorKey.state), state_loss.sum() / selected.sum().clamp_min(1)
    )
    if selected.any():
        module.track(
            (address, strata, Metric.accuracy, TensorKey.state),
            logits.argmax(-1)[selected].eq(states[selected]).float().mean(),
        )
    decoder = cast(Decoder, module.nodes[address].decoder)
    size = len(decoder.classes.values)
    module.track((address, strata, "vocabulary", "size"), logits.new_tensor(float(size)))
    valued = selected & states.eq(Tokens.valued)
    content = cast(torch.Tensor, prediction.payload[TensorKey.content]).reshape(-1, size)[valued].float()
    targets = cast(torch.Tensor, batch.targets[TensorKey.content]).reshape(-1)[valued]
    objective = torch.nn.functional.cross_entropy(content, targets, reduction="sum")
    metric = cast(Reconstruction, decoder.metrics[f"{strata.value}_metrics"])
    with torch.no_grad():
        correct = content.argmax(-1).eq(targets).sum()
        topk = [content.topk(min(k, size), dim=-1).indices.eq(targets.unsqueeze(-1)).any(-1).sum() for k in metric.topk]
        metric.update(torch.stack((valued.sum(), valued.sum(), correct, *topk)), torch.stack((objective, objective)))
    return result + objective / valued.sum().clamp_min(1)


@enum.register
def output(module: Model, address: Address) -> pa.StructType:
    candidate = pa.struct(
        [
            pa.field(TensorKey.value.name, pa.large_string(), nullable=False),
            pa.field(TensorKey.probability.name, pa.float32(), nullable=False),
        ]
    )
    content = pa.struct(
        [
            pa.field(TensorKey.value.name, pa.large_string()),
            pa.field(TensorKey.probability.name, pa.float32(), nullable=False),
            pa.field(TensorKey.topk.name, pa.list_(candidate), nullable=False),
        ]
    )
    return pa.struct([pa.field(TensorKey.content.name, content, nullable=False)])


@enum.register
def write(module: Model, prediction: Prediction, datatype: pa.StructType) -> pa.StructArray:
    request = cast(Request, module.schema.requests[prediction.address])
    content_type = datatype.field(TensorKey.content.name).type
    candidate_type = content_type.field(TensorKey.topk.name).type.value_type
    vocabulary = pa.array([str(value) for value in request.values], type=pa.large_string())
    logits = cast(torch.Tensor, prediction.payload[TensorKey.content]).reshape(-1, len(request.values))
    probabilities = logits.softmax(-1)
    best_probabilities, best_indices = probabilities.max(-1)
    width = min(max(cast(list[int], request.topk), default=0), len(request.values))
    top_probabilities, top_indices = probabilities.topk(width, dim=-1)
    candidates = struct(
        {
            TensorKey.value.name: pc.take(vocabulary, array(top_indices, pa.int64())),
            TensorKey.probability.name: array(top_probabilities, pa.float32()),
        },
        candidate_type,
    )
    content = struct(
        {
            TensorKey.value.name: pc.take(vocabulary, array(best_indices, pa.int64())),
            TensorKey.probability.name: array(best_probabilities, pa.float32()),
            TensorKey.topk.name: variable(candidates, torch.full((len(logits),), width, dtype=torch.int64)),
        },
        content_type,
    )
    return struct({TensorKey.content.name: content}, datatype)


__all__ = [
    "Request",
    "State",
    "Classes",
    "TensorField",
    "Embedder",
    "Decoder",
    "enum",
    "observe",
    "bind",
    "learn",
    "loss",
    "output",
    "write",
]
