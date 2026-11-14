from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from dataclasses import InitVar
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Unpack, cast, override

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from beartype import beartype
from pydantic import StrictInt
from tensordict import TensorDict, tensorclass

from relflow.data.ragged import RaggedField
from relflow.structs.enums import Component, Metric, Strata, TensorKey, Tokens
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
from relflow.tensorfields.output import array, fixed, struct
from relflow.tensorfields.shared.channels import width
from relflow.tensorfields.shared.jitter import Jitter
from relflow.tensorfields.shared.reconstruction import Errors, Metrics, Scores, errors

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch


vector: Extension = Extension(name="vector", types=(int | float,), callbacks=(Metrics,))


class Objective(enum.StrEnum):
    l1 = "l1"
    l2 = "l2"

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean error over every coordinate of the selected vectors."""
        if self == Objective.l1:
            return torch.nn.functional.l1_loss(input=inputs, target=targets)

        return torch.nn.functional.mse_loss(input=inputs, target=targets)


class Joint:
    """A single query: every element must be finite, or the whole vector is not a measurement."""

    strict: ClassVar[bool] = True

    def size(self, n_dim: int) -> int:
        """Embedder input width for a vector of ``n_dim`` coordinates."""
        return n_dim

    def usable(self, encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the content and a Boolean per vector that is a measurement."""
        return encoded, np.asarray(np.isfinite(encoded).all(axis=1))

    def embed(self, content: torch.Tensor) -> torch.Tensor:
        """Embedder input for the coordinates."""
        return content

    def select(self, predicted: torch.Tensor, observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Coordinates that enter the loss."""
        return predicted, observed


class Channels(Joint):
    """A tuple query: a null or non-finite channel is missing, and the others stay valid."""

    strict: ClassVar[bool] = False

    @override
    def size(self, n_dim: int) -> int:
        return n_dim * 2

    @override
    def usable(self, encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        finite = np.isfinite(encoded)
        return np.where(finite, encoded, math.nan), np.asarray(finite.any(axis=1))

    @override
    def embed(self, content: torch.Tensor) -> torch.Tensor:
        observed = torch.isfinite(content)
        return torch.cat([content.masked_fill(~observed, 0.0), observed.to(content.dtype)], dim=-1)

    @override
    def select(self, predicted: torch.Tensor, observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        finite = torch.isfinite(observed)
        return predicted[finite], observed[finite]


READOUTS: Mapping[bool, Joint] = MappingProxyType({False: Joint(), True: Channels()})


@vector.register(Component.Request)
class Request(RequestBase):
    """Fixed-width numeric vector, such as a precomputed embedding.

    Every valued input must contain exactly ``n_dim`` non-null numeric elements;
    content is stored as float32 without fitting a normalizer. ``jitter`` adds
    training-only noise in the supplied coordinate units and preserves targets.
    ``objective`` selects mean ``"l1"`` or ``"l2"`` reconstruction loss across
    coordinates. ``mask=True`` makes the vector a supervised target.
    Predictions contain a reconstructed vector of the same width. A tuple
    ``query`` reads one source per element as the channels of the vector, and
    ``n_dim`` follows from its length.
    """

    type: Literal["vector"] = "vector"
    stacked: ClassVar[bool] = True
    n_dim: Annotated[StrictInt, pydantic.Field(gt=0)]
    jitter: Jitter = pydantic.Field(default_factory=Jitter)
    objective: Objective = Objective.l2

    @property
    def readout(self) -> Joint:
        """How elements become coordinates, chosen once by the kind of ``query``.

        A tuple ``query`` stacks channels that each keep their own validity: a
        channel that is null or non-finite is missing, its content is NaN, the
        embedder is told which channels are missing, and the loss skips it. A
        vector with no finite channel takes the ``other`` state.
        """
        return READOUTS[isinstance(self.query, tuple)]

    @pydantic.model_validator(mode="before")
    @classmethod
    def check_n_dim(cls, data: Any) -> Any:
        """Fill ``n_dim`` from a tuple ``query`` before validation.

        Args:
            data: Raw request input.

        Returns:
            The input with ``n_dim`` set to the number of tuple sources, if any.

        Raises:
            ValueError: If the tuple query is empty or disagrees with an explicit ``n_dim``.
        """
        return width(data=data)

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            n_dim: int = ...,
            jitter: Jitter = ...,
            objective: Objective | Literal["l1", "l2"] = Objective.l2,
            type: Literal["vector"] = "vector",
            **options: Unpack[FieldOptions],
        ) -> None: ...


def dense(values: pa.Array, *, request: Request, address: Address) -> np.ndarray:
    """Float32 ``(vectors, n_dim)`` matrix of a list array; null channels are NaN when the request allows them.

    Args:
        values: One Arrow list per valued record.
        request: The vector request.
        address: Field address, for error messages.

    Returns:
        The matrix, a fresh writable array.

    Raises:
        ValueError: If a value is not a list of ``n_dim`` numeric elements, or has a null element
            under a single query.
    """
    if not len(values):
        return np.empty((0, request.n_dim), dtype=np.float32)
    if not (
        pa.types.is_list(values.type) or pa.types.is_large_list(values.type) or pa.types.is_fixed_size_list(values.type)
    ):
        raise ValueError(f"vector field at '{address}' expects an Arrow list, got {values.type}")
    lengths = pc.list_value_length(values)
    # pyarrow-stubs incorrectly decorates not_equal as an overload.
    if pc.any(pc.not_equal(lengths, request.n_dim)).as_py():  # pyrefly: ignore [no-matching-overload]
        raise ValueError(f"vector field at '{address}' expects every value to have length {request.n_dim}")
    flattened = pc.list_flatten(values)
    if flattened.null_count and request.readout.strict:
        raise ValueError(f"vector field at '{address}' expects non-null numeric elements")
    try:
        flattened = pc.cast(flattened, pa.float32(), safe=True)
    except pa.ArrowException as error:
        raise ValueError(f"vector field at '{address}' could not be converted to float32") from error
    return flattened.to_numpy(zero_copy_only=False).reshape(-1, request.n_dim).copy()


@vector.register(Component.TensorField)
@tensorclass
class TensorField(TensorFieldBase[torch.Tensor]):
    content: torch.Tensor
    state: torch.Tensor
    present: torch.Tensor
    trainable: torch.Tensor
    inferred: torch.Tensor
    targets: TensorDict

    if TYPE_CHECKING:
        # TensorClass metadata is accepted by its generated initializer.
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
        request: Request = cast(Request, schema.requests[address])

        def encode(field: RaggedField) -> tuple[torch.Tensor, torch.Tensor]:
            """Dense content and state; an unusable vector is not a measurement.

            Such a vector takes the ``other`` state and zero content, so it is
            neither projected as a value nor regressed against.
            """
            values = field.values.combine_chunks() if isinstance(field.values, pa.ChunkedArray) else field.values
            state = field.dense.copy()
            encoded = dense(values, request=request, address=address)
            encoded, usable = request.readout.usable(encoded)
            if not usable.all():
                encoded[~usable] = 0.0
                state[field.place(np.logical_not(usable), fill=False)] = Tokens.other.value
            content = torch.from_numpy(field.place(encoded, fill=0.0, value_shape=(request.n_dim,)))
            return content, torch.from_numpy(state)

        content, state = encode(input)
        targets = TensorDict({}, batch_size=input.shape)
        if address in schema.objectives:
            target_content, target_state = encode(target)
            targets = TensorDict(
                {TensorKey.state: target_state, TensorKey.content: target_content}, batch_size=input.shape
            )

        return cls(
            content=content,
            state=state,
            present=present,
            trainable=trainable,
            inferred=inferred,
            targets=targets,
            batch_size=input.batch_size,
        )


@vector.register(Component.Embedder)
class Embedder(EmbedderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = cast(Request, schema.requests[address])
        self.origin: Address = address
        self.destination: Address = cast("Branch", request.parent).address
        self.jitter: Jitter = request.jitter
        self.readout: Joint = request.readout

        self.embeddings = torch.nn.Embedding(
            num_embeddings=len(Tokens),
            embedding_dim=schema.d_model,
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.readout.size(request.n_dim), out_features=schema.d_model),
            torch.nn.GELU(approximate="tanh"),
        )

    @beartype
    def forward(self, inputs: TensorInput) -> Parcel:
        N, *dims = inputs.state.shape
        D = math.prod((N, *dims))

        state = inputs.state.reshape(D)
        content = cast(torch.Tensor, inputs.content).reshape(D, -1)

        if self.training:
            eligible = state.eq(Tokens.valued).unsqueeze(-1) & torch.isfinite(content)
            content = self.jitter.apply(content, eligible)

        projected = self.linear(self.readout.embed(content))
        embeddings = self.embeddings(state)

        return Parcel(
            payload=(projected + embeddings).reshape(N, *dims, -1),
            present=torch.ones(N, dtype=torch.bool, device=projected.device),
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )


@vector.register(Component.Decoder)
class Decoder(DecoderBase):
    coordinatewise = True

    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = cast(Request, schema.requests[address])

        self.classification = torch.nn.Linear(
            in_features=schema.d_model,
            out_features=len(Tokens),
        )
        self.regression = torch.nn.Linear(
            in_features=schema.d_model,
            out_features=request.n_dim,
        )
        self.metrics = Scores(address, Errors)

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict:
        return TensorDict(
            source={
                TensorKey.state: self.classification(pooled),
                TensorKey.content: self.regression(pooled),
            }
        )


@vector.register(Component.loss)
def loss(
    module: Model,
    prediction: Prediction,
    batch: TensorFieldBase,
    strata: Strata,
) -> torch.Tensor:
    address: Address = prediction.address
    request: Request = cast(Request, module.schema.requests[address])
    decoder: Decoder = cast(Decoder, module.nodes[address].decoder)

    trainable = batch.trainable.reshape(-1)
    state_targets = cast(torch.Tensor, batch.targets[TensorKey.state]).reshape(-1)
    state_inputs = cast(torch.Tensor, prediction.payload[TensorKey.state]).reshape(-1, len(Tokens))

    output: torch.Tensor = module.track(
        (address, strata, Metric.loss, TensorKey.state),
        value=(
            torch.nn.functional.cross_entropy(
                input=state_inputs,
                target=state_targets,
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
    inputs = cast(torch.Tensor, prediction.payload[TensorKey.content]).reshape(-1, request.n_dim)
    targets = cast(torch.Tensor, batch.targets[TensorKey.content]).reshape(-1, request.n_dim)
    predicted, observed = request.readout.select(inputs[valued], targets[valued])
    if predicted.numel():
        output += module.track(
            (address, strata, Metric.loss, TensorKey.content),
            value=request.objective.loss(inputs=predicted, targets=observed),
        )

    decoder.metrics.record(module, strata, *errors(predicted, observed))

    return output


@vector.register(Component.output)
def output(module: Model, address: Address) -> pa.StructType:
    request: Request = cast(Request, module.schema.requests[address])
    content = pa.list_(pa.field("element", pa.float32()), request.n_dim)
    return pa.struct([pa.field(TensorKey.content.name, content, nullable=False)])


@vector.register(Component.write)
def write(module: Model, prediction: Prediction, datatype: pa.StructType) -> pa.StructArray:
    request: Request = cast(Request, module.schema.requests[prediction.address])
    content = cast(torch.Tensor, prediction.payload[TensorKey.content]).detach().float().clone()
    non_valued = cast(torch.Tensor, prediction.payload[TensorKey.state]).argmax(dim=-1).ne(Tokens.valued.value)
    content[non_valued] = 0.0
    values = fixed(array(content, pa.float32()), request.n_dim, datatype=datatype.field(TensorKey.content.name).type)
    return struct({TensorKey.content.name: values}, datatype)
