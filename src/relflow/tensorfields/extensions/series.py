from __future__ import annotations

import enum
import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import InitVar
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self, Unpack, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from beartype import beartype
from pydantic import StrictInt
from tensordict import TensorDict, tensorclass

from relflow.architecture.convolution import Block
from relflow.data.ragged import RaggedField
from relflow.structs.enums import Component, Direction, Metric, Overflow, Strata, TensorKey, Tokens
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
from relflow.tensorfields.shared import channels
from relflow.tensorfields.shared.jitter import Jitter
from relflow.tensorfields.shared.normalizer import GlobalOnlineNormalizer, moments, precise
from relflow.tensorfields.shared.reconstruction import Errors, Metrics, Scores, errors

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch


series: Extension = Extension(name="series", types=(int | float,), callbacks=(Metrics,))

VALUES = "values"
OBSERVED = "observed"
SCALED = "scaled"


class Objective(enum.StrEnum):
    l1 = "l1"
    l2 = "l2"

    def loss(self, *, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean error over every selected step and channel.

        Args:
            inputs: Predicted values.
            targets: Observed values of the same shape.

        Returns:
            The scalar mean error of this objective.
        """
        return LOSSES[self](input=inputs, target=targets)


LOSSES: Mapping[Objective, Callable[..., torch.Tensor]] = MappingProxyType(
    {Objective.l1: torch.nn.functional.l1_loss, Objective.l2: torch.nn.functional.mse_loss}
)


@series.register(Component.Request)
class Request(RequestBase):
    """One multichannel time series per record.

    Each record holds one numeric value per channel and step, observed at
    consecutive positions; steps are ordered but not timestamped. ``n_dim`` is
    the number of channels. A tuple ``query`` reads one source per channel,
    sets ``n_dim`` and names the forecast step members by its query text; a
    single query reads ``list<number>`` (``n_dim=1``) or lists of steps that
    each hold ``n_dim`` values. A step with a missing channel is unobserved.
    ``length`` caps the retained steps with an ``overflow`` policy as
    ``Branch`` does; ``None`` keeps the whole series and pads each batch to its
    longest. ``n_blocks`` sets the number of dilated convolution blocks. As an
    input, the series embeds as one pooled token. As a reconstruction target
    (``mask=True`` or a reconstructing ancestor) the series is a forecast
    target and ``horizon`` gives the number of steps to predict; ``horizon`` is
    required there and rejected elsewhere. ``jitter`` adds training-only noise
    to input steps, and ``objective`` selects mean ``"l1"`` or ``"l2"`` error
    over predicted steps. Predictions contain ``horizon`` steps of one value
    per channel in source units.
    """

    type: Literal["series"] = "series"
    stacked: ClassVar[bool] = True
    n_dim: Annotated[StrictInt, pydantic.Field(gt=0)]
    length: Annotated[StrictInt | None, pydantic.Field(gt=0)] = None
    overflow: Overflow | None = None
    horizon: Annotated[StrictInt | None, pydantic.Field(gt=0)] = None
    n_blocks: Annotated[StrictInt, pydantic.Field(gt=0)] = 6
    jitter: Jitter = pydantic.Field(default_factory=Jitter)
    objective: Objective = Objective.l2

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            n_dim: int = ...,
            length: int | None = None,
            overflow: Overflow | Literal["head", "tail", "random", "error"] | None = None,
            horizon: int | None = None,
            n_blocks: int = 6,
            jitter: Jitter = ...,
            objective: Objective | Literal["l1", "l2"] = Objective.l2,
            type: Literal["series"] = "series",
            **options: Unpack[FieldOptions],
        ) -> None: ...

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
        return channels.width(data=data)

    @pydantic.model_validator(mode="after")
    def check_bounds(self) -> Self:
        """Default a bounded series to ``head`` overflow and reject overflow without a length.

        Returns:
            This request with its overflow policy resolved.

        Raises:
            ValueError: If an unbounded series configures an overflow policy.
        """
        if self.length is None and self.overflow is not None:
            message = (
                f"unbounded series field cannot configure overflow={self.overflow!r}; "
                "set a finite length or remove the overflow policy"
            )
            raise ValueError(message)
        if self.length is not None and self.overflow is None:
            self.overflow = Overflow.head

        return self


def floats(values: pa.Array, *, address: Address) -> np.ndarray:
    """Float64 elements with null as NaN; integers beyond ±2^53 are rejected."""
    try:
        values = pc.cast(values, pa.float64(), safe=True)
    except pa.ArrowInvalid as error:
        message = (
            f"series field at '{address}' holds numbers float64 cannot represent exactly "
            "(integers beyond ±2^53); scale them in a preprocessor"
        )
        raise ValueError(message) from error
    filled = cast(pa.Array, pc.fill_null(values, pa.scalar(math.nan, pa.float64())))
    return filled.to_numpy(zero_copy_only=False)


def listlike(datatype: pa.DataType) -> bool:
    """Whether an Arrow type is a variable, large or fixed-size list.

    Args:
        datatype: The Arrow type to test.

    Returns:
        True for any list layout.
    """
    return pa.types.is_list(datatype) or pa.types.is_large_list(datatype) or pa.types.is_fixed_size_list(datatype)


def steps(values: pa.Array, *, width: int, address: Address, stacked: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Every step of every series back to back as ``(steps, channels)`` float64, with the step count per series.

    A null step or element is NaN; a step reads as observed only when all
    its channels are finite.

    Args:
        values: One Arrow list of steps per record.
        width: Number of channels per step.
        address: Field address, for error messages.
        stacked: Whether the values come from a tuple query, whose sources must each select a list.

    Returns:
        The step matrix and the number of steps of each record.

    Raises:
        ValueError: If a record is not a list, a tuple query selects no list level, or a step does not hold
            ``width`` values.
    """
    if not len(values):
        return np.empty((0, width)), np.empty(0, dtype=np.int64)
    if not listlike(values.type):
        message = f"series field at '{address}' expects an Arrow list per record, got {values.type}"
        raise ValueError(message)

    lengths = pc.list_value_length(values).to_numpy(zero_copy_only=False).astype(np.int64)
    flat = pc.list_flatten(values)
    if stacked and not listlike(flat.type):
        message = (
            f"series field at '{address}' reads a tuple query of scalar values; "
            "each tuple source must select a list per record, one element per step"
        )
        raise ValueError(message)
    if not listlike(flat.type):
        if width != 1:
            message = (
                f"series field at '{address}' has scalar steps but expects n_dim={width} values per step; "
                "nest each step as a list"
            )
            raise ValueError(message)
        return floats(flat, address=address).reshape(-1, 1), lengths

    widths = pc.list_value_length(flat)
    wrong = pc.filter(widths, pc.not_equal(widths, pa.scalar(width, widths.type)))
    if len(wrong):
        message = f"series field at '{address}' expects n_dim={width} values per step, got {wrong[0].as_py()}"
        raise ValueError(message)
    matrix = np.full((len(flat), width), math.nan)
    matrix[pc.is_valid(flat).to_numpy(zero_copy_only=False)] = floats(pc.list_flatten(flat), address=address).reshape(
        -1, width
    )
    return matrix, lengths


def combine(field: RaggedField) -> pa.Array:
    """The values of a field as one contiguous Arrow array."""
    return field.values.combine_chunks() if isinstance(field.values, pa.ChunkedArray) else field.values


def keep_head(*, excess: np.ndarray, **_: object) -> np.ndarray:
    """Start each series at its first step, dropping the tail."""
    return np.zeros_like(excess)


def keep_tail(*, excess: np.ndarray, **_: object) -> np.ndarray:
    """Start each series so that its last steps remain."""
    return excess


def keep_window(*, excess: np.ndarray, strata: Strata, salt: int, address: Address, **_: object) -> np.ndarray:
    """Start each series at a random offset while training and keep the tail otherwise.

    The seed derives from the salt and the address, so crops repeat for a fixed salt.
    """
    if strata != Strata.train:
        return excess
    seed = hashlib.sha256(f"relflow-series-overflow-v1:{salt}:{address}".encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(seed, "big")).integers(0, excess + 1)


def reject(*, lengths: np.ndarray, cap: int | None, address: Address, **_: object) -> np.ndarray:
    """Refuse a series longer than the cap."""
    message = (
        f"series field at '{address}' holds {int(lengths.max())} steps but length={cap} with overflow='error'; "
        "raise length or choose another overflow"
    )
    raise ValueError(message)


OVERFLOW: Mapping[Overflow, Callable[..., np.ndarray]] = MappingProxyType(
    {
        Overflow.head: keep_head,
        Overflow.tail: keep_tail,
        Overflow.random: keep_window,
        Overflow.error: reject,
    }
)


def crop(
    lengths: np.ndarray,
    *,
    excess: np.ndarray,
    cap: int | None,
    request: Request,
    address: Address,
    strata: Strata,
    salt: int,
    target: bool,
) -> np.ndarray:
    """First retained step of every series.

    Args:
        lengths: Step count of each record.
        excess: Steps beyond the cap of each record.
        cap: Retained step limit, ``None`` when unbounded.
        request: Field options, whose ``overflow`` picks the policy.
        address: Field address, for error messages and the crop seed.
        strata: Current stage; random crops apply only to training.
        salt: Run salt for the crop seed.
        target: Whether the field is a forecast target, which never crops.

    Returns:
        The offset of the first kept step in each record.

    Raises:
        ValueError: If a target exceeds ``horizon`` or ``overflow='error'`` meets an excess.
    """
    if not excess.any():
        return np.zeros_like(lengths)
    if target:
        message = (
            f"series target at '{address}' holds {int(lengths.max())} steps but horizon={cap}; "
            "shorten it or raise horizon"
        )
        raise ValueError(message)
    return OVERFLOW[cast(Overflow, request.overflow)](
        lengths=lengths, excess=excess, cap=cap, address=address, strata=strata, salt=salt
    )


def encode(
    field: RaggedField,
    *,
    request: Request,
    address: Address,
    strata: Strata,
    salt: int,
    target: bool,
) -> tuple[TensorDict, torch.Tensor]:
    """Dense steps, their observed mask and record state tokens of one field.

    Inputs keep at most ``length`` steps by ``overflow`` and pad the batch to
    its longest series, never below one step. A target pads to exactly
    ``horizon`` steps and rejects a longer series. A non-empty series without
    one observed step takes the ``other`` state; an empty one stays valued.
    """

    matrix, lengths = steps(
        values=combine(field=field), width=request.n_dim, address=address, stacked=isinstance(request.query, tuple)
    )
    cap = request.horizon if target else request.length
    kept = lengths if cap is None else np.minimum(lengths, cap)
    start = crop(
        lengths,
        excess=lengths - kept,
        cap=cap,
        request=request,
        address=address,
        strata=strata,
        salt=salt,
        target=target,
    )

    width = cast(int, cap) if target else max(1, int(kept.max(initial=0)))
    record = np.repeat(np.arange(len(lengths)), kept)
    offset = np.arange(len(record)) - np.repeat(np.cumsum(kept) - kept, kept)
    picked = matrix[(np.cumsum(lengths) - lengths)[record] + start[record] + offset]
    finite = np.isfinite(picked).all(axis=1)

    content = np.zeros((len(lengths), width, request.n_dim))
    observed = np.zeros((len(lengths), width), dtype=np.bool_)
    content[record[finite], offset[finite]] = picked[finite]
    observed[record[finite], offset[finite]] = True

    state = field.dense.copy()
    state[field.place((kept > 0) & ~observed.any(axis=1), fill=False)] = Tokens.other.value
    encoded = TensorDict(
        {
            VALUES: torch.from_numpy(field.place(content, fill=0.0, value_shape=(width, request.n_dim))),
            OBSERVED: torch.from_numpy(field.place(observed, fill=False, value_shape=(width,))),
        },
        batch_size=field.shape,
    )
    return encoded, torch.from_numpy(state)


@series.register(Component.TensorField)
@tensorclass
class TensorField(TensorFieldBase[TensorDict]):
    content: TensorDict
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

        content, state = encode(input, request=request, address=address, strata=strata, salt=context.salt, target=False)
        targets = TensorDict({}, batch_size=input.shape)
        if address in schema.objectives:
            target_content, target_state = encode(
                target, request=request, address=address, strata=strata, salt=context.salt, target=True
            )
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


@series.register(Component.observe)
def observe(
    field: RaggedField,
    *,
    address: Address,
    schema: Schema,
    state: object | None,
    learn: bool,
) -> TensorDict | None:
    """Summarize every observed step of the complete pristine series, per channel."""

    if not learn:
        return None
    request = cast(Request, schema.requests[address])
    matrix, _ = steps(
        values=combine(field=field), width=request.n_dim, address=address, stacked=isinstance(request.query, tuple)
    )
    observed = torch.from_numpy(matrix[np.isfinite(matrix).all(axis=1)])
    return TensorDict({TensorKey.content: moments(observed, width=request.n_dim)}, batch_size=[])


@series.register(Component.Embedder)
class Embedder(EmbedderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = cast(Request, schema.requests[address])
        if address in schema.objectives and request.horizon is None:
            message = (
                f"series field at '{address}' is a reconstruction target and requires "
                "horizon=<steps to predict>; set horizon or remove mask"
            )
            raise ValueError(message)
        if address not in schema.objectives and request.horizon is not None:
            message = (
                f"series field at '{address}' sets horizon={request.horizon}, but horizon applies only to "
                "reconstruction targets; remove horizon or add mask=True"
            )
            raise ValueError(message)

        self.origin: Address = address
        self.destination: Address = cast("Branch", request.parent).address
        self.jitter: Jitter = request.jitter

        self.normalizer = GlobalOnlineNormalizer(width=request.n_dim)
        self.projection = torch.nn.Linear(in_features=request.n_dim, out_features=schema.d_model)
        self.blocks = torch.nn.ModuleList(
            [
                Block(width=schema.d_model, dilation=2**depth, direction=Direction.bidirectional)
                for depth in range(request.n_blocks)
            ]
        )
        self.embeddings = torch.nn.Embedding(num_embeddings=len(Tokens), embedding_dim=schema.d_model)

    @beartype
    def forward(self, inputs: TensorInput) -> Parcel:
        """Pool each record's observed steps to one token.

        Values standardize per channel in the widest float the device holds,
        then the network runs float32. Jitter perturbs observed steps of valued
        records, in standardized units when ``normalize`` is set.
        """

        state = inputs.state
        content = cast(TensorDict, inputs.content)
        observed = cast(torch.Tensor, content[OBSERVED]) & state.eq(Tokens.valued).unsqueeze(-1)
        keep = observed.unsqueeze(-1)

        with precise(state.device):
            values = cast(torch.Tensor, content[VALUES]).to(dtype=self.normalizer.mean.dtype)
            eligible = keep.expand_as(values)
            if self.training and not self.jitter.normalize:
                values = self.jitter.apply(values, eligible)
            values = (values - self.normalizer.mean) / self.normalizer.scale
            if self.training and self.jitter.normalize:
                values = self.jitter.apply(values, eligible)
            values = values.masked_fill(~eligible, 0.0).float()

        hidden = self.projection(values).masked_fill(~keep, 0.0)
        for block in self.blocks:
            hidden = block(hidden, observed=observed)
        # Records with no observed step pool to zeros, so neither the value nor its gradient sees -inf.
        pooled = hidden.masked_fill(~keep, -torch.inf).amax(dim=1)
        pooled = torch.where(observed.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))

        return Parcel(
            payload=pooled + self.embeddings(state),
            present=torch.ones(state.shape[0], dtype=torch.bool, device=state.device),
            origin=self.origin,
            destination=self.destination,
            batch_size=state.shape[0],
        )


@series.register(Component.learn)
def learn(
    module: Model,
    observation: TensorDict,
    *,
    address: Address,
    strata: Strata,
) -> None:
    """Apply pristine per-channel statistics."""

    if strata != Strata.train:
        message = f"series learner at '{address}' requires train strata, got {strata}"
        raise ValueError(message)
    embedder: Embedder = cast(Embedder, module.nodes[address].embedder)
    embedder.normalizer.learn(cast(TensorDict, observation[TensorKey.content]))
    if address in module.schema.objectives:
        decoder: Decoder = cast(Decoder, module.nodes[address].decoder)
        decoder.normalizer.learn(cast(TensorDict, observation[TensorKey.content]))


@series.register(Component.Decoder)
class Decoder(DecoderBase):
    coordinatewise = True

    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = cast(Request, schema.requests[address])
        self.classification = torch.nn.Linear(in_features=schema.d_model, out_features=len(Tokens))
        self.forecast: bool = address in schema.objectives
        if not self.forecast:
            return

        horizon = cast(int, request.horizon)
        self.horizon: int = horizon
        self.width: int = request.n_dim
        self.regression = torch.nn.Linear(in_features=schema.d_model, out_features=horizon * request.n_dim)
        # Target statistics per channel: the forecast starts at their mean and predicts standardized deviations.
        self.normalizer: GlobalOnlineNormalizer = GlobalOnlineNormalizer(width=request.n_dim)
        self.metrics = Scores(address=address, factory=Errors)

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict:
        state = self.classification(pooled)
        if not self.forecast:
            return TensorDict(source={TensorKey.state: state})

        with precise(pooled.device):
            scaled = self.regression(pooled.float()).reshape(*pooled.shape[:-1], self.horizon, self.width)
            scaled = scaled.to(dtype=self.normalizer.mean.dtype)
            content = self.normalizer.mean + self.normalizer.scale * scaled
        return TensorDict(source={TensorKey.state: state, TensorKey.content: content, SCALED: scaled})


@series.register(Component.loss)
def loss(
    module: Model,
    prediction: Prediction,
    batch: TensorFieldBase,
    strata: Strata,
) -> torch.Tensor:
    """Cross-entropy of the record state plus the objective over observed forecast steps."""
    address: Address = prediction.address
    request: Request = cast(Request, module.schema.requests[address])
    decoder: Decoder = cast(Decoder, module.nodes[address].decoder)
    horizon, width = decoder.horizon, request.n_dim

    size: int = cast(torch.Tensor, batch.targets[TensorKey.state]).numel()
    trainable = batch.trainable.reshape(size)
    state_targets = cast(torch.Tensor, batch.targets[TensorKey.state]).reshape(size)
    state_inputs = cast(torch.Tensor, prediction.payload[TensorKey.state]).reshape(size, len(Tokens))

    output: torch.Tensor = module.track(
        (address, strata, Metric.loss, TensorKey.state),
        value=torch.nn.functional.cross_entropy(input=state_inputs, target=state_targets, reduction="none")
        .masked_select(trainable)
        .mean(),
    )
    module.track(
        (address, strata, Metric.accuracy, TensorKey.state),
        value=state_inputs.argmax(dim=1).eq(state_targets).masked_select(trainable).float().mean(),
    )

    target = cast(TensorDict, batch.targets[TensorKey.content])
    scaled = cast(torch.Tensor, prediction.payload[SCALED]).reshape(size, horizon, width)
    values = cast(torch.Tensor, target[VALUES]).reshape(size, horizon, width).to(dtype=scaled.dtype)
    observed = cast(torch.Tensor, target[OBSERVED]).reshape(size, horizon)
    entries = (trainable & state_targets.eq(Tokens.valued.value)).unsqueeze(-1) & observed
    if entries.any():
        with precise(scaled.device):
            standardized = (values - decoder.normalizer.mean) / decoder.normalizer.scale
        output += module.track(
            (address, strata, Metric.loss, TensorKey.content),
            value=request.objective.loss(inputs=scaled[entries], targets=standardized[entries]).to(dtype=output.dtype),
        )
    else:
        output += scaled.sum().to(dtype=output.dtype) * 0.0

    # Error metrics report source units whatever the training scale.
    content = cast(torch.Tensor, prediction.payload[TensorKey.content]).reshape(size, horizon, width)
    counts, sums = errors(predicted=content[entries], observed=values[entries])
    decoder.metrics.record(module=module, strata=strata, counts=counts, sums=sums)

    return output


def step_type(*, names: tuple[str, ...] | None) -> pa.DataType:
    """Arrow type of one forecast step.

    Args:
        names: The query text of each channel for a tuple query, else ``None``.

    Returns:
        A list of float64 values without names, or a struct with one
        non-null float64 member per name, in name order.
    """
    if names is None:
        return pa.list_(pa.field("element", pa.float64()))
    return pa.struct([pa.field(name, pa.float64(), nullable=False) for name in names])


@series.register(Component.output)
def output(module: Model, address: Address) -> pa.StructType:
    """Arrow type of a prediction: ``horizon`` steps of one float64 value per channel per record.

    A single query writes each step as a list of values; a tuple query writes
    each step as a struct with one member per query text, in tuple order.

    Args:
        module: The model that owns the field.
        address: Field address.

    Returns:
        A struct with a non-null ``content`` list of steps.
    """
    query = cast(Request, module.schema.requests[address]).query
    step = step_type(names=query if isinstance(query, tuple) else None)
    return pa.struct([pa.field(TensorKey.content.name, pa.list_(pa.field("element", step)), nullable=False)])


def struct_steps(*, content: torch.Tensor, step: pa.StructType) -> pa.Array:
    """Write steps as structs with one member per named channel.

    Args:
        content: The forecast tensor ``[..., width]``.
        step: The struct step type with one float64 member per named channel.

    Returns:
        One struct per step.
    """
    return struct(
        values={field.name: array(values=content[..., index], dtype=pa.float64()) for index, field in enumerate(step)},
        dtype=step,
    )


def list_steps(*, content: torch.Tensor, step: pa.ListType) -> pa.Array:
    """Write steps as lists with one value per channel.

    Args:
        content: The forecast tensor ``[..., width]``.
        step: The list step type.

    Returns:
        One list of ``width`` float64 values per step.
    """
    *_, width = content.shape
    return variable(
        values=array(values=content, dtype=pa.float64()),
        counts=torch.full((content.numel() // width,), width),
        datatype=step,
    )


# The keys are the step types that ``output`` writes.
STEP_WRITERS: Mapping[type[pa.DataType], Callable[..., pa.Array]] = MappingProxyType(
    {pa.StructType: struct_steps, pa.ListType: list_steps}
)


@series.register(Component.write)
def write(module: Model, prediction: Prediction, datatype: pa.StructType) -> pa.StructArray:
    """Write ``horizon`` steps of every channel for every coordinate; the rich result's state decides validity.

    Args:
        module: The model that owns the field.
        prediction: Decoded forecast whose content is ``[..., horizon, width]``.
        datatype: The type returned by ``output``; a struct step type writes named channels.

    Returns:
        A struct array with the ``content`` forecast.
    """
    content = cast(torch.Tensor, prediction.payload[TensorKey.content])
    *coordinates, horizon, _ = content.shape
    nested = datatype.field(TensorKey.content.name).type
    step = cast(pa.StructType | pa.ListType, nested.value_type)  # the type written by ``output``
    forecast = variable(
        values=STEP_WRITERS[type(step)](content=content, step=step),
        counts=torch.full((math.prod(coordinates),), horizon),
        datatype=nested,
    )
    return struct(values={TensorKey.content.name: forecast}, dtype=datatype)
