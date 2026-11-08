from __future__ import annotations

import enum
import hashlib
import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Literal, Self, Unpack, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from beartype import beartype
from pydantic import StrictInt, StrictStr
from tensordict import TensorDict, tensorclass

from relflow.data.ragged import RaggedField
from relflow.structs.enums import Component, Metric, Overflow, Strata, TensorKey, Tokens
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
from relflow.tensorfields.shared.jitter import Jitter
from relflow.tensorfields.shared.normalizer import GlobalOnlineNormalizer, moments, precise
from relflow.tensorfields.shared.reconstruction import Errors, Metrics, Scores, errors

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch


def numeric(datatype: pa.DataType) -> bool:
    """Whether an Arrow type is a number or a list of numbers.

    Args:
        datatype: The Arrow type to test.

    Returns:
        True for an integer or floating type, or one list level of them.
    """
    # Every list layout exposes value_type; pyarrow types it only on ListType.
    datatype = cast(pa.ListType, datatype).value_type if listlike(datatype=datatype) else datatype
    return pa.types.is_integer(datatype) or pa.types.is_floating(datatype)


def channelled(datatype: pa.DataType) -> bool:
    """Whether an Arrow type is a struct of channels.

    Args:
        datatype: The Arrow type to test.

    Returns:
        True for a non-empty struct whose members are all numbers (one step)
        or all lists of numbers (one list per channel).
    """
    if not pa.types.is_struct(datatype) or not datatype.num_fields:
        return False
    kinds = [field.type for field in datatype]
    return all(numeric(datatype=kind) and not listlike(datatype=kind) for kind in kinds) or all(
        listlike(datatype=kind) and numeric(datatype=kind) for kind in kinds
    )


series: Extension = Extension(name="series", types=(int | float, dict), arrow={dict: channelled}, callbacks=(Metrics,))

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


@dataclass(frozen=True, kw_only=True)
class Channels:
    """The channels of one step.

    Attributes:
        width: Number of channels per step.
        names: Struct member names that are the channels, in channel order; ``None`` for a plain count.
    """

    width: int
    names: tuple[str, ...] | None


@series.register(Component.Request)
class Request(RequestBase):
    """One multichannel time series per record.

    Each record holds one numeric value per channel and step, observed at
    consecutive positions; steps are ordered but not timestamped. ``channels``
    is either a positive count or a tuple of distinct struct member names; a
    tuple sets the count and the channel order and labels forecast steps. ``length``
    caps the retained steps with an ``overflow`` policy as ``Branch`` does;
    ``None`` keeps the whole series and pads each batch to its longest.
    ``n_blocks`` sets the number of dilated convolution blocks. As an input, the
    series embeds as one pooled token. As a reconstruction target (``mask=True``
    or a reconstructing ancestor) the series is a forecast target and
    ``horizon`` gives the number of steps to predict; ``horizon`` is required
    there and rejected elsewhere. ``jitter`` adds
    training-only noise to input steps, and ``objective`` selects mean ``"l1"``
    or ``"l2"`` error over predicted steps. Predictions contain ``horizon``
    steps of one value per channel in source units.
    """

    type: Literal["series"] = "series"
    channels: Annotated[StrictInt, pydantic.Field(gt=0)] | tuple[StrictStr, ...]
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
            channels: int | tuple[str, ...],
            length: int | None = None,
            overflow: Overflow | Literal["head", "tail", "random", "error"] | None = None,
            horizon: int | None = None,
            n_blocks: int = 6,
            jitter: Jitter = ...,
            objective: Objective | Literal["l1", "l2"] = Objective.l2,
            type: Literal["series"] = "series",
            **options: Unpack[FieldOptions],
        ) -> None: ...

    @pydantic.field_validator("channels")
    @classmethod
    def check_channels(cls, channels: int | tuple[str, ...]) -> int | tuple[str, ...]:
        """Reject an empty, blank or duplicated channel name list.

        Args:
            channels: A channel count or the channel names.

        Returns:
            The unchanged channels.

        Raises:
            ValueError: If names are empty, blank or repeated.
        """
        if isinstance(channels, int):
            return channels
        repeated = sorted({name for name in channels if channels.count(name) > 1})
        if not channels or not all(name.strip() for name in channels) or repeated:
            message = f"series channels must be a non-empty tuple of distinct, non-blank names, got {channels!r}" + (
                f" with repeated {repeated}" if repeated else ""
            )
            raise ValueError(message)
        return channels

    @property
    def layout(self) -> Channels:
        """The channels of one step.

        Returns:
            The channel count, with the member names when ``channels`` is a tuple of names.
        """
        if isinstance(self.channels, int):
            return Channels(width=self.channels, names=None)
        return Channels(width=len(self.channels), names=self.channels)

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


def members(datatype: pa.StructType, *, channels: Channels, address: Address) -> tuple[str, ...]:
    """Struct member names to read, in channel order.

    Args:
        datatype: The Arrow struct type of a step or of a record.
        channels: The channel count, or the member names to select.
        address: Field address, for error messages.

    Returns:
        The member names, in the struct order for a count and in tuple order for names.

    Raises:
        ValueError: If a count differs from the member count or a name is not a member.
    """
    names = tuple(field.name for field in datatype)
    if channels.names is None and len(names) != channels.width:
        message = (
            f"series field at '{address}' has channels={channels.width} but its struct has {len(names)} members "
            f"{names}; name the members in channels or select them in a preprocessor"
        )
        raise ValueError(message)
    if channels.names is None:
        return names
    missing = [name for name in channels.names if name not in names]
    if missing:
        message = (
            f"series field at '{address}' names channels {missing} absent from its struct members {names}; "
            "check the names in channels or rename the members in a preprocessor"
        )
        raise ValueError(message)
    return channels.names


def children(datatype: pa.StructType, *, flat: list[pa.Array]) -> dict[str, pa.Array]:
    """Map member names of a struct to their arrays.

    Args:
        datatype: The struct type whose member names label the arrays.
        flat: The member arrays in struct order, from ``StructArray.flatten()``.

    Returns:
        A dict from member name to member array.
    """
    return dict(zip((field.name for field in datatype), flat, strict=True))


def rows(values: pa.Array, *, channels: Channels, address: Address) -> tuple[np.ndarray, np.ndarray]:
    """Steps of lists of structs, one struct per step.

    A null step struct or a null member makes the whole step NaN.

    Args:
        values: One Arrow list of step structs per record.
        channels: The channel count, or the member names to select.
        address: Field address, for error messages.

    Returns:
        The step matrix and the number of steps of each record.

    Raises:
        ValueError: If the members do not match ``channels``.
    """
    flat = cast(pa.StructArray, pc.list_flatten(values))
    names = members(flat.type, channels=channels, address=address)
    by_name = children(flat.type, flat=flat.flatten())
    matrix = np.column_stack([floats(values=by_name[name], address=address) for name in names])
    return matrix, pc.list_value_length(values).to_numpy(zero_copy_only=False).astype(np.int64)


def columns(values: pa.StructArray, *, channels: Channels, address: Address) -> tuple[np.ndarray, np.ndarray]:
    """Steps of structs with one list per channel.

    A null channel list is a missing channel: it warns, and the steps of its
    record are NaN. A record whose channel lists are all null has no steps.
    The warning text is the same for every batch, so Python shows it once per
    field and member.

    Args:
        values: One Arrow struct of channel lists per record.
        channels: The channel count, or the member names to select.
        address: Field address, for error messages.

    Returns:
        The step matrix and the number of steps of each record.

    Raises:
        ValueError: If the channel lists of a record differ in length or the members do not match ``channels``.
    """
    names = members(values.type, channels=channels, address=address)
    by_name = children(values.type, flat=values.flatten())
    lists = [by_name[name] for name in names]
    counts = np.stack(
        [floats(values=pc.list_value_length(member), address=address) for member in lists]
    )  # NaN marks a null list
    missing = np.isnan(counts)
    for name, absent in zip(names, missing, strict=True):
        if absent.any():
            message = (
                f"series field at '{address}' has a null channel list '{name}' in some records; "
                "its steps are unobserved. Supply a list of nulls with the record step count, "
                "or drop or fill the record in a preprocessor"
            )
            warnings.warn(message, UserWarning, stacklevel=2)
    longest = np.where(missing, -1, counts).max(axis=0)
    shortest = np.where(missing, np.inf, counts).min(axis=0)
    differing = np.flatnonzero(~missing.all(axis=0) & (longest != shortest))
    if len(differing):
        message = (
            f"series field at '{address}' has channel lists {names} with different step count "
            f"in record {int(differing[0])}: {counts[:, differing[0]].tolist()}; "
            "pad the shorter lists with nulls or trim them in a preprocessor"
        )
        raise ValueError(message)
    lengths = np.where(missing.all(axis=0), 0, longest).astype(np.int64)
    matrix = np.full((int(lengths.sum()), len(names)), math.nan)
    for index, (member, absent) in enumerate(zip(lists, missing, strict=True)):
        matrix[np.repeat(~absent, lengths), index] = floats(values=pc.list_flatten(member), address=address)
    return matrix, lengths


def steps(values: pa.Array, *, channels: Channels, address: Address) -> tuple[np.ndarray, np.ndarray]:
    """Every step of every series back to back as ``(steps, channels)`` float64, with the step count per series.

    A null step or element is NaN; a step reads as observed only when all
    its channels are finite.

    Args:
        values: One Arrow list of steps per record, a list of step structs, or a struct of channel lists.
        channels: Values expected in each step, or the struct member names that are the channels.
        address: Field address, for error messages.

    Returns:
        The step matrix and the number of steps of each record.

    Raises:
        ValueError: If a record is not a list or struct, named channels meet a list input,
            or a step does not hold ``channels`` values.
    """
    width = channels.width
    if not len(values):
        return np.empty((0, width)), np.empty(0, dtype=np.int64)
    if pa.types.is_struct(values.type):
        return columns(cast(pa.StructArray, values), channels=channels, address=address)
    if not listlike(values.type):
        message = f"series field at '{address}' expects an Arrow list or struct per record, got {values.type}"
        raise ValueError(message)
    if pa.types.is_struct(cast(pa.ListType, values.type).value_type):
        return rows(values, channels=channels, address=address)
    if channels.names is not None:
        message = (
            f"series field at '{address}' names channels {channels.names} but its input has no struct members; "
            "provide a list of structs or a struct of lists, or set channels to a count"
        )
        raise ValueError(message)

    lengths = pc.list_value_length(values).to_numpy(zero_copy_only=False).astype(np.int64)
    flat = pc.list_flatten(values)
    if not listlike(flat.type):
        if width != 1:
            message = (
                f"series field at '{address}' has scalar steps but expects channels={width} values per step; "
                "nest each step as a list"
            )
            raise ValueError(message)
        return floats(flat, address=address).reshape(-1, 1), lengths

    widths = pc.list_value_length(flat)
    wrong = pc.filter(widths, pc.not_equal(widths, pa.scalar(width, widths.type)))
    if len(wrong):
        message = f"series field at '{address}' expects channels={width} values per step, got {wrong[0].as_py()}"
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

    matrix, lengths = steps(values=combine(field=field), channels=request.layout, address=address)
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

    content = np.zeros((len(lengths), width, request.layout.width))
    observed = np.zeros((len(lengths), width), dtype=np.bool_)
    content[record[finite], offset[finite]] = picked[finite]
    observed[record[finite], offset[finite]] = True

    state = field.dense.copy()
    state[field.place((kept > 0) & ~observed.any(axis=1), fill=False)] = Tokens.other.value
    encoded = TensorDict(
        {
            VALUES: torch.from_numpy(field.place(content, fill=0.0, value_shape=(width, request.layout.width))),
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
    matrix, _ = steps(values=combine(field=field), channels=request.layout, address=address)
    observed = torch.from_numpy(matrix[np.isfinite(matrix).all(axis=1)])
    return TensorDict({TensorKey.content: moments(observed, width=request.layout.width)}, batch_size=[])


class Block(torch.nn.Module):
    """Pre-norm residual block of two dilated convolutions over observed steps.

    Unobserved steps are zero on entry to every convolution, so a series
    embeds the same whatever the batch pads it with.
    """

    def __init__(self, *, width: int, dilation: int):
        super().__init__()
        self.norm = torch.nn.RMSNorm(width)
        self.first = torch.nn.Conv1d(width, width, kernel_size=3, padding=dilation, dilation=dilation)
        self.second = torch.nn.Conv1d(width, width, kernel_size=3, padding=dilation, dilation=dilation)
        self.activation = torch.nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor, *, observed: torch.Tensor) -> torch.Tensor:
        """Apply the block to ``x`` of shape ``(batch, steps, width)`` with its ``observed`` step mask."""
        gone = ~observed.unsqueeze(-1)
        hidden = self.activation(self.norm(x)).transpose(1, 2)
        hidden = self.activation(self.first(hidden).transpose(1, 2)).masked_fill(gone, 0.0)
        hidden = self.second(hidden.transpose(1, 2)).transpose(1, 2)
        return (x + hidden).masked_fill(gone, 0.0)


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

        self.normalizer = GlobalOnlineNormalizer(width=request.layout.width)
        self.projection = torch.nn.Linear(in_features=request.layout.width, out_features=schema.d_model)
        self.blocks = torch.nn.ModuleList(
            [Block(width=schema.d_model, dilation=2**depth) for depth in range(request.n_blocks)]
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
        self.width: int = request.layout.width
        self.regression = torch.nn.Linear(in_features=schema.d_model, out_features=horizon * request.layout.width)
        # Target statistics per channel: the forecast starts at their mean and predicts standardized deviations.
        self.normalizer: GlobalOnlineNormalizer = GlobalOnlineNormalizer(width=request.layout.width)
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
    horizon, width = decoder.horizon, request.layout.width

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


@series.register(Component.output)
def output(module: Model, address: Address) -> pa.StructType:
    """Arrow type of a prediction: ``horizon`` steps of one float64 value per channel per record.

    A channel count writes each step as a list of values; channel names write
    each step as a struct with one member per name, in tuple order.

    Args:
        module: The model that owns the field.
        address: Field address.

    Returns:
        A struct with a non-null ``content`` list of steps.
    """
    names = cast(Request, module.schema.requests[address]).layout.names
    step = pa.list_(pa.field("element", pa.float64()))
    if names is not None:
        step = pa.struct([pa.field(name, pa.float64(), nullable=False) for name in names])
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
