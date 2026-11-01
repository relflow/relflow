from __future__ import annotations

import enum
import hashlib
import math
from dataclasses import InitVar
from typing import TYPE_CHECKING, Annotated, Literal, Unpack, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from pydantic import StrictInt
from tensordict import TensorDict, tensorclass

from relflow.data.ragged import RaggedField
from relflow.structs.enums import Component, Overflow, Strata, TensorKey, Tokens
from relflow.structs.tree import Address, FieldOptions
from relflow.tensorfields.base import Context, Extension, RequestBase, TensorFieldBase
from relflow.tensorfields.shared.jitter import Jitter
from relflow.tensorfields.shared.normalizer import moments
from relflow.tensorfields.shared.reconstruction import Metrics

if TYPE_CHECKING:
    from relflow.structs.experiment import Schema

series: Extension = Extension(name="series", types=(int | float,), callbacks=(Metrics,))


class Objective(enum.StrEnum):
    l1 = "l1"
    l2 = "l2"

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean error over every selected step and channel."""
        if self == Objective.l1:
            return torch.nn.functional.l1_loss(input=inputs, target=targets)

        return torch.nn.functional.mse_loss(input=inputs, target=targets)


@series.register(Component.Request)
class Request(RequestBase):
    """One multichannel time series per record.

    Each record holds ``channels`` numeric values per step, observed at
    consecutive positions; steps are ordered but not timestamped. ``length``
    caps the retained steps with an ``overflow`` policy as ``Branch`` does;
    ``None`` keeps the whole series and pads each batch to its longest.
    ``n_blocks`` sets the number of dilated convolution blocks. As an input, the
    series embeds as one pooled token. With ``mask=True`` the series is a
    forecast target and ``horizon`` gives the number of steps to predict;
    ``horizon`` is required there and rejected elsewhere. ``jitter`` adds
    training-only noise to input steps, and ``objective`` selects mean ``"l1"``
    or ``"l2"`` error over predicted steps. Predictions contain ``horizon``
    steps of ``channels`` values in source units.
    """

    type: Literal["series"] = "series"
    channels: Annotated[int, pydantic.Field(gt=0)]
    length: Annotated[StrictInt | None, pydantic.Field(gt=0)] = None
    overflow: Overflow | None = None
    horizon: Annotated[StrictInt | None, pydantic.Field(gt=0)] = None
    n_blocks: Annotated[int, pydantic.Field(gt=0)] = 6
    jitter: Jitter = pydantic.Field(default_factory=Jitter)
    objective: Objective = Objective.l2

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            channels: int,
            length: int | None = None,
            overflow: Overflow | Literal["head", "tail", "random", "error"] | None = None,
            horizon: int | None = None,
            n_blocks: int = 6,
            jitter: Jitter = ...,
            objective: Objective | Literal["l1", "l2"] = Objective.l2,
            type: Literal["series"] = "series",
            **options: Unpack[FieldOptions],
        ) -> None: ...

    @pydantic.model_validator(mode="after")
    def check_bounds(self):
        if self.length is None and self.overflow is not None:
            raise ValueError(
                f"unbounded series field cannot configure overflow={self.overflow!r}; "
                "set a finite length or remove the overflow policy"
            )
        if self.length is not None and self.overflow is None:
            self.overflow = Overflow.head

        return self


def floats(values: pa.Array, *, address: Address) -> np.ndarray:
    """Float64 elements with null as NaN; integers beyond ±2^53 are rejected."""
    try:
        values = pc.cast(values, pa.float64(), safe=True)
    except pa.ArrowInvalid as error:
        raise ValueError(
            f"series field at '{address}' holds numbers float64 cannot represent exactly "
            "(integers beyond ±2^53); scale them in a preprocessor"
        ) from error
    filled = cast(pa.Array, pc.fill_null(values, pa.scalar(math.nan, pa.float64())))
    return filled.to_numpy(zero_copy_only=False)


def listlike(datatype: pa.DataType) -> bool:
    return pa.types.is_list(datatype) or pa.types.is_large_list(datatype) or pa.types.is_fixed_size_list(datatype)


def steps(values: pa.Array, *, channels: int, address: Address) -> tuple[np.ndarray, np.ndarray]:
    """Every series' steps back to back as ``(steps, channels)`` float64, and each series' step count.

    A null step or element is NaN; a step reads as observed only when all
    its channels are finite.
    """

    if not len(values):
        return np.empty((0, channels)), np.empty(0, dtype=np.int64)
    if not listlike(values.type):
        raise ValueError(f"series field at '{address}' expects an Arrow list per record, got {values.type}")

    lengths = pc.list_value_length(values).to_numpy(zero_copy_only=False).astype(np.int64)
    flat = pc.list_flatten(values)
    if not listlike(flat.type):
        if channels != 1:
            raise ValueError(
                f"series field at '{address}' has scalar steps but expects channels={channels} values per step; "
                "nest each step as a list"
            )
        return floats(flat, address=address).reshape(-1, 1), lengths

    widths = pc.list_value_length(flat)
    wrong = pc.filter(widths, pc.not_equal(widths, channels))  # pyrefly: ignore [no-matching-overload]
    if len(wrong):
        raise ValueError(
            f"series field at '{address}' expects channels={channels} values per step, got {wrong[0].as_py()}"
        )
    matrix = np.full((len(flat), channels), math.nan)
    matrix[pc.is_valid(flat).to_numpy(zero_copy_only=False)] = floats(pc.list_flatten(flat), address=address).reshape(
        -1, channels
    )
    return matrix, lengths


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

    values = field.values.combine_chunks() if isinstance(field.values, pa.ChunkedArray) else field.values
    matrix, lengths = steps(values, channels=request.channels, address=address)
    cap = request.horizon if target else request.length
    kept = lengths if cap is None else np.minimum(lengths, cap)
    excess = lengths - kept
    start = np.zeros_like(lengths)
    if excess.any():
        policy = Overflow.error if target else request.overflow
        longest = int(lengths.max())
        match policy:
            case Overflow.error if target:
                raise ValueError(
                    f"series target at '{address}' holds {longest} steps but horizon={cap}; shorten it or raise horizon"
                )
            case Overflow.error:
                raise ValueError(
                    f"series field at '{address}' holds {longest} steps but length={cap} with overflow='error'; "
                    "raise length or choose another overflow"
                )
            case Overflow.tail:
                start = excess
            case Overflow.random if strata == Strata.train:
                seed = hashlib.sha256(f"relflow-series-overflow-v1:{salt}:{address}".encode()).digest()[:8]
                start = np.random.default_rng(int.from_bytes(seed, "big")).integers(0, excess + 1)
            case Overflow.random:
                start = excess

    width = cast(int, cap) if target else max(1, int(kept.max(initial=0)))
    record = np.repeat(np.arange(len(lengths)), kept)
    offset = np.arange(len(record)) - np.repeat(np.cumsum(kept) - kept, kept)
    picked = matrix[(np.cumsum(lengths) - lengths)[record] + start[record] + offset]
    finite = np.isfinite(picked).all(axis=1)

    content = np.zeros((len(lengths), width, request.channels))
    observed = np.zeros((len(lengths), width), dtype=np.bool_)
    content[record[finite], offset[finite]] = picked[finite]
    observed[record[finite], offset[finite]] = True

    state = field.dense.copy()
    state[field.place((kept > 0) & ~observed.any(axis=1), fill=False)] = Tokens.other.value
    encoded = TensorDict(
        {
            "values": torch.from_numpy(field.place(content, fill=0.0, value_shape=(width, request.channels))),
            "observed": torch.from_numpy(field.place(observed, fill=False, value_shape=(width,))),
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
    values = field.values.combine_chunks() if isinstance(field.values, pa.ChunkedArray) else field.values
    matrix, _ = steps(values, channels=request.channels, address=address)
    observed = torch.from_numpy(matrix[np.isfinite(matrix).all(axis=1)])
    return TensorDict({TensorKey.content: moments(observed, width=request.channels)}, batch_size=[])
