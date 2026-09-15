from __future__ import annotations

import difflib
import enum
import math
import re
from dataclasses import InitVar
from datetime import date, datetime
from typing import TYPE_CHECKING, Annotated, Any, Callable, ClassVar, Literal, Unpack, cast, overload

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from relflow.data.arrow import variants
from relflow.data.ragged import RaggedField
from relflow.helpers import Jitter
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

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch


class DatePart(enum.StrEnum):
    REGISTRY: ClassVar[dict["DatePart", Callable[..., tuple[np.ndarray, np.ndarray]]]]
    day_of_year = "day_of_year"
    week_of_year = "week_of_year"
    month_of_year = "month_of_year"
    day_of_month = "day_of_month"
    week_of_month = "week_of_month"
    day_of_week = "day_of_week"
    hour_of_day = "hour_of_day"
    minute_of_hour = "minute_of_hour"
    second_of_minute = "second_of_minute"

    def register(
        self, func: Callable[..., tuple[np.ndarray, np.ndarray]]
    ) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
        cls = type(self)

        if not hasattr(cls, "REGISTRY"):
            cls.REGISTRY = {}

        if self in cls.REGISTRY:
            raise ValueError(f"{self.name} already has a registered function.")

        cls.REGISTRY[self] = func

        return func

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        func = getattr(type(self), "REGISTRY", {}).get(self)

        if func is None:
            raise RuntimeError(f"No function registered for {self.name}")

        return func(*args, **kwargs)


def normalize_datepart_key(value: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value.strip())
    value = re.sub(r"[^0-9A-Za-z]+", "_", value)
    return value.strip("_").casefold()


def datepart_lookup() -> dict[str, DatePart]:
    lookup: dict[str, DatePart] = {}
    for datepart in DatePart:
        normalized = normalize_datepart_key(datepart.value)
        lookup[normalized] = datepart
        lookup[normalized.replace("_", "")] = datepart

    return lookup


@DatePart.day_of_month.register
def day_of_month(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_value = 31
    month_start = arr.astype("datetime64[M]")
    value = (arr - month_start).astype("timedelta64[D]").astype(int) + 1
    radians = 2 * np.pi * value / max_value
    return (np.sin(radians), np.cos(radians))


@DatePart.day_of_year.register
def day_of_year(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_value = 366
    year_start = arr.astype("datetime64[Y]")
    value = (arr - year_start).astype("timedelta64[D]").astype(int) + 1
    radians = 2 * np.pi * value / max_value
    return (np.sin(radians), np.cos(radians))


@DatePart.month_of_year.register
def month_of_year(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_value = 12
    value = (arr.astype("datetime64[M]") - arr.astype("datetime64[Y]")).astype(int) + 1
    radians = 2 * np.pi * value / max_value
    return (np.sin(radians), np.cos(radians))


@DatePart.week_of_year.register
def week_of_year(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_value = 53
    year_start = arr.astype("datetime64[Y]")
    value = ((arr.astype("datetime64[W]") - year_start.astype("datetime64[W]")).astype(int) + 1).astype(int)
    radians = 2 * np.pi * value / max_value
    return (np.sin(radians), np.cos(radians))


@DatePart.day_of_week.register
def day_of_week(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_value = 7
    value = (arr.astype("datetime64[D]").astype(int) + 4) % 7
    radians = 2 * np.pi * value / max_value
    return (np.sin(radians), np.cos(radians))


@DatePart.week_of_month.register
def week_of_month(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_value = 6
    month_start = arr.astype("datetime64[M]")
    month_start_dow = (month_start.astype("datetime64[D]").astype(int) + 4) % 7
    day_offset = (arr - month_start).astype("timedelta64[D]").astype(int)
    value = ((day_offset + month_start_dow) // 7) + 1
    radians = 2 * np.pi * value / max_value
    return (np.sin(radians), np.cos(radians))


@DatePart.hour_of_day.register
def hour_of_day(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_value = 24
    day_start = arr.astype("datetime64[D]")
    value = (arr - day_start).astype("timedelta64[h]").astype(int)
    radians = 2 * np.pi * value / max_value
    return (np.sin(radians), np.cos(radians))


@DatePart.minute_of_hour.register
def minute_of_hour(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_value = 60
    hour_start = arr.astype("datetime64[h]")
    value = (arr - hour_start).astype("timedelta64[m]").astype(int)
    radians = 2 * np.pi * value / max_value
    return (np.sin(radians), np.cos(radians))


@DatePart.second_of_minute.register
def second_of_minute(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_value = 60
    minute_start = arr.astype("datetime64[m]")
    value = (arr - minute_start).astype("timedelta64[s]").astype(int)
    radians = 2 * np.pi * value / max_value
    return (np.sin(radians), np.cos(radians))


dateparts: Extension = Extension(name="dateparts", types=(str | date | datetime,))


@overload
def parse(values: pa.Array, pattern: str | None) -> pa.Array: ...


@overload
def parse(values: pa.ChunkedArray, pattern: str | None) -> pa.ChunkedArray: ...


def parse(
    values: pa.Array | pa.ChunkedArray,
    pattern: str | None,
) -> pa.Array | pa.ChunkedArray:
    """Convert a heterogeneous date union to second-resolution timestamps."""

    datatype = pa.timestamp("s")
    if isinstance(values, pa.ChunkedArray):
        return pa.chunked_array([parse(chunk, pattern) for chunk in values.chunks], type=datatype)
    if pa.types.is_dictionary(values.type):
        values = cast(pa.DictionaryArray, values)
        return parse(pc.take(values.dictionary, values.indices), pattern)
    if isinstance(values, pa.ExtensionArray):
        return parse(values.storage, pattern)
    if not pa.types.is_union(values.type):
        if pa.types.is_string(values.type) or pa.types.is_large_string(values.type):
            return pc.strptime(values, format=pattern, unit="s") if pattern is not None else pc.cast(values, datatype)
        return pc.cast(values, datatype, safe=True)

    values = cast(pa.UnionArray, values)
    codes, offsets = variants(values)
    result = pa.nulls(len(values), type=datatype)
    for index, code in enumerate(values.type.type_codes):
        selected = codes == code
        positions = np.flatnonzero(selected).astype(np.int64, copy=False)
        if not len(positions):
            continue
        indices = offsets[selected] if offsets is not None else positions
        child = pc.take(values.field(index), pa.array(indices, type=pa.int64()))
        # Arrow exposes scatter dynamically; pyarrow-stubs omits it.
        placed = pc.scatter(  # pyrefly: ignore [missing-attribute]
            parse(child, pattern),
            pa.array(positions, type=pa.int64()),
            max_index=len(values) - 1,
        )
        result = pc.coalesce(result, placed)
    return result


@dateparts.register
class Request(RequestBase):
    """Calendar components represented as cyclical sine/cosine pairs.

    ``dateparts`` is a nonempty, unique sequence of ``DatePart`` members or
    names such as ``"day_of_week"``; names normalize to a list of enum members.
    Inputs may be dates, datetimes, or strings, with optional ``pattern``
    specifying string parsing. ``jitter`` perturbs training coordinate pairs.
    Reconstructing masks train state and per-part cosine losses. Prediction
    exports a field embedding with ``embed=True``; no datetime is decoded.
    """

    type: Literal["dateparts"] = "dateparts"
    dateparts: list[DatePart]
    pattern: Annotated[str | None, pydantic.Field(default=None)] = None
    jitter: Jitter = pydantic.Field(default_factory=Jitter)

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            dateparts: list[DatePart | str] | tuple[DatePart | str, ...],
            pattern: str | None = None,
            jitter: Jitter = ...,
            type: Literal["dateparts"] = "dateparts",
            **options: Unpack[FieldOptions],
        ) -> None: ...

    @pydantic.field_validator("dateparts", mode="before", check_fields=False)
    @classmethod
    def check_dateparts_input(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            return value

        lookup = datepart_lookup()
        canonical = [datepart.value for datepart in DatePart]
        dateparts: list[DatePart] = []
        for item in value:
            if isinstance(item, DatePart):
                dateparts.append(item)
                continue

            if not isinstance(item, str):
                raise ValueError(f"datepart values must be strings, got {type(item).__name__}")

            key = normalize_datepart_key(item)
            match = lookup.get(key) or lookup.get(key.replace("_", ""))
            if match is not None:
                dateparts.append(match)
                continue

            suggestions = difflib.get_close_matches(key, canonical, n=1)
            suggestion = f"; did you mean '{suggestions[0]}'?" if suggestions else ""
            raise ValueError(f"unknown datepart '{item}'{suggestion}")

        return dateparts

    @pydantic.field_validator("dateparts", check_fields=False)
    @classmethod
    def check_dateparts(cls, v):
        if not v:
            raise ValueError("dateparts cannot be empty")

        if not len(v) == len(set(v)):
            raise ValueError("dateparts must be unique")

        return v

    @pydantic.field_validator("pattern", check_fields=False)
    @classmethod
    def check_date_pattern(cls, v):
        if v is None:
            return v

        regex: re.Pattern = re.compile(r"^(?:%%| %(?:[aAwdbBmyYHIpMSfzZjUWcxXGuV])|[^%])+$", re.VERBOSE)

        if not bool(regex.fullmatch(v)):
            raise ValueError(f"{v} is not a valid format pattern")

        return v


@dateparts.register
@tensorclass
class TensorField(TensorFieldBase[TensorDict]):
    state: torch.Tensor
    content: TensorDict
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

        def encode(field: RaggedField) -> TensorDict:
            try:
                if not len(field.values):
                    date_values = np.empty(0, dtype="datetime64[s]")
                else:
                    if pa.types.is_union(field.values.type):
                        values = parse(field.values, request.pattern)
                    else:
                        values = (
                            field.values.combine_chunks() if isinstance(field.values, pa.ChunkedArray) else field.values
                        )
                    if request.pattern is not None and not pa.types.is_timestamp(values.type):
                        values = pc.strptime(values, format=request.pattern, unit="s")
                    if isinstance(values, pa.ChunkedArray):
                        values = values.combine_chunks()
                    date_values = values.to_numpy(zero_copy_only=False).astype("datetime64[s]")
            except (pa.ArrowException, TypeError, ValueError) as error:
                raise ValueError(f"dateparts field at '{address}' contains invalid date values") from error

            dateparts: dict[DatePart, torch.Tensor] = {}
            for datepart in request.dateparts:
                sin, cos = datepart(date_values)
                embeddings = np.stack([sin, cos], axis=-1)
                dateparts[datepart] = torch.from_numpy(field.place(embeddings, fill=0.0, value_shape=(2,))).to(
                    dtype=torch.float
                )
            # TensorDict accepts this narrower dictionary; its input type is invariant.
            return TensorDict(dateparts, batch_size=field.shape)  # pyrefly: ignore [bad-argument-type]

        state = torch.from_numpy(input.dense)

        return cls(
            content=encode(input),
            state=state,
            present=present,
            trainable=trainable,
            inferred=inferred,
            targets=TensorDict(
                {
                    TensorKey.state: torch.from_numpy(target.dense),
                    TensorKey.content: encode(target),
                },
                batch_size=input.shape,
            ),
            batch_size=input.batch_size,
        )


@dateparts.register
class Embedder(EmbedderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request = cast(Request, schema.requests[address])
        self.origin: Address = address
        self.destination: Address = cast("Branch", request.parent).address
        self.jitter: Jitter = request.jitter

        self.embeddings = torch.nn.Embedding(
            num_embeddings=len(Tokens),
            embedding_dim=schema.d_model,
        )

        self.dateparts = torch.nn.ModuleDict()

        for datepart in request.dateparts:
            self.dateparts[datepart] = torch.nn.Linear(in_features=2, out_features=schema.d_model)

    @beartype
    def forward(self, inputs: TensorInput) -> Parcel:
        N, *dims = inputs.state.shape
        D = math.prod(tuple([N, *dims]))

        state = inputs.state.reshape(D)
        valued = state.eq(Tokens.valued).unsqueeze(-1)
        embeddings: torch.Tensor = self.embeddings(state)

        for datepart in self.dateparts:
            projection = cast(torch.nn.Linear, self.dateparts[datepart])
            content = cast(torch.Tensor, cast(TensorDict, inputs.content)[datepart]).reshape(D, 2)
            if self.training:
                eligible = valued & torch.isfinite(content)
                content = self.jitter.apply(content, eligible)
            embeddings = embeddings + projection(content)

        return Parcel(
            payload=embeddings.reshape(N, *dims, -1),
            present=torch.ones(N, dtype=torch.bool, device=embeddings.device),
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )


@dateparts.register
class Decoder(DecoderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        self.linear = torch.nn.Linear(
            in_features=schema.d_model,
            out_features=len(Tokens),
        )

        self.dateparts = torch.nn.ModuleDict()

        for datepart in cast(Request, schema.requests[address]).dateparts:
            self.dateparts[datepart] = torch.nn.Linear(in_features=schema.d_model, out_features=2)

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict:
        content: dict[str, torch.Tensor] = {}
        for datepart in self.dateparts:
            content[datepart] = self.dateparts[datepart](pooled)

        return TensorDict(
            source={
                TensorKey.state: self.linear(pooled),
                # TensorDict accepts string-keyed tensor dictionaries.
                TensorKey.content: TensorDict(content, batch_size=pooled.shape[0]),  # pyrefly: ignore [bad-argument-type]
            }
        )


@dateparts.register
def loss(
    module: Model,
    prediction: Prediction,
    batch: TensorFieldBase,
    strata: Strata,
) -> torch.Tensor:
    numel: int = cast(torch.Tensor, batch.targets[TensorKey.state]).numel()

    trainable = batch.trainable.reshape(numel)

    loss: torch.Tensor = module.track(
        (prediction.address, strata, Metric.loss, TensorKey.state),
        value=(
            torch.nn.functional.cross_entropy(
                input=(inputs := cast(torch.Tensor, prediction.payload[TensorKey.state]).reshape(numel, -1)),
                target=(targets := cast(torch.Tensor, batch.targets[TensorKey.state]).reshape(numel)),
                reduction="none",
            )
            .masked_select(mask=trainable)
            .mean()
        ),
    )

    module.track(
        (prediction.address, strata, Metric.accuracy, TensorKey.state),
        value=inputs.argmax(dim=1).eq(targets).masked_select(trainable).float().mean(),
    )

    request: Request = cast(Request, module.schema.requests[prediction.address])

    losses: list[torch.Tensor] = []

    for datepart in request.dateparts:
        pred_raw = cast(torch.Tensor, cast(TensorDict, prediction.payload[TensorKey.content])[datepart]).reshape(
            numel, 2
        )
        target = cast(torch.Tensor, cast(TensorDict, batch.targets[TensorKey.content])[datepart]).reshape(numel, 2)

        pred: torch.Tensor = pred_raw / pred_raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cosine: torch.Tensor = (pred * target).sum(dim=-1)

        losses.append(
            module.track(
                (prediction.address, strata, Metric.loss, TensorKey.content, datepart),
                value=(1.0 - cosine).masked_select(trainable).mean(),
            )
        )

        module.track(
            (prediction.address, strata, Metric.mae, TensorKey.content, datepart),
            value=cosine.clamp(min=-1.0, max=1.0).arccos().masked_select(trainable).mean(),
        )

    loss += torch.stack(losses).mean()

    return loss
