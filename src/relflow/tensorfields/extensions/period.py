from __future__ import annotations

import difflib
import enum
import math
import re
from dataclasses import InitVar
from datetime import date, datetime
from functools import partial
from typing import TYPE_CHECKING, Annotated, Any, Callable, ClassVar, Literal, Self, Unpack, cast, overload

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from relflow.data.arrow import variants
from relflow.data.ragged import RaggedField
from relflow.structs.enums import Component, Metric, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address, FieldOptions
from relflow.tensorfields.base import (
    OMITTED,
    Context,
    DecoderBase,
    EmbedderBase,
    Extension,
    RequestBase,
    TensorFieldBase,
    TensorInput,
    masks,
    resolve,
)
from relflow.tensorfields.shared.jitter import Jitter
from relflow.tensorfields.shared.reconstruction import Metrics, Scores, Totals

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.data.ragged import Sources
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch


class Part(enum.StrEnum):
    REGISTRY: ClassVar[dict["Part", Callable[..., tuple[np.ndarray, np.ndarray]]]]
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


def normalize_part_key(value: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value.strip())
    value = re.sub(r"[^0-9A-Za-z]+", "_", value)
    return value.strip("_").casefold()


def part_lookup() -> dict[str, Part]:
    lookup: dict[str, Part] = {}
    for part in Part:
        normalized = normalize_part_key(part.value)
        lookup[normalized] = part
        lookup[normalized.replace("_", "")] = part

    return lookup


@Part.day_of_month.register
def day_of_month(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_value = 31
    month_start = arr.astype("datetime64[M]")
    value = (arr - month_start).astype("timedelta64[D]").astype(int) + 1
    radians = 2 * np.pi * value / max_value
    return (np.sin(radians), np.cos(radians))


@Part.day_of_year.register
def day_of_year(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_value = 366
    year_start = arr.astype("datetime64[Y]")
    value = (arr - year_start).astype("timedelta64[D]").astype(int) + 1
    radians = 2 * np.pi * value / max_value
    return (np.sin(radians), np.cos(radians))


@Part.month_of_year.register
def month_of_year(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_value = 12
    value = (arr.astype("datetime64[M]") - arr.astype("datetime64[Y]")).astype(int) + 1
    radians = 2 * np.pi * value / max_value
    return (np.sin(radians), np.cos(radians))


@Part.week_of_year.register
def week_of_year(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Sunday-start weeks, as in day_of_week and week_of_month. The week holding
    # January 1 is week 1, so a leap year starting on Saturday reaches week 54.
    max_value = 54
    year_start = arr.astype("datetime64[Y]")
    year_start_dow = (year_start.astype("datetime64[D]").astype(int) + 4) % 7
    day_offset = (arr - year_start).astype("timedelta64[D]").astype(int)
    value = ((day_offset + year_start_dow) // 7) + 1
    radians = 2 * np.pi * value / max_value
    return (np.sin(radians), np.cos(radians))


@Part.day_of_week.register
def day_of_week(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_value = 7
    value = (arr.astype("datetime64[D]").astype(int) + 4) % 7
    radians = 2 * np.pi * value / max_value
    return (np.sin(radians), np.cos(radians))


@Part.week_of_month.register
def week_of_month(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_value = 6
    month_start = arr.astype("datetime64[M]")
    month_start_dow = (month_start.astype("datetime64[D]").astype(int) + 4) % 7
    day_offset = (arr - month_start).astype("timedelta64[D]").astype(int)
    value = ((day_offset + month_start_dow) // 7) + 1
    radians = 2 * np.pi * value / max_value
    return (np.sin(radians), np.cos(radians))


@Part.hour_of_day.register
def hour_of_day(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_value = 24
    day_start = arr.astype("datetime64[D]")
    value = (arr - day_start).astype("timedelta64[h]").astype(int)
    radians = 2 * np.pi * value / max_value
    return (np.sin(radians), np.cos(radians))


@Part.minute_of_hour.register
def minute_of_hour(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_value = 60
    hour_start = arr.astype("datetime64[h]")
    value = (arr - hour_start).astype("timedelta64[m]").astype(int)
    radians = 2 * np.pi * value / max_value
    return (np.sin(radians), np.cos(radians))


@Part.second_of_minute.register
def second_of_minute(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_value = 60
    minute_start = arr.astype("datetime64[m]")
    value = (arr - minute_start).astype("timedelta64[s]").astype(int)
    radians = 2 * np.pi * value / max_value
    return (np.sin(radians), np.cos(radians))


period: Extension = Extension(name="period", types=(str | date | datetime,), callbacks=(Metrics,))


class Arcs(Totals):
    """Mean angular error per calendar part over valued targets."""

    def __init__(self, parts: tuple[str, ...]):
        super().__init__(counts=1, sums=len(parts))
        self.parts = parts

    def compute(self) -> dict[str, torch.Tensor]:
        valued = self.counts[0]
        return {
            f"{Metric.mae.value}.{TensorKey.content.value}.{part}": total / valued
            for part, total in zip(self.parts, self.sums.unbind(), strict=True)
        }


@overload
def parse(values: pa.Array, pattern: str | None) -> pa.Array: ...


@overload
def parse(values: pa.ChunkedArray, pattern: str | None) -> pa.ChunkedArray: ...


def wall_clock(values: pa.Array) -> pa.Array:
    """Zone-aware timestamps read in their own zone: calendar parts describe local time, not UTC.

    Strings that carry an offset are normalized to UTC by Arrow before this
    point; give such columns a zone instead.
    """
    if pa.types.is_timestamp(values.type) and cast(pa.TimestampType, values.type).tz is not None:
        # Arrow exposes this kernel dynamically; pyarrow-stubs omits it.
        return pc.local_timestamp(values)  # pyrefly: ignore [no-matching-overload]
    return values


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
        return pc.cast(wall_clock(values), datatype, safe=True)

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
        gather = np.zeros(len(values), dtype=np.int64)
        gather[positions] = np.arange(len(positions))
        placed = pc.take(parse(child, pattern), pa.array(gather, type=pa.int64(), mask=~selected))
        result = pc.coalesce(result, placed)
    return result


@period.register(Component.Request)
class Request(RequestBase):
    """Calendar components represented as cyclical sine/cosine pairs.

    ``Period("day_of_week", "hour_of_day")`` selects a nonempty, unique
    sequence of ``Part`` members or names. Names normalize to a list of
    enum members. The ``parts=`` keyword accepts the same selection.
    Inputs may be dates, datetimes, or strings, with optional ``pattern``
    specifying string parsing. Zone-aware timestamps describe their own wall
    clock; strings carrying an offset are read as UTC. ``jitter`` perturbs
    training coordinate pairs.
    Reconstructing masks train state and per-part cosine losses. Prediction
    exports a field embedding with ``embed=True``; no datetime is decoded.
    """

    type: Literal["period"] = "period"
    parts: list[Part]
    pattern: Annotated[str | None, pydantic.Field(default=None)] = None
    jitter: Jitter = pydantic.Field(default_factory=Jitter)

    if TYPE_CHECKING:

        def __init__(
            self,
            *selected: Part | str,
            parts: list[Part | str] | tuple[Part | str, ...] = ...,
            pattern: str | None = None,
            jitter: Jitter = ...,
            type: Literal["period"] = "period",
            **options: Unpack[FieldOptions],
        ) -> None: ...
    else:

        def __init__(
            self,
            *selected: Part | str,
            parts: list[Part | str] | tuple[Part | str, ...] = cast(list[Part | str] | tuple[Part | str, ...], OMITTED),
            **options: Any,
        ) -> None:
            if selected and parts is not OMITTED:
                raise TypeError("Period selects calendar parts positionally or with parts=; use one form")
            super().__init__(parts=selected if parts is OMITTED else parts, **options)

    @pydantic.field_validator("parts", mode="before", check_fields=False)
    @classmethod
    def check_parts_input(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            return value

        lookup = part_lookup()
        canonical = [part.value for part in Part]
        parts: list[Part] = []
        for item in value:
            if isinstance(item, Part):
                parts.append(item)
                continue

            if not isinstance(item, str):
                raise ValueError(f"Period part values must be strings, got {type(item).__name__}")

            key = normalize_part_key(item)
            match = lookup.get(key) or lookup.get(key.replace("_", ""))
            if match is not None:
                parts.append(match)
                continue

            suggestions = difflib.get_close_matches(key, canonical, n=1)
            suggestion = f"; did you mean '{suggestions[0]}'?" if suggestions else ""
            raise ValueError(f"unknown Period part '{item}'{suggestion}")

        return parts

    @pydantic.field_validator("parts", check_fields=False)
    @classmethod
    def check_parts(cls, v):
        if not v:
            raise ValueError("Period parts cannot be empty")

        if not len(v) == len(set(v)):
            raise ValueError("Period parts must be unique")

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


@period.register(Component.TensorField)
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
        sources: Sources,
        *,
        schema: Schema,
        context: Context,
    ) -> tuple[Self, TensorDict | None, tuple[object, ...]]:
        address = sources.address
        projection, context, observation, binding = resolve(sources, schema=schema, context=context)
        input, target = projection.split()
        present, trainable, inferred = masks(projection)
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
                    date_values = wall_clock(values).to_numpy(zero_copy_only=False).astype("datetime64[s]")
            except (pa.ArrowException, TypeError, ValueError) as error:
                raise ValueError(f"Period field at '{address}' contains invalid date values") from error

            encoded: dict[Part, torch.Tensor] = {}
            for part in request.parts:
                sin, cos = part(date_values)
                embeddings = np.stack([sin, cos], axis=-1)
                encoded[part] = torch.from_numpy(field.place(embeddings, fill=0.0, value_shape=(2,))).to(
                    dtype=torch.float
                )
            # TensorDict accepts this narrower dictionary; its input type is invariant.
            return TensorDict(encoded, batch_size=field.shape)  # pyrefly: ignore [bad-argument-type]

        state = torch.from_numpy(input.dense)

        result = cls(
            content=encode(input),
            state=state,
            present=present,
            trainable=trainable,
            inferred=inferred,
            targets=TensorDict(
                {
                    TensorKey.state: torch.from_numpy(target.dense),
                    TensorKey.content: encode(target),
                }
                if sources.needs_targets(schema)
                else {},
                batch_size=input.shape,
            ),
            batch_size=input.batch_size,
        )
        return result, observation, binding


@period.register(Component.Embedder)
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

        self.parts = torch.nn.ModuleDict()

        for part in request.parts:
            self.parts[part] = torch.nn.Linear(in_features=2, out_features=schema.d_model)

    @beartype
    def forward(self, inputs: TensorInput) -> Parcel:
        N, *dims = inputs.state.shape
        D = math.prod(tuple([N, *dims]))

        state = inputs.state.reshape(D)
        valued = state.eq(Tokens.valued).unsqueeze(-1)
        embeddings: torch.Tensor = self.embeddings(state)

        for part in self.parts:
            projection = cast(torch.nn.Linear, self.parts[part])
            content = cast(torch.Tensor, cast(TensorDict, inputs.content)[part]).reshape(D, 2)
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


@period.register(Component.Decoder)
class Decoder(DecoderBase):
    coordinatewise = True

    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        self.linear = torch.nn.Linear(
            in_features=schema.d_model,
            out_features=len(Tokens),
        )

        self.parts = torch.nn.ModuleDict()

        request = cast(Request, schema.requests[address])
        for part in request.parts:
            self.parts[part] = torch.nn.Linear(in_features=schema.d_model, out_features=2)
        self.metrics = Scores(address, partial(Arcs, tuple(part.value for part in request.parts)))

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict:
        content: dict[str, torch.Tensor] = {}
        for part in self.parts:
            content[part] = self.parts[part](pooled)

        return TensorDict(
            source={
                TensorKey.state: self.linear(pooled),
                # TensorDict accepts string-keyed tensor dictionaries.
                TensorKey.content: TensorDict(content, batch_size=pooled.shape[0]),  # pyrefly: ignore [bad-argument-type]
            }
        )


@period.register(Component.loss)
def loss(
    module: Model,
    prediction: Prediction,
    batch: TensorFieldBase,
    strata: Strata,
) -> torch.Tensor:
    numel: int = cast(torch.Tensor, batch.targets[TensorKey.state]).numel()

    trainable = batch.trainable.reshape(numel)

    inputs = cast(torch.Tensor, prediction.payload[TensorKey.state]).reshape(numel, -1)
    targets = cast(torch.Tensor, batch.targets[TensorKey.state]).reshape(numel)
    indices = trainable.nonzero(as_tuple=True)[0]
    logits = inputs.index_select(0, indices)
    states = targets.index_select(0, indices)
    loss = module.track(
        (prediction.address, strata, Metric.loss, TensorKey.state),
        value=torch.nn.functional.cross_entropy(logits, states),
    )
    module.track(
        (prediction.address, strata, Metric.accuracy, TensorKey.state),
        value=logits.argmax(dim=1).eq(states).float().mean(),
    )

    request: Request = cast(Request, module.schema.requests[prediction.address])
    decoder: Decoder = cast(Decoder, module.nodes[prediction.address].decoder)

    # Only valued targets lie on the unit circle; null and padded coordinates
    # train the state head alone.
    valued = trainable & targets.eq(Tokens.valued.value)
    losses: list[torch.Tensor] = []
    angles: list[torch.Tensor] = []

    for part in request.parts:
        pred_raw = cast(torch.Tensor, cast(TensorDict, prediction.payload[TensorKey.content])[part]).reshape(numel, 2)[
            valued
        ]
        target = cast(torch.Tensor, cast(TensorDict, batch.targets[TensorKey.content])[part]).reshape(numel, 2)[valued]

        pred: torch.Tensor = pred_raw / pred_raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cosine: torch.Tensor = (pred * target).sum(dim=-1)

        if cosine.numel():
            losses.append(
                module.track(
                    (prediction.address, strata, Metric.loss, TensorKey.content, part),
                    value=(1.0 - cosine).mean(),
                )
            )
        angles.append(cosine.detach().clamp(min=-1.0, max=1.0).arccos().sum())

    decoder.metrics.record(module, strata, valued.sum().unsqueeze(0), torch.stack(angles))
    if losses:
        loss += torch.stack(losses).mean()

    return loss
