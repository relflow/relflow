"""Ordered scalar values modeled as numeric ranks, including unseen integer ranks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Unpack, cast

import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from tensordict import TensorDict

from relflow.data.query import Projection
from relflow.helpers.state import compatible
from relflow.structs.enums import Component, Strata, TensorKey
from relflow.structs.packages import Prediction
from relflow.structs.tree import Address, FieldOptions
from relflow.tensorfields.base import Extension, RequestBase, TensorFieldBase
from relflow.tensorfields.extensions import enum, number
from relflow.tensorfields.output import array, struct
from relflow.tensorfields.shared.counter import Counter
from relflow.tensorfields.shared.jitter import Jitter
from relflow.tensorfields.shared.vocabulary import canonical

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.structs.experiment import Schema
    from relflow.tensorfields.shared.resize import Resize

EXACT = 2**53
ordinal = Extension("ordinal", types=(int,), callbacks=number.number.callback_factories)


@ordinal.register(Component.Request)
class Request(number.Numeric):
    """A numeric rank or one label from an explicitly ordered declaration.

    ``values=("low", "medium", "high")`` maps labels to ranks 0, 1, and 2.
    Without ``values``, integers are ranks directly; negative and unseen ranks
    remain valid. Numeric gaps retain their size: 10 to 20 is a distance of ten.
    All modeling options come from Number. Prediction rounds to the nearest
    rank with ties to even, returning a declared label or a native integer.
    Declared predictions clamp to their first and last labels; implicit integer
    predictions clamp to the exactly supported range, -2**53 through 2**53.
    """

    type: Literal["ordinal"] = "ordinal"
    values: tuple[enum.Value, ...] | None = None

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            values: tuple[enum.Value, ...] | None = None,
            jitter: Jitter = ...,
            n_bands: int = 8,
            offset: int = 4,
            alpha: float | None = None,
            objective: number.Objective | Literal["mae", "mse", "huber"] = number.Objective.mae,
            type: Literal["ordinal"] = "ordinal",
            **options: Unpack[FieldOptions],
        ) -> None: ...

    @pydantic.field_validator("values", mode="before")
    @classmethod
    def check_values(cls, values: Any) -> tuple[enum.Value, ...] | None:
        if values is None:
            return None
        try:
            return enum.Request.check_values(values)
        except (ValueError, TypeError) as error:
            raise ValueError(str(error).replace("Enum", "Ordinal")) from error

    @pydantic.field_serializer("values", when_used="json")
    def serialize_values(self, values: tuple[enum.Value, ...] | None) -> list[Any] | None:
        return (
            None
            if values is None
            else [{"bytes": value.hex()} if isinstance(value, bytes) else value for value in values]
        )

    @pydantic.model_validator(mode="after")
    def check_source(self) -> Request:
        if isinstance(self.query, tuple):
            raise ValueError("Ordinal reads one source; provide one query string rather than a tuple")
        return self


@dataclass(frozen=True)
class Order:
    """Preserve rank meanings across worker batches and model restoration."""

    values: tuple[enum.Value, ...] | None
    address: Address

    @property
    def signature(self) -> tuple[str, tuple[enum.Value, ...]] | None:
        return None if self.values is None else (type(self.values[0]).__name__, self.values)

    def batch(self, values: pa.Array | pa.ChunkedArray) -> tuple[Order, Order]:
        return self, self


class Ordered(torch.nn.Module):
    """Reject state whose learned ranks refer to a different declaration."""

    order: Order

    def get_extra_state(self) -> tuple[str, tuple[enum.Value, ...]] | None:
        return self.order.signature

    def set_extra_state(self, state: tuple[str, tuple[enum.Value, ...]] | None) -> None:
        if state != self.order.signature:
            raise ValueError(f"Ordinal checkpoint at '{self.order.address}' has a different values declaration")

    def rebuild_state(self, previous: torch.nn.Module) -> dict[str, Any]:
        if not isinstance(previous, type(self)) or previous.order.signature != self.order.signature:
            return {}
        resources = dict(previous.named_modules())
        for name, resource in self.named_modules():
            if isinstance(resource, Counter) and name in resources:
                resource.rebuild_state(resources[name])
        return compatible(self.state_dict(), previous.state_dict())


@ordinal.register(Component.Embedder)
class Embedder(Ordered, number.Embedder):
    def __init__(self, schema: Schema, address: Address) -> None:
        super().__init__(schema=schema, address=address)
        self.order = Order(cast(Request, schema.requests[address]).values, address)

    @property
    def context(self) -> Order:
        return self.order


@ordinal.register(Component.Decoder)
class Decoder(Ordered, number.Decoder):
    def __init__(self, schema: Schema, address: Address) -> None:
        super().__init__(schema=schema, address=address)
        self.order = Order(cast(Request, schema.requests[address]).values, address)


@ordinal.register(Component.bind)
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
    order = cast(Embedder, module.nodes[address].embedder).order
    if not isinstance(binding, Order) or binding.address != address or binding.signature != order.signature:
        raise ValueError(f"Ordinal batch at '{address}' has a different values declaration; encode it again")


@ordinal.register(Component.derive)
def derive(inputs: tuple[Projection, ...], *, request: RequestBase, address: Address, rows: int) -> Projection:
    """Map labels to ranks before numeric observation, masking, or tensorization."""
    request = cast(Request, request)
    if len(inputs) != 1:
        raise ValueError(f"Ordinal field at '{address}' requires exactly one source query, got {len(inputs)}")
    selected = inputs[0]
    values = selected.values
    if pa.types.is_dictionary(values.type):
        values = pc.dictionary_decode(values)  # pyrefly: ignore [missing-attribute]
    values = pc.if_else(selected.present, values, None)
    if pa.types.is_null(values.type):
        return Projection(values=pc.cast(values, pa.int64()), present=selected.present)
    if request.values is None:
        if not pa.types.is_integer(values.type):
            raise ValueError(
                f"Ordinal field at '{address}' expects integer ranks, got {values.type}; "
                "provide ordered values for labels or use Number for continuous inputs"
            )
        low = (
            pc.less(values, pa.scalar(-EXACT, type=pa.int64()))
            if pa.types.is_signed_integer(values.type)
            else pa.repeat(pa.scalar(False), len(values))
        )
        upper = pa.scalar(EXACT, type=pa.uint64() if pa.types.is_unsigned_integer(values.type) else pa.int64())
        if pc.any(pc.or_(low, pc.greater(values, upper))).as_py():
            raise ValueError(
                f"Ordinal field at '{address}' has ranks outside [-2**53, 2**53]; "
                "rescale ranks or declare an ordered values tuple"
            )
        ranks = pc.cast(values, pa.int64(), safe=True)
    else:
        kind = type(request.values[0])
        if not enum.enum.matches(kind, values.type):
            raise ValueError(f"Ordinal field at '{address}' expects declared {kind.__name__} labels, got {values.type}")
        labels = canonical(pa.array(cast(tuple[Any, ...], request.values)))
        try:
            values = canonical(pc.cast(values, labels.type, safe=True))
            ranks = pc.index_in(values, value_set=labels)
        except pa.ArrowException as error:
            raise ValueError(
                f"Ordinal field at '{address}' contains labels outside its declared scalar type"
            ) from error
        unknown = pc.and_(pc.is_valid(values), pc.is_null(ranks))
        if pc.any(unknown).as_py():
            value = pc.filter(values, unknown)[0].as_py()
            raise ValueError(
                f"Ordinal field at '{address}' contains undeclared value {value!r}; "
                "declare it in values or normalize the source"
            )
        ranks = pc.cast(ranks, pa.int64())
    return Projection(values=ranks, present=selected.present)


@ordinal.register(Component.output)
def output(module: Model, address: Address) -> pa.StructType:
    request = cast(Request, module.schema.requests[address])
    datatype = pa.int64() if request.values is None else pa.array(cast(tuple[Any, ...], request.values)).type
    return pa.struct([pa.field(TensorKey.content.name, datatype, nullable=False)])


@ordinal.register(Component.write)
def write(module: Model, prediction: Prediction, datatype: pa.StructType) -> pa.StructArray:
    request = cast(Request, module.schema.requests[prediction.address])
    ranks = array(cast(torch.Tensor, prediction.payload[TensorKey.content]), pa.float64())
    if pc.any(pc.invert(pc.is_finite(ranks))).as_py():
        raise ValueError(
            f"Ordinal prediction at '{prediction.address}' contains nonfinite ranks; "
            "inspect model weights and numeric training stability"
        )
    lower, upper = (-EXACT, EXACT) if request.values is None else (0, len(request.values) - 1)
    bounded = pc.max_element_wise(
        pc.min_element_wise(ranks, pa.scalar(upper, type=pa.float64())), pa.scalar(lower, type=pa.float64())
    )
    indices = cast(
        pa.Int64Array, pc.cast(pc.round(bounded, ndigits=0, round_mode="half_to_even"), pa.int64(), safe=True)
    )
    content = indices if request.values is None else pc.take(pa.array(cast(tuple[Any, ...], request.values)), indices)
    return struct({TensorKey.content.name: content}, datatype)


for role, component in number.number.components.items():
    if role not in {Component.Request, Component.Embedder, Component.Decoder, Component.output, Component.write}:
        ordinal.register(role, component)
