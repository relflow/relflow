"""Elapsed-time inputs reuse Number's encodings, statistics, objectives, and output."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Literal, Self, Unpack, cast

import pyarrow as pa
import pyarrow.compute as pc
import pydantic

from relflow.data.query import Projection
from relflow.structs.enums import Component
from relflow.structs.tree import Address, FieldOptions
from relflow.tensorfields.base import Extension, RequestBase
from relflow.tensorfields.extensions import number
from relflow.tensorfields.shared.jitter import Jitter
from relflow.tensorfields.shared.temporal import difference

duration = Extension("duration", types=(int | float,), callbacks=number.number.callback_factories)


@duration.register(Component.Request)
class Request(number.Numeric):
    """A signed elapsed time, ``end - start``, measured in fixed units.

    ``query=(start_path, end_path)`` selects both endpoints. A single query
    pairs its source with one configured ``start`` or ``end`` timestamp. With
    a configured endpoint and no query, the field reads its same-named member.
    Two configured timestamps produce a constant duration. Ancestor references
    begin with ``../`` and broadcast within explicit Branch geometry.

    ``pattern`` uses the same format syntax as Period. Zone-aware endpoints
    describe instants, and mixed aware/naive endpoints are rejected. Inputs
    never read the wall clock. Months and years have no fixed elapsed length
    and are not supported. All numeric modeling options come from Number.
    """

    type: Literal["duration"] = "duration"
    start: datetime | date | str | None = None
    end: datetime | date | str | None = None
    granularity: Literal["seconds", "minutes", "hours", "days", "weeks"] = "days"
    pattern: str | None = None

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            start: datetime | date | str | None = None,
            end: datetime | date | str | None = None,
            granularity: Literal["seconds", "minutes", "hours", "days", "weeks"] = "days",
            pattern: str | None = None,
            jitter: Jitter = ...,
            n_bands: int = 8,
            offset: int = 4,
            alpha: float | None = None,
            objective: number.Objective | Literal["mae", "mse", "huber"] = number.Objective.mae,
            type: Literal["duration"] = "duration",
            **options: Unpack[FieldOptions],
        ) -> None: ...

    @pydantic.field_validator("start", "end", mode="before")
    @classmethod
    def check_endpoint(cls, value: object) -> object:
        if value is None or isinstance(value, (date, datetime)):
            return value
        if isinstance(value, dict):
            if set(value) != {"type", "value"} or value["type"] not in ("date", "datetime"):
                raise ValueError("Duration endpoint tags must contain exactly type ('date' or 'datetime') and value")
            if not isinstance(value["value"], str):
                raise ValueError("Duration endpoint tag value must be an ISO timestamp string")
            try:
                return (
                    datetime.fromisoformat(value["value"])
                    if value["type"] == "datetime"
                    else date.fromisoformat(value["value"])
                )
            except ValueError as error:
                raise ValueError(f"Duration endpoint tag requires a valid ISO {value['type']} value") from error
        if isinstance(value, str) and value.strip():
            return value
        raise ValueError("Duration endpoints must be date, datetime, or nonempty timestamp strings")

    @pydantic.field_serializer("start", "end", when_used="json")
    def serialize_endpoint(self, value: datetime | date | str | None) -> dict[str, str] | str | None:
        """Preserve native types independently of source string formats and resolution."""
        if isinstance(value, datetime):
            return {"type": "datetime", "value": value.isoformat()}
        if isinstance(value, date):
            return {"type": "date", "value": value.isoformat()}
        return value

    @pydantic.field_validator("pattern")
    @classmethod
    def check_pattern(cls, value: str | None) -> str | None:
        if value is not None and not value:
            raise ValueError("Duration pattern must be a nonempty timestamp format")
        return value

    @pydantic.model_validator(mode="after")
    def check_endpoints(self) -> Self:
        constants = int(self.start is not None) + int(self.end is not None)
        if self.query is None and constants == 2:
            self.query = ()
        if isinstance(self.query, tuple):
            if len(self.query) == 2 and constants == 0 or len(self.query) == 0 and constants == 2:
                return self
            raise ValueError(
                "Duration needs query=(start_path, end_path) without endpoint constants, "
                "or both configured start and end without queries"
            )
        if constants != 1:
            raise ValueError("Duration with one input needs exactly one configured start or end timestamp")
        return self


@duration.register(Component.derive)
def derive(inputs: tuple[Projection, ...], *, request: RequestBase, address: Address, rows: int) -> Projection:
    """Derive numeric values before Number observes statistics or applies masks."""
    request = cast(Request, request)
    if len(inputs) == 2:
        start, end = (item.values for item in inputs)
    elif len(inputs) == 1:
        start = inputs[0].values if request.start is None else request.start
        end = inputs[0].values if request.end is None else request.end
    else:
        start, end = request.start, request.end
    try:
        values = difference(start, end, unit=request.granularity, pattern=request.pattern)
    except (TypeError, ValueError, pa.ArrowException) as error:
        raise ValueError(f"Duration field at '{address}' cannot derive elapsed time: {error}") from error
    if isinstance(values, pa.Scalar):
        values = pa.repeat(cast(pa.DoubleScalar, values), rows)
    present = pa.repeat(pa.scalar(True), rows)
    for item in inputs:
        present = pc.and_(present, item.present)
    return Projection(values=values, present=present)


for role, component in number.number.components.items():
    if role != Component.Request:
        duration.register(role, component)
