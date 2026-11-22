"""Elapsed-time inputs reuse Number's encodings, statistics, objectives, and output."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal, Self, Unpack, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
from tensordict import TensorDict, tensorclass

from relflow.data.arrow import nullify
from relflow.data.query import Projection, Query, gather, listed, origin
from relflow.structs.enums import Component
from relflow.structs.tree import Address, FieldOptions
from relflow.tensorfields.base import Context, Extension
from relflow.tensorfields.extensions import number
from relflow.tensorfields.shared.jitter import Jitter
from relflow.tensorfields.shared.temporal import difference

if TYPE_CHECKING:
    from relflow.data.ragged import Collection, Sources
    from relflow.structs.experiment import Schema

duration = Extension.inherit(number.Request, name="duration", queries=slice(0, 3))


@duration.register(Component.Request)
class Request(number.Request):
    """A signed elapsed time, ``end - start``, measured in fixed units.

    ``query=(start_path, end_path)`` selects both endpoints. A single query
    pairs its source with one configured ``start`` or ``end`` timestamp. With
    a configured endpoint and no query, the field reads its same-named member.
    Two configured timestamps produce a constant duration. Ancestor references
    begin with ``../`` and broadcast within explicit Branch geometry.

    ``delta=True`` instead measures each timestamp against the preceding
    timestamp in its Branch collection. Records keep their source order and
    the lookup precedes cropping. A record without a predecessor has a null
    duration, so its position stays aligned with the other branch fields.

    ``pattern`` uses the same format syntax as Period. Zone-aware endpoints
    describe instants, and mixed aware/naive endpoints are rejected. Inputs
    never read the wall clock. Months and years have no fixed elapsed length
    and are not supported. All numeric modeling options come from Number.
    """

    # A specialized Request deliberately replaces its parent's discriminator.
    type: Literal["duration"] = "duration"  # pyrefly: ignore [bad-override-mutable-attribute]
    start: datetime | date | str | None = None
    end: datetime | date | str | None = None
    granularity: Literal["seconds", "minutes", "hours", "days", "weeks"] = "days"
    pattern: str | None = None
    delta: pydantic.StrictBool = False

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            start: datetime | date | str | None = None,
            end: datetime | date | str | None = None,
            granularity: Literal["seconds", "minutes", "hours", "days", "weeks"] = "days",
            pattern: str | None = None,
            delta: bool = False,
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
        count = 1 if self.query is None else len(self.query)
        if self.delta:
            if count != 1 or constants:
                raise ValueError(
                    "Duration with delta=True measures one timestamp query against the preceding record; "
                    "remove the configured start or end and use a single query"
                )
            return self
        if self.query is None and constants == 2:
            self.query = ()
            count = 0
        if count == 2 and constants == 0 or count == 0 and constants == 2 or count == 1 and constants == 1:
            return self
        raise ValueError(
            "Duration needs two queries without endpoint constants, one query and one endpoint, "
            "or both configured endpoints without queries"
        )

    def post_bind_validate(self) -> None:
        super().post_bind_validate()
        if self.delta:
            (expression,) = self.queries()
            depth, _ = origin(expression, address=str(self.address))
            if depth:
                raise ValueError(f"Duration delta at '{self.address}' requires a query in its own Branch")
            if sum(node.type == "branch" for node in self.path) < 2:
                raise ValueError(f"Duration delta at '{self.address}' requires a Branch of timestamp records")


def previous(collections: tuple[Collection, ...], expression: Query, *, address: Address) -> pa.Array | pa.ChunkedArray:
    """Select the preceding timestamp for each retained record in its original collection."""
    chunks = []
    for collection in collections:
        total = int(collection.counts.sum())
        parents = np.repeat(np.arange(len(collection.counts), dtype=np.int64), collection.counts)
        boundaries = np.repeat(np.cumsum(collection.counts) - collection.counts, collection.counts)
        slots = np.arange(total, dtype=np.int64) - boundaries
        positions = collection.starts[parents] + slots
        if collection.valid is not None:
            positions = positions[collection.valid]
            parents = parents[collection.valid]
        prior = positions - 1
        reachable = prior >= collection.first[parents].astype(np.int64, copy=False)
        indices = pa.array(np.where(reachable, prior - collection.base, 0), mask=~reachable, type=pa.int64())
        selected = gather(collection.records, expression, indices, address=str(address))
        chunks.append(pc.if_else(selected.present, selected.values, None))
    if not chunks:
        raise RuntimeError(f"Duration delta at '{address}' has no branch collection")
    return chunks[0] if len(chunks) == 1 else pa.chunked_array(chunks)


@duration.register(Component.TensorField)
@tensorclass
class TensorField(number.TensorField):
    """Construct numeric tensors from the selected timestamp sources."""

    @classmethod
    def new(
        cls,
        sources: Sources,
        *,
        schema: Schema,
        context: Context,
    ) -> tuple[Self, TensorDict | None, tuple[object, ...]]:
        address = sources.address
        request = cast(Request, schema.requests[address])
        if sources.vacant:
            return super().new(sources, schema=schema, context=context)
        inputs = sources.inputs
        rows = sources.rows
        if request.delta:
            (current,) = inputs
            if listed(current.values.type):
                raise ValueError(
                    f"Duration delta at '{address}' expects one timestamp per Branch record, got {current.values.type}"
                )
            _, expression = origin(request.queries()[0], address=str(address))
            start = previous(sources.collections, expression, address=address)
            present = current.present if current.present.type == pa.bool_() else pc.is_valid(current.present)
            end = nullify(current.values, present)
        elif len(inputs) == 2:
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
        if not request.delta:
            for item in inputs:
                present = pc.and_(present, item.present)
        return super().new(
            replace(sources, inputs=(Projection(values, present),)),
            schema=schema,
            context=context,
        )
