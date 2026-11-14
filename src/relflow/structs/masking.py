"""Frozen schema configurations for row selectors of a branch mask."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

import pydantic

from relflow.structs.reduction import StrictPositiveInt

__all__ = ["Forecast", "SelectorConfig", "Spans"]


class Forecast(pydantic.BaseModel):
    """Select the last ``horizon`` rows of each record as a forecast window.

    In training, validation and test the rows hide every child except the
    ``time_index`` child and the ``known`` children, nested collections
    included; only the ``targets`` children become reconstruction targets. At
    prediction, future rows hide the same children. Automatic slots are
    appended only when ``known`` is empty; otherwise the caller appends future
    rows that carry the ``known`` children. Requires ``direction=causal`` on
    the branch and ``reconstruct=True`` on the mask.

    Attributes:
        horizon: Number of trailing rows to forecast.
        targets: Direct leaf children to predict.
        known: Direct leaf children that stay visible in the forecast window.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    type: Literal["forecast"] = "forecast"
    horizon: StrictPositiveInt
    targets: tuple[str, ...] = pydantic.Field(min_length=1)
    known: tuple[str, ...] = ()


class Spans(pydantic.BaseModel):
    """Hide about ``rate`` of the rows, every child of a hidden row included.

    Rows hide in runs of ``length`` rows during training only; overlapping
    runs merge, and ``length=1`` hides independent rows.

    Attributes:
        rate: Fraction of rows hidden, strictly between 0 and 1.
        length: Rows in one hidden run.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    type: Literal["spans"] = "spans"
    rate: Annotated[float, pydantic.Field(strict=True, gt=0.0, lt=1.0)]
    length: StrictPositiveInt = 5


SelectorConfig: TypeAlias = Annotated[Forecast | Spans, pydantic.Field(discriminator="type")]
