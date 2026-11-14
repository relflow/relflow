"""Input distribution warning policy."""

from typing import Annotated

import pydantic

__all__ = ["Drift"]


class Drift(pydantic.BaseModel):
    """Per-field effect-size thresholds; these diagnostics are not hypothesis tests."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    threshold: Annotated[float, pydantic.Field(gt=0, lt=1, allow_inf_nan=False)] = 0.2
    min_samples: Annotated[int, pydantic.Field(strict=True, ge=2)] = 128
