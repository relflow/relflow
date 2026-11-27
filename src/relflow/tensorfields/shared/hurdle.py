"""Hurdle states for numeric fields with a point mass, such as exact zeros."""

import math
import numbers
from typing import Annotated, TypeAlias

import numpy as np
import pydantic

from relflow.structs.enums import Tokens


def check_hurdle(value: object) -> float | None:
    """Accept one finite real number; booleans and numeric strings are not hurdles."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"hurdle must be a finite number, got {value!r}")
    return float(value)


Hurdle: TypeAlias = Annotated[float | None, pydantic.BeforeValidator(check_hurdle)]


def mark(data: np.ndarray, state: np.ndarray, hurdle: float | None) -> None:
    """Move valued positions equal to ``hurdle`` to the hurdle state, in place.

    Like ``other``, a hurdle position keeps zero content: it is neither embedded
    as a value, counted in statistics, nor regressed against; the state head
    learns its probability instead.
    """
    if hurdle is None:
        return
    selected = (data == hurdle) & (state == Tokens.valued.value)
    state[selected] = Tokens.hurdle.value
    data[selected] = 0
