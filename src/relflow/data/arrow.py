"""Canonical Arrow data structures and ingress helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pyarrow as pa
from tensordict import TensorDict

from relflow.structs.tree import Address


def mappings(
    values: Sequence[Mapping[str, Any]],
    *,
    schema: pa.Schema | None = None,
    context: str = "Python records",
) -> list[dict[str, Any]]:
    """Align bounded Python mappings before their one Arrow conversion."""

    if any(not isinstance(value, Mapping) for value in values):
        raise TypeError(f"{context} must contain only mappings")

    if schema is None and values:
        first = values[0]
        names = tuple(first)
        if (
            names
            and all(isinstance(name, str) for name in names)
            and isinstance(first, dict)
            and all(isinstance(value, dict) and value.keys() == first.keys() for value in values)
        ):
            return cast(list[dict[str, Any]], list(values))

    keys: list[str] = []
    seen: set[str] = set()
    for value in values:
        for key in value:
            if not isinstance(key, str):
                raise TypeError(f"{context} mapping keys must be strings")
            if key not in seen:
                seen.add(key)
                keys.append(key)

    if schema is not None:
        if len(set(schema.names)) != len(schema.names):
            raise ValueError(f"{context} Arrow schema must have unique field names")
        extra = [key for key in keys if key not in schema.names]
        if extra:
            raise TypeError(f"{context} introduced field(s) {extra}; provide a complete arrow_schema")
        keys = list(schema.names)
    elif values and not keys:
        raise ValueError(f"{context} cannot infer an Arrow schema from mappings without fields")

    return [{key: value.get(key) for key in keys} for value in values]


def variants(values: pa.Array) -> tuple[np.ndarray, np.ndarray | None]:
    """Return logical type codes and child offsets for a sliced Arrow union."""

    if not isinstance(values, pa.UnionArray):
        raise TypeError(f"variants values must be an Arrow UnionArray, got {type(values).__name__}")
    buffers = values.buffers()
    codes = pa.Array.from_buffers(
        pa.int8(),
        len(values),
        [None, buffers[1]],
        offset=values.offset,
    ).to_numpy(zero_copy_only=False)
    if values.type.mode != "dense":
        return codes, None
    offsets = pa.Array.from_buffers(
        pa.int32(),
        len(values),
        [None, buffers[2]],
        offset=values.offset,
    ).to_numpy(zero_copy_only=False)
    return codes, offsets


@dataclass(frozen=True, slots=True)
class Encoded:
    """Tensor payload, pristine observations, and its untouched Arrow source."""

    tensors: TensorDict
    source: pa.Table
    retain: tuple[str, ...] | Literal["*"] = ()
    observations: Mapping[Address, TensorDict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.tensors, TensorDict):
            raise TypeError(f"Encoded tensors must be a TensorDict, got {type(self.tensors).__name__}")
        if not isinstance(self.source, pa.Table):
            raise TypeError(f"Encoded source must be a pyarrow.Table, got {type(self.source).__name__}")
        if not isinstance(self.observations, Mapping):
            raise TypeError(f"Encoded observations must be a mapping, got {type(self.observations).__name__}")
        normalized: dict[Address, TensorDict] = {}
        for address, observation in self.observations.items():
            if not isinstance(address, str) or not address:
                raise TypeError("Encoded observation addresses must be non-empty strings")
            if not isinstance(observation, TensorDict):
                raise TypeError(
                    f"Encoded observation at '{address}' must be a TensorDict, got {type(observation).__name__}"
                )
            normalized[Address(address)] = observation
        object.__setattr__(self, "observations", normalized)
        if self.retain != "*" and (
            not isinstance(self.retain, tuple)
            or any(not isinstance(name, str) or not name for name in self.retain)
            or len(set(self.retain)) != len(self.retain)
        ):
            raise ValueError("Encoded retain must be '*', or a tuple of unique non-empty column names")


__all__: list[str] = []
