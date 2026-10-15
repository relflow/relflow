"""Canonical Arrow data structures and ingress helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import pyarrow as pa
import torch
from tensordict import TensorClass, TensorDict

from relflow.structs.tree import Address

Retain: TypeAlias = tuple[str, ...] | Literal["*"]


class Geometry(TensorClass):
    """Retained collection slots, independent of tensorfield values and masks.

    Lengths and collection validity use the parent grid. ``slots`` adds the
    resolved branch axis and marks its non-null object records.
    """

    lengths: torch.Tensor
    present: torch.Tensor
    valid: torch.Tensor
    slots: torch.Tensor


def retention(names: Retain, *, context: str = "retain") -> Retain:
    """Validate one selection of processed input columns."""

    if isinstance(names, str) and names == "*":
        return names
    if not isinstance(names, tuple):
        raise TypeError(f"{context} must be a tuple of column names or '*', got {type(names).__name__}")
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError(f"{context} entries must be non-empty strings, got {names!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"{context} entries must be unique, got {names!r}")
    return names


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
    retain: Retain = ()
    observations: Mapping[Address, TensorDict] = field(default_factory=dict)
    bindings: Mapping[Address, object] = field(default_factory=dict)

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
        if not isinstance(self.bindings, Mapping):
            raise TypeError(f"Encoded bindings must be a mapping, got {type(self.bindings).__name__}")
        if any(not isinstance(address, str) or not address for address in self.bindings):
            raise TypeError("Encoded binding addresses must be non-empty strings")
        object.__setattr__(self, "bindings", {Address(address): value for address, value in self.bindings.items()})
        retention(self.retain, context="Encoded retain")

    def pin_memory(self) -> Encoded:
        """Pin every tensor payload while leaving Arrow metadata untouched."""

        return Encoded(
            tensors=self.tensors.pin_memory(),
            source=self.source,
            retain=self.retain,
            observations={address: value.pin_memory() for address, value in self.observations.items()},
            bindings=self.bindings,
        )


__all__: list[str] = []
