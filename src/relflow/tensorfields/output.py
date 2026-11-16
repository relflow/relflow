"""Arrow builders shared by tensorfield output extensions and the model runtime.

Extensions return flat coordinate arrays. The runtime assembles their record
geometry and separately writes requested embeddings. List builders retain the
declared element field so Arrow IPC and compliant Parquet share one schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import numpy as np
import pyarrow as pa
import torch

from relflow.structs.enums import Tokens

if TYPE_CHECKING:
    from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel

STATE = pa.struct([pa.field(token.name, pa.float32(), nullable=False) for token in Tokens])


def array(values: torch.Tensor, dtype: pa.DataType) -> pa.Array:
    """Expose a detached tensor as one flat Arrow primitive array.

    CPU tensors cross through a contiguous NumPy view. Device tensors are
    copied to CPU once; PyArrow may then reuse the NumPy primitive buffer.
    """
    tensor = values.detach()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    numpy = tensor.contiguous().cpu().numpy().reshape(-1)
    if numpy.dtype == object:
        raise TypeError("tensor output cannot use an object-dtype NumPy buffer")
    return pa.array(numpy, type=dtype, from_pandas=False)


def fixed(
    values: pa.Array,
    size: int,
    *,
    datatype: pa.FixedSizeListType | None = None,
) -> pa.FixedSizeListArray:
    """Wrap values in a canonical fixed-width list, retaining its declared field."""
    if size <= 0:
        raise ValueError("fixed list size must be positive")
    if len(values) % size:
        raise ValueError(f"cannot wrap {len(values)} values in fixed lists of size {size}")
    declared = pa.list_(pa.field("element", values.type), size) if datatype is None else datatype
    if not isinstance(declared, pa.FixedSizeListType):
        raise TypeError(f"fixed list output requires a FixedSizeListType, got {declared}")
    if declared.list_size != size or declared.value_type != values.type:
        raise TypeError(f"fixed list output requires {size} elements of {values.type}, got {declared}")
    if declared.value_field.name != "element":
        raise ValueError(f"fixed list output child must be named 'element', got {declared.value_field.name!r}")
    if not declared.value_field.nullable and values.null_count:
        raise ValueError("fixed list output has null values but its element field is non-nullable")
    # pyarrow-stubs omits the supported declared-type overload.
    return pa.FixedSizeListArray.from_arrays(values, type=declared)  # pyrefly: ignore [no-matching-overload]


def shape(values: pa.Array, axes: tuple[int, ...]) -> pa.Array:
    """Wrap a flat coordinate array in model axes, innermost first."""
    shaped = values
    for size in reversed(axes):
        shaped = fixed(shaped, size)
    return shaped


def struct(values: Mapping[str, pa.Array], dtype: pa.StructType) -> pa.StructArray:
    """Build a struct array and enforce its declared field order and types."""
    names = tuple(values)
    expected = tuple(field.name for field in dtype)
    if names != expected:
        raise ValueError(f"struct fields must be {expected}, got {names}")

    arrays = list(values.values())
    lengths = {len(value) for value in arrays}
    if len(lengths) > 1:
        raise ValueError(f"struct fields have inconsistent lengths: {sorted(lengths)}")
    for field, value in zip(dtype, arrays, strict=True):
        if value.type != field.type:
            raise TypeError(f"struct field {field.name!r} must have type {field.type}, got {value.type}")

    return pa.StructArray.from_arrays(arrays, fields=list(dtype))


def offsets(counts: torch.Tensor) -> pa.Int32Array:
    """Return Arrow list offsets for non-negative per-coordinate counts."""
    flat = counts.detach().reshape(-1).to(dtype=torch.int64, device="cpu")
    if flat.lt(0).any():
        raise ValueError("list counts must be non-negative")
    cumulative = torch.empty(flat.numel() + 1, dtype=torch.int64)
    cumulative[0] = 0
    cumulative[1:] = flat.cumsum(dim=0)
    if cumulative[-1].item() > np.iinfo(np.int32).max:
        raise OverflowError("Arrow list output exceeds the int32 offset limit")
    return cast(pa.Int32Array, array(cumulative, pa.int32()))


def variable(values: pa.Array, counts: torch.Tensor, *, datatype: pa.ListType | None = None) -> pa.ListArray:
    """Wrap values in canonical lists, retaining the declared element field."""
    boundaries = offsets(counts)
    if boundaries[-1].as_py() != len(values):
        raise ValueError("list counts do not consume the supplied values")
    declared = pa.list_(pa.field("element", values.type)) if datatype is None else datatype
    if not isinstance(declared, pa.ListType) or declared.value_type != values.type:
        raise TypeError(f"list output requires elements of {values.type}, got {declared}")
    if declared.value_field.name != "element":
        raise ValueError(f"list output child must be named 'element', got {declared.value_field.name!r}")
    if not declared.value_field.nullable and values.null_count:
        raise ValueError("list output has null values but its element field is non-nullable")
    return pa.ListArray.from_arrays(boundaries, values, type=declared)


def labels(vocabulary: OnlineVocabularyModel) -> pa.Array:
    """Return one cached canonical large-string vocabulary array."""
    values = vocabulary.labels()

    if not isinstance(values, pa.Array) or values.type != pa.large_string():
        raise TypeError("vocabulary labels must be a large_string Arrow array")
    return values


def state(logits: torch.Tensor) -> pa.StructArray:
    """Convert final-axis state logits into the shared probability struct."""
    if logits.ndim == 0 or logits.shape[-1] != len(Tokens):
        raise ValueError(f"state logits must have a final dimension of {len(Tokens)}")
    probabilities = logits.detach().float().softmax(dim=-1).reshape(-1, len(Tokens)).cpu()
    return struct(
        {token.name: array(probabilities[:, token.value], pa.float32()) for token in Tokens},
        STATE,
    )


def inferred(values: torch.Tensor) -> pa.BooleanArray:
    """Convert a tensor mask into one flat Arrow boolean array."""
    return cast(pa.BooleanArray, array(values.bool(), pa.bool_()))


def embedding(values: torch.Tensor, *, datatype: pa.FixedSizeListType | None = None) -> pa.FixedSizeListArray:
    """L2-normalize embeddings and preserve their final fixed-width axis."""
    if values.ndim == 0 or values.shape[-1] <= 0:
        raise ValueError("embedding output requires a non-empty final dimension")
    normalized = torch.nn.functional.normalize(values.detach().float(), p=2, dim=-1, eps=1e-12)
    return fixed(array(normalized, pa.float32()), normalized.shape[-1], datatype=datatype)


__all__ = [
    "STATE",
    "array",
    "embedding",
    "fixed",
    "inferred",
    "labels",
    "offsets",
    "shape",
    "state",
    "struct",
    "variable",
]
