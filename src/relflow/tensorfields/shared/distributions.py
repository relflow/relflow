"""Reusable Arrow projections selected explicitly by tensorfield extensions."""

from __future__ import annotations

from typing import cast

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

from relflow.data.ragged import RaggedField
from relflow.distributions import Distribution
from relflow.structs.tree import Leaf
from relflow.tensorfields.shared.vocabulary import canonical, fold

__all__ = ["numeric", "categorical", "boolean", "sets", "text", "vector"]


def numeric(field: RaggedField, *, request: Leaf) -> dict[str, Distribution]:
    values = pc.cast(field.values, pa.float64()).to_numpy(zero_copy_only=False)
    status = np.zeros(len(values), dtype=np.int64)
    status[np.isnan(values)] = 1
    status[np.isposinf(values)] = 2
    status[np.isneginf(values)] = 3
    return {"value": Distribution.numeric(field.values), "nonfinite": Distribution.categorical(status, bins=4)}


def labels(values: pa.Array | pa.ChunkedArray, *, strip: bool, cased: bool) -> Distribution:
    """Fixed identity buckets avoid retaining labels or growing a vocabulary."""
    values = canonical(fold(values, strip=strip, cased=cased))
    if pa.types.is_floating(values.type):
        values = pc.filter(values, pc.invert(pc.is_nan(values)))
    series = cast(pl.Series, pl.from_arrow(values))
    if series.dtype.is_numeric() or series.dtype == pl.Boolean:
        # Python vocabulary equality identifies 1, 1.0, and True. Preserve that
        # equality across Arrow widths without collapsing large integer IDs.
        numbers = series.cast(pl.Float64)
        indices = numbers.hash(seed=1).to_numpy() % 128
        if series.dtype.is_integer():
            remainder = (series.cast(pl.Int128) - numbers.cast(pl.Int128)).cast(pl.Int64).to_numpy()
            indices = (indices + (remainder % 128).astype(np.uint64) * 131) % 128
    else:
        indices = series.hash(seed=0).to_numpy() % 128
    return Distribution.categorical(indices, bins=128, encoding=f"polars:{pl.__version__}")


def categorical(field: RaggedField, *, request: Leaf) -> dict[str, Distribution]:
    # These options are supplied only by extensions that declare label folding.
    return {
        "label": labels(
            field.values, strip=bool(getattr(request, "strip", False)), cased=bool(getattr(request, "cased", True))
        )
    }


def boolean(field: RaggedField, *, request: Leaf) -> dict[str, Distribution]:
    return {"value": Distribution.categorical(field.values.to_numpy(zero_copy_only=False), bins=2)}


def sets(field: RaggedField, *, request: Leaf) -> dict[str, Distribution]:
    values = cast(pl.Series, pl.from_arrow(field.values))
    if not isinstance(values.dtype, (pl.List, pl.Array)):
        values = values.reshape((-1, 1))
    if isinstance(values.dtype, pl.Array):
        values = values.arr.to_list()
    # Fold before deduplicating: Set membership follows its vocabulary equality.
    flattened = pc.list_flatten(values.to_arrow())
    folded = fold(flattened, strip=bool(getattr(request, "strip", False)), cased=bool(getattr(request, "cased", True)))
    lists = cast(pa.LargeListArray, values.to_arrow())
    folded = folded.combine_chunks() if isinstance(folded, pa.ChunkedArray) else folded
    # Flattening compacts the retained values, while sliced lists retain source offsets.
    offsets = pc.subtract(lists.offsets, lists.offsets[0])
    lists = pa.LargeListArray.from_arrays(offsets, folded)
    values = cast(pl.Series, pl.from_arrow(lists)).list.drop_nulls()
    if cast(pl.List, values.dtype).inner.is_float():
        values = values.list.eval(pl.element().filter(pl.element().is_not_nan()))
    values = values.list.unique()
    members = values.explode().drop_nulls().to_arrow()
    return {
        "member": labels(members, strip=False, cased=True),
        "size": Distribution.numeric(values.list.len().to_arrow()),
    }


def text(field: RaggedField, *, request: Leaf) -> dict[str, Distribution]:
    return {"characters": Distribution.numeric(pc.utf8_length(cast(pa.StringArray, field.values)))}


def vector(field: RaggedField, *, request: Leaf) -> dict[str, Distribution]:
    values = cast(pl.Series, pl.from_arrow(field.values))
    if isinstance(values.dtype, pl.Array):
        values = values.arr.to_list()
    if not isinstance(values.dtype, pl.List):
        return {}  # The vector codec owns the useful shape error.
    values = values.cast(pl.List(pl.Float32))
    finite = values.list.eval(pl.element().is_finite()).list.all().fill_null(False)
    valid = values.filter(finite).cast(pl.List(pl.Float64))
    # Norm uses scaled components to avoid overflow during squaring.
    scale = valid.list.eval(pl.element().abs()).list.max()
    norm = (valid / scale.replace(0, 1)).list.eval(pl.element().pow(2)).list.sum().sqrt() * scale
    return {
        "mean": Distribution.numeric(valid.list.mean().to_arrow()),
        "norm": Distribution.numeric(norm.to_arrow()),
        "finite": Distribution.categorical(finite.to_numpy(), bins=2),
    }
