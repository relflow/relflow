"""RelFlow's Arrow-backed merging t-digest, independent of Arrow's quantile kernel.

Compression uses the canonical k1 scale, k(q) = compression * asin(2q-1)
/ (2*pi). Whole weighted centroids are greedily merged while their combined
k-span is at most one. An indivisible existing centroid can exceed that span
after distribution shifts; it is retained intact, never split into fake data.
The two extreme values are retained separately, with their exact masses.

CDF interpolates centroid midranks, preserving jumps for centroids known to
contain one distinct value; inverse quantiles use the same knots. Equal
centroid values are coalesced before compression.
This is approximation state, not quantile samples or retained raw history.
"""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

__all__ = ["Digest"]

SCHEMA = pa.schema([("mean", pa.float64()), ("weight", pa.int64()), ("point", pa.bool_())])


def numeric(values: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """Extract a float64 vector at the digest's numerical boundary."""
    if not isinstance(values, (pa.Array, pa.ChunkedArray)):
        raise TypeError("digest values must be an Arrow Array or ChunkedArray")
    array = pc.cast(values, pa.float64())
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    return array.to_numpy(zero_copy_only=False)


def interpolate(query: np.ndarray, coordinates: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Interpolate locally without overflowing opposite-sign float64 bounds."""
    right = np.clip(np.searchsorted(coordinates, query, side="right"), 1, len(coordinates) - 1)
    left = right - 1
    low, high = coordinates[left], coordinates[right]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        width = high - low
        fraction = (query - low) / width
        wide = np.isinf(width)
        scale = np.maximum(np.abs(low[wide]), np.abs(high[wide]))
        fraction[wide] = (query[wide] / scale - low[wide] / scale) / (high[wide] / scale - low[wide] / scale)
    fraction = np.clip(np.nan_to_num(fraction, nan=0.0), 0, 1)
    result = (1 - fraction) * values[left] + fraction * values[right]
    result = np.clip(result, values[left], values[right])
    result[query <= coordinates[0]] = values[0]
    result[query >= coordinates[-1]] = values[-1]
    return result


class Digest:
    """Bounded weighted-centroid sketch; updates ignore nonfinite inputs."""

    def __init__(self, compression: int = 100):
        if isinstance(compression, bool) or not isinstance(compression, Integral) or compression < 10:
            raise ValueError("digest compression must be an integer of at least 10")
        self.compression = int(compression)
        self.centroids = pa.Table.from_arrays([pa.array([], type=t.type) for t in SCHEMA], schema=SCHEMA)
        self.count = 0

    @property
    def minimum(self) -> float:
        return float(self.centroids["mean"][0].as_py()) if self.count else math.nan

    @property
    def maximum(self) -> float:
        return float(self.centroids["mean"][-1].as_py()) if self.count else math.nan

    def update(self, values: pa.Array | pa.ChunkedArray) -> None:
        incoming = numeric(values)
        incoming = incoming[np.isfinite(incoming)]
        if incoming.size:
            self.compress(incoming, np.ones(incoming.size, dtype=np.int64), np.ones(incoming.size, dtype=bool))

    def merge(self, other: Digest) -> None:
        if not isinstance(other, Digest):
            raise TypeError("digest merge requires another Digest")
        if other.count:
            self.compress(
                other.centroids["mean"].to_numpy(),
                other.centroids["weight"].to_numpy(),
                other.centroids["point"].to_numpy(),
            )

    def compress(self, means: np.ndarray, weights: np.ndarray, points: np.ndarray) -> None:
        """Sort/coalesce in bulk, then choose each output centroid by rank."""
        added = int(weights.sum(dtype=np.int64))
        if self.count + added > np.iinfo(np.int64).max:
            raise OverflowError("digest count exceeds int64 capacity")
        means = np.concatenate((self.centroids["mean"].to_numpy(), means))
        weights = np.concatenate((self.centroids["weight"].to_numpy(), weights))
        points = np.concatenate((self.centroids["point"].to_numpy(), points))
        order = np.argsort(means, kind="stable")
        means, weights, points = means[order], weights[order], points[order]
        starts = np.r_[0, np.flatnonzero(means[1:] != means[:-1]) + 1]
        means = means[starts]
        weights = np.add.reduceat(weights, starts)
        points = np.logical_and.reduceat(points, starts)
        total = self.count + added

        if len(means) > 2:
            cumulative = np.r_[np.int64(0), np.cumsum(weights, dtype=np.int64)]
            merged_means = [float(means[0])]
            merged_weights = [int(weights[0])]
            merged_points = [bool(points[0])]
            start = 1
            while start < len(means) - 1:
                rank = float(cumulative[start]) / total
                angle = math.asin(min(1.0, max(-1.0, 2 * rank - 1))) + 2 * math.pi / self.compression
                limit = total if angle >= math.pi / 2 else total * (math.sin(angle) + 1) / 2
                stop = int(np.searchsorted(cumulative, limit, side="right")) - 1
                stop = min(len(means) - 1, max(start + 1, stop))
                mass = int(cumulative[stop] - cumulative[start])
                segment = means[start:stop]
                # Per-centroid native reduction avoids cancellation from subtracting
                # prefix moments and overflow from summing large raw magnitudes.
                scale = max(abs(float(segment[0])), abs(float(segment[-1])))
                mean = 0.0 if scale == 0 else float(np.dot(segment / scale, weights[start:stop] / mass)) * scale
                merged_means.append(float(np.clip(mean, segment[0], segment[-1])))
                merged_weights.append(mass)
                merged_points.append(stop == start + 1 and bool(points[start]))
                start = stop
            merged_means.append(float(means[-1]))
            merged_weights.append(int(weights[-1]))
            merged_points.append(bool(points[-1]))
            means = np.asarray(merged_means, dtype=np.float64)
            weights = np.asarray(merged_weights, dtype=np.int64)
            points = np.asarray(merged_points, dtype=bool)

        self.centroids = pa.Table.from_arrays([pa.array(means), pa.array(weights), pa.array(points)], schema=SCHEMA)
        self.count = total

    def knots(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Represent point masses as jumps and mixed centroids as midranks."""
        weights = self.centroids["weight"].to_numpy()
        after = np.cumsum(weights, dtype=np.int64) / self.count
        before = (np.cumsum(weights, dtype=np.int64) - weights) / self.count
        midranks = (after + before) / 2
        points = self.centroids["point"].to_numpy()
        ranks = np.stack((np.where(points, before, midranks), np.where(points, after, midranks)), axis=1)
        return np.repeat(self.centroids["mean"].to_numpy(), 2), ranks.reshape(-1), midranks

    def cdf(self, values: pa.Array) -> pa.Array:
        query = numeric(values)
        valid = np.isfinite(query)
        result = np.full(query.shape, 0.5, dtype=np.float64)
        if self.count:
            means, ranks, midranks = self.knots()
            result[valid & (query < means[0])] = 0
            result[valid & (query > means[-1])] = 1
            inside = valid & (query >= means[0]) & (query <= means[-1])
            result[inside] = interpolate(query[inside], means, ranks)
            positions = np.minimum(np.searchsorted(means[::2], query[inside]), len(midranks) - 1)
            tied = means[::2][positions] == query[inside]
            result[inside] = np.where(tied, midranks[positions], result[inside])
        return pa.array(result, mask=~valid, type=pa.float64())

    def quantile(self, probabilities: pa.Array) -> pa.Array:
        query = numeric(probabilities)
        valid = np.isfinite(query)
        if np.any(valid & ((query < 0) | (query > 1))):
            raise ValueError("digest quantile probabilities must lie in [0, 1]")
        if not self.count:
            return pa.nulls(len(query), type=pa.float64())
        means, ranks, _ = self.knots()
        result = np.full(query.shape, np.nan, dtype=np.float64)
        result[valid] = interpolate(query[valid], ranks, means)
        return pa.array(result, mask=~valid, type=pa.float64())

    def serialize(self) -> bytes:
        table = self.centroids.replace_schema_metadata(
            {b"relflow.digest": b"1", b"compression": str(self.compression).encode("ascii")}
        )
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        return sink.getvalue().to_pybytes()

    @classmethod
    def deserialize(cls, payload: bytes) -> Digest:
        table = pa.ipc.open_stream(payload).read_all()
        metadata = table.schema.metadata or {}
        if metadata.get(b"relflow.digest") != b"1" or not table.schema.remove_metadata().equals(SCHEMA):
            raise ValueError("invalid Arrow digest snapshot schema or version")
        digest = cls(compression=int(metadata[b"compression"]))
        means, weights = table["mean"].to_numpy(), table["weight"].to_numpy()
        if (
            table["mean"].null_count
            or table["weight"].null_count
            or table["point"].null_count
            or np.any(~np.isfinite(means))
            or np.any(weights <= 0)
            or np.any(means[1:] <= means[:-1])
            or len(means) > digest.compression + 3
            or (len(means) and not (table["point"][0].as_py() and table["point"][-1].as_py()))
        ):
            raise ValueError("invalid Arrow digest centroid state")
        digest.count = sum(map(int, weights))
        if digest.count > np.iinfo(np.int64).max:
            raise ValueError("Arrow digest snapshot count exceeds int64 capacity")
        digest.centroids = table.replace_schema_metadata(None)
        return digest
