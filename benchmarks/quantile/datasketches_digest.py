"""Arrow boundary for Apache DataSketches 5.2's native double t-digest."""

from __future__ import annotations

import datasketches
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


class Digest:
    """Mergeable empirical ranks; null and nonfinite input values are missing."""

    def __init__(self, compression: int = 100):
        if isinstance(compression, bool) or not isinstance(compression, int) or not 10 <= compression <= 65535:
            raise ValueError("compression must be an integer from 10 through 65535")
        self.compression = compression
        self.sketch = datasketches.tdigest_double(compression)

    @property
    def count(self) -> int:
        return self.sketch.get_total_weight()

    def update(self, values: pa.Array | pa.ChunkedArray) -> None:
        chunks = values.chunks if isinstance(values, pa.ChunkedArray) else (values,)
        for chunk in chunks:
            data = pc.cast(chunk, pa.float64()).to_numpy(zero_copy_only=False, writable=True)
            finite = np.isfinite(data)
            if finite.all():
                self.sketch.update(data)
            elif finite.any():
                self.sketch.update(data[finite])

    def merge(self, other: Digest) -> None:
        self.sketch.merge(other.sketch)

    def cdf(self, values: pa.Array) -> pa.Array:
        data = pc.cast(values, pa.float64()).to_numpy(zero_copy_only=False)
        finite = np.isfinite(data)
        output = np.full(data.shape, 0.5, dtype=np.float64)
        if self.count and finite.any():
            minimum = self.sketch.get_min_value()
            maximum = self.sketch.get_max_value()
            selected = data[finite]
            if minimum == maximum:
                ranks = np.where(selected < minimum, 0.0, np.where(selected > maximum, 1.0, 0.5))
            else:
                unique, inverse = np.unique(selected, return_inverse=True)
                ranks = np.asarray(self.sketch.get_cdf(unique), dtype=np.float64)[:-1][inverse]
            output[finite] = ranks
        return pa.array(output, mask=~finite)

    def quantile(self, probabilities: pa.Array) -> pa.Array:
        data = pc.cast(probabilities, pa.float64()).to_numpy(zero_copy_only=False)
        finite = np.isfinite(data)
        if np.any((data[finite] < 0.0) | (data[finite] > 1.0)):
            raise ValueError("quantile probabilities must lie between 0 and 1")
        if not self.count:
            return pa.nulls(len(data), type=pa.float64())
        output = np.zeros(data.shape, dtype=np.float64)
        output[finite] = np.fromiter((self.sketch.get_quantile(float(p)) for p in data[finite]), dtype=np.float64)
        return pa.array(output, mask=~finite)

    def serialize(self) -> bytes:
        return self.sketch.serialize()

    @classmethod
    def deserialize(cls, payload: bytes) -> Digest:
        sketch = datasketches.tdigest_double.deserialize(payload)
        digest = cls(sketch.k)
        digest.sketch = sketch
        return digest
