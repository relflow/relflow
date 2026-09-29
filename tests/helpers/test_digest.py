"""Numerical and persistence contracts for the percentile normalization state."""

import math

import numpy as np
import pyarrow as pa
import pytest

from relflow.helpers.digest import SCHEMA, Digest


def snapshot(means, weights, points=None, *, metadata=None) -> bytes:
    """Encode supplied centroid state to exercise the persistence boundary."""
    table = pa.Table.from_arrays(
        [
            pa.array(means, type=pa.float64()),
            pa.array(weights, type=pa.int64()),
            pa.array([True] * len(means) if points is None else points, type=pa.bool_()),
        ],
        schema=SCHEMA,
    ).replace_schema_metadata({b"relflow.digest": b"1", b"compression": b"100"} if metadata is None else metadata)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


@pytest.mark.parametrize(
    ("distribution", "limit"), [("uniform", 0.008), ("normal", 0.01), ("lognormal", 0.02), ("duplicates", 1e-14)]
)
def test_empirical_cdf_and_inverse_rank_error(distribution, limit):
    rng = np.random.default_rng(418)
    if distribution == "uniform":
        values = rng.uniform(-10, 10, 60_000)
    elif distribution == "normal":
        values = rng.normal(size=60_000)
    elif distribution == "lognormal":
        values = rng.lognormal(0, 2.5, 60_000)
    else:
        values = rng.choice([0.0, 1.0, 2.0, 10.0, 1000.0], 60_000, p=[0.7, 0.1, 0.1, 0.05, 0.05])
    digest = Digest()
    for batch in np.array_split(values, 30):
        digest.update(pa.array(batch))

    ordered = np.sort(values)
    probabilities = np.unique(np.r_[np.linspace(0, 1, 501), 0.0001, 0.001, 0.999, 0.9999])
    queries = np.unique(np.quantile(ordered, probabilities))
    lower = np.searchsorted(ordered, queries, side="left") / len(values)
    upper = np.searchsorted(ordered, queries, side="right") / len(values)
    ranks = digest.cdf(pa.array(queries)).to_numpy()
    inverse = digest.quantile(pa.array(probabilities)).to_numpy()
    inverse_lower = np.searchsorted(ordered, inverse, side="left") / len(values)
    inverse_upper = np.searchsorted(ordered, inverse, side="right") / len(values)
    # These are regression bounds on fixed representative samples, not a
    # distribution-independent accuracy guarantee for t-digests.
    assert np.max(np.abs(ranks - (lower + upper) / 2)) < limit
    assert np.max(np.maximum(inverse_lower - probabilities, probabilities - inverse_upper)) < limit
    assert np.all(np.isfinite(ranks)) and np.all((ranks >= 0) & (ranks <= 1))
    assert np.all(ranks[1:] >= ranks[:-1])
    assert np.all(np.isfinite(inverse)) and np.all(inverse[1:] >= inverse[:-1])
    assert inverse[0] == ordered[0] and inverse[-1] == ordered[-1]
    assert digest.count == len(values)


@pytest.mark.parametrize("compression", [100, 200])
def test_shard_merging_preserves_mass_sources_and_bounded_error(compression):
    values = np.sort(np.random.default_rng(91).normal(size=32_000))
    shards = []
    for batch in np.array_split(values, 16):
        shard = Digest(compression)
        shard.update(pa.array(batch))
        shards.append(shard)
    snapshots = [shard.serialize() for shard in shards]
    queries = np.quantile(values, np.linspace(0, 1, 501))
    truth = np.searchsorted(values, queries) / len(values)
    orders = [range(16), reversed(range(16)), np.random.default_rng(37).permutation(16)]
    for order in orders:
        merged = Digest(compression)
        for position in order:
            merged.merge(shards[position])
        merged.merge(Digest(compression))
        assert merged.count == len(values)
        assert merged.minimum == values[0] and merged.maximum == values[-1]
        assert merged.centroids.num_rows <= compression + 3
        ranks = merged.cdf(pa.array(queries)).to_numpy()
        assert np.max(np.abs(ranks - truth)) < 0.02
        assert np.all(ranks[1:] >= ranks[:-1])
    # Merge order can change an approximation; it must never consume sources.
    assert [shard.serialize() for shard in shards] == snapshots


@pytest.mark.parametrize(
    "values",
    [
        [-1e308, 0.0, 1e308],
        [-1e308, 1e-300, 1e-200, 1e308],
        [1e308, np.nextafter(1e308, math.inf), np.finfo(np.float64).max],
    ],
)
def test_float64_extrema_have_finite_monotone_ranks_and_inverse(values):
    digest = Digest()
    digest.update(pa.array(np.repeat(values, 1000)))
    queries = np.sort(np.r_[values, 0.0, 1e-250])
    ranks = digest.cdf(pa.array(queries)).to_numpy()
    inverse = digest.quantile(pa.array(np.linspace(0, 1, 1001))).to_numpy()
    assert np.isfinite(ranks).all() and np.all((ranks >= 0) & (ranks <= 1))
    assert np.all(ranks[1:] >= ranks[:-1])
    assert np.isfinite(inverse).all() and np.all(inverse[1:] >= inverse[:-1])
    np.testing.assert_allclose(digest.cdf(pa.array(values)).to_numpy(), (np.arange(len(values)) + 0.5) / len(values))
    assert inverse[0] == min(values) and inverse[-1] == max(values)


def test_empty_and_chunked_inputs_preserve_null_and_nonfinite_policy():
    digest = Digest()
    query = pa.array([None, math.nan, math.inf, -math.inf, -1.0, 0.0, 1.0])
    assert digest.cdf(query).to_pylist() == [None, None, None, None, 0.5, 0.5, 0.5]
    assert digest.quantile(pa.array([None, math.nan, 0.0, 0.5, 1.0])).to_pylist() == [None] * 5
    assert math.isnan(digest.minimum) and math.isnan(digest.maximum)
    digest.update(pa.chunked_array([[None, math.nan, math.inf], [], [-math.inf]], type=pa.float64()))
    assert digest.count == 0
    digest.update(pa.chunked_array([[None, -1], [0, 1]], type=pa.int64()))
    assert digest.count == 3
    assert digest.cdf(query).to_pylist()[:4] == [None] * 4
    assert digest.cdf(pa.array([], type=pa.float64())).type == pa.float64()
    assert digest.quantile(pa.array([], type=pa.float64())).to_pylist() == []


def test_constant_and_dominant_ties_retain_point_mass():
    digest = Digest()
    digest.update(pa.array([4.0] * 700))
    assert digest.cdf(pa.array([3.0, 4.0, 5.0, None])).to_pylist() == [0.0, 0.5, 1.0, None]
    assert digest.quantile(pa.array([0.0, 0.5, 1.0])).to_pylist() == [4.0] * 3
    digest.update(pa.array([5.0] * 200 + [9.0] * 100))
    np.testing.assert_allclose(digest.cdf(pa.array([4.0, 4.5, 5.0, 8.0, 9.0])).to_numpy(), [0.35, 0.7, 0.8, 0.9, 0.95])
    assert digest.quantile(pa.array([0.1, 0.5, 0.6, 0.8, 0.95])).to_pylist() == [4.0, 4.0, 4.0, 5.0, 9.0]


@pytest.mark.parametrize("probability", [-0.001, 1.001])
def test_invalid_probability_is_rejected_even_before_fitting(probability):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Digest().quantile(pa.array([probability]))


def test_snapshot_restores_inference_and_continued_updates():
    original = Digest(150)
    original.update(pa.array(np.random.default_rng(7).lognormal(size=10_000)))
    restored = Digest.deserialize(original.serialize())
    queries = pa.array([None, 0.0, 0.25, 0.5, 1.0])
    assert restored.compression == original.compression
    assert restored.count == original.count
    assert restored.cdf(queries).equals(original.cdf(queries))
    assert restored.quantile(queries).equals(original.quantile(queries))
    for digest in [original, restored]:
        digest.update(pa.array([0.5, 2.0, 1000.0]))
    assert restored.cdf(queries).equals(original.cdf(queries))
    assert restored.quantile(queries).equals(original.quantile(queries))
    assert Digest.deserialize(Digest().serialize()).count == 0


@pytest.mark.parametrize(
    "payload",
    [
        snapshot([math.nan], [1]),
        snapshot([0.0], [0]),
        snapshot([0.0], [-1]),
        snapshot([1.0, 0.0], [1, 1]),
        snapshot([0.0, 0.0], [1, 1]),
        snapshot([0.0], [1], [False]),
        snapshot([0.0], [None]),
        snapshot([0.0], [1], metadata={b"relflow.digest": b"2", b"compression": b"100"}),
    ],
    ids=["nonfinite", "zero_mass", "negative_mass", "unordered", "duplicate", "mixed_extreme", "null", "version"],
)
def test_corrupt_snapshot_is_rejected(payload):
    with pytest.raises(ValueError):
        Digest.deserialize(payload)


def test_large_counts_remain_exact_and_overflow_does_not_mutate_state():
    maximum = np.iinfo(np.int64).max
    digest = Digest.deserialize(snapshot([0.0, 1.0], [maximum - 2, 1]))
    assert digest.count == maximum - 1
    digest.update(pa.array([0.0]))
    assert digest.count == maximum
    assert digest.minimum == 0.0 and digest.maximum == 1.0
    assert Digest.deserialize(digest.serialize()).count == maximum
    inverse = digest.quantile(pa.array([0.0, 0.25, 0.5, 0.75, 1.0])).to_numpy()
    assert np.isfinite(inverse).all() and np.all(inverse[1:] >= inverse[:-1])
    before = digest.serialize()
    with pytest.raises(OverflowError, match="int64"):
        digest.update(pa.array([0.0]))
    addition = Digest()
    addition.update(pa.array([2.0]))
    with pytest.raises(OverflowError, match="int64"):
        digest.merge(addition)
    assert digest.count == maximum and digest.serialize() == before
    assert addition.count == 1
    with pytest.raises(ValueError, match="int64"):
        Digest.deserialize(snapshot([0.0, 1.0], [maximum, 1]))
