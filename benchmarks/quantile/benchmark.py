"""Reproducible comparison; all data movement inside adapters is timed."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import inspect
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
from datasketches_digest import Digest as SketchDigest

from relflow.helpers.digest import Digest as ArrowDigest

BACKENDS = {"arrow": ArrowDigest, "datasketches": SketchDigest}


def numpy(values):
    return values.to_numpy(zero_copy_only=False)


def timed(call):
    start = time.perf_counter()
    result = call()
    return time.perf_counter() - start, result


def summary(samples):
    return {"median_s": statistics.median(samples), "samples_s": samples}


def correctness(cls, compression):
    result = {}
    digest = cls(compression)
    assert digest.count == 0
    assert digest.cdf(pa.array([None, -1.0, 0.0, 1.0])).to_pylist() == [None, 0.5, 0.5, 0.5]
    assert digest.quantile(pa.array([0.0, 0.5, 1.0, None])).to_pylist() == [None] * 4
    digest.update(pa.chunked_array([[None, 0.0, float("nan")], [1.0, float("inf"), float("-inf"), 2.0]]))
    assert digest.count == 3
    values = pa.array([None, float("nan"), float("inf"), -1.0, 0.0, 0.5, 1.0, 2.0, 3.0])
    ranks = digest.cdf(values)
    assert ranks.to_pylist()[:3] == [None] * 3
    finite = numpy(ranks)[3:]
    assert np.isfinite(finite).all() and np.all(np.diff(finite) >= 0)
    assert finite[0] == 0 and finite[-1] == 1
    assert digest.quantile(pa.array([0.0, 1.0])).to_pylist() == [0.0, 2.0]
    for probability in [-0.001, 1.001]:
        try:
            digest.quantile(pa.array([probability]))
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-range probabilities accepted")
    other = cls(compression)
    other.update(pa.array([4.0] * 100))
    assert other.cdf(pa.array([3.0, 4.0, 5.0, None])).to_pylist() == [0.0, 0.5, 1.0, None]
    other_before = other.cdf(values).to_pylist()
    digest.merge(other)
    assert digest.count == 103 and other.count == 100
    assert other.cdf(values).to_pylist() == other_before
    digest.merge(cls(compression))
    assert digest.count == 103
    restored = cls.deserialize(digest.serialize())
    assert restored.count == digest.count
    np.testing.assert_allclose(numpy(restored.cdf(values)), numpy(digest.cdf(values)), equal_nan=True)
    probs = pa.array([0.0, 0.001, 0.1, 0.5, 0.9, 0.999, 1.0, None])
    np.testing.assert_allclose(numpy(restored.quantile(probs)), numpy(digest.quantile(probs)), equal_nan=True)
    result["standard_checks"] = "pass: empty/null/nonfinite/count/constant/bounds/merge/persistence"
    extreme = cls(compression)
    extreme.update(pa.array(np.repeat([-1e308, 0.0, 1e308], 1000)))
    ranks = numpy(extreme.cdf(pa.array([-1e308, 0.0, 1e308])))
    quantiles = numpy(extreme.quantile(pa.array(np.linspace(0, 1, 101))))
    result["float64_extreme_stress"] = {
        "count": extreme.count,
        "finite_cdf": bool(np.isfinite(ranks).all()),
        "finite_quantiles": bool(np.isfinite(quantiles).all()),
        "cdf_max_rank_error": float(np.max(np.abs(ranks - np.array([1 / 6, 0.5, 5 / 6])))),
        "quantile_nonfinite_count": int(np.sum(~np.isfinite(quantiles))),
    }
    return result


def stream(cls, values, batch_size, compression):
    digest = cls(compression)
    update_s = cdf_s = 0.0
    for offset in range(0, len(values), batch_size):
        batch = values.slice(offset, batch_size)
        elapsed, _ = timed(lambda: digest.update(batch))
        update_s += elapsed
        elapsed, _ = timed(lambda: digest.cdf(batch))
        cdf_s += elapsed
    assert digest.count == len(values)
    return update_s, cdf_s, len(digest.serialize())


def merged(cls, values, compression, shards=32):
    digests = []
    build_s = 0.0
    for offset in range(0, len(values), (len(values) + shards - 1) // shards):
        digest = cls(compression)
        elapsed, _ = timed(lambda: digest.update(values.slice(offset, (len(values) + shards - 1) // shards)))
        build_s += elapsed
        digests.append(digest)
    target = cls(compression)
    start = time.perf_counter()
    for digest in digests:
        target.merge(digest)
    merge_s = time.perf_counter() - start
    assert target.count == len(values)
    return build_s, merge_s, len(target.serialize())


def errors(cls, data, probabilities, compression):
    values = pa.array(data)
    ordered = np.sort(data)
    digest = cls(compression)
    for offset in range(0, len(data), 10_000):
        digest.update(values.slice(offset, 10_000))
    queries = np.unique(np.quantile(ordered, probabilities))
    left = np.searchsorted(ordered, queries, side="left") / len(data)
    right = np.searchsorted(ordered, queries, side="right") / len(data)
    truth = (left + right) / 2
    predicted = numpy(digest.cdf(pa.array(queries)))
    cdf_error = np.abs(predicted - truth)
    inverse = numpy(digest.quantile(pa.array(probabilities)))
    lower = np.searchsorted(ordered, inverse, side="left") / len(data)
    upper = np.searchsorted(ordered, inverse, side="right") / len(data)
    inverse_error = np.maximum(np.maximum(lower - probabilities, probabilities - upper), 0)
    tail = (truth <= 0.01) | (truth >= 0.99)
    return {
        "cdf_max_rank_error": float(cdf_error.max()),
        "cdf_mean_rank_error": float(cdf_error.mean()),
        "cdf_tail_max_rank_error": float(cdf_error[tail].max()) if tail.any() else None,
        "inverse_max_rank_error": float(inverse_error.max()),
        "inverse_mean_rank_error": float(inverse_error.mean()),
        "finite_queries": bool(np.isfinite(predicted).all() and np.isfinite(inverse).all()),
        "monotone_cdf": bool(np.all(np.diff(predicted) >= -1e-15)),
        "monotone_inverse": bool(np.all(np.diff(inverse) >= 0)),
        "count": digest.count,
        "serialized_bytes": len(digest.serialize()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compression", type=int, default=100)
    parser.add_argument("--output", default=str(Path(__file__).with_name("results.json")))
    args = parser.parse_args()
    rng = np.random.default_rng(20260921)
    data = pa.array(rng.normal(size=args.total))
    result = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "datasketches": importlib.metadata.version("datasketches"),
        },
        "parameters": vars(args),
        "correctness": {},
        "stream": {},
        "merge": {},
        "query": {},
        "accuracy": {},
    }

    def save():
        Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    for name, cls in BACKENDS.items():
        result["correctness"][name] = correctness(cls, args.compression)
        print("correctness", name, result["correctness"][name], flush=True)
        warm = cls(args.compression)
        warm.update(data.slice(0, 1000))
        warm.cdf(data.slice(0, 1000))
        warm.quantile(pa.array(np.linspace(0, 1, 1000)))
    save()
    for batch_size in [128, 256, 1000, 10_000, 100_000]:
        samples = {name: [] for name in BACKENDS}
        for repeat in range(args.repeats):
            order = list(BACKENDS) if repeat % 2 == 0 else list(reversed(BACKENDS))
            for name in order:
                gc.collect()
                measurement = stream(BACKENDS[name], data, batch_size, args.compression)
                samples[name].append(measurement)
        result["stream"][str(batch_size)] = {
            name: {
                "update": summary([r[0] for r in rows]),
                "cdf": summary([r[1] for r in rows]),
                "combined": summary([r[0] + r[1] for r in rows]),
                "serialized_bytes": [r[2] for r in rows],
            }
            for name, rows in samples.items()
        }
        print(
            "stream",
            batch_size,
            {name: row["combined"]["median_s"] for name, row in result["stream"][str(batch_size)].items()},
            flush=True,
        )
        save()
    samples = {name: [] for name in BACKENDS}
    for repeat in range(args.repeats):
        for name in list(BACKENDS) if repeat % 2 == 0 else list(reversed(BACKENDS)):
            samples[name].append(merged(BACKENDS[name], data, args.compression))
    result["merge"] = {
        name: {
            "build_shards": summary([r[0] for r in rows]),
            "merge_only": summary([r[1] for r in rows]),
            "serialized_bytes": [r[2] for r in rows],
        }
        for name, rows in samples.items()
    }
    print("merge", {name: row["merge_only"]["median_s"] for name, row in result["merge"].items()}, flush=True)
    save()
    for name, cls in BACKENDS.items():
        digest = cls(args.compression)
        digest.update(data)
        result["query"][name] = {}
        for size in [1000, 10_000, 100_000]:
            values = pa.array(np.random.default_rng(923).normal(size=size))
            probabilities = pa.array(np.random.default_rng(924).uniform(size=size))
            cdf_samples, inverse_samples = [], []
            for repeat in range(args.repeats):
                elapsed, _ = timed(lambda: digest.cdf(values))
                cdf_samples.append(elapsed)
                elapsed, _ = timed(lambda: digest.quantile(probabilities))
                inverse_samples.append(elapsed)
            result["query"][name][str(size)] = {"cdf": summary(cdf_samples), "inverse": summary(inverse_samples)}
        print("query", name, result["query"][name]["100000"], flush=True)
    save()
    probabilities = np.unique(
        np.concatenate([np.linspace(0, 1, 1001), np.logspace(-6, -2, 101), 1 - np.logspace(-6, -2, 101)])
    )
    generators = {
        "uniform": lambda random: random.uniform(size=args.total),
        "normal": lambda random: random.normal(size=args.total),
        "lognormal": lambda random: random.lognormal(0, 2, size=args.total),
        "duplicates": lambda random: random.choice([0.0, 1.0, 2.0, 10.0], size=args.total, p=[0.7, 0.2, 0.09, 0.01]),
        "extremes": lambda random: (
            random.choice([-1.0, 1.0], size=args.total) * np.power(10.0, random.uniform(-150, 150, size=args.total))
        ),
    }
    for distribution, generator in generators.items():
        values = generator(np.random.default_rng(19283))
        result["accuracy"][distribution] = {}
        for name, cls in BACKENDS.items():
            measures = [errors(cls, values, probabilities, args.compression) for repeat in range(args.repeats)]
            result["accuracy"][distribution][name] = {
                "median": {
                    key: statistics.median([measure[key] for measure in measures])
                    if isinstance(measures[0][key], (float, int)) and not isinstance(measures[0][key], bool)
                    else measures[0][key]
                    for key in measures[0]
                },
                "samples": measures,
            }
        print(
            "accuracy",
            distribution,
            {name: rows["median"]["cdf_max_rank_error"] for name, rows in result["accuracy"][distribution].items()},
            flush=True,
        )
        save()
    result["source_lines"] = {
        name: len(Path(inspect.getfile(cls)).read_text().splitlines()) for name, cls in BACKENDS.items()
    }
    save()
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
