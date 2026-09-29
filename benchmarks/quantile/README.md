# Quantile backend comparison

RelFlow uses the Arrow-backed digest. The comparison found faster native updates in DataSketches 5.2.0, especially for small batches, but its inverse quantiles decreased as the requested percentile increased on ordinary uniform data. The Arrow implementation preserved monotonicity, has vectorized queries, and needs no additional dependency. It is a custom weighted-centroid digest with Arrow state and NumPy computation; it does not obtain mergeable state from `pyarrow.compute.tdigest`.

Run from the repository root:

```bash
uv run --with datasketches==5.2.0 python benchmarks/quantile/benchmark.py
```

DataSketches is an optional benchmark dependency. It is not a RelFlow dependency. The benchmark imports the production Arrow digest, uses the same Arrow inputs for both implementations, and writes all samples and checks to [results.json](results.json). `--total`, `--repeats`, `--compression`, and `--output` can change the run. A short execution check is:

```bash
uv run --with datasketches==5.2.0 python benchmarks/quantile/benchmark.py --total 20000 --repeats 1 --output /tmp/quantile-smoke.json
```

The recorded run used Python 3.12.10, macOS ARM64, PyArrow 24.0.0, NumPy 2.4.6, and DataSketches 5.2.0. It used three repetitions, fixed seeds, one million observations, and compression 100. Timed operations include Arrow conversion, copies, sorting, and result construction inside the adapters; input generation and exact reference sorting are excluded. Backend execution order alternates across streaming and merge repetitions. These are local CPU measurements, not training throughput measurements.

| Normal stream batch size | Arrow update + CDF, seconds | DataSketches update + CDF, seconds |
| ---: | ---: | ---: |
| 128 | 2.852 | 0.282 |
| 256 | 1.487 | 0.213 |
| 1,000 | 0.456 | 0.170 |
| 10,000 | 0.175 | 0.164 |
| 100,000 | 0.160 | 0.174 |

Each row processes one million values and queries each batch after adding it. Arrow costs about 0.37 ms per batch at sizes 128–256. DataSketches buffers native updates, giving it a substantial advantage at these sizes. Arrow becomes competitive for larger batches.

| Operation | Arrow, milliseconds | DataSketches, milliseconds |
| --- | ---: | ---: |
| CDF of 100,000 values | 6.76 | 13.34 |
| Inverse of 100,000 probabilities | 3.89 | 18.05 |
| Merge 32 prebuilt shards, one million total values | 7.64 | 0.37 |

The released DataSketches API supplies bulk updates and sorted CDF queries, but only scalar inverse queries. Its adapter sorts unique CDF values and restores their original positions; inverse calls loop at the external library boundary. Bulk update requires a writable NumPy array in this wheel, so the adapter copies Arrow input. [Official Python API](https://apache.github.io/datasketches-python/5.2.0/quantiles/tdigest.html)

Compression parameters are not equivalent memory or accuracy budgets. At compression 100 the normal stream in batches of 10,000 retains roughly 55 Arrow centroids versus 158 native centroids. Serialized state is 1,456 versus 2,560 bytes, including each format's headers. The tested Arrow implementation occupied 207 source lines; the DataSketches adapter occupied 72 lines and added a compiled dependency. Source lines include comments and blank lines and do not measure algorithmic maintenance cost.

| Distribution | Arrow maximum CDF rank error | DataSketches maximum CDF rank error |
| --- | ---: | ---: |
| Uniform [0, 1] | 0.000524 | 0.001227 |
| Standard normal | 0.004221 | 0.001176 |
| Lognormal, log standard deviation 2 | 0.001588 | 0.002563 |
| Four repeated values, 70% at zero | 0 | 0.026061 |
| Random signs and magnitudes from 10^-150 to 10^150 | 0.335334 | 0.381846 |

These are empirical results on fixed samples, not error bounds. Exact CDF references use the midpoint of each observed value's probability mass. Inverse error measures distance from the requested probability to the returned value's empirical rank interval, so a tied value's whole probability interval is valid. Probes combine a uniform percentile grid with logarithmically spaced tail probabilities. No parameter was tuned against these results. Both digests perform poorly across the 300-decade stress distribution; finite output alone does not ensure useful percentile accuracy. Apache also documents t-digest's distribution-dependent accuracy and lack of formal error guarantees. [Official t-digest overview](https://datasketches.apache.org/docs/tdigest/tdigest.html)

Both implementations passed count, missing-value, empty, constant, endpoint, merge-mass, source-merge preservation, and serialized round-trip checks. Arrow CDF and inverse queries were monotone on all five distributions. DataSketches inverse queries were nonmonotone on all five; `results.json` records those failures rather than treating them as successful accuracy checks. On repeated finite values `[-1e308, 0, 1e308]`, Arrow remained finite and preserved the point masses; DataSketches returned 76 nonfinite inverse values among 101 probes.

The ordinary-data failure reproduces without this adapter:

```python
import datasketches
import numpy as np

sketch = datasketches.tdigest_double(100)
sketch.update(np.random.default_rng(19283).uniform(size=1_000_000))
values = np.array([sketch.get_quantile(float(p)) for p in np.linspace(0, 1, 1001)])
print(np.count_nonzero(np.diff(values) < 0))  # 921
print(sketch.get_quantile(0.48), sketch.get_quantile(0.50))
# 0.50659424545968, 0.48674654204327356
```

The tagged C++ source passes interpolation weights in an order consistent with this observed reversal; that is an inference from source inspection, while the failing values above were directly measured. Adopting this release would require an inverse workaround or maintaining a patched native dependency. [DataSketches C++ 5.2.0 implementation](https://github.com/apache/datasketches-cpp/blob/5.2.0/tdigest/include/tdigest_impl.hpp#L137-L181)
