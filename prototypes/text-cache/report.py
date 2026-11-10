"""Render saved native-cache receipts; this never imports relflow or trains."""

import argparse
import json
import statistics
from pathlib import Path


def report(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(args.receipts.glob("*-r*.json"))
    runs = [dict(run=path.stem, **json.loads(path.read_text())) for path in files]
    assert len(runs) == 36
    assert all(run["timed_encoder_calls"] == 0 for run in runs)
    assert len({run["observations_sha256"] for run in runs}) == 1
    summary = {}
    for backend in ("python", "owner", "full"):
        for profile, workers in (("gpu", 0), ("gpu", 2), ("cpu-128", 0), ("cpu-8", 0)):
            label = f"{backend}-{profile}-w{workers}"
            values = [run for run in runs if run["run"].rsplit("-r", 1)[0] == label]
            assert len(values) == 3
            speeds = [value["rows_per_second"] for value in values]
            summary[label] = dict(
                rows_per_second=statistics.median(speeds),
                min_rows_per_second=min(speeds),
                max_rows_per_second=max(speeds),
                stdev_rows_per_second=statistics.stdev(speeds),
                tree_pss_mib=statistics.median(value["peak_tree_pss_mib"] for value in values),
                cpu_cores=statistics.median(value["average_cpu_cores"] for value in values),
            )
    micro = json.loads((args.receipts / "micro.json").read_text())
    validation = json.loads((args.receipts / "validation.json").read_text())
    build = json.loads((args.receipts / "build-info.json").read_text())
    manifest = json.loads((args.receipts / "manifest.json").read_text())
    profiles = {
        name: json.loads((args.receipts / f"profile-{name}/operators.json").read_text()) for name in ("python", "full")
    }
    identity = json.loads((args.receipts / "identity-check.json").read_text())
    result = dict(
        date="2026-11-10",
        baseline="PR #94 at 132cb118a6029c923482c0b2c3aa22a6f593fb9a, with the Arrow-boundary/benchmark selector changes captured in the source manifest",
        summary=summary,
        raw_runs=runs,
        micro=micro,
        validation=validation,
        build=build,
        manifest=manifest,
        profiles=profiles,
        identity_check=identity,
        scope="Warm user-badge text-only data loop with the same Torch GPU table and tiny backward head. Median of three separate process runs, randomly interleaved. Disk pages OS-cache warm. Native SQLite 3.51.1 versus Python SQLite 3.46.0; compares complete implementations rather than isolating programming language alone.",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    rows = []
    for profile, workers in (("gpu", 0), ("gpu", 2), ("cpu-128", 0), ("cpu-8", 0)):
        keys = [f"{backend}-{profile}-w{workers}" for backend in ("python", "owner", "full")]
        speeds = [summary[key]["rows_per_second"] for key in keys]
        gain = 100 * (speeds[2] / speeds[0] - 1)
        label = {
            "gpu": "GPU cache",
            "cpu-128": "128 MiB shared RAM, GPU cache off",
            "cpu-8": "8 MiB shared RAM + disk, GPU cache off",
        }[profile]
        rows.append(f"| {label} | {workers} | {speeds[0]:,.0f} | {speeds[1]:,.0f} | {speeds[2]:,.0f} | {gain:+.1f}% |")
    native = summary["full-gpu-w0"]
    python = summary["python-gpu-w2"]
    saving = 100 * (1 - native["tree_pss_mib"] / python["tree_pss_mib"])
    micro_base = next(r for r in micro["results"] if r["backend"] == "python" and r["batch_size"] == 1024)
    micro_native = next(r for r in micro["results"] if r["backend"] == "full" and r["batch_size"] == 1024)
    readme = f"""# Native shared text-cache prototype

Built and tested on `lab`: a standalone Rust cache owner plus an Arrow-aware
PyO3 client, packaged with Maturin. The native client improves isolated text-to-ID
lookup by **{micro_native["texts_per_second"] / micro_base["texts_per_second"]:.1f} times** at batch size 1,024. Complete warm-loop gains are smaller:
**about 10% with no workers, 8% for shared-RAM vectors, and 22% with an 8 MiB
RAM budget and disk fallback**. The two-worker GPU-cache case is essentially
unchanged within the observed run spread.

![Native cache throughput and resource tradeoffs](results/comparison.svg)

The baseline here is the automatic Python/Moka cache from PR #94, rather than
the earlier NPY store. These are user snapshots per second:

| Profile | Workers | Python owner/client | Rust owner, Python client | Rust owner, native client | Full native gain |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Each cell is the median of three separate process runs, randomly interleaved
with seed 7302. The plot's whiskers show the observed minimum and maximum,
not confidence intervals. Every run completed at least two epochs and five
seconds of timing after a full warm-up epoch. The comparison executed 36 runs.

The practical resource result is **{native["rows_per_second"]:,.0f} snapshots/s with no native-path workers**
at **{native["tree_pss_mib"] / 1024:.2f} GiB process-tree PSS**, comparable to the Python path's
**{python["rows_per_second"]:,.0f}/s with two workers** at **{python["tree_pss_mib"] / 1024:.2f} GiB**:
about **{saving:.0f}% less process memory**. Worker count changes batch padding and
layout; exact parity checks therefore compare matching worker counts.

## What runs in Rust

- `src/store.rs`: byte-weighted Moka admission, stable IDs, canonical token
  rows, SQLite lookups/transactions, in-flight encoding claims, and counters.
  The wire protocol, SQLite layout, logical cache weights and page budget match
  the current Python implementation. It holds no corpus-sized in-memory index.
- `src/bin/owner.rs`: the authenticated local cache process and client lifecycle.
- `src/python.rs`: a batch PyO3 client. Arrow C Data imports provide direct
  access to UTF-8/offset buffers. Hashing and socket work release the GIL.
  The hot path does not materialize Python strings or call Rust once per item.
- `python/relflow_text_native/adapter.py`: an experimental `text_native`
  extension that reuses the existing Text tensorfield, masks, geometry,
  projections, encoder, and Torch device cache. Python strings are created
  only for tokenizer misses. This uses extension registration, with no runtime
  function replacements in the throughput loop.

The small package change passes Arrow values to the encoding context; the
existing Python context still performs its previous conversion. The benchmark
adds `--native-backend python|owner|full`. The owner-only mode is an ablation
that keeps Python hashing and transport.

Vectors still travel as bytes over local TCP on CPU-tier misses. This is not
shared-memory IPC. Frozen-encoder execution, Python configuration, and the GPU
lookup/projection path remain in Torch/Python. Native packaging is isolated
from relflow's existing build system.

## Why a faster cache does not yield a proportional loop speedup

The isolated warm text-to-ID test used real `user-badge` strings and checked
that no tokenizer ran. At 1,024 strings per call, Python delivered
**{micro_base["texts_per_second"] / 1e6:.2f} million text IDs/s**, versus **{micro_native["texts_per_second"] / 1e6:.2f} million/s** for full native.
That microbenchmark is one three-second run per backend/batch-size combination.

A separate eight-batch CPU profile left `text.resolve` at about **75 ms** for
Python and **74 ms** for full native. Both recorded **3,249 CUDA kernel launches**,
**726 stream synchronization calls**, and **363 `aten::nonzero` calls**. The native
cache does not remove this Torch work or the remaining loader work. A separate
warm encoder-fingerprint check took about **{identity["p50_us"]:.0f} microseconds per call**, excluding
the initial weight hash. These diagnostics point to metadata reuse, batch
preparation, and device lookup/indexing as further targets; they do not establish
a speedup for those unimplemented optimizations.

For warm two-worker GPU throughput alone, these results do not justify replacing
the current service wholesale. The native approach is more useful when avoiding
workers or serving vectors through the CPU/disk tiers. The prototype keeps that
option measurable without changing the production default.

## Correctness and scope

- **9 native integration tests passed**: Arrow UTF-8/large-string/string-view
  slices and Unicode, cold fills, inference-to-training gradients, precision
  namespaces, pickle/deepcopy, disconnect recovery, durable reopen, malformed
  requests, spawn/fork workers, and real CPU/CUDA BERT parity.
- **2 Rust tests passed**, including a persisted 1,000-vector corpus under a
  2 KiB hot budget, canonical deduplication, transaction rollback, and claims.
- Full-corpus validation matched **58,167 string/length IDs** and **58,089
  vectors byte-for-byte**. The first populated per-field batches matched exactly
  across all **36 runs**, using references with the same worker count.
- The existing Python Text tests passed **77 tests**; the core suite passed
  **2,807 tests**. Ruff and Pyrefly passed.
- All timed frozen-encoder call counts were zero. These are text-only Arrow
  loader/lookup loops with a tiny backward head, not full model training or
  predictive-quality measurements. The same 8,192-snapshot fixture contains
  17 text leaves, six branches, and 149,019 text occurrences, with seed 7302.
- Fixture SHA-256:
  `eb14614a0afe1af4e30688506d1eff830128c3d763a5847cf032c0e0e1f3e396`.
  Vectors alone occupy about 85 MiB. The disk-fallback pages were OS-cache warm;
  no cold-drive or multi-terabyte throughput is claimed.

PSS includes trainer, workers and owner, sampled every 250 ms. It excludes GPU
memory and the kernel file cache; CUDA allocation is recorded separately.
Budgets limit charged hot state rather than whole-process RSS. Disk quota and
pruning are not implemented. The native engine uses bundled SQLite **3.51.1**;
Python uses SQLite **3.46.0**, so this compares complete implementations rather
than isolating the language change alone.

The host has 32 CPUs, 125 GiB RAM and an RTX 3090. Python 3.12.6, Torch 2.12.0+cu130,
Transformers 5.9.0, Polars 2.1.0, PyArrow 24.0.0 and NumPy 2.4.6 were used.
The native build used Rust 1.99.1, PyO3 0.29.3, Arrow 60.1.0, Moka 0.12.16,
rusqlite 0.38.0, and Maturin 1.16.0. Only Linux/Python 3.12 was exercised.

[Raw runs, counters and provenance](results/results.json),
[build hashes](results/build-info.json), [measured source snapshot](results/measured-source.tar.gz),
and [operator profile](results/profile-native.txt) are retained. The source
snapshot preserves the measured files before Rust formatting; the executable
and wheel checksums identify the exact binaries. Full traces and executable
artifacts remain at `/tmp/relflow-native-text-20261110` on the lab.

## Build and replay

Start from a relflow environment with `relflow[text]`, CUDA, cached MiniLM
assets, and the saved `user-badge` fixture. Install Rust and Maturin, then:

```bash
export NATIVE_PYTHON="$PWD/.venv/bin/python"
cd prototypes/text-cache
bash build-lab.sh
uv pip install --python "$NATIVE_PYTHON" --no-deps --target /tmp/native-text-deps wheels/*.whl
cd ../..
export RELFLOW_NATIVE_OWNER="$PWD/prototypes/text-cache/target/release/text-cache-owner"
export PYTHONPATH="$PWD/src:/tmp/native-text-deps"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false POLARS_MAX_THREADS=2
```

The wheel contains the client/adapter; the standalone executable is built
separately and supplied through `RELFLOW_NATIVE_OWNER`. This is prototype
packaging, not a published cross-platform dependency.

```bash
.venv/bin/python -m pytest prototypes/text-cache/tests -q
.venv/bin/python scripts/benchmark_text_cache.py \\
  --input /tmp/user-badge-input --store /tmp/native-text-store \\
  --native-backend full --workers 0 --output /tmp/native-text.json --save-vectors
```

For the full paired sweep, first populate a Python cache on that same fixture,
then clone its database for every profile. The driver does those backups and
runs the same benchmark script in all modes:

```bash
.venv/bin/python prototypes/text-cache/sweep.py \\
  --source "$PWD" --input /tmp/user-badge-input \\
  --database /tmp/python-text-store/cache.sqlite3 --output /tmp/native-text-comparison
```

The workload can be regenerated using
`scripts/benchmarks/text-cache/prepare.py`. Install `psutil` for the benchmark
harness. `micro.py`, `profile_loop.py`, and `validate.py` accept the same saved
input and database; their `--help` documents the additional output arguments.
"""
    (args.output.parent / "README.md").write_text(readme)
    colors = ["#747980", "#088779", "#306cc0"]
    labels = ["Python owner/client", "Rust owner", "Rust owner + native client"]
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    profiles_order = [("gpu", 0), ("gpu", 2), ("cpu-128", 0), ("cpu-8", 0)]
    for index, backend in enumerate(("python", "owner", "full")):
        values = [summary[f"{backend}-{p}-w{w}"] for p, w in profiles_order]
        x = [group + (index - 1) * 0.24 for group in range(4)]
        y = [v["rows_per_second"] for v in values]
        err = [
            [v["rows_per_second"] - v["min_rows_per_second"] for v in values],
            [v["max_rows_per_second"] - v["rows_per_second"] for v in values],
        ]
        axes[0].bar(x, y, width=0.22, color=colors[index], label=labels[index], yerr=err, capsize=2)
        xmem, ymem = [], []
        for workers in (0, 2):
            point = summary[f"{backend}-gpu-w{workers}"]
            xmem.append(point["tree_pss_mib"] / 1024)
            ymem.append(point["rows_per_second"])
            axes[1].annotate(
                str(workers),
                (xmem[-1], ymem[-1]),
                xytext=(4, 5 if index != 1 else -12),
                textcoords="offset points",
                color=colors[index],
            )
        axes[1].plot(xmem, ymem, "o-", color=colors[index], alpha=0.85)
        micro_values = [
            next(r for r in micro["results"] if r["backend"] == backend and r["batch_size"] == n)["texts_per_second"]
            / 1e6
            for n in (128, 1024, 4096)
        ]
        axes[2].bar([group + (index - 1) * 0.24 for group in range(3)], micro_values, width=0.22, color=colors[index])
    axes[0].set(
        xticks=range(4),
        xticklabels=["GPU cache\n0 workers", "GPU cache\n2 workers", "128 MiB RAM\nGPU off", "8 MiB + disk\nGPU off"],
        ylabel="User snapshots / second",
        ylim=(0, 4000),
        title="Complete warm loop: median and range",
    )
    axes[1].set(
        xlabel="Process-tree PSS (GiB)",
        ylabel="User snapshots / second",
        title="GPU cache: labels show worker count",
        ylim=(2900, 3550),
        xlim=(1.4, 3.2),
    )
    axes[2].set(
        xticks=range(3),
        xticklabels=["128", "1,024", "4,096"],
        xlabel="Strings per native/client call",
        ylabel="Million text IDs / second",
        title="Isolated warm text-to-ID operation",
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(labels, loc="outside upper center", ncols=3)
    fig.savefig(args.output / "comparison.svg")
    fig.savefig(args.receipts / "comparison.png", dpi=140)
    print("Wrote", args.output / "results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    report(parser.parse_args())
