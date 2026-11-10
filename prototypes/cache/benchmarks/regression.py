"""Paired user-badge regression against the archived text-specific prototype.

Both implementations run the real loader, frozen Text lookups, and a tiny
backward head. Encoder fills happen only during an excluded warm-up epoch.
The different SQLite formats are warmed independently; outputs are compared
numerically, never by IDs from different databases.
"""

import argparse
import hashlib
import json
import os
import random
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [
        (backend, profile, repeat)
        for backend in ("text", "generic")
        for profile in ("gpu", "spill")
        for repeat in range(args.repeats)
    ]
    random.Random(args.seed).shuffle(jobs)
    files = [
        *sorted((args.source / "prototypes/cache/src").rglob("*.rs")),
        *sorted((args.source / "prototypes/cache/python").rglob("*.py")),
        args.source / "prototypes/cache/benchmarks/regression.py",
        args.source / "scripts/benchmark_text_cache.py",
    ]
    fingerprints = {str(path.relative_to(args.source)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    if args.resume:
        previous = json.loads((args.output / "results.json").read_text())
        for name, fingerprint in fingerprints.items():
            if name != "prototypes/cache/benchmarks/regression.py":
                assert previous["source_sha256"][name] == fingerprint, f"runtime source changed: {name}"
        assert (previous["seed"], previous["repeats"], previous["minimum_seconds"]) == (
            args.seed,
            args.repeats,
            args.seconds,
        )
    records, preparations = [], []
    for backend, profile, repeat in jobs:
        store = args.output / "stores" / f"{backend}-{profile}"
        store.mkdir(parents=True, exist_ok=True)
        if backend == "text" and not (store / "cache.sqlite3").exists():
            with sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True) as source:
                with sqlite3.connect(store / "cache.sqlite3") as target:
                    source.backup(target)
        source_root = args.source if backend == "generic" else args.previous_source
        output = args.output / f"{backend}-{profile}-r{repeat}.json"
        command = [
            sys.executable,
            str(source_root / "scripts/benchmark_text_cache.py"),
            "--input",
            str(args.input),
            "--store",
            str(store),
            "--native-backend",
            "full",
            "--workers",
            "0",
            "--memory",
            "64" if profile == "gpu" else "8",
            "--epochs",
            "2",
            "--seconds",
            str(args.seconds),
            "--seed",
            str(args.seed),
            "--output",
            str(output),
            "--save-vectors",
        ]
        if profile == "spill":
            command += ["--resident", "0"]
        environment = dict(
            os.environ,
            HF_HUB_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            RELFLOW_CACHE_OWNER=str(args.owner),
            RELFLOW_NATIVE_OWNER=str(args.previous_owner),
        )
        if backend == "text":
            environment["PYTHONPATH"] = f"{args.previous_deps}:{source_root / 'src'}:{source_root}:" + environment.get(
                "PYTHONPATH", ""
            )
        result = json.loads(output.read_text()) if args.resume and output.exists() else None
        if result is None:
            print("START", output.stem, flush=True)
            with output.with_suffix(".log").open("w") as log:
                subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads(output.read_text())
        if result["warm_encoder_calls"]:
            # A separate preparation process avoids retaining cold-fill allocator
            # memory in only one side of a warm-cache resource comparison.
            preparation = output.with_name("prepare-" + output.name)
            for suffix in (".json", ".log", ".vectors.npz"):
                output.with_suffix(suffix).rename(preparation.with_suffix(suffix))
            preparations.append(dict(file=preparation.name, encoder_calls=result["warm_encoder_calls"]))
            print("PREPARED; REPLAY", output.stem, flush=True)
            with output.with_suffix(".log").open("w") as log:
                subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads(output.read_text())
        assert result["warm_encoder_calls"] == 0
        assert result["timed_encoder_calls"] == 0
        records.append(dict(backend=backend, profile=profile, repeat=repeat, result=result))
        print(
            "DONE",
            output.stem,
            round(result["rows_per_second"], 1),
            "rows/s",
            round(result["peak_tree_pss_mib"], 1),
            "MiB PSS",
            flush=True,
        )
    summaries = []
    max_error = 0.0
    for profile in ("gpu", "spill"):
        reference = np.load(args.output / f"text-{profile}-r0.vectors.npz")
        for backend, candidate_profile, repeat in jobs:
            if candidate_profile != profile:
                continue
            actual = np.load(args.output / f"{backend}-{profile}-r{repeat}.vectors.npz")
            assert actual.files == reference.files
            for name in actual.files:
                np.testing.assert_allclose(actual[name], reference[name], rtol=1e-5, atol=2e-6)
                if actual[name].size:
                    max_error = max(max_error, float(np.abs(actual[name] - reference[name]).max()))
        metrics = {}
        for backend in ("text", "generic"):
            results = [r["result"] for r in records if r["backend"] == backend and r["profile"] == profile]
            rates = [r["rows_per_second"] for r in results]
            metrics[backend] = dict(
                rows_per_second=float(np.median(rates)),
                min_rows_per_second=min(rates),
                max_rows_per_second=max(rates),
                pss_mib=float(np.median([r["peak_tree_pss_mib"] for r in results])),
                head_ceiling_rows_per_second=float(np.median([r["head_ceiling_rows_per_second"] for r in results])),
            )
        summaries.append(
            dict(
                profile=profile,
                metrics=metrics,
                throughput_change_pct=100
                * (metrics["generic"]["rows_per_second"] / metrics["text"]["rows_per_second"] - 1),
            )
        )
    receipt = dict(
        scope="Warm user-badge loader and tiny backward head; generic versus previous text-specific native cache; two SQLite formats warmed independently; OS disk page cache warm; Linux lab only.",
        seed=args.seed,
        repeats=args.repeats,
        minimum_seconds=args.seconds,
        order=jobs,
        preparations=preparations,
        preparation_replays_after_initial_round=args.resume,
        maximum_output_error=max_error,
        summaries=summaries,
        records=records,
        source_sha256=fingerprints,
    )
    (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "previous-source",
        "previous-deps",
        "previous-owner",
        "owner",
        "input",
        "database",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--seed", type=int, default=7302)
    parser.add_argument(
        "--resume", action="store_true", help="Reuse matching runtime receipts and replay cold preparation runs"
    )
    run(parser.parse_args())
