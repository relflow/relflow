"""Measure drift profiling overhead against identical models with profiling disabled.

This is a steady-state, in-process cost benchmark, not learning evidence.
Input generation, statistics warmup, model placement, and compilation warmup are
outside timing. Every measured pair uses the same model weights and Arrow data.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import io
import json
import logging
import math
import os
import pickle
import platform
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

import relflow as rf
from relflow.data.iterables import encode
from relflow.data.ragged import coalesce
from relflow.distributions import Distributions

SEED = 22001
SIZES = {"numbers": (64, 256, 1024), "identities": (64, 256, 1024), "nested": (32, 128, 512)}
KINDS = ("Number", "Quantile", "Boolean", "Category", "Enum", "Hash", "Cluster", "Set", "Vector", "Period", "Text")


def records(workload, rows, seed):
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        if workload == "numbers":
            row = {
                f"number{i}": None if rng.random() < 0.05 else float(rng.normal(1000 + 100 * i, 20)) for i in range(8)
            }
        elif workload == "identities":
            row = {}
            for i in range(2):
                row[f"category{i}"] = None if rng.random() < 0.05 else f"label-{rng.integers(64)}"
                row[f"enum{i}"] = None if rng.random() < 0.05 else f"class-{rng.integers(16)}"
                row[f"hash{i}"] = None if rng.random() < 0.05 else 2**60 + int(rng.integers(128))
                row[f"cluster{i}"] = None if rng.random() < 0.05 else f"label-{rng.integers(64)}"
            row["flag"] = bool(rng.integers(2))
        else:
            row = {
                "amount": None if rng.random() < 0.05 else float(rng.lognormal(3, 1)),
                "merchant": f"merchant-{rng.integers(64)}",
                "country": f"class-{rng.integers(8)}",
                "customer": 2**60 + int(rng.integers(128)),
                "flag": bool(rng.integers(2)),
                "tags": [f"tag-{x}" for x in rng.choice(32, int(rng.integers(6)), replace=False)],
                "vector": rng.normal(size=32).tolist(),
                "happened": datetime(2026, 1, 1) + timedelta(days=int(rng.integers(365))),
                "events": [
                    {
                        "amount": None if rng.random() < 0.05 else float(rng.normal(100, 10)),
                        "merchant": f"merchant-{rng.integers(64)}",
                        "items": [
                            {"sku": f"sku-{rng.integers(64)}", "quantity": float(rng.integers(1, 11))}
                            for _ in range(int(rng.integers(5)))
                        ],
                    }
                    for _ in range(int(rng.integers(17)))
                ],
            }
        row["target"] = bool(rng.integers(2))
        yield row


def build(workload, enabled):
    if workload == "numbers":
        fields = {f"number{i}": rf.Number(drift=True) for i in range(8)}
    elif workload == "identities":
        fields = {}
        for i in range(2):
            fields[f"category{i}"] = rf.Category(drift=True)
            fields[f"enum{i}"] = rf.Enum(values=tuple(f"class-{j}" for j in range(16)), drift=True)
            fields[f"hash{i}"] = rf.Hash(n_hashes=2, drift=True)
            fields[f"cluster{i}"] = rf.Cluster(bounds=4, revive_temperature=0, drift=True)
        fields["flag"] = rf.Boolean(drift=True)
    else:
        fields = {
            "amount": rf.Number(drift=True),
            "merchant": rf.Category(drift=True),
            "country": rf.Enum(values=tuple(f"class-{i}" for i in range(8)), drift=True),
            "customer": rf.Hash(n_hashes=2, drift=True),
            "flag": rf.Boolean(drift=True),
            "tags": rf.Set(drift=True),
            "vector": rf.Vector(n_dim=32, drift=True),
            "happened": rf.Period("month_of_year", "day_of_week", drift=True),
            "events": rf.Branch(
                length=16,
                n_layers=1,
                n_heads=4,
                amount=rf.Number(drift=True),
                merchant=rf.Category(drift=True),
                items=rf.Branch(
                    length=4, n_layers=1, n_heads=4, sku=rf.Category(drift=True), quantity=rf.Number(drift=True)
                ),
            ),
        }
    fields["target"] = rf.Boolean(mask=True, drift=True)
    return rf.Model(d_model=64, n_layers=2, n_heads=4, fields=fields, drift=enabled)


def snapshot(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }


def same(left, right):
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def synchronize(cuda):
    if cuda:
        torch.cuda.synchronize()


def elapsed(operation, iterations, cuda=False):
    gc.collect()
    synchronize(cuda)
    baseline = torch.cuda.memory_allocated() if cuda else 0
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    gc.disable()
    cpu = time.process_time()
    started = time.perf_counter()
    try:
        for _ in range(iterations):
            operation()
        synchronize(cuda)
    finally:
        seconds = time.perf_counter() - started
        cpu_seconds = time.process_time() - cpu
        gc.enable()
    return {
        "wall_ms": seconds * 1000 / iterations,
        "cpu_ms": cpu_seconds * 1000 / iterations,
        "gpu_peak_delta_bytes": torch.cuda.max_memory_allocated() - baseline if cuda else None,
    }


def paired(operations, *, repetitions, target_ms, warmup, rows, seed, cuda=False):
    for operation in operations.values():
        for _ in range(warmup):
            operation()
    calibration = {name: elapsed(operation, 2, cuda)["wall_ms"] for name, operation in operations.items()}
    iterations = max(1, min(64, math.ceil(target_ms / max(calibration.values()))))
    rng = random.Random(seed)
    samples = []
    for repetition in range(repetitions):
        order = list(operations)
        rng.shuffle(order)
        measured = {name: elapsed(operations[name], iterations, cuda) for name in order}
        off, on = measured["off"]["wall_ms"], measured["on"]["wall_ms"]
        samples.append(
            {
                "repetition": repetition,
                "order": order,
                "iterations": iterations,
                "off": measured["off"],
                "on": measured["on"],
                "ratio": on / off,
                "extra_ms": on - off,
            }
        )
    ratios = np.array([sample["ratio"] for sample in samples])
    return {
        "iterations": iterations,
        "calibration_ms": calibration,
        "samples": samples,
        "off_ms": float(np.median([sample["off"]["wall_ms"] for sample in samples])),
        "on_ms": float(np.median([sample["on"]["wall_ms"] for sample in samples])),
        "extra_ms": float(np.median([sample["extra_ms"] for sample in samples])),
        "overhead_percent": float((np.median(ratios) - 1) * 100),
        "paired_ratio_median": float(np.median(ratios)),
        "paired_ratio_min": float(ratios.min()),
        "paired_ratio_max": float(ratios.max()),
        "rows_per_batch": rows,
    }


class Operation:
    """Cycle the same finite Arrow pool through one specified runtime phase."""

    def __init__(self, model, batches, encoded, phase):
        self.model, self.batches, self.encoded, self.phase = model, batches, encoded, phase
        self.ordinal = 0
        self.policies = {address: request.drift for address, request in model.schema.active_requests.items()}

    def __call__(self):
        index = self.ordinal % len(self.batches)
        self.ordinal += 1
        if self.phase == "worker_encode":
            return encode(
                self.batches[index],
                self.model.schema,
                rf.Strata.predict,
                self.model.interprocess_encoding_context,
                drift=self.model.drift,
            )
        if self.phase == "public_encode":
            return self.model.encode(self.batches[index])
        if self.phase == "public_train_encode":
            return self.model.encode(self.batches[index], strata="train")
        if self.phase.startswith("predict_"):
            return self.model.predict(self.batches[index])
        return self.model.distributions.update(
            self.encoded[index].profiles,
            strata=rf.Strata.train if self.phase == "commit_train_three_strata" else rf.Strata.predict,
            policies=self.policies,
        )


def storage(model, encoded):
    buffers = 0
    centroids = 0
    features = 0
    for strata in model.distributions.profiles.values():
        for profile in strata.values():
            for feature in profile.values():
                features += 1
                if feature.digest is not None:
                    buffers += feature.digest.centroids.nbytes
                    centroids = max(centroids, feature.digest.centroids.num_rows)
                else:
                    buffers += feature.counts.nbytes
    stream = io.BytesIO()
    torch.save(model.distributions.state_dict(), stream)
    return {
        "feature_profiles": features,
        "retained_buffer_bytes": buffers,
        "max_numeric_centroids": centroids,
        "checkpoint_metadata_bytes": stream.tell(),
        "batch_profile_pickle_bytes": len(pickle.dumps(encoded.profiles, protocol=5)),
        "encoded_batch_pickle_bytes": len(pickle.dumps(encoded, protocol=5)),
    }


def pipeline(workload, size, options, output):
    seed = SEED + ("numbers", "identities", "nested").index(workload) * 1000 + size
    source = pa.Table.from_pylist(list(records(workload, size * options.pool_batches, seed)))
    train = [source.take(np.arange(i * size, (i + 1) * size)) for i in range(options.pool_batches)]
    inputs = [table.drop(["target"]) for table in train]
    models = {}
    for enabled in (False, True):
        torch.manual_seed(SEED)
        model = build(workload, enabled)
        for _ in range(2):
            for table in train:
                model.encode(table, strata="train")
        model.eval()
        models["on" if enabled else "off"] = model
    assert same(snapshot(models["off"]), snapshot(models["on"])), "model tensors differ before comparison"
    encoded = {
        name: [
            encode(table, model.schema, rf.Strata.predict, model.interprocess_encoding_context, drift=model.drift)
            for table in inputs
        ]
        for name, model in models.items()
    }
    assert models["off"].predict(inputs[0]).equals(models["on"].predict(inputs[0]), check_metadata=True), (
        "prediction outputs differ"
    )
    assert not models["on"].distributions.warned, "same-population warmup unexpectedly warned"
    baseline = {name: model.distributions.state_dict() for name, model in models.items()}
    rows = []
    phases = (
        "worker_encode",
        "commit_predict",
        "commit_train_three_strata",
        "public_encode",
        "public_train_encode",
        "predict_cpu",
    )
    if options.cuda and torch.cuda.is_available():
        phases += ("predict_cuda",)
    if options.smoke:
        phases = ("worker_encode", "commit_predict", "public_encode", "public_train_encode", "predict_cpu")
        if options.cuda and torch.cuda.is_available():
            phases += ("predict_cuda",)
    for phase in phases:
        for name, model in models.items():
            model.distributions = Distributions.from_state(baseline[name])
            if phase == "commit_train_three_strata":
                policies = {address: request.drift for address, request in model.schema.active_requests.items()}
                for split in (rf.Strata.validate, rf.Strata.test, rf.Strata.predict):
                    for batch in encoded[name]:
                        model.distributions.update(batch.profiles, strata=split, policies=policies)
        if phase == "predict_cuda":
            for model in models.values():
                model.to("cuda")
            assert models["off"].predict(inputs[0]).equals(models["on"].predict(inputs[0]), check_metadata=True), (
                "CUDA prediction outputs differ"
            )
        batches = train if phase == "public_train_encode" else inputs
        operations = {name: Operation(model, batches, encoded[name], phase) for name, model in models.items()}
        result = paired(
            operations,
            repetitions=options.repetitions,
            target_ms=options.target_ms,
            warmup=options.warmup,
            rows=size,
            seed=seed,
            cuda=phase == "predict_cuda",
        )
        assert same(snapshot(models["off"]), snapshot(models["on"])), "model tensors diverged during paired timing"
        assert not models["on"].distributions.warned, "quiet benchmark unexpectedly warned"
        measured = {
            "workload": workload,
            "phase": phase,
            "data_seed": seed,
            "pool_batches": options.pool_batches,
            "data_rows": source.num_rows,
            "source_arrow_bytes": source.nbytes,
            "batch_arrow_bytes": inputs[0].nbytes,
            "schema": str(inputs[0].schema),
            "active_fields": len(models["on"].schema.active_requests) - 1,
            "valued_leaf_slots_per_batch": sum(
                feature["state"].count - int(feature["state"].counts[rf.Tokens.null])
                for feature in encoded["on"][0].profiles.values()
            ),
            "parameters": sum(parameter.numel() for parameter in models["on"].parameters()),
            "storage": {name: storage(model, encoded[name][0]) for name, model in models.items()},
            **result,
        }
        rows.append(measured)
        with (output / "cases.jsonl").open("a") as handle:
            handle.write(json.dumps(measured) + "\n")
        print(
            json.dumps(
                {
                    key: measured[key]
                    for key in ("workload", "phase", "rows_per_batch", "off_ms", "on_ms", "overhead_percent")
                }
            ),
            flush=True,
        )
    # These comparisons warm statistics but perform no optimizer update.
    assert same(snapshot(models["off"]), snapshot(models["on"]))
    memory = []
    on = models["on"]
    on.distributions = Distributions.from_state(baseline["on"])
    policies = {address: request.drift for address, request in on.schema.active_requests.items()}
    for iteration in range(256):
        on.distributions.update(
            encoded["on"][iteration % len(inputs)].profiles, strata=rf.Strata.predict, policies=policies
        )
        if iteration + 1 in (1, 16, 256):
            memory.append({"batches": iteration + 1, "rows": (iteration + 1) * size, **storage(on, encoded["on"][0])})
    del models, encoded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, {"workload": workload, "rows_per_batch": size, "growth": memory}


def atoms(kind, rows, seed):
    rng = np.random.default_rng(seed)
    if kind in ("Number", "Quantile"):
        values = rng.normal(1000, 20, rows).tolist()
        request = getattr(rf, kind)()
    elif kind == "Boolean":
        values, request = (rng.random(rows) < 0.5).tolist(), rf.Boolean()
    elif kind in ("Category", "Cluster", "Enum"):
        values = [f"label-{i}" for i in rng.integers(0, 64, rows)]
        request = (
            rf.Cluster(bounds=4)
            if kind == "Cluster"
            else rf.Enum(values=tuple(f"label-{i}" for i in range(64)))
            if kind == "Enum"
            else rf.Category()
        )
    elif kind == "Hash":
        values, request = (2**60 + rng.integers(0, 128, rows)).tolist(), rf.Hash()
    elif kind == "Set":
        values, request = [[f"tag-{v}" for v in rng.choice(32, 4, replace=False)] for _ in range(rows)], rf.Set()
    elif kind == "Vector":
        values, request = rng.normal(size=(rows, 32)).tolist(), rf.Vector(n_dim=32)
    elif kind == "Period":
        values = [datetime(2026, 1, 1) + timedelta(days=int(i)) for i in rng.integers(365, size=rows)]
        request = rf.Period("month_of_year", "day_of_week")
    else:
        vocabulary = ("Structured observations and typed features. ", "Small records with repeated histories. ")
        values, request = [vocabulary[int(i)] * 4 for i in rng.integers(2, size=rows)], rf.Text()
    values = [None if missing else value for value, missing in zip(values, rng.random(rows) < 0.05, strict=True)]
    schema = rf.Schema.from_tree(d_model=64, n_layers=2, n_heads=4, value=request)
    request = schema.requests["/value"]
    projection = coalesce(pa.table({"value": values}), schema=schema, strata=rf.Strata.predict).fields["/value"]
    return projection.pristine, request


def feature_costs(options):
    results = []
    for rows in (64,) if options.smoke else (256, 2048):
        for kind in KINDS:
            field, request = atoms(kind, rows, SEED + rows)
            component = rf.TENSORFIELDS[request.type].components[rf.Component.profile]

            def profile():
                states = field.state.to_numpy(zero_copy_only=False)
                states = states[states != rf.Tokens.padded]
                return {
                    "state": rf.Distribution.categorical(states, bins=len(rf.Tokens)),
                    **component(field, request=request),
                }

            # Fixed per-leaf computation only, without codec, encoder, or merging.
            for _ in range(options.warmup):
                profile()
            probe = elapsed(profile, 2)["wall_ms"]
            iterations = max(1, min(64, math.ceil(options.target_ms / probe)))
            samples = [elapsed(profile, iterations) for _ in range(options.repetitions)]
            results.append(
                {
                    "datatype": kind,
                    "rows": rows,
                    "samples": samples,
                    "iterations": iterations,
                    "profile_ms": float(np.median([sample["wall_ms"] for sample in samples])),
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--target-ms", type=float, default=80)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--pool-batches", type=int, default=8)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    if options.repetitions < 1 or options.target_ms <= 0 or options.warmup < 1 or options.pool_batches < 2:
        parser.error("repetitions, target duration, warmup, and pool size must be positive")
    options.output.mkdir(parents=True, exist_ok=True)
    if (options.output / "cases.jsonl").exists() or (options.output / "results.json").exists():
        parser.error("choose a new output directory; benchmark records are immutable")
    torch.set_num_threads(1)
    logging.getLogger("relflow").setLevel(logging.ERROR)
    if options.smoke:
        options.repetitions, options.target_ms, options.warmup = 1, 5, 1
    metadata = {
        "seed": SEED,
        "mode": "smoke" if options.smoke else "full",
        "repetitions": options.repetitions,
        "target_ms": options.target_ms,
        "warmup_batches": options.warmup,
        "pool_batches": options.pool_batches,
        "model": {"d_model": 64, "n_layers": 2, "n_heads": 4, "precision": "float32", "optimizer_steps": 0},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host": platform.node(),
        "packages": {
            name: importlib.metadata.version(name) for name in ("relflow", "torch", "polars", "pyarrow", "numpy")
        },
        "relflow_path": str(Path(rf.__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "thread_policy": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS")
        },
        "scope": "In-memory synthetic pools reused cyclically; no loader workers, disk/network IO, optimizer, or Text encoder; CUDA synchronized at timing boundaries.",
    }
    with (options.output / "configuration.json").open("x") as handle:
        json.dump(metadata, handle, indent=2)
    cases, memory = [], []
    for workload in ("numbers",) if options.smoke else SIZES:
        for size in (32,) if options.smoke else SIZES[workload]:
            measured, footprint = pipeline(workload, size, options, options.output)
            cases.extend(measured)
            memory.append(footprint)
    features = feature_costs(options)
    result = {"configuration": metadata, "cases": cases, "features": features, "memory": memory}
    with (options.output / "results.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({"complete": True, "cases": len(cases), "features": len(features)}), flush=True)


if __name__ == "__main__":
    main()
