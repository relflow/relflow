"""Render a checksum-bound report from a completed drift-cost benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def render(root: Path):
    manifest = json.loads((root / "inputs/manifest.json").read_text())
    result = json.loads((root / "full/results.json").read_text())
    config = result["configuration"]
    if config["mode"] != "full" or config["script_sha256"] != manifest["files"]["scripts/drift_costs.py"]:
        raise ValueError("report requires a full run of the frozen benchmark script")
    output = root / "report"
    output.mkdir(exist_ok=True)
    cases = result["cases"]
    medium = {"numbers": 256, "identities": 256, "nested": 128}
    names = {"numbers": "8 Numbers", "identities": "9 identity/Boolean fields", "nested": "Nested mixed fields"}
    stages = {
        "worker_encode": "Worker encoding",
        "public_encode": "Public encode",
        "public_train_encode": "Train encode/statistics",
        "predict_cpu": "CPU prediction",
        "predict_cuda": "CUDA prediction",
        "commit_predict": "Profile merge + compare",
        "commit_train_three_strata": "Train merge + 3 strata",
    }
    selected = [case for case in cases if case["rows_per_batch"] == medium[case["workload"]]]
    columns = [
        "workload",
        "phase",
        "rows_per_batch",
        "off_ms",
        "on_ms",
        "extra_ms",
        "overhead_percent",
        "paired_ratio_median",
        "paired_ratio_min",
        "paired_ratio_max",
    ]
    with (output / "timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({name: case[name] for name in columns} for case in cases)
    summary = {
        "scope": config["scope"],
        "host": config["host"],
        "gpu": config["gpu"],
        "cpu_affinity": config["cpu_affinity"],
        "thread_policy": config["thread_policy"],
        "repetitions": config["repetitions"],
        "model": config["model"],
        "wheel_sha256": manifest["wheels"],
        "source_archive_sha256": manifest["archive_sha256"],
        "results_sha256": hashlib.sha256((root / "full/results.json").read_bytes()).hexdigest(),
        "medium_batch_cases": [{name: case[name] for name in columns} for case in selected],
        "per_datatype_profile_ms_at_2048": {
            entry["datatype"]: entry["profile_ms"] for entry in result["features"] if entry["rows"] == 2048
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Relative cost of drift detection",
        "",
        "Paired measurements on the lab with the same model tensors, same prepared Arrow pools, and equal prediction outputs. Each case uses seven randomized on/off pairs; ranges are observed paired ratios, not confidence intervals.",
        "",
        "## Medium batch sizes",
        "",
        "| Workload | Rows/batch | Stage | Off ms | On ms | Extra ms | Latency overhead |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for case in selected:
        if case["phase"] not in (
            "worker_encode",
            "public_encode",
            "public_train_encode",
            "predict_cpu",
            "predict_cuda",
        ):
            continue
        lines.append(
            f"| {names[case['workload']]} | {case['rows_per_batch']} | {stages[case['phase']]} | {case['off_ms']:.3f} | {case['on_ms']:.3f} | {case['extra_ms']:.3f} | {case['overhead_percent']:+.1f}% |"
        )
    lines += [
        "",
        "Off/on milliseconds are separate medians; extra milliseconds and percentages are medians of paired changes, so the columns need not match by subtraction or division. Public encode includes observation ingress, tensorization, binding, profile accumulation, and comparisons. Worker encoding excludes model-owned profile commits. Prediction includes CPU input preparation, neural forward, and Arrow writing. Train encode/statistics includes field statistics updates but no forward, backward, or optimizer.",
        "",
        "![Latency overhead by workload and execution path](overhead.svg)",
        "",
        "## Comparison costs",
        "",
        "| Workload | Merge + prediction comparison ms | Train merge + three comparison strata ms |",
        "|---|---:|---:|",
    ]
    for workload in medium:
        entries = {case["phase"]: case for case in selected if case["workload"] == workload}
        lines.append(
            f"| {names[workload]} | {entries['commit_predict']['on_ms']:.3f} | {entries['commit_train_three_strata']['on_ms']:.3f} |"
        )
    lines += [
        "",
        "Commit timings use already-computed profiles. They isolate merging/comparison work and cannot be added mechanically to independently measured stage medians. The disabled commit is nearly a no-op, so its relative percentage is not useful.",
        "",
        "## Profile construction by datatype",
        "",
        "These times isolate the registered Arrow profiler plus structural-state counts for one leaf at 2,048 observations. They exclude coalescing, tensorization, merge/comparison, and neural computation. Text measures character-length profiling only; no frozen text encoder is run. Vector observations have 32 components; sets have four unique members; Period uses month and weekday.",
        "",
        "| Datatype | Profile construction ms |",
        "|---|---:|",
    ]
    for kind, value in sorted(
        summary["per_datatype_profile_ms_at_2048"].items(), key=lambda item: item[1], reverse=True
    ):
        lines.append(f"| {kind} | {value:.3f} |")
    lines += [
        "",
        "## Retained profiles and serialization",
        "",
        "| Workload | Retained histogram/digest buffers KiB | Added serialized checkpoint metadata KiB | Batch profile pickle KiB |",
        "|---|---:|---:|---:|",
    ]
    for workload in medium:
        entry = next(case for case in selected if case["workload"] == workload and case["phase"] == "public_encode")
        off, on = entry["storage"]["off"], entry["storage"]["on"]
        lines.append(
            f"| {names[workload]} | {on['retained_buffer_bytes'] / 1024:.1f} | {(on['checkpoint_metadata_bytes'] - off['checkpoint_metadata_bytes']) / 1024:.1f} | {on['batch_profile_pickle_bytes'] / 1024:.1f} |"
        )
    lines += [
        "",
        "Retained buffers exclude Python object overhead. Checkpoint bytes are an actual serialized metadata delta. Profile pickle bytes are a transport-payload proxy, not measured loader-worker overhead. Buffer growth is recorded after 1, 16, and 256 additional batches in the raw results.",
        "",
        "![Retained buffer growth](memory.svg)",
        "",
        "## Protocol and limits",
        "",
        "Models use d_model=64, two layers, four heads, float32, and zero optimizer updates. Statistics and vocabularies are warmed with two cycles of the same eight-batch pool. Inputs use fixed recorded seeds and are reused cyclically; generation and model placement are outside timing. CPU Torch/BLAS use one thread, Polars two threads, with two CPU cores allowed. CUDA is synchronized at timing boundaries. Garbage collection is outside timed sections. Same-population timing asserts that no drift warning was triggered.",
        "",
        "Numbers are eight Gaussian fields with 5% nulls and differing offsets. Identities combine two each of Category, Enum, Hash, and Group plus Boolean, using 64/16/128/64-label domains. Nested records combine scalar types, sets, vectors, calendar parts, up to 16 events, and up to four items per event. Exact field counts, observed values, Arrow sizes, model parameters, seeds, timing iterations, paired order, CPU time, and CUDA allocation peaks are retained per case.",
        "",
        "No disk/network IO, data-loader workers, distributed reduction, optimizer, training quality, or text encoder is measured. Results establish relative cost for these synthetic schemas and this machine; larger neural models can change the percentage substantially. Startup, first warning IO, and compilation are outside timing.",
        "",
        "All raw pairs are in ../full/results.json and ../full/cases.jsonl. Inputs and library payloads are checksum-bound in ../inputs/manifest.json. Reproduce the measured commands with ../inputs/execute.py in an equivalent prepared lab environment. See the retained benchmark source ../inputs/source.tar.gz or scripts/drift_costs.py in the checkout.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    labels = list(medium)
    colors = ("#4d8b31", "#206bba", "#b35b16")
    for offset, (phase, color) in enumerate(zip(("public_encode", "predict_cpu", "predict_cuda"), colors, strict=True)):
        entries = [
            next((case for case in selected if case["workload"] == workload and case["phase"] == phase), None)
            for workload in labels
        ]
        values = [case["overhead_percent"] if case else float("nan") for case in entries]
        low = [
            max(0, (case["paired_ratio_median"] - case["paired_ratio_min"]) * 100) if case else 0 for case in entries
        ]
        high = [
            max(0, (case["paired_ratio_max"] - case["paired_ratio_median"]) * 100) if case else 0 for case in entries
        ]
        axis.bar(
            np.arange(len(labels)) + (offset - 1) * 0.24,
            values,
            width=0.22,
            color=color,
            label=stages[phase],
            yerr=[low, high],
            capsize=3,
        )
    axis.set_xticks(range(len(labels)), [f"{names[name]}\n{medium[name]} rows/batch" for name in labels])
    axis.set_ylabel("Latency overhead with drift enabled (%)")
    axis.set_title("Seven randomized paired repeats · Same data and model tensors")
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.legend(frameon=False)
    figure.savefig(output / "overhead.svg")
    figure.savefig(output / "overhead.png", dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for entry in result["memory"]:
        if entry["rows_per_batch"] != medium[entry["workload"]]:
            continue
        axis.plot(
            [value["rows"] for value in entry["growth"]],
            [value["retained_buffer_bytes"] / 1024 for value in entry["growth"]],
            marker="o",
            label=names[entry["workload"]],
        )
    axis.set_xscale("log")
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Additional observed rows")
    axis.set_ylabel("Retained histogram/digest buffers (KiB)")
    axis.set_title("Training reference retained; prediction profile accumulated")
    axis.legend(frameon=False)
    figure.savefig(output / "memory.svg")
    figure.savefig(output / "memory.png", dpi=150)
    plt.close(figure)
    (output / "renderer.py").write_bytes(Path(__file__).read_bytes())
    files = [
        root / "full/results.json",
        root / "full/cases.jsonl",
        root / "full/configuration.json",
        root / "inputs/manifest.json",
        root / "preflight.json",
        root / "status.json",
    ]
    files += [path for path in output.iterdir() if path.is_file() and path.name != "manifest.json"]
    (output / "manifest.json").write_text(
        json.dumps(
            {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)},
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    render(parser.parse_args().root.resolve())
