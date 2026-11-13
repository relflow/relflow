"""Compare complete drift cost runs with matched data, models, and timing policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def compare(root: Path) -> dict:
    results = {arm: json.loads((root / arm / "results.json").read_text()) for arm in ("baseline", "candidate")}
    inputs = [root / "inputs/baseline", root / "candidate-runtime/inputs"]
    for arm, source in zip(results, inputs, strict=True):
        manifest = json.loads((source / "manifest.json").read_text())
        if hashlib.sha256((source / "source.tar.gz").read_bytes()).hexdigest() != manifest["archive_sha256"]:
            raise ValueError(f"{arm} source archive does not match its manifest")
        for name, digest in manifest["wheels"].items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"{arm} wheel does not match its manifest")
        if results[arm]["configuration"]["script_sha256"] != manifest["files"]["scripts/drift_costs.py"]:
            raise ValueError(f"{arm} timing script differs from its frozen source")
    configurations = [result["configuration"] for result in results.values()]
    for key in (
        "seed",
        "mode",
        "repetitions",
        "target_ms",
        "warmup_batches",
        "pool_batches",
        "model",
        "python",
        "platform",
        "host",
        "packages",
        "script_sha256",
        "cpu_affinity",
        "thread_policy",
    ):
        if configurations[0][key] != configurations[1][key]:
            raise ValueError(f"cost comparison requires matching {key}")
    if configurations[0]["mode"] != "full" or configurations[0]["repetitions"] != 7:
        raise ValueError("cost comparison requires full runs with seven on/off pairs")
    for arm in results:
        status = json.loads((root / f"{arm}-status.json").read_text())
        if status["state"] != "complete" or status["returncode"] != 0:
            raise ValueError(f"{arm} cost execution did not complete successfully")
    cases = {
        arm: {(case["workload"], case["rows_per_batch"], case["phase"]): case for case in result["cases"]}
        for arm, result in results.items()
    }
    if cases["baseline"].keys() != cases["candidate"].keys() or len(cases["baseline"]) != 54:
        raise ValueError("cost comparison requires the same 54 CPU cases")
    comparisons = []
    for key, before in cases["baseline"].items():
        after = cases["candidate"][key]
        for attribute in (
            "data_seed",
            "pool_batches",
            "data_rows",
            "source_arrow_bytes",
            "batch_arrow_bytes",
            "schema",
            "active_fields",
            "valued_leaf_slots_per_batch",
            "parameters",
        ):
            if before[attribute] != after[attribute]:
                raise ValueError(f"{key}: mismatched {attribute}")
        comparisons.append(
            {
                "workload": key[0],
                "rows": key[1],
                "phase": key[2],
                "before": {name: before[name] for name in ("off_ms", "on_ms", "extra_ms", "overhead_percent")},
                "after": {name: after[name] for name in ("off_ms", "on_ms", "extra_ms", "overhead_percent")},
                "added_overhead_reduction_percent": 100 * (1 - after["extra_ms"] / before["extra_ms"])
                if before["extra_ms"] > 0
                else None,
            }
        )
    output = root / "report"
    output.mkdir(exist_ok=True)
    summary = {"scope": "matched CPU cost measurements; not a CUDA or production claim", "cases": comparisons}
    (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    sizes = {"numbers": 256, "identities": 256, "nested": 128}
    selected = [case for case in comparisons if case["rows"] == sizes[case["workload"]]]
    names = {"numbers": "8 Numbers", "identities": "9 identity/Boolean fields", "nested": "Nested mixed fields"}
    lines = [
        "# Drift overhead optimization",
        "",
        "Fresh before/after runs use identical model definitions, data seeds, eight-batch Arrow pools, "
        "seven randomized on/off pairs, CPU affinity 0/2, and thread settings. Both revisions assert identical "
        "model tensors and prediction outputs between monitoring on and off. The GPU was occupied and is excluded.",
        "",
        "## Medium batch sizes",
        "",
        "| Workload | Rows | Stage | Extra ms before | Extra ms after | Added overhead reduction |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for case in selected:
        if case["phase"] in {"public_encode", "public_train_encode", "predict_cpu"}:
            lines.append(
                f"| {names[case['workload']]} | {case['rows']} | {case['phase']} | "
                f"{case['before']['extra_ms']:.3f} | {case['after']['extra_ms']:.3f} | "
                f"{case['added_overhead_reduction_percent']:.1f}% |"
            )
    lines += [
        "",
        "Extra milliseconds are the median of paired on-minus-off timings within each revision. "
        "The reduction compares those medians across revisions; it is not a whole-model speedup. "
        "All 54 cases, including unfavorable changes and noisy near-zero differences, remain in comparison.json. "
        "Raw pairs and calibration iteration counts remain in each arm's results.json and cases.jsonl. "
        "Ranges from seven pairs are observations, not confidence intervals.",
        "",
        "![Added encode overhead before and after optimization](encode-overhead.svg)",
        "",
        "The implementation keeps every batch, the same compression and warning thresholds, and bounded metadata. "
        "It batches centroid reductions, reuses unchanged Arrow fields, and only compares a stratum when its "
        "reference or observed profile changes. Training updates still revisit every comparison stratum. "
        "Floating-point reduction order can change last-bit numeric sketch values.",
        "",
        "These are in-process steady-state CPU measurements, without loader workers, optimizer steps, "
        "disk/network IO, or a Text encoder. The independent full-budget detection studies and runtime tests "
        "are retained under candidate-runtime. Check each record's actual measured source and wheel; "
        "these local experiments do not replace paired hosted Evidence CI.",
        "",
        "Render again with `uv run --no-project --with matplotlib==3.10.7 --with numpy==2.4.6 "
        "python scripts/drift_optimization_report.py /absolute/path/to/run-root`.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    encode = [case for case in selected if case["phase"] == "public_encode"]
    positions = np.arange(len(encode))
    figure, axis = plt.subplots(figsize=(9, 4), layout="constrained")
    for offset, arm, color in [(-0.18, "before", "#888888"), (0.18, "after", "#0c2d5e")]:
        bars = axis.bar(positions + offset, [case[arm]["extra_ms"] for case in encode], 0.36, label=arm, color=color)
        axis.bar_label(bars, fmt="%.2f ms", padding=3)
    axis.set_xticks(positions, [f"{names[case['workload']]}\n{case['rows']} rows/batch" for case in encode])
    axis.set_ylabel("Additional milliseconds per public encode")
    axis.set_title("Same observations and detection policy; less profiling work")
    axis.legend()
    axis.margins(y=0.2)
    for suffix in ("svg", "png"):
        figure.savefig(output / f"encode-overhead.{suffix}", dpi=150)
    plt.close(figure)
    (output / "renderer.py").write_bytes(Path(__file__).read_bytes())
    assets = [root / f"{arm}-status.json" for arm in results]
    assets += [source / "manifest.json" for source in inputs]
    assets += [root / "candidate-runtime/preflight.json", root / "inputs/baseline/preflight.json"]
    assets += [path for arm in results for path in (root / arm).glob("*.json*") if path.is_file()]
    assets += [path for path in output.iterdir() if path.is_file() and path.name != "manifest.json"]
    (output / "manifest.json").write_text(
        json.dumps(
            {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(assets)},
            indent=2,
        )
        + "\n"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(compare(parser.parse_args().root.resolve()), indent=2))
