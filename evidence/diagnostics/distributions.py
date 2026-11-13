"""Render local drift research receipts without replacing published measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build(root: Path) -> dict:
    """Bind every readout and figure to the exact wheel, script, and result bytes."""
    manifest = json.loads((root / "inputs/manifest.json").read_text())
    preflight = json.loads((root / "preflight.json").read_text())
    scripts = {
        "P101": "numeric_shifts.py",
        "P102": "typed_shifts.py",
        "P103": "policies_and_checkpoints.py",
        "P104": "enum_shifts.py",
        "P105": "hash_shifts.py",
        "P106": "cluster_shifts.py",
        "P113": "group_shifts.py",
    }
    requested = sorted(path.name.removeprefix("full-") for path in (root / "results").glob("full-*") if path.is_dir())
    if not requested or set(requested) - scripts.keys():
        raise ValueError("local drift results require recognized full-budget study directories")
    selected = {}
    for identifier in requested:
        script = scripts[identifier]
        paths = list((root / "results" / f"full-{identifier}").glob("*.json"))
        if len(paths) != 1:
            raise ValueError(f"{identifier}: expected exactly one full-budget research record, got {len(paths)}")
        path = paths[0]
        record = json.loads(path.read_text())
        provenance = record["provenance"]
        expected = manifest["files"][f"evidence/research/distributions/{script}"]
        if (
            record["mode"] != "full"
            or record["steps_override"] is not None
            or provenance["script_sha256"] != expected
            or provenance["wheel_sha256"] not in manifest["wheels"].values()
            or provenance["evidence_revision"] != "working-tree:" + manifest["archive_sha256"]
            or provenance["host"] != preflight["host"]
        ):
            raise ValueError(f"{identifier}: record does not match the frozen lab inputs")
        selected[identifier] = (path, record)
    output = root / "report"
    output.mkdir(exist_ok=True)
    summary = {
        "scope": "local CPU research on lab; candidate wheel only; not the paired hosted CI gate",
        "base_commit": manifest["base_commit"],
        "working_tree": manifest["working_tree"],
        "prior_attempt": manifest.get("prior_attempt"),
        "correction": manifest.get("reason"),
        "source_archive_sha256": manifest["archive_sha256"],
        "wheel_sha256": manifest["wheels"],
        "hardware": {key: preflight[key] for key in ("host", "cpu_model", "gpu")},
        "experiments": {
            identifier: {
                "record": str(path.relative_to(root)),
                "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "outcome": record["outcome"],
                "checks_met": sum(record.get("checks", {}).values()),
                "checks_total": len(record.get("checks", {})),
                "unmet": [name for name, passed in record.get("checks", {}).items() if not passed],
                "error": record.get("error"),
                "seed_panel": record.get("metrics", {}).get("seed_panel"),
                "detection_rate": record.get("metrics", {}).get("detection_rate"),
                "shift_cases": record.get("metrics", {}).get("shift_cases"),
                "control_cases": record.get("metrics", {}).get("control_cases"),
                "control_warnings": record.get("metrics", {}).get("control_warnings"),
                "duration_seconds": record["duration_seconds"],
            }
            for identifier, (path, record) in selected.items()
        },
    }
    # Plotting is an explicit execution-time dependency; static builds never import it.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    numeric = selected.get("P101", (None, {}))[1]
    if numeric.get("metrics"):
        conditions = ["same_population", "location", "spread", "shape", "correlation"]
        figure, axis = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
        for index, condition in enumerate(conditions):
            arms = [trial["arms"][f"test/{condition}"] for trial in numeric["metrics"]["trials"]]
            x = index + np.linspace(-0.12, 0.12, len(arms))
            axis.scatter(
                x,
                [arm["distance"] for arm in arms],
                color="#206bba",
                label="Measured CDF distance" if index == 0 else None,
            )
            axis.scatter(
                x,
                [0.2 + arm["allowance"] for arm in arms],
                marker="_",
                s=160,
                color="#b35b16",
                label="Warning cutoff" if index == 0 else None,
            )
        axis.set_xticks(
            range(len(conditions)), ["Same population", "+4 location", "8× spread", "Bimodal shape", "Correlation flip"]
        )
        axis.set_ylabel("Amount marginal CDF distance")
        axis.set_ylim(-0.03, 1.05)
        axis.set_title("P101 · Independent test cohorts · Three fixed seeds")
        axis.legend(frameon=False)
        figure.savefig(output / "numeric-shifts.svg")
        figure.savefig(output / "numeric-shifts.png", dpi=150)
        plt.close(figure)
    typed = selected.get("P102", (None, {}))[1]
    if typed.get("metrics"):
        conditions = ["same_population", "labels", "boolean", "members", "set_size", "vector", "calendar", "nulls"]
        trials = typed["metrics"]["trials"]
        counts = [sum(bool(trial["arms"][condition]["detected"]) for trial in trials) for condition in conditions]
        figure, axis = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
        axis.bar(range(len(conditions)), counts, color=["#777777", *(["#206bba"] * 7)])
        axis.set_xticks(
            range(len(conditions)),
            [
                "Same\npopulation",
                "Labels",
                "Boolean",
                "Set\nmembers",
                "Set\nsize",
                "Vector\nnorm",
                "Calendar",
                "Nested\nnulls",
            ],
        )
        axis.set_yticks(range(len(trials) + 1))
        axis.set_ylim(0, len(trials) + 0.5)
        axis.set_ylabel("Seed trials detecting the declared feature")
        axis.set_title("P102 · 2,048 prediction records per condition")
        figure.savefig(output / "typed-shifts.svg")
        figure.savefig(output / "typed-shifts.png", dpi=150)
        plt.close(figure)
    identities = {
        identifier: selected[identifier][1]
        for identifier in ("P104", "P105", "P106", "P113")
        if identifier in selected and selected[identifier][1].get("metrics")
    }
    if identities:
        labels = {"P104": "Enum", "P105": "Hash", "P106": "Cluster", "P113": "Group"}
        conditions = ["same_population", "folded", "frequencies", "novel", "nulls", "batched_frequencies"]
        matrix, rows = [], []
        for identifier, record in identities.items():
            for split in ("validate", "test", "predict"):
                values = []
                for condition in conditions:
                    arms = [trial["arms"].get(f"{split}/{condition}") for trial in record["metrics"]["trials"]]
                    if any(arm is None for arm in arms):
                        values.append(np.nan)
                    else:
                        values.append(
                            sum(
                                arm["detected"] if arm["expected_feature"] is not None else not arm["warnings"]
                                for arm in arms
                            )
                            / len(arms)
                        )
                matrix.append(values)
                rows.append(f"{labels[identifier]} / {split}")
        figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
        colors = plt.get_cmap("Blues").copy()
        colors.set_bad("#dddddd")
        axis.imshow(np.asarray(matrix), vmin=0, vmax=1, cmap=colors, aspect="auto")
        for row, values in enumerate(matrix):
            for column, value in enumerate(values):
                axis.text(
                    column,
                    row,
                    "N/A" if np.isnan(value) else f"{value:.0%}",
                    ha="center",
                    va="center",
                    color="white" if value >= 0.6 else "black",
                )
        axis.set_yticks(range(len(rows)), rows)
        axis.set_xticks(
            range(len(conditions)),
            ["Stable: quiet", "Folding: quiet", "Frequency: warn", "Novel: warn", "Nulls: warn", "Batches: once"],
            rotation=20,
            ha="right",
        )
        axis.set_title("Identity drift · Fraction of three fixed seeds meeting each expectation")
        figure.savefig(output / "identity-strata.svg")
        figure.savefig(output / "identity-strata.png", dpi=150)
        plt.close(figure)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Drift evidence on lab",
        "",
        summary["scope"] + ".",
        "",
        "Each study uses three fixed seeds and 16 real optimizer steps per seed, with 4,096/1,024 source training/validation rows and 2,048 held-out records per condition. No predictive-accuracy claim is made.",
        "",
        "| Study | Outcome | Checks met | Seeds |",
        "|---|---|---|---|",
    ]
    for identifier, entry in summary["experiments"].items():
        lines.append(
            f"| {identifier} | {entry['outcome']} | {entry['checks_met']}/{entry['checks_total']} | {entry['seed_panel']} |"
        )
        if entry["unmet"]:
            lines.extend(["", "Unmet criteria: " + "; ".join(entry["unmet"])])
        if entry["error"]:
            lines.extend(["", "Execution error retained in the raw record."])
    if summary["prior_attempt"]:
        lines.extend(
            [
                "",
                "The original attempt is retained in ../../"
                + summary["prior_attempt"]
                + "/report/README.md. Correction: "
                + summary["correction"],
            ]
        )
    if identities:
        lines.extend(
            [
                "",
                "![Identity monitoring across validation, test, and prediction](identity-strata.svg)",
                "",
                "Cells show the fraction of three fixed seeds meeting the declared expectation. Stable and folded controls must be quiet; shifted populations must warn. Five 64-row batches must accumulate profiles, wait for the sample minimum, and produce one label warning per field and window.",
                "",
                "Enum additionally rejects undeclared labels in every stratum. Hash checks that training salts change tensor hashes without changing profile frequencies. Identity studies check frozen vocabulary, assignment probabilities, and learned tensors.",
            ]
        )
    lines.extend(
        [
            "",
            (
                "The correlation flip is an intentional detection boundary: nearly +1 to nearly -1 correlation can preserve both marginal populations and remain quiet. Identity buckets can collide, vector mean/norm omit rotations, and profiles remain rank-local."
                if "P101" in selected
                else "These studies monitor marginal input-identity frequencies and structural states. Hash buckets can collide; Enum rejects undeclared labels separately from drift warnings. Group and historical Cluster studies profile source labels rather than learned assignment usage. Profiles remain rank-local."
            ),
            "",
            "The figures use exact recorded distances and warning events. They show these synthetic populations only and do not estimate a production false-positive rate. Smoke results are retained separately and are not capability evidence.",
            "",
            "Inputs, wheel, source archive, environment receipt, full records, smoke records, and execution logs are retained alongside this report. summary.json binds each full record to its source and wheel. manifest.json binds every report artifact and raw record by SHA256.",
            "",
            f"Replay the experiments from the repository root using the retained wheel: `uv run --project docs --with /absolute/path/to/inputs/{next(iter(manifest['wheels']))} python evidence/experiments/run.py --lane research "
            + " ".join(requested)
            + " --threads 1 --accelerator cpu --output-dir /new/result/directory`. This requires the same declared dependencies; inputs/execute.py records the actual lab commands and thread policy.",
            "",
            "Render this receipt again with `uv run --no-project --with matplotlib==3.10.7 --with numpy==2.4.6 python evidence/diagnostics/distributions.py /absolute/path/to/run-root`.",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n")
    (output / "renderer.py").write_bytes(Path(__file__).read_bytes())
    assets = [root / "inputs/manifest.json", root / "preflight.json", root / "status.json"]
    assets.extend(path for path in (root / "results").rglob("*.json") if path.is_file())
    assets.extend(path for path in output.iterdir() if path.is_file() and path.name != "manifest.json")
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(assets)}
    (output / "manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(build(parser.parse_args().root.resolve()), indent=2))
