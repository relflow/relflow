"""Shared plot primitives and renderers; experiments own custom diagnostic components."""

from __future__ import annotations

import ast
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from evidence.diagnostics.data import Bundle, human

COLORS = ("#178772", "#c97843", "#587fbd", "#b3678d", "#778b3e", "#5f8f9a")
MARKERS = ("o", "s", "^", "D", "v", "P")


def finite(values) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("A plotted measurement is nonfinite; preserve its missing/error state instead")
    return result


def caption(text: str) -> str:
    return textwrap.fill(text, 52)


def axes_pair(*, width: float = 9.5, height: float = 4.5):
    return plt.subplots(1, 2, figsize=(width, height), layout="constrained")


def dots(ax, labels, series, *, xlabel: str, reference: tuple | None = None, percent: bool = False):
    positions = np.arange(len(labels))
    width = min(0.18, 0.65 / max(1, len(series)))
    for index, (name, values) in enumerate(series.items()):
        if len(values) != len(labels):
            raise ValueError("A comparison series does not match its declared conditions")
        values = np.asarray([np.nan if value is None else value for value in values], dtype=float)
        offset = (index - (len(series) - 1) / 2) * width
        ax.scatter(
            values,
            positions + offset,
            color=COLORS[index % len(COLORS)],
            marker=MARKERS[index % len(MARKERS)],
            s=35,
            zorder=4,
            label=name,
        )
        for j in np.flatnonzero(np.isnan(values)):
            ax.text(
                0.98,
                positions[j] + offset,
                "unrecorded",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=8,
            )
    if reference:
        ax.axvline(reference[0], color="gray", ls=":", lw=1.2, label=reference[1])
    ax.set(yticks=positions, yticklabels=[textwrap.fill(str(label), 24) for label in labels], xlabel=xlabel)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.22)
    if percent:
        ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(fontsize=8, loc="best")


def numeric(bundle: Bundle, spec: dict, *, pairs=None) -> tuple[plt.Figure, str, dict]:
    pairs = bundle.pairs(spec["target"]) if pairs is None else pairs
    pairs = [pair for pair in pairs if pair["actual"].dtype.kind in "bifu" and pair["predicted"].dtype.kind in "bifu"]
    if not pairs:
        raise ValueError(f"{bundle.manifest['id']}: no bound numeric predictions for {spec['target']}")
    test = [pair for pair in pairs if pair["population"] == "test"] or pairs
    field = spec["target"]
    compared = [pair for pair in test if pair["field"] == field]
    if not compared:
        raise ValueError(f"{bundle.manifest['id']}: requested target {field} has no captured predictions")
    primary = compared[0]
    actual, predicted = (finite(primary["actual"]), finite(primary["predicted"]))
    center = (
        math.floor(float(np.median(actual)) / 1000000.0) * 1000000.0
        if abs(float(np.median(actual))) > 100000000.0
        else 0.0
    )
    a, p = (actual - center, predicted - center)
    fig, axes = axes_pair()
    bound = np.r_[a, p]
    lower, upper = (float(bound.min()), float(bound.max()))
    pad = max((upper - lower) * 0.06, 0.05)
    if len(a) > 600:
        density = axes[0].hexbin(a, p, gridsize=35, mincnt=1, bins="log", cmap="viridis", rasterized=True)
        fig.colorbar(density, ax=axes[0], fraction=0.035, label="Records (log count)")
    else:
        axes[0].scatter(a, p, color=COLORS[0], s=13, alpha=0.45)
    axes[0].plot([lower - pad, upper + pad], [lower - pad, upper + pad], color="gray", ls="--", lw=1)
    suffix = f" minus {center:.0f}" if center else ""
    axes[0].set(
        title=caption(primary["label"]),
        xlabel=f"Actual {field.replace('_', ' ')}{suffix}",
        ylabel="Model prediction" + suffix,
        xlim=(lower - pad, upper + pad),
        ylim=(lower - pad, upper + pad),
    )
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].text(
        0.03,
        0.97,
        f"n = {len(a):,}\nRMSE = {np.sqrt(np.mean((actual - predicted) ** 2)):.4g}",
        transform=axes[0].transAxes,
        va="top",
        fontsize=9,
    )
    selected = compared[:5]
    for index, pair in enumerate(selected):
        errors = np.sort(np.abs(finite(pair["actual"]) - finite(pair["predicted"])))
        axes[1].step(
            errors,
            np.arange(1, len(errors) + 1) / len(errors),
            where="post",
            color=COLORS[index],
            lw=1.8,
            label=textwrap.fill(pair["label"], 33),
        )
    axes[1].set(
        title="How often is the error small?",
        xlabel="Absolute prediction error (source units)",
        ylabel="Fraction of scored coordinates",
        ylim=(0, 1.02),
    )
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].grid(alpha=0.18)
    axes[1].legend(fontsize=8, loc="lower right")
    samples = []
    for pair in compared[:12]:
        indices = list(range(min(16, len(pair["actual"]))))
        records = []
        for index in indices:
            coordinate = pair["coordinates"][index] if pair["coordinates"] else None
            observation = pair["rows"][coordinate[0]] if coordinate and pair["rows"] is not None else None
            records.append(
                {
                    "coordinate": list(coordinate) if coordinate else [index],
                    "input": observation,
                    "actual": float(pair["actual"][index]),
                    "predicted": float(pair["predicted"][index]),
                }
            )
        samples.append(
            {
                "label": field.replace("_", " ") + " · " + pair["label"],
                "field": pair["field"],
                "population": pair["population"],
                "n": len(pair["actual"]),
                "records": records,
            }
        )
    note = f"The left panel uses {len(actual):,} captured coordinates; its dashed line marks exact prediction. The right panel uses every captured coordinate in each displayed condition."
    if len(compared) > len(selected):
        note += f" The example viewer contains {min(12, len(compared))} of the {len(compared)} captured conditions; this figure displays the first {len(selected)} in execution order."
    if center:
        note += f" Both scatter axes subtract {center:.0f}; errors use the original float64 values."
    return (fig, note, {"kind": "numeric", "conditions": samples})


def classification(bundle: Bundle, spec: dict) -> tuple[plt.Figure, str, dict]:
    pairs = bundle.pairs(spec["target"])
    pairs = [pair for pair in pairs if pair["population"] == "test"] or pairs
    if not pairs:
        raise ValueError(f"{bundle.manifest['id']}: no bound classification predictions")
    pair = next((p for p in pairs if p["population"] == "test"), pairs[0])
    a, p = (pair["actual"], pair["predicted"])
    fig, axes = axes_pair()
    is_probability = p.dtype.kind in "f" and a.dtype.kind in "biu" and (set(np.unique(a)) <= {0, 1})
    if is_probability:
        scores = finite(p)
        for value, label, color in [(0, "Negative answers", COLORS[2]), (1, "Positive answers", COLORS[0])]:
            selected = scores[a == value]
            axes[0].hist(
                selected,
                bins=np.linspace(0, 1, 26),
                density=True,
                histtype="step",
                lw=1.8,
                color=color,
                label=f"{label} (n={len(selected):,})",
            )
        axes[0].set(
            title="Do probabilities separate the answers?",
            xlabel="Predicted positive probability",
            ylabel="Density",
            xlim=(0, 1),
        )
        axes[0].legend(fontsize=8)
        for index, item in enumerate(pairs[:4]):
            truth, probability = (np.asarray(item["actual"], dtype=bool), finite(item["predicted"]))
            fpr, tpr = roc(probability[~truth], probability[truth])
            axes[1].plot(fpr, tpr, color=COLORS[index], lw=1.8, label=textwrap.fill(item["label"], 29))
        axes[1].plot([0, 1], [0, 1], ls=":", color="gray")
        axes[1].set(
            title="Every point comes from retained scores",
            xlabel="False-positive rate",
            ylabel="True-positive rate",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axes[1].xaxis.set_major_formatter(PercentFormatter(1))
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1].legend(fontsize=8, loc="lower right")
    else:
        labels = sorted(set(map(str, a)) | set(map(str, p)))
        if len(labels) > 24:
            raise ValueError("Class figure requires an explicit grouping for more than 24 labels")
        counts = np.array(
            [[np.sum((a.astype(str) == left) & (p.astype(str) == right)) for right in labels] for left in labels]
        )
        proportions = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        image = axes[0].imshow(proportions, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axes[0].set(
            xticks=range(len(labels)),
            xticklabels=labels,
            yticks=range(len(labels)),
            yticklabels=labels,
            xlabel="Predicted class",
            ylabel="Actual class",
            title="Which classes are confused?",
        )
        axes[0].tick_params(axis="x", rotation=40, labelsize=8)
        axes[0].tick_params(axis="y", labelsize=8)
        fig.colorbar(image, ax=axes[0], fraction=0.035, label="Fraction within actual class")
        if len(labels) <= 8:
            for i, j in np.ndindex(counts.shape):
                axes[0].text(
                    j,
                    i,
                    str(counts[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if proportions[i, j] < 0.5 else "#18231e",
                )
        shown = pairs[:6]
        dots(
            axes[1],
            [item["label"] for item in shown],
            {"Exact accuracy": [float(np.mean(item["actual"] == item["predicted"])) for item in shown]},
            xlabel="Fraction of captured answers predicted correctly",
            percent=True,
        )
        axes[1].set_xlim(-0.02, 1.02)
    return (
        fig,
        f"The left panel shows {pair['label']} with all {len(a):,} captured answers. Class counts and observed score curves are computed directly from retained predictions.",
        {
            "kind": "classification",
            "conditions": [
                {
                    "label": item["label"],
                    "n": len(item["actual"]),
                    "records": [
                        {
                            "coordinate": [i],
                            "input": None,
                            "actual": str(item["actual"][i]),
                            "predicted": str(item["predicted"][i]),
                            **({"positive_probability": float(item["predicted"][i])} if is_probability else {}),
                        }
                        for i in range(min(16, len(item["actual"])))
                    ],
                }
                for item in pairs[:12]
            ],
        },
    )


def roc(normal, anomalous):
    normal, anomalous = (np.sort(finite(normal)), np.sort(finite(anomalous)))
    if not len(normal) or not len(anomalous):
        raise ValueError("A measured ROC needs both outcome populations")
    thresholds = np.r_[np.inf, np.unique(np.r_[normal, anomalous])[::-1], -np.inf]
    return (
        1 - np.searchsorted(normal, thresholds, side="right") / len(normal),
        1 - np.searchsorted(anomalous, thresholds, side="right") / len(anomalous),
    )


def trajectory(bundle: Bundle, spec: dict):
    points = bundle.record["metrics"]["trajectory"]
    fig, axes = axes_pair()
    if any(("n_assigned_groups" in point for point in points)):
        keys = [("n_assigned_groups", "Assigned groups"), ("perplexity", "Usage perplexity")]
    elif any(("assigned_groups" in point for point in points)):
        keys = [("assigned_groups", "Assigned groups"), ("adjusted_rand", "Adjusted Rand index")]
    else:
        keys = [("n_committed", "Committed groups"), ("perplexity", "Usage perplexity")]
        if not any(("perplexity" in point for point in points)):
            keys[1] = ("adherence", "Input adherence")
    for ax, (key, label) in zip(axes, keys, strict=True):
        selected = [point for point in points if key in point]
        if not selected:
            ax.text(0.5, 0.5, "No trajectory measurement recorded", ha="center", va="center", transform=ax.transAxes)
            continue
        ax.plot(
            [point["epoch"] for point in selected],
            finite([point[key] for point in selected]),
            marker="o",
            ms=3,
            color=COLORS[0],
        )
        ax.set(xlabel="Training epoch", ylabel=label, title=label)
        ax.grid(alpha=0.18)
    return (
        fig,
        "Each point is an actual recorded epoch, not an independent seed. Group counts and usage are descriptive diagnostics; they do not establish a recovered partition.",
        {},
    )


def coverage(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    fig, axes = axes_pair()
    keys = sorted(
        [key for key in m if key.startswith("known_") and key.split("_")[-1].isdigit()],
        key=lambda key: -int(key.split("_")[1]),
    )
    if keys:
        x = finite([m[key]["coverage.content"] for key in keys])
        for index, (key, label) in enumerate(
            [("accuracy.content", "Accuracy on known answers"), ("accuracy.all", "Accuracy on all answers")]
        ):
            values = [m[k].get(key) for k in keys]
            present = [i for i, value in enumerate(values) if value is not None]
            axes[0].plot(
                x[present], [values[i] for i in present], marker=MARKERS[index], color=COLORS[index], label=label
            )
        axes[0].plot(x, x, ls=":", color="gray", label="Coverage ceiling for perfect known answers")
        axes[0].set(
            title="Coverage and conditional accuracy differ",
            xlabel="Fraction of targets in the vocabulary",
            ylabel="Observed accuracy",
            xlim=(-0.03, 1.03),
            ylim=(-0.03, 1.03),
        )
        axes[0].xaxis.set_major_formatter(PercentFormatter(1))
        axes[0].yaxis.set_major_formatter(PercentFormatter(1))
        axes[0].legend(fontsize=8)
        known = finite([m[key]["targets.known"] for key in keys])
        unknown = finite([m[key]["targets.unavailable"] for key in keys])
        axes[1].barh(range(len(keys)), known, color=COLORS[0], label="Known answers")
        axes[1].barh(range(len(keys)), unknown, left=known, color=COLORS[1], label="Unavailable answers")
        axes[1].set(
            yticks=range(len(keys)),
            yticklabels=[f"{value:.0%} coverage" for value in x],
            xlabel="Selected target count",
            title="Which answers contribute to each score?",
        )
        axes[1].invert_yaxis()
        axes[1].legend(fontsize=8)
        return (
            fig,
            "Undefined conditional accuracy at zero coverage is omitted, rather than replaced by zero. Counts come from the proof's selected-target measurements.",
            {},
        )
    raise ValueError(f"{bundle.manifest['id']}: vocabulary comparison needs an authored view")


def temporal(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    labels = [condition["label"] for condition in spec["conditions"]]
    keys = [condition["metric"] for condition in spec["conditions"]]
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    dots(
        ax,
        labels,
        {"Held-out accuracy": [m[key] for key in keys]},
        xlabel="Correct answers on the independent calendar split",
        percent=True,
    )
    ax.set_xlim(-0.02, 1.03)
    return (
        fig,
        "Each point comes from a separately fitted arm with its declared inputs. The plot shows the tested information conditions; lines between arms would not represent a learning trajectory.",
        {},
    )


def metric_leaves(value, path=()):
    if isinstance(value, dict):
        return [item for key, child in value.items() for item in metric_leaves(child, (*path, key))]
    if isinstance(value, (float, int)) and (not isinstance(value, bool)):
        return [(path, float(value))]
    return []


def calibration(bundle: Bundle, spec: dict):
    fig, axes = axes_pair()
    candidates = []
    for frame in bundle.frames:
        if frame["function"] != "score":
            continue
        values = frame["values"]
        p = bundle.value(values.get("p", values.get("predicted")))
        oracle = bundle.value(values.get("oracle", values.get("expected")))
        if p is None or oracle is None or isinstance(p, dict) or isinstance(oracle, dict):
            continue
        p, oracle = (np.asarray(p), np.asarray(oracle))
        if p.shape == oracle.shape and p.ndim == 1 and (len(p) > 1):
            candidates.append((frame, finite(p), finite(oracle)))
    if not candidates:
        plt.close(fig)
        return summary(bundle, spec)
    frame, p, oracle = candidates[0]
    density = axes[0].hexbin(
        oracle, p, gridsize=25, extent=(0, 1, 0, 1), mincnt=1, bins="log", cmap="viridis", rasterized=True
    )
    fig.colorbar(density, ax=axes[0], fraction=0.035, label="Records (log count)")
    axes[0].plot([0, 1], [0, 1], ls="--", color="gray")
    axes[0].set(
        title=caption(bundle.label(frame)),
        xlabel="Generator's conditional probability",
        ylabel="Model positive probability",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].set_aspect("equal")
    selected = candidates[:6]
    labels = [bundle.label(frame) for frame, _, _ in selected]
    errors = [float(np.sqrt(np.mean((p - oracle) ** 2))) for _, p, oracle in selected]
    dots(axes[1], labels, {"Probability RMSE": errors}, xlabel="RMSE against the known conditional probability")
    for frame, p, oracle in selected:
        result = bundle.value(frame["result"])
        key = "oracle_rmse" if "oracle_rmse" in result else "probability_rmse"
        if key in result:
            np.testing.assert_allclose(np.sqrt(np.mean((p - oracle) ** 2)), result[key], rtol=1e-10, atol=1e-12)
    return (
        fig,
        "The diagonal compares probabilities with an evaluator oracle, not with a single random label. Probability RMSE is recomputed from captured arrays and checked against the proof's own score.",
        {},
    )


def mutation(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    fig, axes = axes_pair(width=10, height=4.8)
    targets = spec["targets"]
    for ax, target in zip(axes, targets, strict=True):
        for index, (arm, value) in enumerate(m["arms"].items()):
            curve = value.get("validation_curve", [])
            if not curve:
                continue
            selected = []
            for point in curve:
                scores = point["scores"]
                if target in {"u", "v", "w"}:
                    if target not in scores:
                        continue
                    measured = scores[target]["nrmse"]
                else:
                    measured = scores["only_a_distance_nrmse" if target else "nrmse"]
                selected.append((point["step"], measured))
            if selected:
                ax.plot(
                    *zip(*selected, strict=True),
                    marker=MARKERS[index % len(MARKERS)],
                    ms=4,
                    lw=1.7,
                    color=COLORS[index % len(COLORS)],
                    label=human(arm),
                )
        ax.set(
            xlabel="Optimizer updates after the edit",
            ylabel="Validation normalized error",
            title=f"Target {target}"
            if target in {"u", "v", "w"}
            else "Distance to remaining-input oracle"
            if target
            else "Error against original target",
        )
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8)
    return (
        fig,
        "Every curve uses the same fixed validation split at the recorded update checkpoints. The trained source's earlier updates are additional history; a lower continuation curve does not imply an equal cumulative budget.",
        {},
    )


def composition(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    identifier = bundle.manifest["id"]
    cells = []
    for arm, values in m.items():
        if not isinstance(values, dict):
            continue
        for length in ["familiar_lengths", "unseen_lengths"]:
            if length not in values:
                continue
            for request in ["familiar_requests", "unseen_requests"]:
                if request in values[length]:
                    cells.append(
                        (human(arm), human(length), human(request), values[length][request]["normalized_rmse"], False)
                    )
    if not cells:
        raise ValueError(f"{identifier}: no authored composition cells matched the captured record")
    panels = list(dict.fromkeys((item[0] for item in cells)))
    rows = list(dict.fromkeys((item[1] for item in cells)))
    columns = list(dict.fromkeys((item[2] for item in cells)))
    fig = plt.figure(figsize=(max(7.8, len(panels) * 4.5), 4.8), layout="constrained")
    grid = fig.add_gridspec(1, len(panels) + 1, width_ratios=[*[1] * len(panels), 0.045])
    maximum = max((value for _, _, _, value, _ in cells))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#8a9691")
    for index, panel in enumerate(panels):
        ax = fig.add_subplot(grid[0, index])
        matrix = np.full((len(rows), len(columns)), np.nan)
        flags = {}
        for p, row, column, value, flag in cells:
            if p == panel:
                matrix[rows.index(row), columns.index(column)] = value
                flags[rows.index(row), columns.index(column)] = flag
        image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(maximum, 0.1), aspect="auto")
        ax.set(
            xticks=range(len(columns)),
            xticklabels=[human(c) for c in columns],
            yticks=range(len(rows)),
            yticklabels=[human(r) for r in rows],
            title=caption(panel),
        )
        ax.tick_params(axis="x", labelsize=9, rotation=15)
        for i, j in np.ndindex(matrix.shape):
            value = matrix[i, j]
            text = "unrecorded" if np.isnan(value) else f"{value:.3f}" + (" *" if flags.get((i, j)) else "")
            ax.text(
                j, i, text, ha="center", va="center", fontsize=9, color="white" if value > maximum * 0.65 else "#243b35"
            )
    fig.colorbar(image, cax=fig.add_subplot(grid[0, -1]), label="Normalized RMSE (lower is better)")
    note = "All panels share the same error scale. Every displayed cell is a measured condition; gray cells are unrecorded, rather than zero."
    return (
        fig,
        note,
        {
            "metric_cells": [
                {"panel": p, "row": r, "column": c, "normalized_rmse": value, "withheld": flag}
                for p, r, c, value, flag in cells
            ]
        },
    )


def group_learning(bundle: Bundle, spec: dict):
    """Show recorded assignment and validation trajectories for fixed groups."""
    counts = bundle.record["metrics"]["group_counts"]
    count = "4" if "4" in counts else next(iter(counts))
    arms = counts[count]
    fig, axes = axes_pair(width=10, height=4.5)
    for index, (arm, label) in enumerate((("stable", "Stable identities"), ("no_regime", "No-regime control"))):
        points = arms[arm]["ari_by_step"]
        if not points:
            raise ValueError(f"{bundle.manifest['id']}: no assignment trajectory was recorded")
        x = [point["step"] for point in points]
        for ax, metric in zip(axes, ("adjusted_rand", "validation_rmse"), strict=True):
            selected = [
                (step, point[metric]) for step, point in zip(x, points, strict=True) if point[metric] is not None
            ]
            ax.plot(
                [step for step, _ in selected],
                finite([value for _, value in selected]),
                color=COLORS[index],
                lw=2,
                marker="o",
                ms=2.5,
                label=label,
            )
    axes[0].axhline(0.95, color="#a8524e", linestyle="--", label="Final test ARI target")
    axes[0].set(ylabel="Assignment agreement (ARI)", ylim=(-0.05, 1.04), title="Do identities find the true groups?")
    axes[1].set(ylabel="Validation RMSE (target units)", title="Does prediction improve?")
    for ax in axes:
        ax.set_xlabel("Optimizer updates")
        ax.grid(axis="y", alpha=0.18)
        ax.legend(fontsize=8, loc="best")
    return (
        fig,
        "Points are recorded epoch-end checkpoints. The dashed ARI criterion applies to the final test, not every checkpoint. Prediction error here uses validation observations; final test error is reported separately.",
        {},
    )


def group_composition(bundle: Bundle, spec: dict):
    """Show which generating rules contribute identities to each learned group."""
    counts = bundle.record["metrics"]["group_counts"]
    count = "4" if "4" in counts else next(iter(counts))
    stable = counts[count]["stable"]
    truth, assigned = (np.asarray(stable["expected_partition"]), np.asarray(stable["assignments"]))
    if truth.shape != assigned.shape or np.any(assigned < -1) or np.any(assigned >= int(count)):
        raise ValueError(f"{bundle.manifest['id']}: group composition has invalid identity assignments")
    groups = list(range(int(count)))
    if np.any(assigned == -1):
        groups.append(-1)
    rules = sorted(set(truth))
    values = np.array([[np.sum((assigned == group) & (truth == rule)) for rule in rules] for group in groups])
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    left = np.zeros(len(groups))
    for index, rule in enumerate(rules):
        column = values[:, index]
        ax.barh(
            list(groups), column, left=left, height=0.65, color=COLORS[index % len(COLORS)], label=f"Hidden rule {rule}"
        )
        for index_in_groups, (row, value) in enumerate(zip(groups, column, strict=True)):
            if value:
                ax.text(
                    left[index_in_groups] + value / 2,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )
        left += column
    ax.invert_yaxis()
    ax.set(
        yticks=list(groups),
        yticklabels=["Unassigned" if group < 0 else f"Learned group {group}" for group in groups],
        xlabel="Number of identities",
    )
    ax.set_title("Which hidden rules are mixed within each learned group?", pad=32)
    ax.legend(ncols=min(len(rules), 4), fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.01))
    ax.grid(axis="x", alpha=0.16)
    return (
        fig,
        "Each identity appears once. Colors are the generating rules; learned group numbers are arbitrary. Mixed colors show rules combined, and the same color in several bars shows a rule split across groups.",
        {},
    )


def group_predictions(bundle: Bundle, spec: dict):
    """Distinguish shuffled identities from the stable arm's ordinary test."""
    if "group_counts" not in bundle.record["metrics"]:
        return numeric(bundle, spec)
    pairs = bundle.pairs(spec["target"])
    frames = {frame["ordinal"]: frame for frame in bundle.frames}
    for pair in pairs:
        if pair["population"] != "test":
            continue
        frame = frames[pair["frame"]]
        run = next((caller for caller in frame["context"] if caller["function"] == "run"), None)
        if run is None:
            raise ValueError(f"{bundle.manifest['id']} prediction is missing its run context")
        calls = [
            node
            for node in ast.walk(bundle.syntax)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and (node.func.id == "measure")
            and (node.lineno <= run["line"] <= node.end_lineno)
            and (len(node.args) >= 2)
        ]
        if len(calls) != 1 or not isinstance(calls[0].args[1], ast.Name):
            raise ValueError(f"{bundle.manifest['id']} prediction is missing its measured population")
        population = calls[0].args[1].id
        if population == "shuffled":
            pair["label"] = "Stable model, shuffled test identities"
        elif population == "test" and pair["context"].get("arm") == "stable":
            pair["label"] = "Stable identities"
        elif population == "test" and pair["context"].get("arm") == "no_regime":
            pair["label"] = "No stable identity regimes"
        else:
            raise ValueError(f"{bundle.manifest['id']} prediction has an unrecognized population: {population}")
    return numeric(bundle, spec, pairs=pairs)


def representation_geometry(bundle: Bundle, spec: dict):
    label_candidates = []
    for frame in bundle.frames:
        if frame["function"] not in {"run", "trial", "measure", "evaluate"}:
            continue
        for key in ["test_labels", "labels"]:
            value = bundle.value(frame["values"].get(key))
            if value is None or isinstance(value, dict):
                continue
            values = np.asarray(value)
            if (
                values.ndim == 1
                and len(values) >= 128
                and (values.dtype.kind in "biu")
                and (len(np.unique(values)) <= 16)
            ):
                label_candidates.append((frame, values))
                break
    if not label_candidates:
        raise ValueError("Representation geometry requires its evaluator-only held-out labels")
    label_frame, labels = label_candidates[0]
    frames = [frame for frame in bundle.frames if frame["function"] == "embedding"]
    seed = bundle.context(frames[0]).get("seed", bundle.record["seed"])
    names = spec["geometry"]
    fig, axes = axes_pair(width=10, height=4.8)
    for ax, name in zip(axes, names, strict=True):
        selected = [
            frame
            for frame in frames
            if bundle.context(frame).get("name") == name and bundle.context(frame).get("seed", seed) == seed
        ]
        arrays = [(frame, np.asarray(bundle.value(frame["result"]))) for frame in selected]
        queries = next((value for frame, value in arrays if value.ndim == 2 and len(value) == len(labels)), None)
        reference = next((value for frame, value in arrays if value.ndim == 2 and len(value) != len(labels)), None)
        if queries is None or reference is None:
            raise ValueError(f"{name}: missing paired reference and held-out vectors")
        mean = reference.mean(axis=0)
        _, singular, basis = np.linalg.svd(reference - mean, full_matrices=False)
        projected = (queries - mean) @ basis[:2].T
        variance = singular**2 / max(np.sum(singular**2), 1e-30)
        for label in np.unique(labels):
            mask = labels == label
            ax.scatter(
                projected[mask, 0],
                projected[mask, 1],
                s=8,
                alpha=0.32,
                color=plt.get_cmap("tab10")(int(label) % 10),
                label=f"True group {label}",
                rasterized=True,
            )
        ax.set(
            title=human(name),
            xlabel=f"Reference-fit PC1 ({variance[0]:.0%} variance)",
            ylabel=f"Reference-fit PC2 ({variance[1]:.0%} variance)",
        )
        ax.grid(alpha=0.12)
    axes[1].legend(fontsize=8, loc="best")
    return (
        fig,
        "Illustration for the first declared seed and first captured held-out condition. Each projection is fitted only on that arm's training/reference vectors. True groups color the held-out points but do not fit the projection. Coordinates across panels use different bases; original-space metrics above determine the checks.",
        {},
    )


def summary(bundle: Bundle, spec: dict):
    """Expose remaining measured conditions; every chosen metric stays named."""
    candidates = metric_leaves(bundle.record.get("metrics", {}))
    preferred = (
        "normalized_rmse",
        "nrmse",
        "oracle_rmse",
        "accuracy",
        "exact_accuracy",
        "adjusted_rand",
        "ari",
        "macro_f1",
        "rmse",
        "error",
        "auc",
        "probability_rmse",
    )
    groups = {}
    for path, value in candidates:
        if path[-1] in preferred and math.isfinite(value):
            groups.setdefault(path[-1], []).append((path, value))
    if not groups:
        raise ValueError(f"{bundle.manifest['id']}: needs an authored readout figure")
    key = max(groups, key=lambda name: (len(groups[name]) >= 2, -preferred.index(name)))
    selected = groups[key][:16]
    fig, ax = plt.subplots(figsize=(8, max(3.8, len(selected) * 0.34)), layout="constrained")
    labels = [" · ".join((human(part) for part in path[:-1])) or human(key) for path, value in selected]
    percent = key in {"accuracy", "exact_accuracy", "macro_f1"}
    dots(
        ax,
        labels,
        {human(key): [value for path, value in selected]},
        xlabel=human(key)
        + (
            " (higher is better)"
            if key in {"accuracy", "exact_accuracy", "ari", "adjusted_rand", "macro_f1", "auc"}
            else " (lower is better)"
        ),
        percent=percent,
    )
    return (
        fig,
        "Each displayed condition retains its metric path in the downloadable plot data. These are separately measured readouts, with no invented intermediate points or uncertainty intervals.",
        {"metric_paths": [list(path) for path, value in selected]},
    )


HANDLERS = {
    "summary": summary,
    "group-learning": group_learning,
    "group-composition": group_composition,
    "group-predictions": group_predictions,
    "representation-geometry": representation_geometry,
    "regression": numeric,
    "classification": classification,
    "trajectory": trajectory,
    "coverage": coverage,
    "temporal": temporal,
    "calibration": calibration,
    "mutation": mutation,
    "composition": composition,
}


def resolve(bundle: Bundle, selection: dict):
    """Resolve a declared shared renderer or an experiment-owned component."""
    if "renderer" in selection:
        return bundle.component(selection["renderer"])
    name = selection["mode"]
    if name not in HANDLERS:
        raise ValueError(f"Unknown evidence-plot renderer: {name}")
    return HANDLERS[name]


def render(bundle: Bundle, spec: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    bundle.labels = spec.get("labels", {})
    bundle.exclude_context = set(spec.get("exclude_context", []))
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "relflow-evidence-diagnostics",
        }
    )
    handler = resolve(bundle, spec)
    result = {
        "id": bundle.manifest["id"],
        "focus": spec["focus"],
        "comparison": spec["comparison"],
        "limit": spec["limit"],
    }
    for theme, bg, ink in [("light", "#f8faf7", "#243b35"), ("dark", "#181e1c", "#e6eee8")]:
        with plt.rc_context(
            {
                "figure.facecolor": bg,
                "axes.facecolor": bg,
                "savefig.facecolor": bg,
                "text.color": ink,
                "axes.labelcolor": ink,
                "xtick.color": ink,
                "ytick.color": ink,
                "axes.edgecolor": "#85958c",
                "grid.color": "#85958c",
                "boxplot.boxprops.color": ink,
                "boxplot.whiskerprops.color": ink,
                "boxplot.capprops.color": ink,
            }
        ):
            fig, note, interactive = handler(bundle, spec)
            fig.savefig(output / f"{bundle.manifest['id']}-{theme}.svg", bbox_inches="tight", metadata={"Date": None})
            if theme == "light":
                fig.savefig(
                    output / f"{bundle.manifest['id']}.png",
                    bbox_inches="tight",
                    dpi=130,
                    metadata={"Software": "relflow proof diagnostics"},
                )
            plt.close(fig)
            result.update(caption=note, interactive=interactive)
            if theme == "light":
                result["figures"] = [{"name": bundle.manifest["id"], "caption": note}]
            supplements = [
                (suffix, resolve(bundle, selection), spec) for suffix, selection in spec.get("supplements", {}).items()
            ]
            supplements.extend(
                (target.replace("/", "-"), numeric, dict(spec, target=target))
                for target in spec.get("additional_targets", [])
            )
            for suffix, supplement, additional in supplements:
                extra, description, _ = supplement(bundle, additional)
                name = bundle.manifest["id"] + "-" + suffix
                extra.savefig(output / f"{name}-{theme}.svg", bbox_inches="tight", metadata={"Date": None})
                if theme == "light":
                    extra.savefig(output / f"{name}.png", bbox_inches="tight", dpi=130)
                    result["figures"].append({"name": name, "caption": description})
                plt.close(extra)
    return result
