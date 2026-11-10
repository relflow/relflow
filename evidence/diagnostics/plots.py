"""Proof-specific figures generated explicitly from verified diagnostic bundles."""

from __future__ import annotations

import ast
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
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
    # One named output per comparison prevents mixing unrelated target scales.
    field = spec["target"]
    compared = [pair for pair in test if pair["field"] == field]
    if not compared:
        raise ValueError(f"{bundle.manifest['id']}: requested target {field} has no captured predictions")
    primary = compared[0]
    actual, predicted = finite(primary["actual"]), finite(primary["predicted"])
    center = math.floor(float(np.median(actual)) / 1e6) * 1e6 if abs(float(np.median(actual))) > 1e8 else 0.0
    a, p = actual - center, predicted - center
    fig, axes = axes_pair()
    bound = np.r_[a, p]
    lower, upper = float(bound.min()), float(bound.max())
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
    return fig, note, {"kind": "numeric", "conditions": samples}


def classification(bundle: Bundle, spec: dict) -> tuple[plt.Figure, str, dict]:
    pairs = bundle.pairs(spec["target"])
    pairs = [pair for pair in pairs if pair["population"] == "test"] or pairs
    if not pairs:
        raise ValueError(f"{bundle.manifest['id']}: no bound classification predictions")
    pair = next((p for p in pairs if p["population"] == "test"), pairs[0])
    a, p = pair["actual"], pair["predicted"]
    fig, axes = axes_pair()
    is_probability = p.dtype.kind in "f" and a.dtype.kind in "biu" and set(np.unique(a)) <= {0, 1}
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
            truth, probability = np.asarray(item["actual"], dtype=bool), finite(item["predicted"])
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
    normal, anomalous = np.sort(finite(normal)), np.sort(finite(anomalous))
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
    if any("n_assigned_groups" in point for point in points):
        keys = [("n_assigned_groups", "Assigned groups"), ("perplexity", "Usage perplexity")]
    elif any("assigned_groups" in point for point in points):
        keys = [("assigned_groups", "Assigned groups"), ("adjusted_rand", "Adjusted Rand index")]
    else:
        keys = [("n_committed", "Committed groups"), ("perplexity", "Usage perplexity")]
        if not any("perplexity" in point for point in points):
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
    if bundle.manifest["id"] == "P056":
        labels = ["Known targets", "All targets", "Unavailable targets", "Known targets, shuffled hints"]
        values = [m["known_accuracy"], m["all_row_accuracy"], m["unknown_accuracy"], m["shuffled_known_accuracy"]]
        dots(axes[0], labels, {"Accuracy": values}, xlabel="Fraction correctly predicted", percent=True)
        axes[0].set_xlim(-0.02, 1.02)
        run = next(frame for frame in reversed(bundle.frames) if frame["function"] == "run")
        probability, known = bundle.values(run, "probability", "known")
        for mask, label, color in [(known, "Known answers", COLORS[0]), (~known, "Unavailable answers", COLORS[1])]:
            axes[1].hist(
                probability[mask], bins=np.linspace(0, 1, 26), histtype="step", lw=1.8, color=color, label=label
            )
        axes[1].set(
            title="Confidence does not establish availability",
            xlabel="Top-answer confidence",
            ylabel="Record count",
            xlim=(0, 1),
        )
        axes[1].legend(fontsize=8)
        return (
            fig,
            "Accuracy conditions and confidence populations stay separate. An unavailable true label cannot be recovered merely by assigning high probability to an available label.",
            {},
        )
    if bundle.manifest["id"] == "P058":
        dots(
            axes[0],
            ["Intact membership", "Shuffled membership"],
            {"Normalized RMSE": [m["nrmse"], m["shuffled_nrmse"]]},
            xlabel="Numeric prediction RMSE / baseline RMSE",
            reference=(1, "Constant guess"),
        )
        names = ["empty", "all_unknown", "null", "partial_unknown"]
        run = next(frame for frame in reversed(bundle.frames) if frame["function"] == "run")
        probes = bundle.value(run["values"]["probes"])
        dots(
            axes[1],
            [human(name) for name in names],
            {
                "Prediction": [m["boundary_predictions"][name] for name in names],
                "Declared probe target": [row["target"] for row in probes],
            },
            xlabel="Numeric answer for the fixed membership probes",
        )
        return (
            fig,
            f"Known-valued sets have {m['known_exact_set_accuracy']:.1%} exact reconstruction accuracy. Boundary probes distinguish null from valued empty content. An unavailable target label has no decoder row, even when its input produces a valid numeric answer.",
            {},
        )
    if bundle.manifest["id"] == "P063":
        dots(
            axes[0],
            ["Red member", "Blue member"],
            {
                "Capacity 2": [m["capacity_2"]["probabilities"][name] for name in ["red", "blue"]],
                "Capacity 512": [m["capacity_512"]["probabilities"][name] for name in ["red", "blue"]],
                "Generating member probability": [0.9, 0.2],
            },
            xlabel="Predicted inclusion probability",
        )
        axes[0].set_xlim(0, 1.02)
        known = [m[f"coverage_{size}"]["targets.known"] for size in [2, 512]]
        unavailable = [m[f"coverage_{size}"]["targets.unavailable"] for size in [2, 512]]
        axes[1].barh([0, 1], known, color=COLORS[0], label="Known target members")
        axes[1].barh([0, 1], unavailable, left=known, color=COLORS[1], label="Unavailable target members")
        axes[1].set(
            yticks=[0, 1],
            yticklabels=["Capacity 2", "Capacity 512"],
            xlabel="Selected target-member count",
            title="Extra capacity does not supply unseen members",
        )
        axes[1].invert_yaxis()
        axes[1].legend(fontsize=8)
        return (
            fig,
            "The generator independently includes red with probability 0.9 and blue with probability 0.2. The right panel uses the separate mixed-availability evaluation; spare decoder capacity does not create a learned target label.",
            {},
        )
    raise ValueError(f"{bundle.manifest['id']}: vocabulary comparison needs an authored view")


def ranking(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    if bundle.manifest["id"] == "P017":
        labels = ["Learnable signal", "Independent noise"]
        values = [m["signal_auc"], m["noise_auc"]]
    elif bundle.manifest["id"] == "P021":
        labels = ["Hash, new identities", "Category, unavailable identities", "Shuffled-label training"]
        values = [m["hash_auc"], m["category_oov_auc"], m["shuffled_auc"]]
    elif bundle.manifest["id"] == "P038":
        fig, axes = axes_pair()
        labels = ["Null state preserved", "Null filled with zero", "Independent labels"]
        dots(
            axes[0],
            labels,
            {"Ranking AUC": [m["state_auc"], m["prefilled_auc"], m["independent_auc"]]},
            xlabel="AUC (higher is better)",
            reference=(0.5, "Chance ranking"),
        )
        dots(
            axes[1],
            labels,
            {"Accuracy": [m["state_accuracy"], m["prefilled_accuracy"], m["independent_accuracy"]]},
            xlabel="Accuracy at the declared decision rule",
            reference=(0.5, "Balanced chance"),
            percent=True,
        )
        for ax in axes:
            ax.set_xlim(-0.02, 1.03)
        return (
            fig,
            "The same three information conditions are shown on separate ranking and accuracy axes. Filling missing values with zero removes the state distinction before learning.",
            {},
        )
    else:
        return classification(bundle, spec)
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    dots(
        ax, labels, {"Held-out AUC": values}, xlabel="Ranking AUC (higher is better)", reference=(0.5, "Chance ranking")
    )
    ax.set_xlim(-0.02, 1.03)
    return (
        fig,
        "These are the proof's independent held-out ranking scores. AUC is not accuracy, and no unobserved ROC curve is drawn from these summary values.",
        {},
    )


def temporal(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    identifier = bundle.manifest["id"]
    choices = {
        "P042": (
            ["Day only", "Hour only", "Day and hour"],
            ["day_only_accuracy", "hour_only_accuracy", "composed_accuracy"],
        ),
        "P043": (["Day only", "Additional calendar context"], ["day_only_accuracy", "with_week_of_month_accuracy"]),
        "P044": (
            ["Actual periodic dates", "Dates permuted between records"],
            ["periodic_accuracy", "permuted_accuracy"],
        ),
        "P045": (["Day of year only", "Weekday context supplied"], ["day_only_accuracy", "with_weekday_accuracy"]),
    }
    labels, keys = choices[identifier]
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
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return [(path, float(value))]
    return []


def calibration(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    identifier = bundle.manifest["id"]
    fig, axes = axes_pair()
    if identifier == "P060":
        labels = ["Capacity 2, known", "Capacity 2, novel", "Capacity 128, known", "Capacity 128, novel"]
        keys = ["capacity_2_known", "capacity_2_novel", "capacity_128_known", "capacity_128_novel"]
        dots(
            axes[0],
            labels,
            {"Mean predicted probability": [m[key]["mean_probability"] for key in keys]},
            xlabel="Probability of yes",
            reference=(m["prior"], "Generating prior"),
        )
        axes[0].set_xlim(0, 1.02)
        dots(
            axes[1],
            labels,
            {"Excess NLL": [m[key]["excess_nll"] for key in keys]},
            xlabel="NLL minus the generating-prior NLL",
        )
        return (
            fig,
            "The reference is the generator's fixed prior. Unused vocabulary capacity and an unavailable label are separate conditions; these mean readouts are not a reliability curve.",
            {},
        )
    if identifier == "P064":
        labels = ["Fallback exposed during training", "Fallback unexposed"]
        dots(
            axes[0],
            labels,
            {
                "Known members": [m["augmented_known"], m["control_known"]],
                "Unknown members": [m["augmented_unknown"], m["control_unknown"]],
            },
            xlabel="RMSE against the generator's answer",
            reference=(m["constant_rmse"], "Constant-guess RMSE"),
        )
        dots(
            axes[1],
            labels,
            {
                "Duplicate/identity drift": [
                    m["augmented_duplicate_and_identity_drift"],
                    m["control_duplicate_and_identity_drift"],
                ]
            },
            xlabel="Maximum output change under the declared invariant",
        )
        return (
            fig,
            "Prediction error tests whether fallback behavior is useful. The separate drift panel tests whether renaming unknown members or duplicating them changes that behavior.",
            {},
        )
    candidates = []
    for frame in bundle.frames:
        if frame["function"] != "score":
            continue
        values = frame["values"]
        p = bundle.value(values.get("p", values.get("predicted")))
        oracle = bundle.value(values.get("oracle", values.get("expected")))
        if p is None or oracle is None or isinstance(p, dict) or isinstance(oracle, dict):
            continue
        p, oracle = np.asarray(p), np.asarray(oracle)
        if p.shape == oracle.shape and p.ndim == 1 and len(p) > 1:
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
    if bundle.manifest["id"] == "P061":
        if len(selected) != 4:
            raise ValueError("Cold-start calibration requires both identity groups at both training rates")
        labels = [
            f"{rate:.0%} unavailable training inputs · {group} identities"
            for rate in [0.01, 0.2]
            for group in ["known", "unseen"]
        ]
        axes[0].set_title(caption(labels[0]))
    errors = [float(np.sqrt(np.mean((p - oracle) ** 2))) for _, p, oracle in selected]
    dots(
        axes[1],
        labels,
        {"Probability RMSE": errors},
        xlabel="RMSE against the known conditional probability",
    )
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


def admission(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    fig, axes = axes_pair()
    if bundle.manifest["id"] == "P059":
        labels = ["Before admission", "Immediately after admission", "After continued training"]
        new = [
            m["new_accuracy_before_admission"],
            m["new_accuracy_immediately_after_admission"],
            m["final_new_accuracy"],
        ]
        old = [m["initial_accuracy"], m["initial_accuracy"], m["final_old_accuracy"]]
        # Initial old-class accuracy is one measurement, not a new post-admission evaluation.
        dots(axes[0], labels, {"New-class accuracy": new}, xlabel="Correct answers for new classes", percent=True)
        dots(
            axes[1],
            ["Before continuation", "After continuation"],
            {"Old-class accuracy": [old[0], old[-1]]},
            xlabel="Correct answers for previously learned classes",
            percent=True,
        )
    else:
        labels = ["Cold model", "After seen-class training", "After mixed continuation", "All-class training control"]
        dots(
            axes[0],
            labels,
            {
                "Withheld-class accuracy": [
                    m[key]["accuracy"]
                    for key in ["cold_withheld", "initial_withheld", "final_withheld", "control_withheld"]
                ]
            },
            xlabel="Accuracy on the initially withheld class",
            percent=True,
        )
        dots(
            axes[1],
            ["After initial training", "After continuation", "All-class control"],
            {"Seen-class accuracy": [m[key]["accuracy"] for key in ["initial_seen", "final_seen", "control_seen"]]},
            xlabel="Accuracy on the initially seen classes",
            percent=True,
        )
    for ax in axes:
        ax.set_xlim(-0.02, 1.02)
    return (
        fig,
        "The stages are discrete measured states. Admission is separated from training, and previously learned classes have their own retention panel. Phase budgets and cumulative training remain distinct in the run record.",
        {},
    )


def cohorts(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    fig, axes = axes_pair()
    if bundle.manifest["id"] == "P086":
        for index, (arm, value) in enumerate(m["arms"].items()):
            history = value["history"]
            axes[0].plot(
                range(len(history)),
                [row["scores"]["A"]["nrmse"] for row in history],
                marker=MARKERS[index],
                color=COLORS[index],
                label=human(arm),
            )
        axes[0].set(
            xticks=[0, 1, 2],
            xticklabels=["After A", "After B", "After C"],
            ylabel="Cohort A normalized RMSE",
            title="What happens to the first cohort?",
        )
        axes[0].legend(fontsize=8)
        arms = list(m["arms"])
        dots(
            axes[1],
            [human(arm) for arm in arms],
            {
                f"Cohort {cohort}": [m["arms"][arm]["history"][-1]["scores"][cohort]["nrmse"] for arm in arms]
                for cohort in ["A", "B", "C"]
            },
            xlabel="Final normalized RMSE, each fixed cohort",
        )
        return (
            fig,
            "Cohort A is evaluated on the same held-out population after every phase. The final panel keeps all cohorts separate, including cohorts the continued-A control never learns.",
            {},
        )
    keys = sorted(m, key=int)
    for index, stage in enumerate(["cold", "admitted", "learned", "shuffled"]):
        present = [key for key in keys if stage in m[key]]
        axes[0].plot(
            [int(key) for key in present],
            [m[key][stage]["oracle_rmse"] for key in present],
            marker=MARKERS[index],
            color=COLORS[index],
            label=human(stage),
        )
    axes[0].set(
        xticks=list(map(int, keys)),
        xlabel="New identity cohort",
        ylabel="RMSE against its conditional oracle",
        title="Availability and learning are separate",
    )
    axes[0].legend(fontsize=8)
    present = [key for key in keys if "admission_max_drift" in m[key]]
    dots(
        axes[1],
        [f"Cohort {key}" for key in present],
        {"Maximum admission drift": [m[key]["admission_max_drift"] for key in present]},
        xlabel="Largest paired output change at admission",
    )
    return (
        fig,
        "Only measured stages are drawn for each cohort. The admission drift compares the same observations immediately before and after admitting their identities.",
        {},
    )


def mutation(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    identifier = bundle.manifest["id"]
    fig, axes = axes_pair(width=10, height=4.8)
    if identifier in {"P046", "P049"}:
        edits = m["edits"]
        if identifier == "P046":
            labels = [row["edit"] for row in edits]
            dots(
                axes[0],
                labels,
                {"Maximum paired drift": [row["max_prediction_drift"] for row in edits]},
                xlabel="Maximum absolute output change",
            )
            dots(
                axes[1],
                ["Learned source", "Complete reset"],
                {"Normalized RMSE": [m["source"]["nrmse"], m["controls"]["complete_reset"]["nrmse"]]},
                xlabel="Test RMSE / constant-guess RMSE",
                reference=(1, "Constant guess"),
            )
        else:
            labels = [row["edit"] for row in edits]
            dots(
                axes[0],
                labels,
                {"Normalized RMSE": [row["scores"]["nrmse"] for row in edits]},
                xlabel="Test RMSE / constant-guess RMSE",
                reference=(1, "Constant guess"),
            )
            names = list(m["controls"])
            dots(
                axes[1],
                [human(name) for name in names],
                {"Normalized RMSE": [m["controls"][name]["nrmse"] for name in names]},
                xlabel="Control error on the same test records",
            )
        fig.set_size_inches(10, max(4.8, len(edits) * 0.3))
        return (
            fig,
            "Each edit is evaluated on matched held-out records without post-edit fitting. Exact restoration is a paired output test; destructive controls and the source prerequisite remain separate.",
            {},
        )
    targets = (
        ["w", "u"] if identifier == "P047" else ["u", "v"] if identifier == "P048" else [None, "only_a_distance_nrmse"]
    )
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


def paired_seeds(bundle: Bundle, spec: dict):
    trials = bundle.record["metrics"]["trials"]
    fig, axes = axes_pair()
    for index, trial in enumerate(trials):
        values = [trial["intact_nrmse"], trial["broken_nrmse"]]
        axes[0].plot([0, 1], values, marker=MARKERS[index], color=COLORS[index], lw=1.8, label=f"Seed {trial['seed']}")
    axes[0].axhline(1, color="gray", ls=":")
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Intact keys", "Broken keys"],
        ylabel="Normalized RMSE",
        title="Every declared seed stays visible",
    )
    axes[0].legend(fontsize=8)
    dots(
        axes[1],
        [f"Seed {trial['seed']}" for trial in trials],
        {"Broken minus intact": [trial["corruption_gap"] for trial in trials]},
        xlabel="Increase in normalized RMSE when keys break",
    )
    return (
        fig,
        "Lines join the two measured conditions for one seed. These are three paired trials, not confidence intervals or points along a training trajectory.",
        {},
    )


def sensor_data(bundle: Bundle) -> dict:
    frames = [frame for frame in bundle.frames if frame["function"] == "prediction"]
    if len(frames) != 19:
        raise ValueError("P074 requires its eight paired visibility/poison controls and three stress readouts")
    result = {}
    for index, frame in enumerate(frames):
        rows = bundle.value(frame["values"]["rows"])
        prediction = finite(bundle.value(frame["result"]))
        if index < 16 and index % 2 == 1:
            continue
        key = "".join("0" if rows[0][f"hide_{name}"] else "1" for name in ["first", "second", "third"])
        if index >= 16:
            key = ["nuisance_permuted", "first_sensor_corrupted", "corrupted_sensor_hidden"][index - 16]
        item = {"rows": rows, "prediction": prediction, "target": finite([row["target"] for row in rows])}
        if index < 16:
            precision = np.array([not rows[0][f"hide_{name}"] for name in ["first", "second", "third"]]) / np.square(
                [0.8, 1, 1.2]
            )
            variance = 1 / (1 + precision.sum())
            readings = finite([[row[name] for name in ["first", "second", "third"]] for row in rows])
            item.update(oracle_mean=readings @ precision * variance, oracle_variance=float(variance))
            metric = bundle.record["metrics"]["patterns"][key]
            np.testing.assert_allclose(
                np.mean((prediction - item["target"]) ** 2), metric["mse"], rtol=1e-10, atol=1e-12
            )
            np.testing.assert_allclose(
                np.mean((prediction - item["oracle_mean"]) ** 2), metric["distance_to_oracle"], rtol=1e-10, atol=1e-12
            )
        result[key] = item
    return result


def fusion(bundle: Bundle, spec: dict):
    data = sensor_data(bundle)
    fig = plt.figure(figsize=(13, 7.2), layout="constrained")
    grid = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.055])
    densities = []
    for index, key in enumerate(["111", "110", "101", "011", "100", "010", "001", "000"]):
        ax = fig.add_subplot(grid[index // 4, index % 4])
        d = data[key]
        if key == "000":
            ax.scatter([row["nuisance"] for row in d["rows"]], d["prediction"], s=3, alpha=0.25, color=COLORS[1])
            ax.axhline(0, color="gray", ls="--")
            ax.set_xlabel("Irrelevant nuisance")
        else:
            densities.append(
                ax.hexbin(
                    d["oracle_mean"],
                    d["prediction"],
                    gridsize=32,
                    extent=(-3.5, 3.5, -3.5, 3.5),
                    mincnt=1,
                    cmap="viridis",
                    rasterized=True,
                )
            )
            ax.plot([-3.5, 3.5], [-3.5, 3.5], ls="--", color="gray", lw=1)
            ax.set_xlabel("Conditional mean")
        ax.set(
            ylabel="Model prediction",
            xlim=(-3.5, 3.5),
            ylim=(-3.5, 3.5),
            title=key
            + (
                ": withheld pattern" if key == "101" else ": no useful sensors" if key == "000" else ": seen visibility"
            ),
        )
        ax.set_aspect("equal")
        ax.tick_params(labelsize=9)
        distance = bundle.record["metrics"]["patterns"][key]["distance_to_oracle"]
        ax.text(0.03, 0.96, f"Oracle distance MSE {distance:.3f}", transform=ax.transAxes, va="top", fontsize=8)
    scale = LogNorm(vmin=1, vmax=max(float(density.get_array().max()) for density in densities))
    for density in densities:
        density.set_norm(scale)
    fig.colorbar(densities[0], cax=fig.add_subplot(grid[:, 4]), label="Records per hexagon (log)")
    samples = {
        key: {
            "prediction": d["prediction"][:16].tolist(),
            "target": d["target"][:16].tolist(),
            "oracle_mean": d["oracle_mean"][:16].tolist(),
            "oracle_variance": d["oracle_variance"],
            "readings": [[row[name] for name in ["first", "second", "third"]] for row in d["rows"][:16]],
        }
        for key, d in data.items()
        if len(key) == 3
    }
    return (
        fig,
        "All eight conditions use the same 2,048 held-out observations. Hexagons share a count scale. With no useful sensor, the optimal mean is zero; that panel exposes any response to nuisance. Per-pattern errors are recomputed from retained predictions and verified against the proof.",
        {"kind": "fusion", "patterns": samples},
    )


def fusion_controls(bundle: Bundle, spec: dict):
    data = sensor_data(bundle)
    fig, axes = axes_pair(width=10, height=4.6)
    cases = [
        ("111", "All sensors clean"),
        ("first_sensor_corrupted", "First sensor +6"),
        ("011", "First sensor hidden"),
        ("corrupted_sensor_hidden", "Corrupted sensor hidden"),
    ]
    errors = [(data[key]["prediction"] - data[key]["target"]) ** 2 for key, _ in cases]
    axes[0].boxplot(
        errors,
        orientation="horizontal",
        tick_labels=[label for _, label in cases],
        showfliers=False,
        medianprops={"color": COLORS[1]},
    )
    axes[0].set(xlabel="Squared prediction error", title="Paired corruption and removal", xscale="symlog")
    for key, reference, label, color in [
        ("first_sensor_corrupted", "111", "Visible corruption", COLORS[1]),
        ("corrupted_sensor_hidden", "011", "Corrupted reading hidden", COLORS[2]),
    ]:
        axes[1].scatter(
            data[reference]["prediction"],
            data[key]["prediction"] - data[reference]["prediction"],
            s=6,
            alpha=0.2,
            color=color,
            label=label,
        )
    axes[1].axhline(0, ls="--", color="gray")
    axes[1].set(
        xlabel="Prediction before intervention",
        ylabel="Paired prediction change",
        title="Does hiding restore the corresponding output?",
    )
    axes[1].legend(fontsize=8)
    return (
        fig,
        "Boxes show medians, quartiles, and 1.5-IQR whiskers, with outliers omitted from that display. The scatter includes every pair. Hidden corruption is compared with the same clean two-sensor condition, which has less evidence than the all-clean condition.",
        {},
    )


def anomaly_data(bundle: Bundle) -> dict:
    frames = [frame for frame in bundle.frames if frame["function"] == "predict"]
    if len(frames) != 6:
        raise ValueError("P076 requires validation, test, and rare-valid predictions for both arms")
    run = next(frame for frame in reversed(bundle.frames) if frame["function"] == "run")
    broken = finite(bundle.value(run["values"]["broken"]))
    result = {}
    for index, frame in enumerate(frames):
        arm = "informative" if index < 3 else "shuffled_training"
        split = ["validation", "test", "tail"][index % 3]
        rows = bundle.value(frame["values"]["rows"])
        prediction = finite(bundle.value(frame["result"]))
        target = finite([row["y"] for row in rows])
        d = {"rows": rows, "prediction": prediction, "target": target, "score": np.abs(prediction - target)}
        if split == "test":
            d.update(anomalous_target=broken, anomalous_score=np.abs(prediction - broken))
        result.setdefault(arm, {})[split] = d
    for arm, d in result.items():
        m = bundle.record["metrics"][arm]
        threshold = np.quantile(d["validation"]["score"], 0.95)
        np.testing.assert_allclose(threshold, m["validation_threshold"], rtol=1e-12)
        for key, score in [
            ("normal_false_positive_rate", d["test"]["score"]),
            ("detection_recall", d["test"]["anomalous_score"]),
            ("tail_false_positive_rate", d["tail"]["score"]),
        ]:
            np.testing.assert_allclose(np.mean(score > threshold), m[key], rtol=1e-12)
    return result


def anomaly(bundle: Bundle, spec: dict):
    data = anomaly_data(bundle)
    fig, axes = axes_pair(width=10, height=4.8)
    d = data["informative"]
    threshold = bundle.record["metrics"]["informative"]["validation_threshold"]
    scores = {"normal": d["test"]["score"], "anomalous": d["test"]["anomalous_score"], "tail": d["tail"]["score"]}
    bins = np.linspace(0, max(np.quantile(scores["anomalous"], 0.995), 1), 65)
    for index, (key, label) in enumerate(
        [("normal", "Normal test"), ("anomalous", "Shuffled-answer test"), ("tail", "Rare but valid test")]
    ):
        axes[0].hist(scores[key], bins=bins, density=True, histtype="step", lw=1.7, color=COLORS[index], label=label)
    axes[0].axvline(threshold, color=COLORS[3], ls="--", label="Validation 95th percentile")
    axes[0].set(
        title="Inconsistency and rarity have different scores",
        xlabel="Absolute reconstruction residual",
        ylabel="Density",
        yscale="symlog",
        xlim=(0, bins[-1]),
    )
    axes[0].legend(fontsize=8)
    for index, (arm, values) in enumerate(data.items()):
        fpr, tpr = roc(values["test"]["score"], values["test"]["anomalous_score"])
        measured = bundle.record["metrics"][arm]["anomaly_auc"]
        np.testing.assert_allclose(np.trapezoid(tpr, fpr), measured, atol=1e-12)
        axes[1].plot(fpr, tpr, color=COLORS[index], lw=1.8, label=f"{human(arm)} (AUC {measured:.3f})")
    m = bundle.record["metrics"]["informative"]
    axes[1].plot([0, 1], [0, 1], ls=":", color="gray", label="Target-only baseline (AUC 0.500)")
    axes[1].scatter(
        m["normal_false_positive_rate"],
        m["detection_recall"],
        color=COLORS[3],
        s=45,
        zorder=5,
        label="Validation-selected threshold",
    )
    axes[1].set(
        title="Ranking and the chosen operating point",
        xlabel="Normal false-alarm rate",
        ylabel="Anomaly recall",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(fontsize=8, loc="lower right")
    return (
        fig,
        "Each test population has 2,048 observations. Histogram display ends at the anomaly-score 99.5th percentile and uses a symlog density axis; ROC curves use every score. Threshold rates and ROC areas are independently recomputed from the retained observations.",
        {"kind": "anomaly", "threshold": threshold, **{key: values.tolist() for key, values in scores.items()}},
    )


def anomaly_context(bundle: Bundle, spec: dict):
    d = anomaly_data(bundle)["informative"]["test"]
    fig, axes = axes_pair(width=10, height=4.4)
    indices = [i for i, row in enumerate(d["rows"]) if row["group"] == "amber"][:256]
    x = finite([d["rows"][i]["x"] for i in indices])
    order = np.argsort(x)
    for index, (ax, key, title) in enumerate(
        zip(
            axes,
            ["target", "anomalous_target"],
            ["Normal records follow context", "Shuffling breaks the relationship"],
            strict=True,
        )
    ):
        ax.scatter(x, d[key][indices], s=12, alpha=0.45, color=COLORS[index], label="Observed answer")
        ax.plot(x[order], d["prediction"][indices][order], lw=1.8, color=COLORS[2], label="Actual model predictions")
        ax.plot([-2, 2], [-1.9, 0.9], ls="--", color="gray", label="Amber generator mean")
        ax.set(title=title, xlabel="Visible x", ylabel="Observed or predicted y", xlim=(-2.08, 2.08), ylim=(-3.5, 3.5))
    axes[1].legend(fontsize=8, loc="lower right")
    return (
        fig,
        "Both panels use the first 256 amber examples in their original execution order. Inputs and predictions stay paired; only the observed answer changes. The complete test population's target marginal is exactly preserved by shuffling.",
        {},
    )


def anomaly_variance(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    fig, axes = axes_pair(width=10, height=4.7)
    groups = list(m["global"]["false_positive_by_group"])
    dots(
        axes[0],
        groups,
        {
            human(arm): [m[arm]["false_positive_by_group"][group] for group in groups]
            for arm in ["global", "conditional", "marginal"]
        },
        xlabel="Normal false-alarm rate within each group",
        percent=True,
    )
    axes[0].set_xlim(
        0,
        max(
            0.12,
            max(
                value
                for arm in ["global", "conditional", "marginal"]
                for value in m[arm]["false_positive_by_group"].values()
            )
            * 1.1,
        ),
    )
    frames = [frame for frame in bundle.frames if frame["function"] == "operating"]
    for index, (frame, arm) in enumerate(zip(frames, ["global", "conditional", "marginal"], strict=True)):
        normal, anomalous = bundle.values(frame, "normal", "anomalous")
        fpr, tpr = roc(normal, anomalous)
        np.testing.assert_allclose(np.trapezoid(tpr, fpr), m[arm]["auc"], atol=1e-12)
        axes[1].plot(fpr, tpr, color=COLORS[index], lw=1.8, label=f"{human(arm)} (AUC {m[arm]['auc']:.3f})")
        axes[1].scatter(m[arm]["false_positive_rate"], m[arm]["recall"], color=COLORS[index], s=35, zorder=5)
    axes[1].set(
        xlabel="Normal false-alarm rate",
        ylabel="Anomaly recall",
        title="Ranking with validation-chosen thresholds",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(fontsize=8, loc="lower right")
    return (
        fig,
        "Group false alarms use fixed held-out memberships. Curves use retained scores and their areas match the proof. Dots use thresholds chosen only from clean validation residuals; conditional thresholds adjust for each group's normal noise level.",
        {},
    )


def composition(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    identifier = bundle.manifest["id"]
    cells = []
    if identifier in {"P027", "P031"}:
        for key, value in m.items():
            parts = key.split("/")
            if parts[-1] != "nrmse":
                continue
            if identifier == "P027":
                panel, row, column = "Intact requests", parts[1], parts[2]
            else:
                panel, row, column = "Request conditions", parts[1], parts[0]
            cells.append((panel, row, column, float(value), False))
    elif identifier == "P071":
        for arm in ["complete", "restricted"]:
            values = m[arm]
            # The proof records one scalar metric group per group/operation cell.
            for path, value in metric_leaves(values):
                if path[-1] not in {"normalized_rmse", "nrmse"}:
                    continue
                tokens = "/".join(path[:-1]).replace("cells/", "").replace("_", "/").split("/")
                group = next((token for token in tokens if token in {"amber", "blue", "coral"}), None)
                operation = next((token for token in tokens if token in {"min", "mean", "max"}), None)
                if group and operation:
                    cells.append(
                        (
                            human(arm) + " training",
                            group,
                            operation,
                            value,
                            (group, operation) in {("amber", "mean"), ("blue", "min"), ("coral", "max")},
                        )
                    )
    else:
        for arm, values in m.items():
            if not isinstance(values, dict):
                continue
            for length in ["familiar_lengths", "unseen_lengths"]:
                if length not in values:
                    continue
                for request in ["familiar_requests", "unseen_requests"]:
                    if request in values[length]:
                        cells.append(
                            (
                                human(arm),
                                human(length),
                                human(request),
                                values[length][request]["normalized_rmse"],
                                False,
                            )
                        )
    if not cells:
        raise ValueError(f"{identifier}: no authored composition cells matched the captured record")
    panels = list(dict.fromkeys(item[0] for item in cells))
    rows = list(dict.fromkeys(item[1] for item in cells))
    columns = list(dict.fromkeys(item[2] for item in cells))
    fig = plt.figure(figsize=(max(7.8, len(panels) * 4.5), 4.8), layout="constrained")
    grid = fig.add_gridspec(1, len(panels) + 1, width_ratios=[*[1] * len(panels), 0.045])
    maximum = max(value for _, _, _, value, _ in cells)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#8a9691")
    for index, panel in enumerate(panels):
        ax = fig.add_subplot(grid[0, index])
        matrix = np.full((len(rows), len(columns)), np.nan)
        flags = {}
        for p, row, column, value, flag in cells:
            if p == panel:
                matrix[rows.index(row), columns.index(column)] = value
                flags[(rows.index(row), columns.index(column))] = flag
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
    if identifier == "P071":
        note += " Asterisks identify combinations omitted from restricted training and included in complete training."
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


def grouping(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    if bundle.manifest["id"] in {"P095", "P112"}:
        conditions = list(m["conditions"])
        fig, axes = axes_pair(width=10, height=4.8)
        first = m["conditions"][conditions[0]]["arms"]
        identity_arm = "group" if "group" in first else "cluster"
        arms = [identity_arm, "category", "oracle", "shuffled"]
        dots(
            axes[0],
            [human(c) for c in conditions],
            {
                human(arm): [m["conditions"][condition]["arms"][arm]["nrmse"] for condition in conditions]
                for arm in arms
            },
            xlabel="Held-out normalized RMSE",
            reference=(1, "Constant-guess scale"),
        )
        dots(
            axes[1],
            [human(c) for c in conditions],
            {
                human(arm): [m["conditions"][condition]["arms"][arm]["rare_identity_nrmse"] for condition in conditions]
                for arm in arms
            },
            xlabel="Mean nRMSE for eight least-exposed identities",
        )
        return (
            fig,
            "Dense, sparse, and skewed training populations are separate fits. The right panel fixes the eight least-exposed identities using training counts; the true-group oracle is a separately informed reference.",
            {},
        )
    fig = plt.figure(figsize=(10, 4.8), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.04])
    if "group_counts" in m:
        counts = m["group_counts"]
        count = "4" if "4" in counts else next(iter(counts))
        m = counts[count]
        width = int(count)
    else:
        width = 8
    matrices = []
    for arm in ["stable", "no_regime"]:
        truth = np.asarray(m[arm]["expected_partition"])
        assigned = np.asarray(m[arm]["assignments"])
        rows = sorted(set(truth))
        columns = list(range(max(width, int(assigned.max()) + 1)))
        if np.any(assigned < 0):
            columns = [-1, *columns]
        matrices.append(
            (
                rows,
                columns,
                np.array([[np.sum((truth == row) & (assigned == column)) for column in columns] for row in rows]),
            )
        )
    maximum = max(int(matrix.max()) for _, _, matrix in matrices)
    for index, (arm, (rows, columns, matrix)) in enumerate(zip(["stable", "no_regime"], matrices, strict=True)):
        ax = fig.add_subplot(grid[0, index])
        image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=maximum, aspect="auto")
        ax.set(
            xticks=range(len(columns)),
            xticklabels=["Missing" if c < 0 else str(c) for c in columns],
            yticks=range(len(rows)),
            yticklabels=[f"True group {r}" for r in rows],
            xlabel="Assigned group (arbitrary label)",
            title=f"{human(arm)}\nARI {m[arm]['adjusted_rand']:.3f}",
        )
        for i, j in np.ndindex(matrix.shape):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if matrix[i, j] < maximum * 0.6 else "#18231e",
            )
    fig.colorbar(image, cax=fig.add_subplot(grid[0, 2]), label="Identity count")
    return (
        fig,
        "Each identity contributes once to the matrix. Both arms use the same count scale. Group labels have no intrinsic order; ARI compares partitions without requiring their names to match.",
        {},
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
    truth, assigned = np.asarray(stable["expected_partition"]), np.asarray(stable["assignments"])
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
            and node.func.id == "measure"
            and node.lineno <= run["line"] <= node.end_lineno
            and len(node.args) >= 2
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


def retrieval(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    fig, axes = axes_pair(width=10, height=4.6)
    for ax, hops in zip(axes, [1, 2], strict=True):
        values = m[f"hops_{hops}"]["ladder"]
        capacities = sorted(int(key) for key in values)
        for index, (condition, label) in enumerate(
            [("intact", "Intact keys"), ("broken", "Broken keys"), ("permuted", "Complete records reordered")]
        ):
            y = []
            for capacity in capacities:
                item = values[str(capacity)]
                measured = item.get(condition, item.get("reordered") if condition == "permuted" else None)
                y.append(None if measured is None else measured["normalized_rmse"])
            present = [i for i, value in enumerate(y) if value is not None]
            if present:
                ax.plot(
                    [capacities[i] for i in present],
                    [y[i] for i in present],
                    marker=MARKERS[index],
                    color=COLORS[index],
                    lw=1.8,
                    label=label,
                )
        source = [
            values[str(c)]["intact"]["source_mean_rmse"] / values[str(c)]["intact"]["baseline_rmse"] for c in capacities
        ]
        ax.plot(capacities, source, ls=":", color="gray", label="Source-mean shortcut")
        ax.axhline(0.5, ls="--", color=COLORS[3], label="Intact learning gate")
        ax.set(
            xlabel="Source collection capacity",
            ylabel="Test RMSE / baseline RMSE",
            title=f"{hops}-hop retrieval",
            xscale="log",
        )
        ax.set_xticks(capacities, labels=capacities)
        ax.minorticks_off()
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8)
    return (
        fig,
        "Every capacity is an evaluated condition. One-hop and two-hop models have separate panels. A broken-key gap and the learning gate are both required; the source-mean shortcut is an explicit baseline, not the correct retrieval answer.",
        {},
    )


def label_efficiency(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    counts = sorted(map(int, m["budgets"]))
    fig, axes = axes_pair(width=10, height=4.8)
    arms = ["frozen", "fine_tuned", "scratch", "permuted", "raw", "shuffled_labels"]
    for index, arm in enumerate(arms):
        axes[0].plot(
            counts,
            [m["budgets"][str(count)][arm]["accuracy"] for count in counts],
            marker=MARKERS[index],
            color=COLORS[index],
            lw=1.6,
            label=human(arm),
        )
    axes[0].set(
        xscale="log",
        xlabel="Labeled examples per class",
        ylabel="Independent test accuracy",
        title="How much supervision does each approach need?",
        ylim=(0, 1.03),
    )
    axes[0].set_xticks(counts, labels=counts)
    axes[0].minorticks_off()
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.18)
    dots(
        axes[1],
        [f"{count} labels per class" for count in counts],
        {
            "Fine-tuned minus scratch": [m["budgets"][str(count)]["fine_tuned_gain_over_scratch"] for count in counts],
            "Frozen minus raw": [m["budgets"][str(count)]["frozen_gain_over_raw"] for count in counts],
        },
        xlabel="Paired gain in test accuracy",
        reference=(0, "Equal accuracy"),
        percent=True,
    )
    return (
        fig,
        "Markers are the evaluated label budgets. The paired gain panel distinguishes an adaptation benefit from a frozen-representation benefit. Pretraining uses additional unlabeled observations and is recorded separately.",
        {},
    )


def lookup(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    fig, axes = axes_pair(width=10, height=4.7)
    conditions = [name for name in ["intact", "reordered", "changed_query"] if name in m]
    if not conditions:
        conditions = [name for name, value in m.items() if isinstance(value, dict) and "matches_0" in value]
    for index, name in enumerate(conditions):
        x = list(range(4))
        values = m[name]
        axes[0].plot(
            x,
            [values[f"matches_{count}"]["found_accuracy"] for count in x],
            marker=MARKERS[index],
            color=COLORS[index],
            label=human(name),
        )
        axes[1].plot(
            x,
            [values[f"matches_{count}"]["payload_rmse"] for count in x],
            marker=MARKERS[index],
            color=COLORS[index],
            label=human(name),
        )
    axes[0].set(
        title="Does a matching key exist?",
        xlabel="Number of matches",
        ylabel="Found/not-found accuracy",
        xticks=range(4),
        ylim=(0, 1.03),
    )
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set(
        title="Is the numeric answer right?",
        xlabel="Number of matches",
        ylabel="Payload RMSE (source units)",
        xticks=range(4),
    )
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.18)
    frame = next(frame for frame in bundle.frames if frame["function"] == "predict")
    rows = bundle.value(frame["values"]["rows"])
    probability, predicted = bundle.value(frame["result"])
    samples = [
        {
            "coordinate": [i],
            "input": row,
            "actual": row["answer"],
            "predicted": float(predicted[i]),
            "found_probability": float(probability[i]),
            "matches": sum(item["entity_id"] == row["query_key"] for item in row["source"]),
        }
        for i, row in enumerate(rows[:16])
    ]
    return (
        fig,
        "No match has a defined zero payload in this generator. Multiple matches require the mean of their matched values, so existence accuracy and payload error remain separate. Source-unit RMSE keeps the zero-match condition defined.",
        {
            "kind": "numeric",
            "conditions": [{"label": "Intact lookup", "field": "answer", "n": len(rows), "records": samples}],
        },
    )


def representation(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    identifier = bundle.manifest["id"]
    fig, axes = axes_pair(width=10, height=4.8)
    if identifier == "P073":
        arms = list(m["arms"])
        dots(
            axes[0],
            [human(arm) for arm in arms] + ["Prepared raw features"],
            {"Probe accuracy": [m["arms"][arm]["accuracy"] for arm in arms] + [m["raw_accuracy"]]},
            xlabel="Accuracy with eight labeled references per class",
            reference=(m["chance_accuracy"], "Four-class chance"),
            percent=True,
        )
        dots(
            axes[1],
            [human(arm) for arm in arms],
            {"Shuffled-probe accuracy": [m["arms"][arm]["shuffled_probe_accuracy"] for arm in arms]},
            xlabel="Accuracy after shuffling reference labels",
            reference=(m["chance_accuracy"], "Four-class chance"),
            percent=True,
        )
        for ax in axes:
            ax.set_xlim(0, 1.02)
        return (
            fig,
            "All representation arms use the same held-out labels and the same small reference-label budget. The raw-feature baseline and shuffled reference labels expose gains that reconstruction loss alone cannot establish.",
            {},
        )
    trials = m["trials"]
    if identifier == "P080":
        arms = list(trials[0]["arms"])
        for ax, key, label in zip(
            axes,
            ["ari", "neighbor_agreement"],
            ["Held-out adjusted Rand agreement", "Held-out neighbor agreement"],
            strict=True,
        ):
            dots(
                ax,
                [human(arm) for arm in arms],
                {f"Seed {trial['seed']}": [trial["arms"][arm][key] for arm in arms] for trial in trials},
                xlabel=label,
                reference=(0, "Chance-adjusted zero") if key == "ari" else None,
            )
        return (
            fig,
            "All three declared seeds are plotted individually. These clustering and neighborhood measurements use the original representation space; they do not rely on a two-dimensional projection.",
            {},
        )
    trained = "trained" if identifier == "P081" else "reconstruction"
    arms = list(trials[0]["arms"])
    conditions = list(trials[0]["arms"][trained]["conditions"])
    rows = [(arm, condition) for condition in conditions for arm in arms]
    fig.set_size_inches(11, max(6, len(rows) * 0.58))
    dots(
        axes[0],
        [human(condition) + "\n" + human(arm) for arm, condition in rows],
        {
            f"Seed {trial['seed']}": [trial["arms"][arm]["conditions"][condition]["ari"] for arm, condition in rows]
            for trial in trials
        },
        xlabel="Original-space held-out ARI",
        reference=(0, "Chance-adjusted zero"),
    )
    dots(
        axes[1],
        [human(condition) for condition in conditions],
        {
            f"Seed {trial['seed']}": [
                trial["arms"][trained]["conditions"][condition]["ari"]
                - trial["arms"]["raw"]["conditions"][condition]["ari"]
                for condition in conditions
            ]
            for trial in trials
        },
        xlabel="Trained minus prepared-raw ARI",
        reference=(0, "Equal grouping quality"),
    )
    return (
        fig,
        "Every declared seed, intervention, and representation arm remains visible. Paired gains compare the same seed and condition. Original-space ARI remains authoritative even when a projection looks separated.",
        {},
    )


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
            if values.ndim == 1 and len(values) >= 128 and values.dtype.kind in "biu" and len(np.unique(values)) <= 16:
                label_candidates.append((frame, values))
                break
    if not label_candidates:
        raise ValueError("Representation geometry requires its evaluator-only held-out labels")
    label_frame, labels = label_candidates[0]
    frames = [frame for frame in bundle.frames if frame["function"] == "embedding"]
    seed = bundle.context(frames[0]).get("seed", bundle.record["seed"])
    names = {
        "P073": ("reconstruction", "untrained"),
        "P080": ("aligned", "untrained"),
        "P081": ("trained", "untrained"),
        "P082": ("reconstruction", "untrained"),
    }[bundle.manifest["id"]]
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


def tradeoff(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    fig, axes = axes_pair(width=10, height=4.8)
    if bundle.manifest["id"] == "P079":
        tasks = ["additive", "narrow", "moderate", "heavy"]
        dots(
            axes[0],
            [human(task) for task in tasks],
            {
                "Number input → Number output": [m[task]["number_number"]["normalized_mae"] for task in tasks],
                "Quantile input → Number output": [m[task]["rank_number"]["normalized_mae"] for task in tasks],
                "Population-median oracle": [m[task]["oracle_mae"] / m[task]["baseline_mae"] for task in tasks],
            },
            xlabel=f"MAE / baseline MAE at {m['budget']} updates",
        )
        tasks = [task for task in tasks if "bulk_oracle_ratio" in m[task]["number_number"]]
        labels = [human(task) + " · " + part for task in tasks for part in ["bulk", "tail"]]
        dots(
            axes[1],
            labels,
            {
                human(arm): [m[task][arm][f"{part}_oracle_ratio"] for task in tasks for part in ["bulk", "tail"]]
                for arm in ["number_number", "rank_number"]
            },
            xlabel="MAE / oracle MAE within each region",
            reference=(1, "Population oracle"),
        )
        return (
            fig,
            "The main comparison uses matched 500-update models and source-unit errors normalized within each task. Bulk and tail errors are separate. Longer-budget diagnostics remain in the full record and are not mixed into these equal-budget points.",
            {},
        )
    amount = [key for key in m["input_tasks"] if key.startswith("amount_")]
    rank = [key for key in m["input_tasks"] if key.startswith("rank_")]
    dots(
        axes[0],
        [human(key.removeprefix("amount_")) for key in amount],
        {
            "Central RMSE": [m["input_tasks"][key]["central_rmse"] for key in amount],
            "Tail RMSE": [m["input_tasks"][key]["tail_rmse"] for key in amount],
        },
        xlabel="Amount prediction RMSE (source units)",
    )
    dots(
        axes[1],
        [human(key.removeprefix("rank_")) for key in rank],
        {
            "Central accuracy": [m["input_tasks"][key]["central_accuracy"] for key in rank],
            "Tail accuracy": [m["input_tasks"][key]["tail_accuracy"] for key in rank],
        },
        xlabel="Rank classification accuracy",
        percent=True,
    )
    axes[1].set_xlim(0, 1.02)
    return (
        fig,
        "Input contamination, input representation, and target task stay distinct. Central and tail populations use the proof's declared partition; regression error and classification accuracy have separate axes.",
        {},
    )


def objective_optima(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    fig, axes = axes_pair(width=10, height=4.5)
    tasks = ["number_mae", "number_mse", "quantile_mae", "quantile_mse"]
    optima = m["population_optima"]
    for index, task in enumerate(tasks):
        values = m["noisy_targets"][task]["group_predictions"]
        x = np.arange(len(values)) + (index - 1.5) * 0.07
        axes[0].scatter(x, values, color=COLORS[index], marker=MARKERS[index], s=35, label=human(task))
        axes[0].scatter(x, optima[task], color=COLORS[index], marker="x", s=40)
    axes[0].set(
        xticks=range(3),
        xticklabels=["Group 1", "Group 2", "Group 3"],
        ylabel="Predicted answer in source units",
        title="Different objectives can prefer different answers",
    )
    axes[0].legend(fontsize=8)
    dots(
        axes[1],
        [human(task) for task in tasks],
        {"Mean relative optimum error": [m["noisy_targets"][task]["mean_relative_optimum_error"] for task in tasks]},
        xlabel="Relative distance to each task's own population optimum",
    )
    return (
        fig,
        "Filled markers are learned group-mean predictions; crosses are the known population optima for the declared representation and loss. Each arm is judged against its own optimum, not against an incompatible objective's answer.",
        {},
    )


def extrapolation(bundle: Bundle, spec: dict):
    pairs = [
        pair
        for pair in bundle.pairs(spec["target"])
        if pair["population"] == "test" and "permutation" not in pair["call"]
    ]
    if not pairs:
        raise ValueError("Extrapolation requires retained source-unit predictions")
    fig, axes = axes_pair(width=10, height=4.8)
    selected = pairs[:3]
    for index, pair in enumerate(selected):
        axes[0].scatter(
            pair["actual"], pair["predicted"], s=5, alpha=0.2, color=COLORS[index], label=pair["label"], rasterized=True
        )
    values = np.concatenate([pair["actual"] for pair in selected])
    lower, upper = float(values.min()), float(values.max())
    axes[0].plot([lower, upper], [lower, upper], ls="--", color="gray")
    axes[0].set(
        xlabel="Actual source-unit answer",
        ylabel="Model source-unit prediction",
        title="Does the prediction reach the requested range?",
    )
    axes[0].legend(fontsize=8)
    axes[1].boxplot(
        [np.abs(pair["actual"] - pair["predicted"]) for pair in selected],
        orientation="horizontal",
        tick_labels=[textwrap.fill(pair["label"], 22) for pair in selected],
        showfliers=False,
        medianprops={"color": COLORS[1]},
    )
    axes[1].set(xlabel="Absolute source-unit error", title="Error remains separate for each range")
    return (
        fig,
        "Every scatter point is a captured prediction. The diagonal denotes exact answers; clipping appears as a plateau. Error boxes show medians, quartiles, and 1.5-IQR whiskers with outliers omitted from that display.",
        {},
    )


def masking(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    fig, axes = axes_pair(width=10, height=4.5)
    if bundle.manifest["id"] == "P054":
        keys = sorted(m["prefixes"], key=int)
        dots(
            axes[0],
            [f"{key} visible events" for key in keys],
            {
                "Actual prefix": [m["prefixes"][key]["nrmse"] for key in keys],
                "Swapped prefix": [m["prefixes"][key]["shuffled_nrmse"] for key in keys],
            },
            xlabel="One-step normalized RMSE",
        )
        dots(
            axes[1],
            ["Autoregressive rollout", "Last-value persistence"],
            {"Source-unit RMSE": [m["rollout"]["rmse"], m["rollout"]["persistence_rmse"]]},
            xlabel="Multi-step rollout RMSE in source units",
        )
        return (
            fig,
            "One-step predictions and repeated rollout are different tests and use separate scales. Only evaluated prefix lengths are shown; future-value poisoning remains an independent masking check.",
            {},
        )
    dots(
        axes[0],
        ["Visible context intact", "Visible context corrupted"],
        {"Normalized RMSE": [m["nrmse"], m["corrupted_context_nrmse"]]},
        xlabel="Error on the selected hidden coordinates",
        reference=(1, "Constant-guess error"),
    )
    dots(
        axes[1],
        ["Hidden values poisoned"],
        {"Maximum output drift": [m["hidden_value_drift"]]},
        xlabel="Largest change to a selected prediction",
    )
    axes[1].set_title(f"{m['selected_coordinates']:,} selected coordinates")
    return (
        fig,
        "Only the query-selected coordinates are evaluated. Corrupting visible context removes information; changing hidden values tests exclusion from input. Neither control is combined with unselected coordinates.",
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
    labels = [" · ".join(human(part) for part in path[:-1]) or human(key) for path, value in selected]
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


def unknown_fallback(bundle: Bundle, spec: dict):
    pairs = bundle.pairs("target")
    chosen = []
    labels = {
        "novel": "Unseen label: conditional mean",
        "renamed": "Unseen spelling changed",
        "nulls": "Null input: its own conditional mean",
        "test": "Known and null inputs: observed answers",
    }
    for argument in ["novel", "renamed", "nulls", "test"]:
        pair = next(item for item in pairs if item["call"] == f"prediction(model, {argument})")
        pair = dict(pair, label=labels[argument])
        if argument != "test":
            pair["actual"] = 2 * np.asarray([row["x"] for row in pair["rows"]]) - (3 if argument == "nulls" else 0)
        chosen.append(pair)
    run = next(frame for frame in bundle.frames if frame["function"] == "run")
    baseline = bundle.value(run["values"]["baseline"])
    measured = np.sqrt(np.mean((chosen[0]["actual"] - chosen[0]["predicted"]) ** 2)) / baseline
    if not np.isclose(measured, bundle.record["metrics"]["unknown"]["fallback_nrmse"]):
        raise ValueError("Unknown fallback plot disagrees with the proof's conditional-mean score")
    fig, note, interactive = numeric(bundle, spec, pairs=chosen)
    note += " For unseen labels the reference is the generator's conditional mean 2 × x; for null it is 2 × x − 3. Copied historical target labels are not the oracle for these changed inputs."
    return fig, note, interactive


def quantile_source_units(bundle: Bundle, spec: dict):
    run = next(frame for frame in bundle.frames if frame["function"] == "run")
    v = bundle.value(run["values"])
    pairs = []
    for name, label, rows, metric in [
        ("output", "Intact held-out inputs", v["visible"], "rmse"),
        (
            "control",
            "Visible inputs shuffled across answers",
            [v["visible"][int(i)] for i in v["shuffled"]],
            "shuffled_rmse",
        ),
    ]:
        a, p = np.asarray(v["targets"]), np.asarray(v[name])
        if not np.isclose(np.sqrt(np.mean((a - p) ** 2)), bundle.record["metrics"][metric]):
            raise ValueError("Source-unit plot disagrees with the recorded full-test RMSE")
        pairs.append(
            {
                "actual": a,
                "predicted": p,
                "field": "cost",
                "label": label,
                "population": "test",
                "rows": rows,
                "coordinates": [(i,) for i in range(len(a))],
            }
        )
    fig, note, interactive = numeric(bundle, spec, pairs=pairs)
    note += " These are the full held-out intact and shuffled evaluations. The separate target-tampering and checkpoint checks remain in the run record; their 128-coordinate probes do not replace the test population."
    return fig, note, interactive


def recent_history(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    populations = ["familiar", "longer_source"]
    fig, axes = axes_pair(width=10, height=4.8)
    dots(
        axes[0],
        ["Familiar source lengths", "Longer source lengths"],
        {
            "Keep recent tail": [m["tail"][p]["normalized_rmse"] for p in populations],
            "Keep oldest head": [m["head"][p]["normalized_rmse"] for p in populations],
            "Tail after retained values change": [m["tail"][p]["retained"]["normalized_rmse"] for p in populations],
        },
        xlabel="Test RMSE / constant-guess RMSE",
        reference=(1, "Constant-guess error"),
    )
    labels = [
        ("Familiar source" if p == "familiar" else "Longer source") + " · " + human(mode)
        for p in populations
        for mode in ["trim", "discarded", "retained"]
    ]
    dots(
        axes[1],
        labels,
        {
            "Paired output drift": [
                m["tail"][p][mode]["maximum_prediction_drift"]
                for p in populations
                for mode in ["trim", "discarded", "retained"]
            ]
        },
        xlabel="Maximum absolute prediction change",
        reference=(1e-5, "Declared invariance tolerance"),
    )
    axes[1].set_xscale("symlog", linthresh=1e-6)
    maximum = max(
        m["tail"][p][mode]["maximum_prediction_drift"]
        for p in populations
        for mode in ["trim", "discarded", "retained"]
    )
    axes[1].set_xlim(-2.5e-7, maximum * 3)
    axes[1].set_xticks([0, 1e-5, 1e-3, 0.1, 1], labels=["0", "10⁻⁵", "10⁻³", "0.1", "1"])
    axes[1].minorticks_off()
    return (
        fig,
        "Tail and head policies use the same fixed training budget and independent test populations. Trim removes discarded events; discarded changes only events outside the retained tail. Retained changes the tail and recomputes its answer. The drift axis is linear around zero and logarithmic beyond 0.000001; exact zero is retained.",
        {},
    )


def precision_offsets(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]["arms"]
    arms = ["zero", "wide", "centered", "float32"]
    fig, axes = axes_pair(width=10, height=4.9)
    dots(
        axes[0],
        ["Zero-centered source", "Float64 source near 10¹²", "Explicitly precentered", "Float32 ingress near 10¹²"],
        {"Intact inputs": [m[a]["nrmse"] for a in arms], "Shuffled inputs": [m[a]["shuffled_nrmse"] for a in arms]},
        xlabel="Residual RMSE / constant-guess RMSE",
        reference=(1, "Constant-guess error"),
    )
    pairs = bundle.pairs("target")
    for index, arm in enumerate(["wide", "float32"]):
        pair = next(p for p in pairs if p["context"].get("arm") == arm and p["call"] == "prediction(model, test)")
        actual, predicted = pair["actual"] - 1e12, pair["predicted"] - 1e12
        if not np.isclose(np.sqrt(np.mean((actual - predicted) ** 2)), m[arm]["residual_rmse"]):
            raise ValueError("Centered precision scatter disagrees with the proof's residual score")
        axes[1].scatter(actual, predicted, s=7, alpha=0.23, color=COLORS[index], label=human(arm), rasterized=True)
    axes[1].plot([-3.2, 3.2], [-3.2, 3.2], ls="--", color="gray")
    axes[1].set(
        xlabel="Actual target minus 10¹²",
        ylabel="Model prediction minus 10¹²",
        title="Predictions after subtracting 10¹²",
        xlim=(-3.3, 3.3),
        ylim=(-3.3, 3.3),
    )
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.18)
    return (
        fig,
        f"All four arms use matched latent draws and the same fixed budget. The wide scatter subtracts 10¹² in float64 on both axes. Float32 ingress leaves {m['float32']['input_unique_values']:,} distinct training input, versus {m['wide']['input_unique_values']:,} in the wide float64 arm; later normalization cannot recover discarded differences.",
        {},
    )


def jitter_noise(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]["arms"]
    arms = ["none", "small", "excessive"]
    conditions = ["clean", "mild_noise", "strong_noise"]
    fig, axes = axes_pair(width=10, height=4.8)
    matrix = np.array([[m[a]["conditions"][c]["nrmse"] for c in conditions] for a in arms])
    image = axes[0].imshow(matrix, cmap="YlOrRd", vmin=0, vmax=max(1, float(matrix.max())), aspect="auto")
    axes[0].set(
        xticks=range(3),
        xticklabels=["Clean", "Mild noise", "Strong noise"],
        yticks=range(3),
        yticklabels=["No jitter", "Small jitter (0.08)", "Excessive jitter (1.5)"],
        xlabel="Held-out input condition",
        title="Every jitter × evaluation condition",
    )
    for i, j in np.ndindex(matrix.shape):
        axes[0].text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="#243b35")
    fig.colorbar(image, ax=axes[0], fraction=0.035, label="Test nRMSE (lower is better)")
    dots(
        axes[1],
        ["No jitter", "Small jitter", "Excessive jitter"],
        {
            "Training RMSE": [m[a]["train_rmse"] for a in arms],
            "Clean held-out RMSE": [m[a]["conditions"]["clean"]["rmse"] for a in arms],
        },
        xlabel="RMSE in original target units",
    )
    return (
        fig,
        "The matrix includes all nine measured conditions, with 1 representing constant-guess error. The right panel separates training and clean held-out error on the same source-unit scale. Jitter stops at evaluation; adding too much can degrade the learned relationship.",
        {},
    )


def context_extrapolation(bundle: Bundle, spec: dict):
    m = bundle.record["metrics"]
    counts = [2, 4, 8]
    fig, axes = axes_pair(width=10, height=4.8)
    for ax, region, title in zip(
        axes,
        ["inside", "outside"],
        ["Queries inside the context range", "Queries outside the context range"],
        strict=True,
    ):
        series = {
            "Intact context": [m["context"][f"count_{n}/{region}"]["normalized_rmse"] for n in counts],
            "Context swapped across tasks": [
                m["context"][f"count_{n}/{region}"]["swapped_context"]["normalized_rmse"] for n in counts
            ],
            "Query-only model": [m["query_only"][f"count_{n}/{region}"]["normalized_rmse"] for n in counts],
            "Least-squares reference": [
                m["context"][f"count_{n}/{region}"]["least_squares"]["normalized_rmse"] for n in counts
            ],
        }
        for index, (name, values) in enumerate(series.items()):
            ax.plot(counts, values, marker=MARKERS[index], color=COLORS[index], label=name, lw=1.7)
        ax.set(
            xticks=counts,
            xlabel="Visible context examples",
            ylabel="RMSE / task baseline RMSE",
            title=title,
            ylim=(0, 1.65),
        )
        ax.axhline(1, color="gray", ls=":")
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8)
    return (
        fig,
        "Markers are the three evaluated context counts; both panels share the same error scale. The least-squares reference uses the supplied examples directly. The query-only control and swapped context isolate dependence on task information. Extrapolation is scored separately from interpolation, including its unmet checks.",
        {},
    )


def duration(bundle: Bundle, spec: dict) -> tuple[plt.Figure, str, dict]:
    """Compare elapsed-time readouts with the prepared and shuffled numeric controls."""
    pairs = bundle.pairs(spec["target"])
    arms = ("duration", "static", "number", "shuffled")
    labels = ("Two-query Duration", "Configured-start Duration", "Prepared Number", "Shuffled training Number")
    measured = bundle.record["metrics"]
    selected = []
    for arm in arms:
        matches = [pair for pair in pairs if pair["context"].get("arm") == arm and "shifted" not in pair["call"]]
        if len(matches) != 1:
            raise ValueError(f"P099: expected one held-out prediction array for {arm}, got {len(matches)}")
        selected.append(matches[0])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    primary = selected[0]
    actual, predicted = finite(primary["actual"]), finite(primary["predicted"])
    density = axes[0].hexbin(actual, predicted, gridsize=30, mincnt=1, bins="log", cmap="viridis")
    fig.colorbar(density, ax=axes[0], fraction=0.04, label="Held-out rows (log count)")
    bounds = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axes[0].plot(bounds, bounds, color="gray", ls="--", lw=1)
    axes[0].set(title="Duration predicts the held-out target", xlabel="Actual target", ylabel="Predicted target")
    axes[0].set_aspect("equal", adjustable="box")
    drift = measured["arms"]["duration"]["joint_endpoint_shift_max_drift"]
    axes[0].text(
        0.03,
        0.97,
        f"n = {len(actual):,}\n371-day endpoint shift\nmax prediction drift = {drift:.3g}",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
    )
    examples = []
    for index, (pair, label) in enumerate(zip(selected, labels, strict=True)):
        truth, prediction = finite(pair["actual"]), finite(pair["predicted"])
        errors = np.sort(np.abs(truth - prediction))
        axes[1].step(
            errors, np.arange(1, len(errors) + 1) / len(errors), where="post", color=COLORS[index], label=label
        )
        examples.append(
            {
                "label": label,
                "field": spec["target"],
                "population": "test",
                "n": len(truth),
                "records": [
                    {
                        "coordinate": [i],
                        "input": pair["rows"][i],
                        "actual": float(truth[i]),
                        "predicted": float(prediction[i]),
                    }
                    for i in range(min(16, len(truth)))
                ],
            }
        )
    axes[1].set(
        title="How much error remains?",
        xlabel="Absolute target error",
        ylabel="Fraction of held-out rows",
        ylim=(0, 1.02),
    )
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].grid(alpha=0.18)
    axes[1].legend(fontsize=8, loc="lower right")
    dots(
        axes[2],
        labels,
        {"Held-out nRMSE": [measured["arms"][arm]["nrmse"] for arm in arms]},
        xlabel="RMSE / training-mean predictor RMSE",
        reference=(1, "Training-mean reference"),
    )
    axes[2].axvline(0.25, color=COLORS[0], ls="--", lw=1, label="Informative-arm criterion")
    axes[2].set(title="Does the input carry information?", xlim=(0, None))
    axes[2].legend(fontsize=8)
    note = (
        "All four arms use 2,048 independent held-out rows and 400 updates. "
        "The shuffled control permutes only training durations; test inputs remain intact. "
        "The dashed diagonal is exact prediction; the error curves retain every held-out row. "
        f"Duration versus prepared Number prediction RMSE is {measured['duration_number_prediction_rmse']:.4g}. "
        "The endpoint-shift annotation uses the same trained model. One seed; no uncertainty interval."
    )
    return fig, note, {"kind": "numeric", "conditions": examples}


def ordinal(bundle: Bundle, spec: dict) -> tuple[plt.Figure, str, dict]:
    """Show familiar and withheld rank accuracy without averaging away individual gaps."""
    measured = bundle.record["metrics"]["domains"]
    arms = ("ordinal", "number", "category")
    labels = ("Ordinal", "Rounded Number", "Category")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained", width_ratios=(1, 1.5))
    for row, domain in enumerate(("integer", "labels")):
        scores = measured[domain]
        dots(
            axes[row, 0],
            ("Familiar ranks", "Withheld ranks"),
            {
                label: [scores[arm][condition]["exact_accuracy"] for condition in ("seen", "unseen")]
                for arm, label in zip(arms, labels, strict=True)
            },
            xlabel="Exact rank accuracy",
            percent=True,
        )
        axes[row, 0].set(
            title="Integer ranks" if domain == "integer" else "Declared ordered labels", xlim=(-0.03, 1.03)
        )
        ranks = sorted(
            {int(rank) for condition in ("seen", "unseen") for rank in scores["ordinal"][condition]["ranks"]}
        )
        withheld = {int(rank) for rank in scores["ordinal"]["unseen"]["ranks"]}
        matrix = np.array(
            [
                [
                    scores[arm]["unseen" if rank in withheld else "seen"]["ranks"][str(rank)]["exact_accuracy"]
                    for rank in ranks
                ]
                for arm in arms
            ]
        )
        image = axes[row, 1].imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axes[row, 1].set(
            yticks=range(3),
            yticklabels=labels,
            xticks=range(len(ranks)),
            xticklabels=[str(rank) + ("*" if rank in withheld else "") for rank in ranks],
            xlabel="Actual rank (* withheld from training)",
            title="Where does interpolation succeed?",
        )
        axes[row, 1].tick_params(axis="x", labelsize=8)
        fig.colorbar(
            image, ax=axes[row, 1], fraction=0.035, format=PercentFormatter(1), label="Exact accuracy within rank"
        )
    examples = []
    for frame in bundle.frames:
        if frame["function"] != "evaluate":
            continue
        context = {
            **bundle.context(frame),
            **{key: bundle.value(frame["values"].get(key)) for key in ("domain", "arm")},
        }
        predicted, expected, rows = bundle.values(frame, "predicted", "expected", "rows")
        if predicted is None or expected is None or rows is None:
            raise ValueError("P100: evaluated ranks must retain predictions, expected ranks and source rows")
        examples.append(
            {
                "label": f"{human(context['domain'])} · {human(context['arm'])} · {human(context['condition'])}",
                "field": "rank",
                "population": "test",
                "n": len(expected),
                "records": [
                    {"coordinate": [i], "input": rows[i], "actual": int(expected[i]), "predicted": int(predicted[i])}
                    for i in range(min(16, len(expected)))
                ],
            }
        )
    if len(examples) != 12:
        raise ValueError(f"P100: expected all twelve held-out rank populations, got {len(examples)}")
    note = (
        "Each decoder receives the same signal and 400 updates. Each familiar/withheld test population has 2,048 rows; "
        "heatmap cells score every row at that rank. Asterisks mark answers excluded from training. "
        "Category cannot emit those absent target classes. Rank positions 0–4 map to the five declared labels. "
        "This tests interpolation within the training range, with one seed and no uncertainty interval."
    )
    return fig, note, {"kind": "numeric", "conditions": examples}


HANDLERS = {
    "duration": duration,
    "ordinal": ordinal,
    "regression": numeric,
    "unknown-fallback": unknown_fallback,
    "quantile-source-units": quantile_source_units,
    "recent-history": recent_history,
    "precision-offsets": precision_offsets,
    "jitter-noise": jitter_noise,
    "context-extrapolation": context_extrapolation,
    "classification": classification,
    "ranking": ranking,
    "state": ranking,
    "trajectory": trajectory,
    "coverage": coverage,
    "temporal": temporal,
    "calibration": calibration,
    "admission": admission,
    "cohorts": cohorts,
    "mutation": mutation,
    "paired-seeds": paired_seeds,
    "fusion": fusion,
    "anomaly": anomaly,
    "anomaly-variance": anomaly_variance,
    "composition": composition,
    "grouping": grouping,
    "retrieval": retrieval,
    "label-efficiency": label_efficiency,
    "lookup": lookup,
    "representation": representation,
    "tradeoff": tradeoff,
    "extrapolation": extrapolation,
    "masking": masking,
}

SUPPLEMENTS = {
    "P074": [("controls", fusion_controls)],
    "P076": [("context", anomaly_context)],
    "P078": [("predictions", group_predictions)],
    "P073": [("geometry", representation_geometry)],
    "P080": [("geometry", representation_geometry)],
    "P081": [("geometry", representation_geometry)],
    "P082": [("geometry", representation_geometry)],
    "P077": [("optima", objective_optima)],
}


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
    group_proof = bundle.manifest["id"] in {"P078", "P108", "P111"} and "group_counts" in bundle.record["metrics"]
    handler = group_learning if group_proof else HANDLERS.get(spec.get("measured_mode", spec["mode"]), summary)
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
            supplements = [(suffix, view, spec) for suffix, view in SUPPLEMENTS.get(bundle.manifest["id"], [])]
            if group_proof:
                supplements = [("groups", group_composition, spec), ("predictions", group_predictions, spec)]
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
