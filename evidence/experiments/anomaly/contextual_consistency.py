# %% [markdown]
# ---
# title: Can it spot an ordinary value in the wrong context?
# categories:
# - Anomaly detection
# description: Learn to reconstruct a numeric answer from its context. Use a large reconstruction error as a warning
#   that the value does not fit.
# evidence-readout:
# - label: Learned context
#   metric:
#   - informative
#   - anomaly_auc
#   format: auc
# - label: Shuffled training
#   metric:
#   - shuffled_training
#   - anomaly_auc
#   format: auc
# - label: False alarms on normal records
#   metric:
#   - informative
#   - normal_false_positive_rate
#   format: percent
# evidence-group: Anomaly
# evidence-diagram:
#   caption: Amber cards are hidden prediction targets. Other fields provide the input.
#   alt: Model tree with record, group, x, y. Amber cards are hidden prediction targets. Other fields provide the
#     input.
# evidence-review:
#   question: Can relflow spot an ordinary value in the wrong context?
#   modeling: Generate ordinary numeric answers from a visible group and x, then train the model to reconstruct those
#     answers. Anomalies swap answers between test records, preserving the overall value distribution but breaking
#     context. Compare anomaly AUC with a model trained on shuffled answers.
#   comparison:
#     kind: Controlled comparison
#     metric: AUC
#     conditions:
#     - label: Shuffled training
#       path:
#       - shuffled_training
#       - anomaly_auc
#       format: auc
#     - label: Learned context
#       path:
#       - informative
#       - anomaly_auc
#       format: auc
#     definition: 'Ranking score: 0.5 is chance and 1 is perfect separation. This is not the percentage of correct
#       predictions.'
#     direction: Higher is better
#     domain:
#     - 0
#     - 1
#   note: Anomaly scores are reconstruction errors. Detection thresholds come from clean validation data; normal and
#     valid-tail false alarms remain separate checks.
# evidence-plot:
#   target: y
#   focus: Can the score identify an ordinary value in the wrong context?
#   comparison: Compare normal, shuffled-answer, and rare-valid residuals, actual ROC curves, and the validation-selected
#     threshold.
#   limit: Exploring a threshold on this test set does not change the proof's preselected operating point.
#   supplements:
#     context:
#       renderer: plot_context
#   renderer: plot
# evidence-cost:
#   baseline: 32
#   candidate: 32
# ---
#
# A value can be common overall but inconsistent with the rest of its record. It scores disagreement
# between an observed value and its reconstruction from context. It asks whether the score responds to
# a broken relationship rather than simply treating large or rare values as anomalies.
#
# ## Example
#
# ```yaml
# group: amber
# x: 1.0
# y: 0.24
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```typst
# #tree(node("record", kind: "root", children: (
#   node("group", type: "Enum", detail: "Selects a relationship"),
#   node("x", type: "Number", detail: "Visible measurement"),
#   node("y", kind: "target", type: "Number", detail: "Hidden for scoring"),
# )))
# ```
#
# For the illustrative amber row, the normal mean is `0.7 × 1 − 0.5 = 0.2`; the shown 0.24 includes a
# possible small noise draw. A value near −1 is normal for another group at the same x, but would
# disagree with amber's context. The model predicts from group and x before the observed y is used to
# calculate the warning score.
#
# ## Comparison
#
# Shuffle answers among test records so their individual distribution stays unchanged. Compare with an
# independently shuffled-training model.
#
# Test anomalies shuffle y between records, preserving its distribution and every visible context. A
# detector looking only at y therefore has chance ranking performance. Shuffled training provides a
# second control that cannot learn the correct relationship. Separate valid-tail records use uncommon
# x values still inside training support to check that rarity alone does not cause excessive warnings.
#
# ## Result
#
# The selected run meets the declared checks for this constructed inconsistency. Its warning score is
# not a calibrated anomaly probability, and rare valid values need their own false-alarm check.
#
# AUROC measures how often an anomalous record ranks above a normal one; it is not the probability
# that a flagged record is anomalous. Recall and false-alarm rates use a threshold chosen only from
# clean validation residuals. The selected checks concern this constructed inconsistency and noise
# model, rather than every possible anomaly or a calibrated deployment threshold.
#
# ## Method
#
# ### Process, examples, and observability
#
# Draw a balanced group and `x ~ Uniform(-2, 2)`. Set
# `y = slope[group] * x + offset[group] + Normal(0, 0.1)`. The model observes
# only group and x when reconstructing y. Slopes and offsets are evaluator
# constants, never model inputs. Each split has its own random stream, with
# run seeds spaced by 100 before assigning stream offsets.
#
# Actual test anomalies permute y between records, preserving its entire distribution. Valid-tail tests restrict absolute x to [1.8, 2], within training support.
#
# ### Training, scoring, and gates
#
# Fit informative and shuffled-y models from the same initialization for 500
# AdamW updates each, batch size 128 and learning rate 0.002. Use 4,096 training
# rows, 2,048 clean validation rows, and separate 2,048-row test and tail sets.
# Select each model's residual threshold at the clean validation 95th percentile.
# No test observations choose that threshold or a checkpoint.
#
# Predeclared capability gates require informative test nRMSE below 0.20,
# anomaly AUROC at least 0.90, detection recall at least 0.75, normal false
# positives at most 0.10, and tail false positives at most 0.15. Shuffled
# training must leave AUROC at most 0.65. A univariate y-deviation baseline
# scores exactly 0.5 AUROC because its two score distributions are identical.
#

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.10",
#     "lightning",
#     "numpy",
#     "polars",
#     "pyarrow",
#     "pyyaml>=6",
#     "relflow>=0.5.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "reliability"
# types = []
# conditions = ["changed-associations"]
# ///

"""P076: detect broken cross-field relationships using hidden-field reconstruction."""

import sys
from collections.abc import Iterator
from functools import partial
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
from policies import CONTROL
from reporting import experiment, report

BUDGET = 500
GROUPS = ("amber", "birch", "cove", "dune")
SLOPES = np.array([0.7, -0.8, 1.2, -1.3])
OFFSETS = np.array([-0.5, 0.5, -0.3, 0.3])
TRAIN_ROWS = 4096
TEST_ROWS = 2048


def records(*, rows: int, seed: int, shuffled: bool = False, tail: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    group = np.resize(np.arange(len(GROUPS)), rows)
    rng.shuffle(group)
    x = rng.uniform(-2, 2, rows)
    if tail:
        x = rng.choice([-1.0, 1.0], rows) * rng.uniform(1.8, 2.0, rows)
    y = SLOPES[group] * x + OFFSETS[group] + rng.normal(0, 0.1, rows)
    if shuffled:
        y = y[rng.permutation(rows)]
    for index in range(rows):
        yield {"group": GROUPS[group[index]], "x": float(x[index]), "y": float(y[index])}


class Model(rf.Model):
    group = rf.Enum(values=GROUPS, p_unavailable=0.0)
    x = rf.Number
    y = rf.Number(mask=True, objective="mse")


def build() -> rf.Model:

    model = Model(preset=CONTROL.name)
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{"group": row["group"], "x": row["x"]} for row in rows]
    result = pl.DataFrame(model.predict(pa.Table.from_pylist(visible)))["predictions"].to_list()
    return np.asarray([row["y"]["content"] for row in result], dtype=float)


def auc(normal: np.ndarray, anomalous: np.ndarray) -> float:
    """Probability that an anomaly scores above a normal row, with half credit for ties."""
    ordered = np.sort(normal)
    left = np.searchsorted(ordered, anomalous, side="left")
    right = np.searchsorted(ordered, anomalous, side="right")
    return float(np.mean((left + right) / (2 * len(normal))))


@experiment("P076", seed=7601)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    stream = seed * 100
    validation = list(records(rows=2048, seed=stream + 2))
    test = list(records(rows=TEST_ROWS, seed=stream + 3))
    anomalous = list(records(rows=TEST_ROWS, seed=stream + 3, shuffled=True))
    tail = list(records(rows=TEST_ROWS, seed=stream + 4, tail=True))
    actual = np.asarray([row["y"] for row in test])
    broken = np.asarray([row["y"] for row in anomalous])
    train_mean = float(np.mean([row["y"] for row in records(rows=TRAIN_ROWS, seed=stream + 1)]))
    baseline_rmse = float(np.sqrt(np.mean((actual - train_mean) ** 2)))
    marginal_auc = auc(np.abs(actual - train_mean), np.abs(broken - train_mean))
    metrics = {"train_rows": TRAIN_ROWS, "test_rows_per_condition": TEST_ROWS, "marginal_auc": marginal_auc}
    checks = {
        "Anomalies preserve the exact target marginal": bool(np.array_equal(np.sort(actual), np.sort(broken))),
        "Anomalies preserve visible contexts": all(
            (a["group"], a["x"]) == (b["group"], b["x"]) for a, b in zip(test, anomalous, strict=True)
        ),
        "Marginal deviation baseline has exactly chance AUROC": abs(marginal_auc - 0.5) < 1e-12,
    }
    for shuffled in (False, True):
        lit.seed_everything(seed, workers=True)
        model = build()
        data = rf.SyntheticDataModule(
            train=partial(records, rows=TRAIN_ROWS, seed=stream + 1, shuffled=shuffled),
            validate=partial(records, rows=2048, seed=stream + 2, shuffled=shuffled),
            seed=seed,
        )
        model.fit(
            data,
            batch_size=128,
            learning_rate=0.002,
            early_stopping=False,
            rollback=False,
            precision="32-true",
            strategy="auto",
            accelerator=accelerator,
            devices=1,
            max_epochs=-1,
            max_steps=BUDGET if steps is None else min(steps, BUDGET),
            deterministic=True,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        trainer = model.trainer
        model.eval()
        valid_error = np.abs(predict(model, validation) - [row["y"] for row in validation])
        threshold = float(np.quantile(valid_error, 0.95))
        prediction = predict(model, test)
        clean_score = np.abs(prediction - actual)
        broken_score = np.abs(prediction - broken)
        tail_score = np.abs(predict(model, tail) - [row["y"] for row in tail])
        key = "shuffled_training" if shuffled else "informative"
        result = {
            "steps": trainer.global_step,
            "validation_threshold": threshold,
            "rmse": float(np.sqrt(np.mean(clean_score**2))),
            "baseline_rmse": baseline_rmse,
            "normalized_rmse": float(np.sqrt(np.mean(clean_score**2)) / baseline_rmse),
            "anomaly_auc": auc(clean_score, broken_score),
            "detection_recall": float(np.mean(broken_score > threshold)),
            "normal_false_positive_rate": float(np.mean(clean_score > threshold)),
            "tail_false_positive_rate": float(np.mean(tail_score > threshold)),
        }
        metrics[key] = result
        checks[f"{key}: every measurement is finite"] = bool(np.isfinite(list(result.values())).all())
        if shuffled:
            checks["Shuffled training anomaly AUROC stays below 0.65"] = result["anomaly_auc"] <= 0.65
        else:
            checks["Informative reconstruction nRMSE below 0.20"] = result["normalized_rmse"] < 0.20
            checks["Contextual anomaly AUROC reaches 0.90"] = result["anomaly_auc"] >= 0.90
            checks["Anomaly recall reaches 0.75"] = result["detection_recall"] >= 0.75
            checks["Normal false positive rate at most 0.10"] = result["normal_false_positive_rate"] <= 0.10
            checks["Rare valid tail false positive rate at most 0.15"] = result["tail_false_positive_rate"] <= 0.15
    return metrics, checks


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    from evidence.diagnostics.data import Bundle, human
    from evidence.diagnostics.plots import COLORS, axes_pair, finite, roc

    def anomaly_data(bundle: Bundle) -> dict:
        frames = [frame for frame in bundle.frames if frame["function"] == "predict"]
        if len(frames) != 6:
            raise ValueError(
                f"{bundle.manifest['id']} requires validation, test, and rare-valid predictions for both arms"
            )
        run = next((frame for frame in reversed(bundle.frames) if frame["function"] == "run"))
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


def plot_context(bundle, spec):
    """Render the captured observations for this experiment."""
    import numpy as np

    from evidence.diagnostics.data import Bundle
    from evidence.diagnostics.plots import COLORS, axes_pair, finite

    def anomaly_data(bundle: Bundle) -> dict:
        frames = [frame for frame in bundle.frames if frame["function"] == "predict"]
        if len(frames) != 6:
            raise ValueError(
                f"{bundle.manifest['id']} requires validation, test, and rare-valid predictions for both arms"
            )
        run = next((frame for frame in reversed(bundle.frames) if frame["function"] == "run"))
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


if __name__ == "__main__":
    report(run)
