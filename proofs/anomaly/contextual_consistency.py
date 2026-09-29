# %% [markdown]
# ---
# title: Detecting Contextual Inconsistency
# categories: [Reconstruction]
# proof-id: P076
# description: Hidden-field reconstruction separates inconsistent records from normal records with identical individual-field distributions.
# execute: {enabled: false, eval: false}
# code-fold: true
# ---
#
# {{< proof P076 status >}}
#
# ## Insights
#
# A reconstruction residual can identify an ordinary value in the wrong context.
# This experiment permutes the numeric answer across otherwise unchanged test
# records: every individual-field distribution remains exactly the same. The
# anomaly label is never a training target. An independently shuffled training
# arm checks whether learning the relationship is necessary.
#
# The score is an absolute residual, not a calibrated probability or a general
# anomaly detector. A threshold selected on clean validation records controls
# the experiment's nominal false-positive rate; rare but valid tail records
# receive their own held-out false-positive measurement.
#
# All three initial CPU seeds met every gate. Contextual AUROC was
# 0.961–0.962, versus 0.501–0.513 after shuffled training and exactly 0.5
# for the marginal baseline. Detection recall was 89.6–90.6%; normal false
# positives were 4.2–5.2% and valid-tail false positives 3.5–6.1%.

# %%
"""P076: detect broken cross-field relationships using hidden-field reconstruction."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
from reporting import report

import relflow as rf

PROOF_ID = "P076"
BUDGET = 500
GROUPS = ("amber", "birch", "cove", "dune")
SLOPES = np.array([0.7, -0.8, 1.2, -1.3])
OFFSETS = np.array([-0.5, 0.5, -0.3, 0.3])
TRAIN_ROWS = 4096
TEST_ROWS = 2048

# %% [markdown]
# ## Process, examples, and observability
#
# Draw a balanced group and `x ~ Uniform(-2, 2)`. Set
# `y = slope[group] * x + offset[group] + Normal(0, 0.1)`. The model observes
# only group and x when reconstructing y. Slopes and offsets are evaluator
# constants, never model inputs. Each split has its own random stream, with
# run seeds spaced by 100 before assigning stream offsets.
#
# ```yaml
# group: amber
# x: 1.0
# y: 0.24
# ```
#
# The conditional mean is 0.2, so this small residual is ordinary.
#
# ```yaml
# group: amber
# x: 1.0
# y: -1.8
# ```
#
# A value borrowed from another record can be individually plausible but
# inconsistent here. Actual test anomalies use a permutation, preserving y's
# entire empirical marginal distribution rather than inserting large values.
#
# ```yaml
# group: amber
# x: 1.95
# y: 0.90
# ```
#
# This is a rare but valid tail record: its expected y is 0.865. Tail tests
# restrict absolute x to [1.8, 2], which is still inside training support.
#
# ```{typst}
# //| label: fig-proof-contextual-consistency
# //| fig-cap: "Visible group and x reconstruct y; the supplied y is used only to score the residual."
# //| fig-alt: "A record contains visible Enum group, Number x, and hidden Number y."
# #tree(node("record", kind: "root", children: (
#   node("group", type: "Enum", detail: "Selects a relationship"),
#   node("x", type: "Number", detail: "Visible measurement"),
#   node("y", kind: "target", type: "Number", detail: "Hidden for scoring"),
# )))
# ```


# %%
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


def build() -> rf.Model:
    model = rf.Model.xs(
        batch_size=128,
        group=rf.Enum(values=GROUPS, p_unavailable=0.0),
        x=rf.Number,
        y=rf.Number(mask=True, objective="mse"),
    )
    model.optimizer = rf.adamw(learning_rate=0.002)
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{"group": row["group"], "x": row["x"]} for row in rows]
    result = model.predict(pa.Table.from_pylist(visible))["predictions"].to_pylist()
    return np.asarray([row["/y"]["content"] for row in result], dtype=float)


def auc(normal: np.ndarray, anomalous: np.ndarray) -> float:
    """Probability that an anomaly scores above a normal row, with half credit for ties."""
    ordered = np.sort(normal)
    left = np.searchsorted(ordered, anomalous, side="left")
    right = np.searchsorted(ordered, anomalous, side="right")
    return float(np.mean((left + right) / (2 * len(normal))))


# %% [markdown]
# ## Training, scoring, and gates
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


# %%
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
            model=model,
            train=partial(records, rows=TRAIN_ROWS, seed=stream + 1, shuffled=shuffled),
            validate=partial(records, rows=2048, seed=stream + 2, shuffled=shuffled),
            seed=seed,
        )
        trainer = lit.Trainer(
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
        trainer.fit(model, datamodule=data)
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


# %% [markdown]
# ## Evidence and limits
#
# {{< proof P076 evidence >}}
#
# Three CPU seeds support this bounded task; the original gates and budgets
# are unchanged and remain provisional until a ten-seed panel is recorded. This is
# one hidden-field consistency score for familiar groups and a stationary
# noise distribution. It does not establish unknown-anomaly recall, semantic
# interpretation of groups, calibrated risk, or a deployment threshold.
#
# ## Reproduce
#
# {{< proof P076 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=7601)
