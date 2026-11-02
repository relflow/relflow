# %% [markdown]
# ---
# title: Can it spot an ordinary value in the wrong context?
# categories:
# - Anomaly detection
# evidence-id: P076
# description: Learn to reconstruct a numeric answer from its context. Use a large reconstruction error as a warning
#   that the value does not fit.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
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
# evidence-result:
#   accelerator: cpu
#   checks:
#     Anomalies preserve the exact target marginal: true
#     Anomalies preserve visible contexts: true
#     Anomaly recall reaches 0.75: true
#     Contextual anomaly AUROC reaches 0.90: true
#     Informative reconstruction nRMSE below 0.20: true
#     Marginal deviation baseline has exactly chance AUROC: true
#     Normal false positive rate at most 0.10: true
#     Rare valid tail false positive rate at most 0.15: true
#     Shuffled training anomaly AUROC stays below 0.65: true
#     'informative: every measurement is finite': true
#     'shuffled_training: every measurement is finite': true
#   duration_seconds: 36.843
#   metrics:
#     informative:
#       anomaly_auc: 0.9616631269454956
#       baseline_rmse: 1.274805492668205
#       detection_recall: 0.8974609375
#       normal_false_positive_rate: 0.04638671875
#       normalized_rmse: 0.0919518293532086
#       rmse: 0.11722069712035982
#       steps: 500
#       tail_false_positive_rate: 0.0732421875
#       validation_threshold: 0.22753207864955435
#     marginal_auc: 0.5
#     shuffled_training:
#       anomaly_auc: 0.5051983594894409
#       baseline_rmse: 1.274805492668205
#       detection_recall: 0.04052734375
#       normal_false_positive_rate: 0.0478515625
#       normalized_rmse: 1.013740516610813
#       rmse: 1.2923219787157683
#       steps: 500
#       tail_false_positive_rate: 0.29345703125
#       validation_threshold: 2.463423328292913
#     test_rows_per_condition: 2048
#     train_rows: 4096
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: afe0d6c5c8edf926b0764a0dd2725dcd2376da26dfaa2876b719521abe2ac327
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 567de0301fb1
#     packages:
#       lightning: 2.6.4
#       numpy: 2.4.6
#       polars: 2.0.0
#       pyarrow: 24.0.0
#       pydantic: 2.13.4
#       relflow: 0.4.1
#       tensordict: 0.12.4
#       torch: 2.12.0+cpu
#       torchmetrics: 1.9.0
#     platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.41
#     python: 3.12.16
#     relflow_path: /tmp/runtime/lib/python3.12/site-packages/relflow
#     relflow_revision: 41d6f5fb74c0ba9f0be104c896913a55adb125f5
#     relflow_source_sha256: 00fdf397aeefb3a4e705d431850e0a6294eef142989513fddcb903bb1aefa873
#     script_sha256: 53f3add42167bca9ea2ef43981395c56c8447bf533eb997fd96c658fd9ea18bc
#     source_sha256: 86a0ca46446416afef44869e115de35c43b458124c98848197741fb32794b305
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 7601
#   started_at: '2026-11-01T01:50:09.075017+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41261482839-1
#   candidate:
#     revision: 41d6f5fb74c0ba9f0be104c896913a55adb125f5
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   baseline:
#     revision: 1add6d8919e686dc63b7fa6c4cdbdc437be2b570
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#   manifest_sha256: f4b3bfe706e43e38785ab5bed94c3f51eb44fe7314d0a1b2f0dd3c6f3f7bef92
#   passed: true
#   github_run:
#     event: pull_request
#     head_branch: dev/evidence-in-docs
#     head_sha: 8bb7764d8186c743eb554cb4a91dedb205ad4cb9
#     html_url: https://github.com/relflow/relflow/actions/runs/41261482839
#     pull_requests:
#     - base_sha: 1add6d8919e686dc63b7fa6c4cdbdc437be2b570
#       head_sha: 8bb7764d8186c743eb554cb4a91dedb205ad4cb9
#       number: 64
#     run_attempt: 1
#     run_id: 41261482839
#   release: https://github.com/relflow/docs/releases/tag/ci-41261482839-attempt-1
#   reports_sha256: bc67a1cb3725cbf9c5a8deb65844297d70f6d2122cd3b7de5c1d15f56603864f
# ---
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
# ```{typst}
# //| label: fig-evidence-contextual-consistency
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with record, group, x, y. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("record", kind: "root", children: (
#   node("group", type: "Enum", detail: "Selects a relationship"),
#   node("x", type: "Number", detail: "Visible measurement"),
#   node("y", kind: "target", type: "Number", detail: "Hidden for scoring"),
# )))
# ```
#
# ## Comparison
#
# Shuffle answers among test records so their individual distribution stays unchanged. Compare with an independently shuffled-training model.
#
# ## Result
#
# {{< evidence P076 status >}}
#
# All three seeds detect this constructed inconsistency. The warning score is not a calibrated anomaly probability, and rare valid values need their own false-alarm check.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P076 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P076 script >}}
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
# ### Complete experiment code
#

# %%
"""P076: detect broken cross-field relationships using hidden-field reconstruction."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P076"
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


def build() -> rf.Model:
    model = rf.Model.xs(
        group=rf.Enum(values=GROUPS, p_unavailable=0.0),
        x=rf.Number,
        y=rf.Number(mask=True, objective="mse"),
    )
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{"group": row["group"], "x": row["x"]} for row in rows]
    result = model.predict(pa.Table.from_pylist(visible))["predictions"].to_pylist()
    return np.asarray([row["y"]["content"] for row in result], dtype=float)


def auc(normal: np.ndarray, anomalous: np.ndarray) -> float:
    """Probability that an anomaly scores above a normal row, with half credit for ties."""
    ordered = np.sort(normal)
    left = np.searchsorted(ordered, anomalous, side="left")
    right = np.searchsorted(ordered, anomalous, side="right")
    return float(np.mean((left + right) / (2 * len(normal))))


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


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7601)

# %% [markdown]
# </details>
