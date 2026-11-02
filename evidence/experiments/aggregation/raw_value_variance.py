# %% [markdown]
# ---
# title: Can it tell how spread out a list is?
# categories:
# - Distribution statistics
# evidence-id: P007
# description: Predict variance, a measure of how far the values lie from their average.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Variance prediction
#   metric:
#   - measured
#   - nrmse
#   format: error
# evidence-group: Aggregation
# evidence-result:
#   accelerator: cpu
#   checks:
#     Matched-mean variance prediction gap above 0.65: true
#     Variance nRMSE below 0.30: true
#   duration_seconds: 34.533
#   metrics:
#     measured:
#       baseline_rmse: 0.4630170659412655
#       nrmse: 0.02902662492538892
#       rmse: 0.013439822707131182
#     paired_prediction:
#     - 0.041193554658819576
#     - 1.241580432336992
#     steps: 900
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 137e0a15c861bc7a596cbbdb5cf615a845d94aefecc70cc78b4dff2849b31085
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: d6a0e25f4d78
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
#     script_sha256: 8769e364f9255af2e8956835301922234293ca9567871a298108af596163657f
#     source_sha256: e793b465775a856eaa4317a6a9ce88a0a989a7afbcdde0f473a33b3fda321c8c
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 3504
#   started_at: '2026-11-01T01:33:20.147900+00:00'
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
# items:
#   - {value: 0.1}
#   - {value: 0.5}
#   - {value: 0.1}
#   - {value: 0.5}
#   - {value: 0.1}
#   - {value: 0.5}
#   - {value: 0.1}
#   - {value: 0.5}
# variance: 0.04
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-raw-value-variance
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, items, value, variance. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("value", type: "Number"),
#   )),
#   node("variance", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Include lists with the same average but different spreads. Their answers should differ.
#
# ## Result
#
# {{< evidence P007 status >}}
#
# The model learns more than the average alone. These results cover the tested list length and do not establish exact variance calculation.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P007 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P007 script >}}
#
# ### How it works
#
# The item branch and root use learned `rf.Attention` reductions. Eight raw
# values arrive without a mean, squared deviation, or variance feature. The
# generator varies location and scale independently and exactly centers its
# basis, preventing location alone from predicting spread.
#
# A matched-mean probe keeps the same normalized shape and mean of 0.3 while
# changing scale from 0.2 to 1.1. Its true population variances are 0.04 and 1.21;
# the predicted high-spread variance must exceed the low-spread one by more than
# 0.65. The [squared-deviation control](supplied-squared-deviations.html) isolates
# averaging after the nonlinear statistic has already been supplied.
#
# ### Remaining work
#
# Repeat across seeds; add variable lengths, missing values, outliers,
# zero-variance rows, and permutation controls. The label uses population variance
# (dividing by item count), not an unbiased sample-variance estimator.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""A raw repeated number should reveal spread beyond its mean.

Learn a second central moment without receiving squared deviations.

Run: uv run python evidence/experiments/run.py P007"""

from __future__ import annotations

from collections.abc import Iterator

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P007"
ITEMS = 8


def dispersion_records(*, rows: int, seed: int) -> Iterator[dict]:
    """Draw rows whose location and spread are statistically independent."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        basis = rng.normal(size=ITEMS)
        basis -= basis.mean()
        basis /= np.sqrt(np.mean(np.square(basis)))
        location = float(rng.uniform(-0.8, 0.8))
        scale = float(rng.uniform(0.12, 1.25))
        values = location + scale * basis
        items = [{"value": float(value)} for value in values]
        yield {"items": items, "variance": scale * scale}


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["variance"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["variance"] for row in test], dtype=np.float64)
    baseline = rmse(actual, float(np.asarray([row["variance"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline, "nrmse": measured / baseline}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(dispersion_records(rows=1280, seed=seed + 1))
    test = list(dispersion_records(rows=640, seed=seed + 3))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        items=rf.Branch(length=ITEMS, n_layers=2, reduction=rf.Attention(n_layers=2), value=rf.Number),
        variance=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        train=lambda: dispersion_records(rows=1280, seed=seed + 1),
        validate=lambda: dispersion_records(rows=320, seed=seed + 2),
        seed=seed,
    )
    model.fit(
        datamodule,
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=900 if steps is None else min(steps, 900),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer

    # Evaluate held-out answers and retain their original labels in corruption controls.
    measured = score(train=train, test=test, predicted=prediction(model, test))
    basis = np.linspace(-1.0, 1.0, ITEMS)
    basis -= basis.mean()
    basis /= np.sqrt(np.mean(np.square(basis)))
    matched = [{"items": [{"value": float(0.3 + scale * value)} for value in basis]} for scale in (0.2, 1.1)]
    paired_prediction = prediction(model, matched)
    metrics = {"measured": measured, "paired_prediction": paired_prediction.tolist(), "steps": trainer.global_step}
    checks = {
        "Variance nRMSE below 0.30": bool(measured["nrmse"] < 0.3),
        "Matched-mean variance prediction gap above 0.65": bool(paired_prediction[1] - paired_prediction[0] > 0.65),
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3504)

# %% [markdown]
# </details>
