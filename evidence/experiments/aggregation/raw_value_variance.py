# %% [markdown]
# ---
# title: Can it tell how spread out a list is?
# categories:
# - Distribution statistics
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
# ---
#
# The model receives eight raw values and predicts their variance: the average squared distance from
# the list's own mean. No mean or squared-distance feature is supplied. The task is to distinguish a
# tightly packed list from a widely spread one, even when their centers coincide.
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
# The illustrated values average to 0.3. Each is 0.2 away from that center, and squaring 0.2 gives
# 0.04, so averaging the eight squared distances still gives 0.04. The sign of each deviation
# disappears when it is squared; positive and negative departures cannot cancel.
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
# Ordinary test records vary their centers and spreads independently. A separate pair of probes has
# the same center and shape but different scale, so a prediction based only on the average would
# give nearly identical answers. The correct probe variances are 0.04 and 1.21, and the high-spread
# prediction must be substantially larger.
#
# ## Result
#
# {{< evidence P007 status >}}
#
# The model learns more than the average alone. These results cover the tested list length and do
# not establish exact variance calculation.
#
#
# Prediction error is normalized against guessing the training-set mean variance, not against the
# mean of the input values. The matched-mean check adds evidence that spread matters. This is the
# population definition, dividing by eight; it does not test a sample estimator that divides by
# seven, nor variable-length lists or extreme outliers.
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
import polars as pl
import relflow as rf
import torch
from reporting import experiment, report

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
    output = pl.DataFrame(model.predict(inputs))["predictions"].to_list()
    return np.asarray([row["variance"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["variance"] for row in test], dtype=np.float64)
    baseline = rmse(actual, float(np.asarray([row["variance"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline, "nrmse": measured / baseline}


@experiment("P007", seed=3504)
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
    report(run)

# %% [markdown]
# </details>
