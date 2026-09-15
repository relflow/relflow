# %% [markdown]
# ---
# title: Variance from raw values
# categories:
# - Distribution statistics
# proof-id: P007
# description: Recover population variance from a repeated numerical field whose mean
#   does not reveal its spread.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P007 status >}}
#
# ## Insights
#
# **The model can learn how spread out values are, even when their averages are the same.** Independently
# varied location and scale prevent the mean alone from supplying the variance label. The matched-mean
# control requires a larger prediction for the more dispersed collection, without providing squared
# deviations as inputs.
#
# This supports learning beyond a single raw-average statistic. It does not mean `rf.Mean()` always erases
# spread: averaging nonlinear encoded features is different from supplying only the arithmetic mean. This
# particular proof uses Attention. The squared-deviation control isolates the easier reduction task. This
# proof uses population variance; a sample-variance target, varying lengths, and missing values need
# separate checks.
#
# ## Setup

# %%
"""A raw repeated number should reveal spread beyond its mean.

Learn a second central moment without receiving squared deviations.

Run: uv run python proofs/run.py P007"""

from __future__ import annotations

from collections.abc import Iterator

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P007"
ITEMS = 8

# %% [markdown]
# ## Examples
#
# Targets use `mask=True`; the labels below are supervision hidden from the encoder.
#
# ### Small spread around a positive mean
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
# The values have mean 0.3 and population variance 0.04.
#
# ### More spread at the same mean
#
# ```yaml
# items:
#   - {value: -0.8}
#   - {value: 1.4}
#   - {value: -0.8}
#   - {value: 1.4}
#   - {value: -0.8}
#   - {value: 1.4}
#   - {value: -0.8}
#   - {value: 1.4}
# variance: 1.21
# ```
#
# The mean remains 0.3, but the variance rises to 1.21. This illustrates the
# matched-mean control; the actual probe uses a normalized linear sequence
# instead of these alternating values.
#
# ### Move the mean without changing spread
#
# ```yaml
# items:
#   - {value: -0.5}
#   - {value: -0.1}
#   - {value: -0.5}
#   - {value: -0.1}
#   - {value: -0.5}
#   - {value: -0.1}
#   - {value: -0.5}
#   - {value: -0.1}
# variance: 0.04
# ```
#
# The mean is now −0.3 while variance remains 0.04. Location is independently
# varied in the generator, so a mean-only rule cannot recover the target.
# This example states the label semantics, not an additional translation gate.
#
# ## Synthetic data and controls


# %%
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
    return np.asarray([row["record/variance"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["variance"] for row in test], dtype=np.float64)
    baseline = rmse(actual, float(np.asarray([row["variance"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline, "nrmse": measured / baseline}


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-raw-value-variance
# //| fig-cap: "Learned Attention reduces eight raw values before decoding hidden population variance."
# //| fig-alt: "Record contains repeated items with value inputs, and hidden variance targets. Root reduction: Attention. Item reduction: Attention; capacity 8."
# #tree(node("record", kind: "root", width: 150pt, body: [
#     - *Reduction:* Attention
#   ], children: (
#   node("items", kind: "branch", repeated: true, width: 155pt, body: [
#       - *Reduction:* Attention
#       - *Capacity:* 8 items
#     ], children: (
#     node("value", type: "Number"),
#   )),
#   node("variance", kind: "target", type: "Number"),
# )))
# ```
#
# ## How it works
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
# ## Training and evaluation


# %%
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
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        items=rf.Branch(length=ITEMS, n_layers=2, reduction=rf.Attention(n_layers=2), value=rf.Number),
        variance=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        model=model,
        train=lambda: dispersion_records(rows=1280, seed=seed + 1),
        validate=lambda: dispersion_records(rows=320, seed=seed + 2),
        seed=seed,
    )
    trainer = lit.Trainer(
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
    trainer.fit(model, datamodule=datamodule)

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


# %% [markdown]
# ## Evidence
#
# {{< proof P007 evidence >}}
#
# ## Remaining work
#
# Repeat across seeds; add variable lengths, missing values, outliers,
# zero-variance rows, and permutation controls. The label uses population variance
# (dividing by item count), not an unbiased sample-variance estimator.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ## Reproduce
#
# {{< proof P007 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=3504)
