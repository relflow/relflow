# %% [markdown]
# ---
# title: Averaging supplied centered products
# categories:
# - Distribution statistics
# proof-id: P008
# description: Isolate covariance reduction after the item-level centered products are
#   already available.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P008 status >}}
#
# ## Insights
#
# **Once centered products are supplied, the model can learn their average as covariance.** The difficult
# information about pairing and centering has already been calculated outside the model. Positive, negative,
# and canceling contributions then share the same reduction task.
#
# This control helps localize a failure in raw-pair covariance. If supplied products are easy but raw pairs
# are not, the missing behavior is more likely in the interaction than in final averaging. Passing here does
# not demonstrate learned centering, multiplication, or sensitivity to raw X–Y pairing. The proof checks
# numerical accuracy for its fixed collection length, without independently testing broader covariance
# identities.
#
# ## Setup

# %%
"""Supplied centered products isolate covariance reduction.

Mean learns covariance once centered products are supplied.

Run: uv run python proofs/run.py P008"""

from __future__ import annotations

from collections.abc import Iterator

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P008"
ITEMS = 8

# %% [markdown]
# ## Examples
#
# Targets use `mask=True`; the labels below are supervision hidden from the encoder.
#
# ### Positive centered products
#
# ```yaml
# items:
#   - {cross_deviation: 1.4}
#   - {cross_deviation: 0.2}
#   - {cross_deviation: 0.2}
#   - {cross_deviation: 1.4}
#   - {cross_deviation: 1.4}
#   - {cross_deviation: 0.2}
#   - {cross_deviation: 0.2}
#   - {cross_deviation: 1.4}
# covariance: 0.8
# ```
#
# The eight supplied products average to a population covariance of 0.8.
#
# ### Negative centered products
#
# ```yaml
# items:
#   - {cross_deviation: -1.4}
#   - {cross_deviation: -0.2}
#   - {cross_deviation: -0.2}
#   - {cross_deviation: -1.4}
#   - {cross_deviation: -1.4}
#   - {cross_deviation: -0.2}
#   - {cross_deviation: -0.2}
#   - {cross_deviation: -1.4}
# covariance: -0.8
# ```
#
# A negative average requires a covariance of −0.8; this target retains its sign.
#
# ### Contributions that cancel
#
# ```yaml
# items:
#   - {cross_deviation: 1.4}
#   - {cross_deviation: -0.2}
#   - {cross_deviation: 0.2}
#   - {cross_deviation: -1.4}
#   - {cross_deviation: 1.4}
#   - {cross_deviation: -0.2}
#   - {cross_deviation: 0.2}
#   - {cross_deviation: -1.4}
# covariance: 0.0
# ```
#
# These nonzero centered products average to zero. The model still receives
# the sufficient statistic directly; none of these records demonstrates learned
# centering or multiplication from raw pairs.
#
# ## Synthetic data and controls


# %%
def covariance_records(*, rows: int, seed: int) -> Iterator[dict]:
    """Draw paired fields with controlled means, scales, and covariance."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        x_basis = rng.normal(size=ITEMS)
        x_basis -= x_basis.mean()
        x_basis /= np.sqrt(np.mean(np.square(x_basis)))
        orthogonal = rng.normal(size=ITEMS)
        orthogonal -= orthogonal.mean()
        orthogonal -= np.mean(orthogonal * x_basis) * x_basis
        orthogonal /= np.sqrt(np.mean(np.square(orthogonal)))
        correlation = float(rng.uniform(-0.88, 0.88))
        y_basis = correlation * x_basis + np.sqrt(1.0 - correlation * correlation) * orthogonal
        x_scale = float(rng.uniform(0.35, 1.45))
        y_scale = float(rng.uniform(0.35, 1.45))
        x_location = float(rng.uniform(-0.8, 0.8))
        y_location = float(rng.uniform(-0.8, 0.8))
        x_values = x_location + x_scale * x_basis
        y_values = y_location + y_scale * y_basis
        yield {
            "items": [{"x": float(x), "y": float(y)} for x, y in zip(x_values, y_values, strict=True)],
            "covariance": correlation * x_scale * y_scale,
            "correlation": correlation,
        }


def covariance_sufficient_records(*, rows: int, seed: int) -> Iterator[dict]:
    """Expose per-item centered products while retaining the same targets."""
    rows_with_pairs = covariance_records(rows=rows, seed=seed)
    for row in rows_with_pairs:
        x = np.asarray([item["x"] for item in row["items"]], dtype=np.float64)
        y = np.asarray([item["y"] for item in row["items"]], dtype=np.float64)
        cross_deviation = (x - x.mean()) * (y - y.mean())
        yield {
            "items": [{"cross_deviation": float(value)} for value in cross_deviation],
            "covariance": row["covariance"],
        }


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["record/covariance"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["covariance"] for row in test], dtype=np.float64)
    baseline = rmse(actual, float(np.asarray([row["covariance"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline, "nrmse": measured / baseline}


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-supplied-cross-deviations
# //| fig-cap: "Mean averages eight supplied centered products without item attention before covariance is decoded."
# //| fig-alt: "Record contains repeated items with cross deviation inputs, and hidden covariance targets. Root reduction: Attention. Item reduction: Mean; capacity 8; branch attention Off."
# #tree(node("record", kind: "root", width: 150pt, body: [
#     - *Reduction:* Attention
#   ], children: (
#   node("items", kind: "branch", repeated: true, width: 155pt, body: [
#       - *Reduction:* Mean
#       - *Branch attention:* Off
#       - *Capacity:* 8 items
#     ], children: (
#     node("cross_deviation", type: "Number"),
#   )),
#   node("covariance", kind: "target", type: "Number"),
# )))
# ```
#
# ## How it works
#
# The generator supplies `(x - mean(x)) * (y - mean(y))` for every item.
# The model sees these eight centered products, not raw `x` and `y` values. The
# item branch has `attention=None` and `rf.Mean()`, followed by learned root
# attention and a covariance decoder.
#
# This tests whether an encoded average can be decoded into the population
# cross-moment. It bypasses centering and pairwise multiplication. Compare it with
# [aligned-pair covariance](aligned-pair-covariance.html) to localize a failure in
# sibling interaction rather than in the final reduction.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(covariance_sufficient_records(rows=768, seed=seed + 1))
    test = list(covariance_sufficient_records(rows=384, seed=seed + 3))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        items=rf.Branch(length=ITEMS, n_layers=2, attention=None, reduction=rf.Mean(), cross_deviation=rf.Number),
        covariance=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        model=model,
        train=lambda: covariance_sufficient_records(rows=768, seed=seed + 1),
        validate=lambda: covariance_sufficient_records(rows=192, seed=seed + 2),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=350 if steps is None else min(steps, 350),
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
    metrics = {"measured": measured, "steps": trainer.global_step}
    checks = {"Covariance nRMSE below 0.20": bool(measured["nrmse"] < 0.2)}
    return metrics, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P008 evidence >}}
#
# ## Remaining work
#
# Repeat across seeds and broaden lengths, missing-value behavior, and
# outliers. This sufficient-statistic diagnostic should remain easier than the raw
# pairing task; it cannot establish that the model learned the supplied operation.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ## Reproduce
#
# {{< proof P008 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=3513)
