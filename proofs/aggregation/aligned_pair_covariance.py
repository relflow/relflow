# %% [markdown]
# ---
# title: Covariance from aligned pairs
# categories:
# - Distribution statistics
# proof-id: P006
# description: Learn a centered cross-moment from sibling values whose item-level pairing
#   carries the answer.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P006 status >}}
#
# ## Insights
#
# **Covariance depends on which X belongs with which Y, not just the two value distributions.** Circularly
# shifting only Y preserves both marginal sets of numbers but breaks accuracy against the original
# covariance labels. That is evidence that the model uses item-level pairing.
#
# Keep aligned fields on the same repeated item so their relationship remains available before aggregation.
# The supplied-cross-product control helps distinguish that interaction from the final averaging step. This
# proof supports learned population covariance in its tested setting; it does not establish Pearson
# correlation, which also requires normalization by both spreads. Complete-pair permutation and varying
# collection lengths remain separate checks.
#
# ## Setup

# %%
"""Aligned sibling fields should support covariance.

Learn the centered cross-moment and respond to pairing corruption.

Run: uv run python proofs/run.py P006"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P006"
ITEMS = 8

# %% [markdown]
# ## Examples
#
# Targets use `mask=True`; the labels below are supervision hidden from the encoder.
#
# ### Positive aligned covariance
#
# ```yaml
# items:
#   - {x: -1, y: -1.4}
#   - {x: 1, y: 0.2}
#   - {x: -1, y: -0.2}
#   - {x: 1, y: 1.4}
#   - {x: -1, y: -1.4}
#   - {x: 1, y: 0.2}
#   - {x: -1, y: -0.2}
#   - {x: 1, y: 1.4}
# covariance: 0.8
# ```
#
# Both fields have mean zero, and their mean centered product is 0.8.
#
# ### Reverse the relationship
#
# ```yaml
# items:
#   - {x: -1, y: 1.4}
#   - {x: 1, y: -0.2}
#   - {x: -1, y: 0.2}
#   - {x: 1, y: -1.4}
#   - {x: -1, y: 1.4}
#   - {x: 1, y: -0.2}
#   - {x: -1, y: 0.2}
#   - {x: 1, y: -1.4}
# covariance: -0.8
# ```
#
# Negating Y changes the population covariance to −0.8.
#
# ### Shift only Y
#
# ```yaml
# items:
#   - {x: -1, y: 1.4}
#   - {x: 1, y: -1.4}
#   - {x: -1, y: 0.2}
#   - {x: 1, y: -0.2}
#   - {x: -1, y: 1.4}
#   - {x: 1, y: -1.4}
#   - {x: -1, y: 0.2}
#   - {x: 1, y: -0.2}
# covariance: 0.8  # Retained original label
# ```
#
# A one-position circular shift preserves both marginal value sets but makes
# the visible covariance −0.8. The corruption retains the original 0.8 target
# to test whether predictions depend on pair alignment.
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


def shuffle_y(observations: list[dict], *, seed: int) -> list[dict]:
    """Break within-row pairing while preserving every marginal value."""
    rng = np.random.default_rng(seed)
    rows = deepcopy(observations)
    changed = 0
    for row in rows:
        items = row["items"]
        offset = int(rng.integers(1, len(items)))
        shifted = np.roll([item["y"] for item in items], offset)
        changed += sum((item["y"] != y for item, y in zip(items, shifted, strict=True)))
        row["items"] = [{"x": item["x"], "y": float(y)} for item, y in zip(items, shifted, strict=True)]
    if changed == 0:
        raise ValueError("pair shuffle did not change any item-local associations")
    return rows


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["/covariance"]["content"] for row in output], dtype=np.float64)


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
# //| label: fig-proof-aligned-pair-covariance
# //| fig-cap: "Item attention mixes aligned X and Y fields while the branch keeps every token for learned covariance decoding."
# //| fig-alt: "Record contains repeated items with x, y inputs, and hidden covariance targets. Root reduction: Attention. Item reduction: Keep all tokens; capacity 8; branch attention MHA."
# #tree(node("record", kind: "root", width: 150pt, body: [
#     - *Reduction:* Attention
#   ], children: (
#   node("items", kind: "branch", repeated: true, width: 155pt, body: [
#       - *Reduction:* Keep all tokens
#       - *Branch attention:* MHA
#       - *Capacity:* 8 items
#     ], children: (
#     node("x", type: "Number"),
#     node("y", type: "Number"),
#   )),
#   node("covariance", kind: "target", type: "Number"),
# )))
# ```
#
# ## How it works
#
# The repeated branch keeps `x` and `y` together and uses `reduction=None`
# to preserve their encoded coordinates. Learned root attention can combine that
# evidence before scalar decoding. Independent locations, scales, and a random
# correlation coefficient prevent either marginal alone from supplying covariance.
#
# The negative control circularly shifts only `y`, preserving the exact values in
# both fields but breaking their alignment. Labels stay unchanged, so error must
# increase when the model responds to the changed pairs. The target is population
# covariance, the mean centered product; Pearson correlation is not tested here.
# [Supplied cross-deviations](supplied-cross-deviations.html) isolate its reduction.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(covariance_records(rows=1536, seed=seed + 1))
    test = list(covariance_records(rows=768, seed=seed + 3))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        items=rf.Branch(length=ITEMS, n_layers=2, reduction=None, x=rf.Number, y=rf.Number),
        covariance=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        model=model,
        train=lambda: covariance_records(rows=1536, seed=seed + 1),
        validate=lambda: covariance_records(rows=384, seed=seed + 2),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=1100 if steps is None else min(steps, 1100),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=datamodule)

    # Evaluate held-out answers and retain their original labels in corruption controls.
    intact = score(train=train, test=test, predicted=prediction(model, test))
    shuffled = shuffle_y(test, seed=seed + 4)
    corrupted = score(train=train, test=shuffled, predicted=prediction(model, shuffled))
    calibration = np.asarray(
        [
            intact["rmse"],
            intact["baseline_rmse"],
            intact["nrmse"],
            corrupted["rmse"],
            corrupted["baseline_rmse"],
            corrupted["nrmse"],
        ]
    )
    metrics = {"intact": intact, "corrupted": corrupted, "steps": trainer.global_step}
    checks = {
        "All measurements finite": bool(np.isfinite(calibration).all()),
        "Baseline RMSE above 0.35": bool(intact["baseline_rmse"] > 0.35),
        "Corruption preserves baseline": bool(
            np.isclose(intact["baseline_rmse"], corrupted["baseline_rmse"], atol=1e-12, rtol=0.0)
        ),
        "Covariance nRMSE below 0.35": bool(intact["nrmse"] < 0.35),
        "Shuffled pairing raises nRMSE by at least 0.35": bool(corrupted["nrmse"] >= intact["nrmse"] + 0.35),
    }
    return metrics, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P006 evidence >}}
#
# ## Remaining work
#
# Repeat across seeds and check complete-pair permutation. Variable lengths,
# missing values, zero variance, and correlation normalization remain separate
# questions. Coordinate-mixing ablations would distinguish which mechanisms are
# necessary for the observed result.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ## Reproduce
#
# {{< proof P006 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=3508)
