# %% [markdown]
# ---
# title: Weighted sums from raw pairs
# categories:
# - Weighted aggregation
# proof-id: P014
# description: Learn each item’s value–weight relationship, then sum the contributions
#   across the collection.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P014 status >}}
#
# ## Insights
#
# **The model uses the weight attached to each value, rather than only the collection’s separate values and
# weights.** Swapping weights between items preserves both sets of numbers but worsens predictions against
# the original labels. Reordering complete items leaves predictions approximately stable, while scaling all
# weights changes the answer proportionally.
#
# Keeping related fields on the same item gives the model a route to learn their interaction before
# aggregation. The supplied-product control helps isolate that interaction when diagnosing a failure. This
# proof supports the raw paired schema for its fixed collection length; it does not establish
# variable-length totals, negative-weight behavior, or exact multiplication.
#
# ## Setup

# %%
"""Raw sibling value and weight fields should support a weighted sum.

Learn products and their sum without a derived contribution field.

Run: uv run python proofs/run.py P014"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P014"
ITEMS = 6

# %% [markdown]
# ## Examples
#
# Targets use `mask=True`; the labels below are supervision hidden from the encoder.
#
# ### Combine each pair
#
# ```yaml
# items:
#   - {value: 0.8, weight: 0.5}
#   - {value: -0.4, weight: 1.0}
#   - {value: 0.2, weight: 1.4}
#   - {value: 0.5, weight: 0.2}
#   - {value: -0.5, weight: 0.4}
#   - {value: 0.9, weight: 0.3}
# weighted_sum: 0.45
# ```
#
# Each product uses the value and weight from the same item; their sum is 0.45.
#
# ### Scale every weight
#
# ```yaml
# items:
#   - {value: 0.8, weight: 0.375}
#   - {value: -0.4, weight: 0.75}
#   - {value: 0.2, weight: 1.05}
#   - {value: 0.5, weight: 0.15}
#   - {value: -0.5, weight: 0.3}
#   - {value: 0.9, weight: 0.225}
# weighted_sum: 0.3375
# ```
#
# Scaling all weights by 0.75 changes the correct sum from 0.45 to 0.3375.
#
# ### Break the pairing
#
# ```yaml
# items:
#   - {value: 0.8, weight: 1.0}
#   - {value: -0.4, weight: 0.5}
#   - {value: 0.2, weight: 1.4}
#   - {value: 0.5, weight: 0.2}
#   - {value: -0.5, weight: 0.4}
#   - {value: 0.9, weight: 0.3}
# weighted_sum: 0.45  # Retained original label
# ```
#
# Only the first two weights are swapped. The visible pairs now imply 1.05,
# but this corruption retains the original 0.45 target. A model following the
# pairs should move away from that retained label.
#
# ## Synthetic data and controls


# %%
def weighted_records(*, rows: int, length: int, seed: int) -> Iterator[dict]:
    """Draw random value/weight pairs and expose raw fields or their products."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        values = rng.uniform(-1.0, 1.0, size=length)
        weights = rng.uniform(0.15, 1.5, size=length)
        contributions = values * weights
        items = [
            {"value": float(value), "weight": float(weight)} for value, weight in zip(values, weights, strict=True)
        ]
        yield {
            "items": items,
            "weighted_sum": float(contributions.sum()),
            "weighted_mean": float(contributions.sum() / weights.sum()),
        }


def permute_weights(observations: list[dict], *, seed: int) -> list[dict]:
    """Break value/weight pairing within rows while preserving both marginals."""
    rng = np.random.default_rng(seed)
    rows = deepcopy(observations)
    changed = 0
    for row in rows:
        items = row["items"]
        permutation = rng.permutation(len(items))
        if np.array_equal(permutation, np.arange(len(items))):
            permutation = np.roll(permutation, 1)
        shuffled = [items[index]["weight"] for index in permutation]
        changed += sum((item["weight"] != weight for item, weight in zip(items, shuffled, strict=True)))
        row["items"] = [
            {"value": item["value"], "weight": weight} for item, weight in zip(items, shuffled, strict=True)
        ]
    if changed == 0:
        raise ValueError("weight permutation did not change any item-local associations")
    return rows


def permute_items(observations: list[dict], *, seed: int) -> list[dict]:
    """Jointly permute complete items without changing any target."""
    rng = np.random.default_rng(seed)
    rows = deepcopy(observations)
    for row in rows:
        items = row["items"]
        row["items"] = [items[index] for index in rng.permutation(len(items))]
    return rows


def scale_weights(observations: list[dict], *, factor: float) -> list[dict]:
    """Scale every raw weight and update targets according to their algebra."""
    if factor <= 0.0:
        raise ValueError(f"weight scale must be positive, got {factor}")
    rows = deepcopy(observations)
    for row in rows:
        row["items"] = [{**item, "weight": factor * item["weight"]} for item in row["items"]]
        if "weighted_sum" in row:
            row["weighted_sum"] *= factor
    return rows


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["/weighted_sum"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["weighted_sum"] for row in test], dtype=np.float64)
    baseline_rmse = rmse(actual, float(np.asarray([row["weighted_sum"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline_rmse, "nrmse": measured / baseline_rmse}


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-raw-value-weight-sum
# //| fig-cap: "Learned Attention reduces six item pairs before the hidden weighted sum is decoded."
# //| fig-alt: "Record contains repeated items with value, weight inputs, and hidden weighted sum targets. Root reduction: Attention. Item reduction: Attention; capacity 6."
# #tree(node("record", kind: "root", width: 150pt, body: [
#     - *Reduction:* Attention
#   ], children: (
#   node("items", kind: "branch", repeated: true, width: 155pt, body: [
#       - *Reduction:* Attention
#       - *Capacity:* 6 items
#     ], children: (
#     node("value", type: "Number"),
#     node("weight", type: "Number"),
#   )),
#   node("weighted_sum", kind: "target", type: "Number"),
# )))
# ```
#
# ## How it works
#
# The item branch and root use learned `rf.Attention` reductions. Each of
# six items contains independently sampled values and positive weights. The model
# receives no product field: it must learn enough about the sibling relationship
# to predict `sum(value * weight)`.
#
# Swapping weights between items preserves both marginal distributions while
# breaking their pairing. Scoring these changed inputs against the original labels
# must worsen error. Reordering complete items should leave the answer stable;
# scaling every weight by 0.75 should scale the prediction by the same factor.
# The [supplied-contribution control](supplied-contribution-sum.html) isolates the
# summation part of this task.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(weighted_records(rows=1024, length=ITEMS, seed=seed + 1))
    test = list(weighted_records(rows=512, length=ITEMS, seed=seed + 3))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        items=rf.Branch(
            length=ITEMS, n_layers=2, reduction=rf.Attention(n_layers=2), value=rf.Number, weight=rf.Number
        ),
        weighted_sum=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        model=model,
        train=lambda: weighted_records(rows=1024, length=ITEMS, seed=seed + 1),
        validate=lambda: weighted_records(rows=256, length=ITEMS, seed=seed + 2),
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
    intact_prediction = prediction(model, test)
    intact = score(train=train, test=test, predicted=intact_prediction)
    corrupted_table = permute_weights(test, seed=seed + 4)
    corrupted = score(train=train, test=test, predicted=prediction(model, corrupted_table))
    permuted_prediction = prediction(model, permute_items(test, seed=seed + 5))
    factor = 0.75
    scaled_table = scale_weights(test, factor=factor)
    scaled_prediction = prediction(model, scaled_table)
    scaled = score(train=scale_weights(train, factor=factor), test=scaled_table, predicted=scaled_prediction)
    target_scale = float(np.std(np.asarray([row["weighted_sum"] for row in test], dtype=np.float64)))
    permutation_drift = rmse(intact_prediction, permuted_prediction) / target_scale
    scaling_error = rmse(factor * intact_prediction, scaled_prediction) / (factor * target_scale)
    metrics = {
        "intact": intact,
        "corrupted": corrupted,
        "scaled": scaled,
        "permutation_drift": permutation_drift,
        "scaling_error": scaling_error,
        "steps": trainer.global_step,
    }
    checks = {
        "Weighted sum nRMSE below 0.30": bool(intact["nrmse"] < 0.3),
        "Shuffled weights raise nRMSE by at least 0.35": bool(corrupted["nrmse"] >= intact["nrmse"] + 0.35),
        "Scaled-weight nRMSE below 0.35": bool(scaled["nrmse"] < 0.35),
        "Permutation drift below 0.08 target SD": bool(permutation_drift < 0.08),
        "Weight scaling error below 0.15 target SD": bool(scaling_error < 0.15),
    }
    return metrics, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P014 evidence >}}
#
# ## Remaining work
#
# Repeat the pairing and scaling controls across seeds. Variable-length sum
# duplication, unseen lengths, and zero, negative, missing, or extreme weights
# remain unestablished for this particular weighted-sum schema.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ## Reproduce
#
# {{< proof P014 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=3304)
