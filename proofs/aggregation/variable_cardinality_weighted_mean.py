# %% [markdown]
# ---
# title: Weighted means across varying lengths
# categories:
# - Weighted aggregation
# proof-id: P016
# description: Learn a weighted mean from raw item pairs while preserving the identities
#   of a normalized aggregate.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P016 status >}}
#
# ## Insights
#
# **The model responds to relative weights while keeping the same answer when all weights are scaled
# together.** The model is reported to preserve its answer when complete items are reordered or duplicated,
# and when all weights are scaled equally. Swapping weights between values instead removes accuracy against
# the original labels.
#
# Those controls distinguish useful invariance from simply ignoring the weights. Keep each weight beside its
# value, and choose the target’s algebra deliberately: duplicating items preserves a weighted mean but
# doubles a weighted sum. This proof covers positive weights and lengths within the trained range; a zero
# denominator, signed weights, or substantially longer bags needs separate treatment.
#
# ## Setup

# %%
"""A weighted mean should remain identifiable when collection length varies.

Learn a normalized weighted mean from two through six raw pairs.

Run: uv run python proofs/run.py P016"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P016"
ITEMS = 6

# %% [markdown]
# ## Examples
#
# Targets use `mask=True`; the labels below are supervision hidden from the encoder.
#
# ### Normalize by total weight
#
# ```yaml
# items:
#   - {value: 0.8, weight: 0.5}
#   - {value: -0.4, weight: 1.25}
#   - {value: 0.2, weight: 2.0}
# weighted_mean: 0.08
# ```
#
# The weighted sum is 0.3 and total weight is 3.75, giving a weighted mean of 0.08.
#
# ### Move weight to another value
#
# ```yaml
# items:
#   - {value: 0.8, weight: 1.25}
#   - {value: -0.4, weight: 0.5}
#   - {value: 0.2, weight: 2.0}
# weighted_mean: 0.08  # Retained original label
# ```
#
# Swapping the first two weights changes the mathematical answer to 0.32.
# The corruption keeps 0.08 as its original target, so using the changed pairing
# should worsen the measured error.
#
# ### Duplicate complete items
#
# ```yaml
# items:
#   - {value: 0.8, weight: 0.5}
#   - {value: -0.4, weight: 1.25}
#   - {value: 0.2, weight: 2.0}
#   - {value: 0.8, weight: 0.5}
#   - {value: -0.4, weight: 1.25}
#   - {value: 0.2, weight: 2.0}
# weighted_mean: 0.08
# ```
#
# Doubling both weighted contribution and total weight keeps the answer at 0.08.
# The six-item result stays within the proof’s branch capacity.
#
# ## Synthetic data and controls


# %%
def variable_weighted_mean_records(*, rows: int, seed: int) -> Iterator[dict]:
    """Draw variable-width rows with broad positive weights."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        length = int(rng.integers(2, ITEMS + 1))
        values = rng.uniform(-1.0, 1.0, size=length)
        weights = np.exp(rng.uniform(np.log(0.03), np.log(3.0), size=length))
        yield {
            "items": [
                {"value": float(value), "weight": float(weight)} for value, weight in zip(values, weights, strict=True)
            ],
            "weighted_mean": float(np.dot(values, weights) / weights.sum()),
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


def duplicate_items(observations: list[dict], *, maximum_original: int) -> list[dict]:
    """Duplicate every item on rows that remain within configured capacity."""
    rows = [row for row in deepcopy(observations) if len(row["items"]) <= maximum_original]
    if not rows:
        raise ValueError(f"no rows contain at most {maximum_original} items")
    for row in rows:
        row["items"] = [*row["items"], *row["items"]]
        if "weighted_sum" in row:
            row["weighted_sum"] *= 2.0
    return rows


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["/weighted_mean"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["weighted_mean"] for row in test], dtype=np.float64)
    baseline_rmse = rmse(actual, float(np.asarray([row["weighted_mean"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline_rmse, "nrmse": measured / baseline_rmse}


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-variable-cardinality-weighted-mean
# //| fig-cap: "Attention reduces raw pairs within a six-item capacity to predict a hidden weighted mean."
# //| fig-alt: "Record contains repeated items with value, weight inputs, and hidden weighted mean targets. Root reduction: Attention. Item reduction: Attention; capacity 6."
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
#   node("weighted_mean", kind: "target", type: "Number"),
# )))
# ```
#
# ## How it works
#
# Learned `rf.Attention` reductions receive two through six item pairs.
# The label is `sum(value * weight) / sum(weight)`. Values vary independently;
# positive weights span 0.03 through 3.0 on a logarithmic scale, making an
# unweighted average a poor shortcut.
#
# Complete-item permutation, scaling every weight by 0.75, and duplicating all
# items should preserve the answer. Duplication is checked only for original
# lengths two and three so the result stays within branch capacity. Permuting
# weights alone must worsen error against unchanged labels: normalization does
# not remove the need to bind each weight to its own value.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(variable_weighted_mean_records(rows=1536, seed=seed + 1))
    test = list(variable_weighted_mean_records(rows=768, seed=seed + 3))
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
        weighted_mean=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        model=model,
        train=lambda: variable_weighted_mean_records(rows=1536, seed=seed + 1),
        validate=lambda: variable_weighted_mean_records(rows=384, seed=seed + 2),
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
    intact_prediction = prediction(model, test)
    intact = score(train=train, test=test, predicted=intact_prediction)
    corrupted_table = permute_weights(test, seed=seed + 4)
    corrupted = score(train=train, test=test, predicted=prediction(model, corrupted_table))
    permuted_prediction = prediction(model, permute_items(test, seed=seed + 5))
    scaled_prediction = prediction(model, scale_weights(test, factor=0.75))
    original_short = [row for row in deepcopy(test) if len(row["items"]) <= ITEMS // 2]
    duplicated_prediction = prediction(model, duplicate_items(test, maximum_original=ITEMS // 2))
    original_short_prediction = prediction(model, original_short)
    target_scale = float(np.std(np.asarray([row["weighted_mean"] for row in test], dtype=np.float64)))
    permutation_drift = rmse(intact_prediction, permuted_prediction) / target_scale
    scaling_drift = rmse(intact_prediction, scaled_prediction) / target_scale
    duplication_drift = rmse(original_short_prediction, duplicated_prediction) / target_scale
    metrics = {
        "intact": intact,
        "corrupted": corrupted,
        "permutation_drift": permutation_drift,
        "scaling_drift": scaling_drift,
        "duplication_drift": duplication_drift,
        "steps": trainer.global_step,
    }
    checks = {
        "Weighted mean nRMSE below 0.40": bool(intact["nrmse"] < 0.4),
        "Shuffled weights raise nRMSE by at least 0.25": bool(corrupted["nrmse"] >= intact["nrmse"] + 0.25),
        "Permutation drift below 0.10 target SD": bool(permutation_drift < 0.1),
        "Weight scaling drift below 0.10 target SD": bool(scaling_drift < 0.1),
        "Duplication drift below 0.10 target SD": bool(duplication_drift < 0.1),
    }
    return metrics, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P016 evidence >}}
#
# ## Remaining work
#
# Repeat every identity across seeds. Zero or signed total weight, missing
# weights, extreme magnitudes, and lengths outside the trained range still need
# explicit semantics and checks.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ## Reproduce
#
# {{< proof P016 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=3313)
