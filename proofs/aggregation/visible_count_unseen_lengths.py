# %% [markdown]
# ---
# title: Extending a visible-count sum
# categories:
# - Cardinality generalization
# proof-id: P005
# description: Test whether a model trained on smaller visible counts can extend the
#   learned amount-times-count relationship.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P005 status >}}
#
# ## Insights
#
# **Supplying count lets this model extend its learned total to counts absent from training.** Mean carries
# the repeated amount, while the visible count carries multiplicity. Doubling both the bag and its count
# checks whether predictions respond approximately like a product.
#
# This separates numerical extrapolation from structural counting: the model is told how many items exist.
# It therefore answers a different question from the Attention proof without a count field. The result
# supports a limited extension of the amount-times-count relationship, using bags whose values are identical
# within each record. It does not establish wider mixed-value sums or reliable behavior at arbitrarily large
# counts.
#
# ## Setup

# %%
"""The explicit-count control should extrapolate more cleanly than hidden count.

Require count-aware sum accuracy and duplication scaling out of range.

Run: uv run python proofs/run.py P005"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P005"
TRAIN_MAX = 6
CAPACITY = 12

# %% [markdown]
# ## Examples
#
# Targets use `mask=True`; the labels below are supervision hidden from the encoder.
#
# ### A count absent from training
#
# ```yaml
# item_count: 8
# items:
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
# total: 4.0
# ```
#
# Eight copies of 0.5 require 4.0. The count is visible, but eight was never
# a training cardinality.
#
# ### A count within the training range
#
# ```yaml
# item_count: 4
# items:
#   - {amount: 0.25}
#   - {amount: 0.25}
#   - {amount: 0.25}
#   - {amount: 0.25}
# total: 1.0
# ```
#
# This four-item bag supplies the starting point for a duplication probe.
#
# ### Double the bag and its count
#
# ```yaml
# item_count: 8
# items:
#   - {amount: 0.25}
#   - {amount: 0.25}
#   - {amount: 0.25}
#   - {amount: 0.25}
#   - {amount: 0.25}
#   - {amount: 0.25}
#   - {amount: 0.25}
#   - {amount: 0.25}
# total: 2.0
# ```
#
# The Mean summary stays the same, while `item_count` changes from four to
# eight and the correct total doubles from 1.0 to 2.0.
#
# ## Synthetic data and controls


# %%
def random_records(*, rows: int, seed: int, minimum: int = 1, maximum: int = TRAIN_MAX) -> Iterator[dict]:
    """Draw variable-length numerical bags and their mean and sum."""
    if not 1 <= minimum <= maximum <= CAPACITY:
        raise ValueError(f"length range must satisfy 1 <= minimum <= maximum <= {CAPACITY}")
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        length = int(rng.integers(minimum, maximum + 1))
        values = np.repeat(rng.uniform(-1.0, 1.0), length)
        row: dict[str, object] = {
            "items": [{"amount": float(value)} for value in values],
            "mean_amount": float(values.mean()),
            "total": float(values.sum()),
        }
        row["item_count"] = length
        yield row


def duplicate(observations: list[dict]) -> list[dict]:
    """Duplicate every complete bag and update its algebraic targets."""
    rows = deepcopy(observations)
    if any((len(row["items"]) * 2 > CAPACITY for row in rows)):
        raise ValueError(f"duplicated collection exceeds configured capacity {CAPACITY}")
    for row in rows:
        row["items"] = [*row["items"], *row["items"]]
        row["total"] *= 2.0
        row["item_count"] *= 2
    return rows


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"], "item_count": row["item_count"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["record/total"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["total"] for row in test], dtype=np.float64)
    baseline = rmse(actual, float(np.asarray([row["total"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline, "nrmse": measured / baseline}


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-visible-count-unseen-lengths
# //| fig-cap: "The branch uses Mean without item attention; a visible count distinguishes repeated amounts at unseen lengths."
# //| fig-alt: "Record contains repeated items with amount inputs, visible item count, and hidden total targets. Root reduction: Attention. Item reduction: Mean; capacity 12; branch attention Off."
# #tree(node("record", kind: "root", width: 150pt, body: [
#     - *Reduction:* Attention
#   ], children: (
#   node("item_count", type: "Number"),
#   node("items", kind: "branch", repeated: true, width: 155pt, body: [
#       - *Reduction:* Mean
#       - *Branch attention:* Off
#       - *Capacity:* 12 items
#     ], children: (
#     node("amount", type: "Number"),
#   )),
#   node("total", kind: "target", type: "Number"),
# )))
# ```
#
# ## How it works
#
# The schema matches the [in-range count control](visible-count-sum.html).
# A Mean item branch has no item attention; its equal-valued bags supply content
# but lose multiplicity. The root receives `item_count` as an ordinary Number.
#
# Training counts are one through six. The test uses seven through ten, so
# accuracy measures count extrapolation. A separate complete-bag duplication
# intervention also doubles the visible count and requires approximately doubled
# predictions. It checks whether the learned relationship respects scaling,
# within a fixed branch capacity of twelve.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(random_records(rows=1024, seed=seed + 1))
    unseen = list(random_records(rows=512, seed=seed + 3, minimum=7, maximum=10))
    short = list(random_records(rows=256, seed=seed + 4, maximum=5))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        items=rf.Branch(length=CAPACITY, attention=None, n_layers=2, reduction=rf.Mean(), amount=rf.Number),
        item_count=rf.Number,
        total=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        model=model,
        train=lambda: random_records(rows=1024, seed=seed + 1),
        validate=lambda: random_records(rows=256, seed=seed + 2),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=700 if steps is None else min(steps, 700),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=datamodule)

    # Evaluate held-out answers and retain their original labels in corruption controls.
    unseen_score = score(train=train, test=unseen, predicted=prediction(model, unseen))
    short_prediction = prediction(model, short)
    duplicated_prediction = prediction(model, duplicate(short))
    duplication_error = rmse(2.0 * short_prediction, duplicated_prediction) / float(
        np.std(np.asarray([row["total"] for row in unseen], dtype=np.float64))
    )
    metrics = {"unseen_score": unseen_score, "duplication_error": duplication_error, "steps": trainer.global_step}
    checks = {
        "All measurements finite": bool(np.isfinite([unseen_score["nrmse"], duplication_error]).all()),
        "Unseen nRMSE below 0.40 and duplication error below 0.20": bool(
            unseen_score["nrmse"] < 0.4 and duplication_error < 0.2
        ),
    }
    return metrics, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P005 evidence >}}
#
# ## Remaining work
#
# Repeat across seeds and broaden the input distributions beyond repeated
# values. This proof establishes a route with an explicitly supplied count;
# [Attention without a count field](attention-sum-unseen-lengths.html) covers the
# separate structural-cardinality claim.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ## Reproduce
#
# {{< proof P005 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=3621)
