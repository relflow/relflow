# %% [markdown]
# ---
# title: Does the count help with longer lists?
# categories:
# - Cardinality generalization
# description: Give the model both a repeated amount and its count, then test larger counts absent from training.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Larger counts
#   metric:
#   - unseen_score
#   - nrmse
#   format: error
# evidence-group: Aggregation
# ---
#
# This is a test of extending an amount-times-count relationship beyond the counts used in training.
# Each bag repeats a single amount, and item_count is visible. The model need not discover the count
# from scratch to solve these records.
#
# ## Example
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
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# Eight copies of 0.5 give the illustrated total 4.0. Counts seven through ten are absent from
# training. To picture the scaling check within the allowed capacity, start with four copies, whose
# answer is 2.0, and duplicate that whole bag while changing item_count from four to eight.
#
# ```{typst}
# //| label: fig-evidence-visible-count-unseen-lengths
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, item_count, items, amount, total. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("item_count", type: "Number"),
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("amount", type: "Number"),
#   )),
#   node("total", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Repeat the list and double the visible count. The expected total doubles too.
#
# Training counts run from one through six, while the main test uses seven through ten. The separate
# duplication intervention changes both the item collection and its visible count consistently;
# leaving the old count in place would create contradictory inputs. Amounts remain within the same
# numerical range, so the change being probed is list length rather than a new value distribution.
#
# ## Result
#
# {{< evidence P005 status >}}
#
# The model extends the amount-times-count relationship to these larger counts. The count is
# supplied, so this does not show that it learned to count items.
#
#
# The reported error compares the longer-list predictions with a constant training-mean guess. The
# scaling error compares the duplicated prediction with twice the original prediction. These are
# complementary checks: low test error alone would not show that individual matched examples scale
# properly. The result is limited to repeated equal values and the tested finite capacity, with
# count explicitly available.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P005 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P005 script >}}
#
# ### How it works
#
# The schema matches the [in-range count control](visible-count-sum.html).
# A Mean item branch has no item attention. Its averaged content is unchanged
# by repeating equal values; its current structural count lane retains multiplicity.
# The root additionally receives `item_count` as an ordinary Number. This is not
# an ablation of the two available count routes.
#
# Training counts are one through six. The test uses seven through ten, so
# accuracy measures count extrapolation. A separate complete-bag duplication
# intervention also doubles the visible count and requires approximately doubled
# predictions. It checks whether the learned relationship respects scaling,
# within a fixed branch capacity of twelve.
#
# ### Remaining work
#
# Repeat across seeds and broaden the input distributions beyond repeated
# values. This evidence establishes a route with an explicitly supplied count;
# [Attention without a count field](attention-sum-unseen-lengths.html) covers the
# separate structural-cardinality claim.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""The explicit-count control should extrapolate more cleanly than hidden count.

Require count-aware sum accuracy and duplication scaling out of range.

Run: uv run python evidence/experiments/run.py P005"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import experiment, report

TRAIN_MAX = 6
CAPACITY = 12


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
    if any(len(row["items"]) * 2 > CAPACITY for row in rows):
        raise ValueError(f"duplicated collection exceeds configured capacity {CAPACITY}")
    for row in rows:
        row["items"] = [*row["items"], *row["items"]]
        row["total"] *= 2.0
        row["item_count"] *= 2
    return rows


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"], "item_count": row["item_count"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["total"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["total"] for row in test], dtype=np.float64)
    baseline = rmse(actual, float(np.asarray([row["total"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline, "nrmse": measured / baseline}


@experiment("P005", seed=3621)
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
        items=rf.Branch(length=CAPACITY, attention=None, n_layers=2, reduction=rf.Mean(), amount=rf.Number),
        item_count=rf.Number,
        total=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        train=lambda: random_records(rows=1024, seed=seed + 1),
        validate=lambda: random_records(rows=256, seed=seed + 2),
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
        max_steps=700 if steps is None else min(steps, 700),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer

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


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
