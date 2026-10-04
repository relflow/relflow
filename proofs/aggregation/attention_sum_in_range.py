# %% [markdown]
# ---
# title: Can the model add a list of numbers?
# categories:
# - Cardinality generalization
# proof-id: P001
# description: Predict the total of a list whose length varies within the range used in training.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: New lists
#   metric:
#   - measured
#   - nrmse
#   format: error
# ---
#
# ## Example
#
# ```yaml
# items:
#   - {amount: 0.5}
#   - {amount: -0.2}
#   - {amount: 0.9}
# total: 1.2
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-proof-attention-sum-in-range
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, items, amount, total. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("amount", type: "Number"),
#   )),
#   node("total", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Test on fresh lists, then reorder the same items. Reordering should preserve the total.
#
# ## Result
#
# {{< proof P001 status >}}
#
# The model learns an approximate total in this range. Longer lists need a separate test.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P001 evidence >}}
#
# ### Run this experiment
#
# {{< proof P001 script >}}
#
# ### How it works
#
# Training, validation, and test records contain one through six independent
# amounts drawn from −1 to 1. Both the repeated item branch and root use learned
# `rf.Attention` reductions. There is no visible count input.
#
# Attention’s reduction retains aggregate evidence and present-token count as
# well as its normalized learned summary. These representations offer a route to
# predicting sums with changing length. Independent values prevent count alone
# from solving the task; a complete-item permutation checks approximate order
# stability. [Unseen-length tests](attention-sum-unseen-lengths.html) separately
# check whether the learned behavior extends beyond interpolation.
#
# ### Remaining work
#
# Repeat across seeds and change capacities, widths, and nested placement.
# This in-range result alone establishes neither arbitrary-length generalization
# nor an exact arithmetic sum; empty and missing-item cases need separate checks.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""Learned Attention can interpolate a sum over seen cardinalities.

Check ordinary in-range accuracy and approximate order invariance.

Run: uv run python proofs/run.py P001"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P001"
TRAIN_MAX = 6
CAPACITY = 12


def random_records(*, rows: int, seed: int, minimum: int = 1, maximum: int = TRAIN_MAX) -> Iterator[dict]:
    """Draw variable-length numerical bags and their mean and sum."""
    if not 1 <= minimum <= maximum <= CAPACITY:
        raise ValueError(f"length range must satisfy 1 <= minimum <= maximum <= {CAPACITY}")
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        length = int(rng.integers(minimum, maximum + 1))
        values = rng.uniform(-1.0, 1.0, size=length)
        row: dict[str, object] = {
            "items": [{"amount": float(value)} for value in values],
            "mean_amount": float(values.mean()),
            "total": float(values.sum()),
        }
        yield row


def permute(observations: list[dict], *, seed: int) -> list[dict]:
    """Jointly permute complete items without changing any target."""
    rng = np.random.default_rng(seed)
    rows = deepcopy(observations)
    for row in rows:
        items = row["items"]
        row["items"] = [items[index] for index in rng.permutation(len(items))]
    return rows


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["/total"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["total"] for row in test], dtype=np.float64)
    baseline = rmse(actual, float(np.asarray([row["total"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline, "nrmse": measured / baseline}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(random_records(rows=1536, seed=seed + 1))
    test = list(random_records(rows=768, seed=seed + 3))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        items=rf.Branch(
            length=CAPACITY, attention="mha", n_layers=2, reduction=rf.Attention(n_layers=2), amount=rf.Number
        ),
        total=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        model=model,
        train=lambda: random_records(rows=1536, seed=seed + 1),
        validate=lambda: random_records(rows=384, seed=seed + 2),
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
    measured = score(train=train, test=test, predicted=intact_prediction)
    permuted_prediction = prediction(model, permute(test, seed=seed + 5))
    target_scale = float(np.asarray([row["total"] for row in test], dtype=np.float64).std())
    permutation_drift = rmse(intact_prediction, permuted_prediction) / target_scale
    metrics = {"measured": measured, "permutation_drift": permutation_drift, "steps": trainer.global_step}
    checks = {
        "Seen-length nRMSE below 0.35": bool(measured["nrmse"] < 0.35),
        "Permutation drift below 0.12 target SD": bool(permutation_drift < 0.12),
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3609)

# %% [markdown]
# </details>
