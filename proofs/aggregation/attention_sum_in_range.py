# %% [markdown]
# ---
# title: Sums within the trained length range
# categories:
# - Cardinality generalization
# proof-id: P001
# description: Learn a total from independent item values when collection lengths vary
#   within the training range.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P001 status >}}
#
# ## Insights
#
# **Attention can learn totals across varying collection lengths without a supplied count field.**
# Independent signed amounts prevent count alone from predicting the answer. The reduction retains additive
# evidence and present-token count alongside its learned summary, providing a route for both content and
# multiplicity to survive.
#
# The held-out accuracy and item-permutation controls support this route within the trained length range.
# They do not establish a general sum rule: a model could still learn relationships specialized to familiar
# lengths. The unseen-length proof separately checks longer bags and duplication. Treat this case as
# evidence for ordinary interpolation, rather than a guarantee that every accepted input shape will be
# handled accurately.
#
# ## Setup

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

# %% [markdown]
# ## Examples
#
# Targets use `mask=True`; the labels below are supervision hidden from the encoder.
#
# ### Sum a short bag
#
# ```yaml
# items:
#   - {amount: 0.5}
#   - {amount: -0.2}
#   - {amount: 0.9}
# total: 1.2
# ```
#
# Independent positive and negative amounts combine into a total of 1.2.
#
# ### A contrasting signed bag
#
# ```yaml
# items:
#   - {amount: -0.5}
#   - {amount: 0.2}
#   - {amount: -0.9}
# total: -1.2
# ```
#
# These amounts instead sum to −1.2. The number of items alone cannot
# determine either answer.
#
# ### Reorder complete items
#
# ```yaml
# items:
#   - {amount: 0.9}
#   - {amount: 0.5}
#   - {amount: -0.2}
# total: 1.2
# ```
#
# This is the first bag in a different order. Its label stays 1.2, and the
# permutation gate checks that predictions remain approximately unchanged.
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
# //| label: fig-proof-attention-sum-in-range
# //| fig-cap: "A single Attention output summarizes a branch with capacity twelve; training and this test use shorter bags."
# //| fig-alt: "Record contains repeated items with amount inputs, and hidden total targets. Root reduction: Attention. Item reduction: Attention (1 token); capacity 12; branch attention MHA."
# #tree(node("record", kind: "root", width: 150pt, body: [
#     - *Reduction:* Attention
#   ], children: (
#   node("items", kind: "branch", repeated: true, width: 155pt, body: [
#       - *Reduction:* Attention (1 token)
#       - *Branch attention:* MHA
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
# ## Training and evaluation


# %%
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


# %% [markdown]
# ## Evidence
#
# {{< proof P001 evidence >}}
#
# ## Remaining work
#
# Repeat across seeds and change capacities, widths, and nested placement.
# This in-range result alone establishes neither arbitrary-length generalization
# nor an exact arithmetic sum; empty and missing-item cases need separate checks.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ## Reproduce
#
# {{< proof P001 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=3609)
