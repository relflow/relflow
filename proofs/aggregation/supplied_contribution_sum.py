# %% [markdown]
# ---
# title: Summing supplied contributions
# categories:
# - Weighted aggregation
# proof-id: P015
# description: Isolate whether a model can sum item contributions once their products
#   are already supplied.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P015 status >}}
#
# ## Insights
#
# **The model can sum contributions when their products have already been calculated.** This removes
# value–weight binding and multiplication from the task, leaving numerical encoding, reduction, and
# decoding. It is useful evidence that the simpler aggregation route works.
#
# A failure on raw pairs alongside success here would point toward the missing interaction, rather than
# summation alone. However, every collection has the same length, so learning an average and a fixed scaling
# factor could also solve this task. The proof has an accuracy gate but no separate duplication or
# permutation gate. Its success should not be described as general arithmetic over arbitrary collection
# lengths.
#
# ## Setup

# %%
"""A supplied item-local product isolates fixed-width sum reduction.

Attention learns a fixed-width sum after products are supplied.

Run: uv run python proofs/run.py P015"""

from __future__ import annotations

from collections.abc import Iterator

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P015"
ITEMS = 6

# %% [markdown]
# ## Examples
#
# Targets use `mask=True`; the labels below are supervision hidden from the encoder.
#
# ### Add supplied products
#
# ```yaml
# items:
#   - {contribution: 0.4}
#   - {contribution: -0.4}
#   - {contribution: 0.3}
#   - {contribution: 0.1}
#   - {contribution: -0.2}
#   - {contribution: 0.3}
# weighted_sum: 0.5
# ```
#
# The six contributions sum to 0.5; the model does not receive their original factors.
#
# ### Accumulate negative contributions
#
# ```yaml
# items:
#   - {contribution: -0.2}
#   - {contribution: -0.2}
#   - {contribution: -0.2}
#   - {contribution: -0.2}
#   - {contribution: -0.2}
#   - {contribution: -0.2}
# weighted_sum: -1.2
# ```
#
# Six contributions of −0.2 require a sum of −1.2, not their average of −0.2.
#
# ### Balance positive and negative mass
#
# ```yaml
# items:
#   - {contribution: 0.4}
#   - {contribution: -0.4}
#   - {contribution: 0.3}
#   - {contribution: -0.3}
#   - {contribution: 0.2}
#   - {contribution: -0.2}
# weighted_sum: 0.0
# ```
#
# These nonzero contributions cancel. This is another supplied-product diagnostic
# record, not evidence of a separately tested cancellation guarantee.
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
        items = [{"contribution": float(contribution)} for contribution in contributions]
        yield {
            "items": items,
            "weighted_sum": float(contributions.sum()),
            "weighted_mean": float(contributions.sum() / weights.sum()),
        }


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
# //| label: fig-proof-supplied-contribution-sum
# //| fig-cap: "Attention reduces six supplied contributions; multiplication is already provided by the inputs."
# //| fig-alt: "Record contains repeated items with contribution inputs, and hidden weighted sum targets. Root reduction: Attention. Item reduction: Attention; capacity 6."
# #tree(node("record", kind: "root", width: 150pt, body: [
#     - *Reduction:* Attention
#   ], children: (
#   node("items", kind: "branch", repeated: true, width: 155pt, body: [
#       - *Reduction:* Attention
#       - *Capacity:* 6 items
#     ], children: (
#     node("contribution", type: "Number"),
#   )),
#   node("weighted_sum", kind: "target", type: "Number"),
# )))
# ```
#
# ## How it works
#
# The generator computes each `contribution = value * weight` before the
# record reaches the model. Learned `rf.Attention` reductions on the item branch
# and root combine six contributions into the masked sum target.
#
# This is a diagnostic control for the
# [raw-pair proof](raw-value-weight-sum.html). Success establishes fixed-length
# summation of supplied products. It does not establish that the model learned
# multiplication or same-item value–weight binding.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(weighted_records(rows=768, length=ITEMS, seed=seed + 1))
    test = list(weighted_records(rows=384, length=ITEMS, seed=seed + 3))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        items=rf.Branch(length=ITEMS, n_layers=2, reduction=rf.Attention(n_layers=2), contribution=rf.Number),
        weighted_sum=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        model=model,
        train=lambda: weighted_records(rows=768, length=ITEMS, seed=seed + 1),
        validate=lambda: weighted_records(rows=192, length=ITEMS, seed=seed + 2),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=500 if steps is None else min(steps, 500),
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
    checks = {"Contribution sum nRMSE below 0.25": bool(measured["nrmse"] < 0.25)}
    return metrics, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P015 evidence >}}
#
# ## Remaining work
#
# Repeat the control across seeds and test variable lengths and complete-item
# duplication. Keep the raw-pair proof alongside it so that supplying products does
# not conceal a missing learned interaction.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ## Reproduce
#
# {{< proof P015 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=3300)
