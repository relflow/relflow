# %% [markdown]
# ---
# title: Subtract a Supplied Peer Mean
# categories:
# - Peer-relative inference
# proof-id: P034
# description: Isolate local subtraction and repeated prediction by supplying each item's
#   peer mean.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# Supplying the mean alongside each value removes the aggregation problem.
# The model only needs to learn subtraction and return the result at the correct
# item coordinate. This is a diagnostic control for the harder peer proofs.
#
# {{< proof P034 status >}}
#
# ## Insights
#
# **Supplying the peer mean makes this a test of local subtraction, not learned
# aggregation.** Each hidden deviation has both operands beside it at the same
# coordinate. The repeated decoder can use those visible siblings together with
# ancestor context to return an item-specific Number.
#
# This control helps localize a failure before asking a model to infer a mean
# or select peers. Its low error cannot establish either of those harder
# capabilities, because the input already supplies the statistic. The recorded
# failed compression contrast was exploratory and is not a general impossibility
# result. Compare the raw-value proof to assess aggregation; use this case to
# check whether local arithmetic and repeated decoding work.
#
# ## Setup

# %%
"""Subtract a supplied peer mean and write each result to its item coordinate.

The mean is deliberately provided beside every value. This diagnostic removes
aggregation and peer selection, isolating local subtraction and repeated
prediction writeback.

Run this file with --help for seed, training-budget, and reporting options.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P034"
ITEMS = 6

# %% [markdown]
# ## Examples
#
# ### Subtract the visible mean
#
# ```yaml
# items:
#   - {value: 1.5, peer_mean: 2.5, deviation: -1.0}
#   - {value: 1.9, peer_mean: 2.5, deviation: -0.6}
#   - {value: 2.3, peer_mean: 2.5, deviation: -0.2}
#   - {value: 2.7, peer_mean: 2.5, deviation: 0.2}
#   - {value: 3.1, peer_mean: 2.5, deviation: 0.6}
#   - {value: 3.5, peer_mean: 2.5, deviation: 1.0}
# ```
#
# Deviations are training supervision hidden by `mask=True` and removed at
# prediction. Each target is its own value minus the supplied mean.
#
# ### A different collection location
#
# ```yaml
# items:
#   - {value: -2.0, peer_mean: -1.0, deviation: -1.0}
#   - {value: -1.6, peer_mean: -1.0, deviation: -0.6}
#   - {value: -1.2, peer_mean: -1.0, deviation: -0.2}
#   - {value: -0.8, peer_mean: -1.0, deviation: 0.2}
#   - {value: -0.4, peer_mean: -1.0, deviation: 0.6}
#   - {value: 0.0, peer_mean: -1.0, deviation: 1.0}
# ```
#
# A new collection can have the same deviations around a different mean. A
# negative raw value does not necessarily imply a negative deviation: −0.4
# is 0.6 above this collection's mean.
#
# ### A smaller spread around the same mean
#
# ```yaml
# items:
#   - {value: 2.0, peer_mean: 2.5, deviation: -0.5}
#   - {value: 2.2, peer_mean: 2.5, deviation: -0.3}
#   - {value: 2.4, peer_mean: 2.5, deviation: -0.1}
#   - {value: 2.6, peer_mean: 2.5, deviation: 0.1}
#   - {value: 2.8, peer_mean: 2.5, deviation: 0.3}
#   - {value: 3.0, peer_mean: 2.5, deviation: 0.5}
# ```
#
# These correct targets are half the first example's deviations. All three
# examples supply the mean explicitly, so they illustrate the local arithmetic
# control rather than evidence for learned collection aggregation.
#
# ## Synthetic data and controls


# %%
def records(*, rows: int, seed: int) -> Iterator[dict]:
    """Generate collections with independent locations and centered residuals."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        basis = rng.normal(size=ITEMS)
        basis -= basis.mean()
        basis /= np.sqrt(np.mean(np.square(basis)))
        scale = float(rng.uniform(0.25, 1.1))
        mean = float(rng.uniform(-3.5, 3.5))
        deviations = scale * basis
        values = mean + deviations
        items: list[dict[str, float]] = []
        for value, deviation in zip(values, deviations, strict=True):
            item = {"value": float(value), "deviation": float(deviation)}
            item["peer_mean"] = mean
            items.append(item)
        yield {"items": items}


def targets(rows: list[dict]) -> np.ndarray:
    return np.asarray([item["deviation"] for row in rows for item in row["items"]])


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    inputs = [
        {"items": [{key: value for key, value in item.items() if key != "deviation"} for item in row["items"]]}
        for row in rows
    ]
    output = model.predict(inputs).to_pylist()
    return np.asarray([value["content"] for row in output for value in row["predictions"]["/items/deviation"]])


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-supplied-peer-mean-control
# //| fig-cap: "Both branch and root keep all tokens. Each masked deviation has its value and supplied mean at the same coordinate."
# //| fig-alt: "Collection contains six repeated items with value and supplied peer mean inputs and a masked Number deviation. The root keeps all tokens. The item branch also keeps all tokens."
# #tree(node("collection", kind: "root", width: 150pt, body: [
#   - *Reduction:* Keep all tokens
# ], children: (
#   node("items", kind: "branch", repeated: true, width: 150pt, body: [
#     - *Capacity:* 6 items
#     - *Reduction:* Keep all tokens
#   ], children: (
#     node("value", type: "Number"),
#     node("peer_mean", type: "Number"),
#     node("deviation", kind: "target", type: "Number", width: 150pt, body: [
#       - *Input:* always hidden
#     ]),
#   )),
# )))
# ```
#
# Both the item branch and root use `reduction=None`. There is no group field
# in this diagnostic rung.
#
# ## How it works
#
# The repeated decoder can use visible siblings at the same coordinate. Since
# both operands are already supplied, a failure would point toward local
# arithmetic or repeated output routing before collection aggregation is tested.
# The generator varies the collection's location independently of its centered
# deviations, so a constant or raw-value shortcut is inadequate.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=48,
        n_layers=3,
        n_heads=4,
        reduction=None,
        batch_size=128,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=1e-3),
        items=rf.Branch(
            length=ITEMS,
            overflow="error",
            n_layers=2,
            reduction=None,
            value=rf.Number,
            peer_mean=rf.Number,
            deviation=rf.Number(mask=True, objective="mse", n_linear=2),
        ),
    )
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, rows=1024, seed=seed + 1),
        validate=partial(records, rows=256, seed=seed + 2),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        max_steps=600 if steps is None else min(steps, 600),
        max_epochs=-1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=data)
    train = list(records(rows=1024, seed=seed + 1))
    test = list(records(rows=512, seed=seed + 3))
    actual = targets(test)
    baseline = rmse(actual, float(targets(train).mean()))
    predicted = predict(model, test)
    intact_error = rmse(actual, predicted)
    intact_nrmse = intact_error / baseline
    metrics = {"intact_rmse": intact_error, "baseline_rmse": baseline, "intact_nrmse": intact_nrmse}
    checks = {"Baseline RMSE > 0.000001": baseline > 1e-6}
    checks["Supplied-mean subtraction nRMSE < 0.25"] = intact_nrmse < 0.25
    checks["Finite calibration and model metrics"] = bool(np.isfinite(list(metrics.values())).all())
    return metrics, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P034 evidence >}}
#
# ## Remaining work
#
# Repeat three core seeds and ten calibration seeds, then test missing operands
# and variable lengths. The [raw ungrouped case](ungrouped-peer-deviation.html)
# removes the supplied mean to test learned aggregation.
#
# ## Reproduce
#
# {{< proof P034 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=3700)
