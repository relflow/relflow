# %% [markdown]
# ---
# title: Keep Each Subtotal with Its Item
# categories:
# - Item alignment
# proof-id: P030
# description: Reconstruct each item's subtotal while preserving variable-length prediction
#   geometry.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# Each subtotal must use the quantity and price at its own item coordinate.
# The proof combines learning this local relationship with checking that empty,
# short, and full collections keep the correct prediction coordinates.
#
# {{< proof P030 status >}}
#
# ## Insights
#
# **The model learns each item's subtotal and keeps its prediction attached to
# the correct coordinate.** Repeated decoder queries can use aligned quantity
# and price inputs as well as ancestor context; they are not limited to one
# identical root representation for every item.
#
# Permuting only target subtotals leaves requests and predictions unchanged but
# raises error against the altered labels. That isolates coordinate alignment.
# Separate checks verify that real items are inferred and padded coordinates
# are not, including empty orders. The arithmetic score is still aggregated over
# lengths, so a weak individual length could be hidden. This establishes local
# learned arithmetic and output geometry, not general cross-item reasoning or
# exact billing calculations.
#
# ## Setup

# %%
"""Reconstruct quantity times unit price at each original item coordinate.

Rows range from empty to five items. Permuting only target subtotals tests local
alignment; fixed output lengths and inferred flags distinguish real coordinates
from padding.

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

PROOF_ID = "P030"
LENGTH = 5

# %% [markdown]
# ## Examples
#
# ### Two items keep their own subtotals
#
# ```yaml
# items:
#   - {quantity: 1.5, unit_price: 1.2, subtotal: 1.8}
#   - {quantity: 0.5, unit_price: -1.0, subtotal: -0.5}
# ```
#
# Subtotals are supervision hidden by `mask=True` and removed before prediction.
# The generator uses the exact product of quantity and unit price, with no
# additional target noise.
#
# ### An empty order has no inferred subtotal
#
# ```yaml
# items: []
# ```
#
# This is a valid length-zero record in the proof. The fixed output schema still
# has five subtotal coordinates, all marked `inferred=False`; there is no real
# item for which a value should be inferred.
#
# ### Exchange the target subtotals
#
# ```yaml
# items:
#   - {quantity: 1.5, unit_price: 1.2, subtotal: -0.5}
#   - {quantity: 0.5, unit_price: -1.0, subtotal: 1.8}
# ```
#
# The control permutes only the first example's subtotals. These are deliberately
# incorrect targets for the visible quantity/price pairs. The requests and model
# predictions stay unchanged, so worse error against these labels demonstrates
# that the intact result was tied to each item's coordinate.
#
# ## Synthetic data and controls


# %%
def records(*, rows: int, seed: int, permute_targets: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    target_rng = np.random.default_rng(seed + 10000)
    for row in range(rows):
        length = row % (LENGTH + 1)
        quantity = rng.uniform(0.25, 2.0, size=length)
        unit_price = rng.uniform(-1.5, 1.5, size=length)
        subtotal = quantity * unit_price
        if permute_targets and length > 1:
            subtotal = subtotal[target_rng.permutation(length)]
        yield {
            "items": [
                {"quantity": float(item_quantity), "unit_price": float(item_price), "subtotal": float(item_subtotal)}
                for item_quantity, item_price, item_subtotal in zip(quantity, unit_price, subtotal, strict=True)
            ]
        }


def inputs(rows: list[dict]) -> list[dict]:
    return [
        {"items": [{key: value for key, value in item.items() if key != "subtotal"} for item in row["items"]]}
        for row in rows
    ]


def targets(rows: list[dict]) -> np.ndarray:
    return np.asarray([item["subtotal"] for row in rows for item in row["items"]])


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-repeated-subtotal-reconstruction
# //| fig-cap: "The branch supports up to five items with masked subtotals. Default attention summaries coexist with each target's aligned quantity and price context."
# //| fig-alt: "Order contains up to five repeated items with quantity and unit-price Numbers and masked Number subtotals. The item branch and root each use one learned attention summary."
# #tree(node("order", kind: "root", width: 150pt, body: [
#   - *Reduction:* `Attention`
#   - *Learned summaries:* 1
# ], children: (
#   node("items", kind: "branch", repeated: true, width: 150pt, body: [
#     - *Capacity:* 5 items
#     - *Reduction:* `Attention`
#     - *Learned summaries:* 1
#   ], children: (
#     node("quantity", type: "Number"),
#     node("unit_price", type: "Number"),
#     node("subtotal", kind: "target", type: "Number", width: 150pt, body: [
#       - *Input:* always hidden
#     ]),
#   )),
# )))
# ```
#
# The branch uses default attention reduction. Related inputs and their target
# stay together instead of being flattened into unrelated rows.
#
# ## How it works
#
# Shared coordinates expose each target's own visible siblings to the decoder.
# A paired control keeps every visible order unchanged but permutes target
# subtotals between items. Predictions are therefore identical, while comparison
# against the corrupted targets should become substantially worse.
#
# Lengths cycle from zero through five. The output uses five coordinates per
# row; real items must be marked `inferred=True`, with every padded coordinate
# marked `inferred=False`.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        name="order",
        d_model=48,
        n_layers=1,
        n_heads=4,
        batch_size=128,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=1e-3),
        items=rf.Branch(
            length=LENGTH,
            overflow="error",
            n_layers=2,
            quantity=rf.Number,
            unit_price=rf.Number,
            subtotal=rf.Number(mask=True, objective="mse"),
        ),
    )
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, rows=4096, seed=seed + 1),
        validate=partial(records, rows=1024, seed=seed + 2),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        max_steps=-1 if steps is None else steps,
        max_epochs=25,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model, datamodule=data)
    train = list(records(rows=4096, seed=seed + 1))
    test = list(records(rows=2048, seed=seed + 3))
    broken = list(records(rows=2048, seed=seed + 3, permute_targets=True))
    output = model.predict(inputs(test)).to_pylist()
    coordinates = [row["predictions"]["order/items/subtotal"] for row in output]
    lengths = [len(row["items"]) for row in test]
    predicted = np.asarray(
        [value["content"] for row, length in zip(coordinates, lengths, strict=True) for value in row[:length]]
    )
    actual = targets(test)
    baseline = rmse(actual, float(targets(train).mean()))
    aligned = rmse(actual, predicted) / baseline
    broken_error = rmse(targets(broken), predicted) / baseline
    inferred = [[bool(value["inferred"]) for value in row] for row in coordinates]
    expected = [[True] * length + [False] * (LENGTH - length) for length in lengths]
    metrics = {"aligned_nrmse": aligned, "permuted_target_nrmse": broken_error, "baseline_rmse": baseline}
    checks = {
        "Target permutation preserves visible inputs": inputs(test) == inputs(broken),
        "Output retains the fixed branch length": all(len(row) == LENGTH for row in coordinates),
        "Inferred flags match real and padded coordinates": inferred == expected,
        "Aligned subtotal nRMSE <= 0.25": aligned <= 0.25,
        "Permuting targets increases nRMSE by >= 0.50": broken_error >= aligned + 0.50,
    }
    return metrics, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P030 evidence >}}
#
# ## Remaining work
#
# Report accuracy separately by nonempty length and repeat three paired core
# seeds plus ten calibration seeds. The proof guide also flags a mismatch between
# its exact-product generator and the noisy-product proposal; that broader noisy
# task has not been established by this case.
#
# ## Reproduce
#
# {{< proof P030 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=31)
