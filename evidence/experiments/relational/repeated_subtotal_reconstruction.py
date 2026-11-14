# %% [markdown]
# ---
# title: Does each subtotal stay with its own item?
# categories:
# - Item alignment
# description: Predict quantity times price separately for every real item in an order.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Correct item answers
#   metric:
#   - aligned_nrmse
#   format: error
# - label: Shuffled item answers
#   metric:
#   - permuted_target_nrmse
#   format: error
# evidence-group: Relational
# ---
#
# ## Example
#
# Each item has its own visible quantity and unit price, and needs its own hidden subtotal. The illustrative
# first answer is `1.5 × 1.2 = 1.8`; the second is `0.5 × -1.0 = -0.5`. Negative prices are permitted by
# this synthetic arithmetic generator. These are two separate predictions at the original item positions,
# rather than one total for the order. Generated orders range from empty to five items.
#
# ```yaml
# items:
#   - {quantity: 1.5, unit_price: 1.2, subtotal: 1.8}
#   - {quantity: 0.5, unit_price: -1.0, subtotal: -0.5}
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-repeated-subtotal-reconstruction
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with order, items, quantity, unit_price, subtotal. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("order", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("quantity", type: "Number"),
#     node("unit_price", type: "Number"),
#     node("subtotal", kind: "target", type: "Number", width: 150pt,),
#   )),
# )))
# ```
#
# ## Comparison
#
# The scoring control shuffles subtotal targets among items inside an order without changing the visible
# quantities or prices. For the illustrated pair, swapping `1.8` and `-0.5` would attach valid numbers to
# the wrong records. Predictions should fit the original pairing better. Independent structural checks also
# require the output list to retain exactly the real items, including zero predictions for an empty order,
# and to mark each predicted coordinate as inferred.
#
# Shuffle the expected subtotals between items while keeping visible inputs fixed. Also check empty orders and retained item positions.
#
# ## Result
#
# Normalized RMSE divides subtotal error by the error of a constant guess based on the training targets.
# Lower error for aligned targets, with a clear increase for shuffled targets, supports learning an item-
# specific relationship. Correct output lengths address a different question: where those predictions were
# written. Neither low average error nor correct alignment makes multiplication exact; the example explains
# the target formula rather than showing a measured prediction.
#
# {{< evidence P030 status >}}
#
# Predictions follow the correct items, and empty orders stay empty. The arithmetic is approximate and should not replace exact billing calculations.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P030 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P030 script >}}
#
# ### How it works
#
# Shared coordinates expose each target's own visible siblings to the decoder.
# A paired control keeps every visible order unchanged but permutes target
# subtotals between items. Predictions are therefore identical, while comparison
# against the corrupted targets should become substantially worse.
#
# Lengths cycle from zero through five. Each output retains exactly the input's
# item count without padding; every real item must be marked `inferred=True`.
#
# ### Remaining work
#
# Report accuracy separately by nonempty length and repeat three paired core
# seeds plus ten calibration seeds. The evidence guide also flags a mismatch between
# its exact-product generator and the noisy-product proposal; that broader noisy
# task has not been established by this case.
#
# ### Complete experiment code
#

# %%
"""Reconstruct quantity times unit price at each original item coordinate.

Rows range from empty to five items. Permuting only target subtotals tests local
alignment; output lengths and inferred flags identify the retained coordinates.

Run this file with --help for seed, training-budget, and reporting options.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
import torch
from relflow.presets.base import resolve
from reporting import experiment, report

LENGTH = 5


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


def prediction(model: rf.Model, rows: list[dict]) -> list[list[dict]]:
    output = pl.DataFrame(model.predict(inputs(rows))).to_dicts()
    return [[item["subtotal"] for item in row["predictions"]["items"]] for row in output]


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


@experiment("P030", seed=31)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Freeze the experimental architecture independently of SDK preset changes.
    control_preset = rf.presets.Preset(
        name="xs",
        d_model=64,
        root=rf.presets.BranchDefaults(n_layers=1, n_heads=2, reduction=rf.Attention()),
        branch=rf.presets.BranchDefaults(n_layers=1, n_heads=2, reduction=rf.Attention()),
        leaf=rf.presets.LeafDefaults(n_heads=2),
    )
    model = rf.Model(
        schema=resolve(
            control_preset,
            items=rf.Branch(
                length=LENGTH,
                overflow="error",
                n_layers=2,
                quantity=rf.Number,
                unit_price=rf.Number,
                subtotal=rf.Number(mask=True, objective="mse"),
            ),
        )
    )
    model.preset = control_preset
    data = rf.SyntheticDataModule(
        train=partial(records, rows=4096, seed=seed + 1),
        validate=partial(records, rows=1024, seed=seed + 2),
        seed=seed,
    )
    model.fit(
        data,
        batch_size=128,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
        accelerator=accelerator,
        max_steps=-1 if steps is None else steps,
        max_epochs=25,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    train = list(records(rows=4096, seed=seed + 1))
    test = list(records(rows=2048, seed=seed + 3))
    broken = list(records(rows=2048, seed=seed + 3, permute_targets=True))
    coordinates = prediction(model, test)
    lengths = [len(row["items"]) for row in test]
    predicted = np.asarray([value["content"] for row in coordinates for value in row])
    actual = targets(test)
    baseline = rmse(actual, float(targets(train).mean()))
    aligned = rmse(actual, predicted) / baseline
    broken_error = rmse(targets(broken), predicted) / baseline
    inferred = [[bool(value["inferred"]) for value in row] for row in coordinates]
    expected = [[True] * length for length in lengths]
    metrics = {"aligned_nrmse": aligned, "permuted_target_nrmse": broken_error, "baseline_rmse": baseline}
    checks = {
        "Target permutation preserves visible inputs": inputs(test) == inputs(broken),
        "Output retains each real item without padding": [len(row) for row in coordinates] == lengths,
        "Inferred flags identify every real coordinate": inferred == expected,
        "Aligned subtotal nRMSE <= 0.25": aligned <= 0.25,
        "Permuting targets increases nRMSE by >= 0.50": broken_error >= aligned + 0.50,
    }
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
