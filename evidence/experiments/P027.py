# %% [markdown]
# ---
# title: Can it choose both a group and an operation?
# categories:
# - Category-conditioned reduction
# description: Ask for a group’s minimum, maximum, average, or total from the same list.
# evidence-readout:
# - label: Group A, maximum
#   metric:
#   - intact/max/A/nrmse
#   format: error
# - label: Group A, average
#   metric:
#   - intact/mean/A/nrmse
#   format: error
# - label: Group A, total
#   metric:
#   - intact/sum/A/nrmse
#   format: error
# evidence-group: Relational
# evidence-diagram:
#   caption: Amber cards are hidden prediction targets. Repeated collections keep their fields together.
#   alt: Model tree with request, items, value, group, answer, operation, selected_group. Amber cards are hidden prediction
#     targets. Repeated collections keep their fields together.
# evidence-review:
#   question: Can relflow answer all twelve group-and-operation requests?
#   modeling: Give each item a group and value, then request one group’s minimum, maximum, average, or total. Hide
#     the requested answer. Generate independent lists for training and testing, and measure every combination of
#     three groups and four operations.
#   comparison:
#     metric: Normalized RMSE
#     kind: Recorded conditions
#     conditions:
#     - label: Min · group A
#       path:
#       - intact/min/A/nrmse
#       format: number
#     - label: Min · group B
#       path:
#       - intact/min/B/nrmse
#       format: number
#     - label: Min · group C
#       path:
#       - intact/min/C/nrmse
#       format: number
#     - label: Max · group A
#       path:
#       - intact/max/A/nrmse
#       format: number
#     - label: Max · group B
#       path:
#       - intact/max/B/nrmse
#       format: number
#     - label: Max · group C
#       path:
#       - intact/max/C/nrmse
#       format: number
#     - label: Mean · group A
#       path:
#       - intact/mean/A/nrmse
#       format: number
#     - label: Mean · group B
#       path:
#       - intact/mean/B/nrmse
#       format: number
#     - label: Mean · group C
#       path:
#       - intact/mean/C/nrmse
#       format: number
#     - label: Sum · group A
#       path:
#       - intact/sum/A/nrmse
#       format: number
#     - label: Sum · group B
#       path:
#       - intact/sum/B/nrmse
#       format: number
#     - label: Sum · group C
#       path:
#       - intact/sum/C/nrmse
#       format: number
#     definition: Root mean squared error divided by the reference scale defined in this experiment. Zero is exact;
#       one equals that reference scale.
#     direction: Lower error is better
#   note: The chart compares errors across the twelve tasks, not an ablation. All twelve must satisfy their own criteria.
#   criteria:
#     operations: group
# evidence-plot:
#   target: answer
#   focus: Can group selection combine with the requested reduction?
#   comparison: Compare every recorded group-operation cell on a shared error scale.
#   limit: The cell grid reports tested combinations; a pooled score cannot establish every operation.
#   renderer: plot
# evidence-cost:
#   baseline: 409
#   candidate: 417
# ---
#
# Two instructions jointly define the answer: which group to select and which operation to apply.
# Every twelve-item bag contains four values from each of A, B, and C. The visible operation
# requests a sum, average, minimum, or maximum of only the selected group's values.
#
# The bag number is generator bookkeeping, not a model input.
#
# ## Example
#
# ```yaml
# bag: 0
# items:
#   - {group: A, value: 0.1}
#   - {group: B, value: -0.1}
#   - {group: C, value: 0.0}
#   - {group: A, value: 0.4}
#   - {group: B, value: 0.3}
#   - {group: C, value: 0.5}
#   - {group: A, value: -1.0}
#   - {group: B, value: -0.9}
#   - {group: C, value: -1.2}
#   - {group: A, value: 1.2}
#   - {group: B, value: 1.3}
#   - {group: C, value: 1.0}
# selected_group: B
# operation: max
# answer: 1.3
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# The illustrated B values are −0.1, 0.3, −0.9, and 1.3. Their maximum is 1.3, so that is the answer
# to the shown request. On the same B values, sum would request 0.6, mean 0.15, and min −0.9.
# Changing the instruction changes the answer without changing the list.
#
# ```typst
# #tree(node("request", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("value", type: "Number"),
#     node("group", type: "Category"),
#   )),
#   node("answer", kind: "target", type: "Number", width: 150pt,),
#   node("operation", type: "Category"),
#   node("selected_group", type: "Category"),
# )))
# ```
#
# ## Comparison
#
# Measure every group-and-operation combination so a good score on one cannot hide a poor score on
# another.
#
# Each bag produces all twelve group–operation requests within the same split, and different splits
# use independently generated bags. Every combination receives its own score, preventing a good
# average from hiding a failed group or operation. The input values are chosen so different
# reductions usually disagree. This version does not corrupt request fields or group membership;
# those would provide stronger direct tests of information use.
#
# ## Result
#
# All tested combinations learn. This exact experiment does not corrupt either request, so stronger
# evidence that both instructions are used remains needed.
#
# Each error percentage uses that particular group's and operation's training-mean baseline. The
# overview shows a few examples, while the checks cover all twelve combinations. Low errors support
# the composed task on these familiar labels and fixed group sizes; they do not show that an unseen
# operation or unseen combination can be composed without training examples.
#
# ## Method
#
# ### How it works
#
# Coordinate-local mixing binds groups to values, retained item slots preserve
# their associations, and visible request fields condition the decoder query.
# Independently generated bags are assigned to each split before expansion into
# requests, so related answers stay together. Measuring all twelve cells prevents
# a favorable average from hiding one failed group or operation.
#
# ### Remaining work
#
# Add label, selected-group, and operation corruptions to this composed case.
# Repeat three core seeds and ten calibration seeds, vary collection length, and
# test absent groups and missing values. Sweep reduction width independently of
# the number of requested cells. See the [filtered-mean control](filtered-mean.html).
#

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.10",
#     "lightning",
#     "numpy",
#     "polars",
#     "pyyaml>=6",
#     "relflow>=0.5.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "structure"
# types = []
# conditions = ["fresh-records"]
# ///

"""Compose a requested group filter with sum, mean, minimum, or maximum.

Each bag contains four items from each of three groups and produces all twelve
requests. Score every group/operation cell against its own training-mean
baseline so one easy operation cannot hide failure on another.

Run this file with --help for seed, training-budget, and reporting options.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from functools import partial
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
import torch
from reporting import experiment, report

OPERATIONS = ("sum", "mean", "min", "max")
GROUPS = ("A", "B", "C")


def reduce(values: np.ndarray, operation: str) -> float:
    """Execute one synthetic reduction outside RelFlow."""

    match operation:
        case "sum":
            return float(values.sum())
        case "mean":
            return float(values.mean())
        case "min":
            return float(values.min())
        case "max":
            return float(values.max())
        case _:
            raise ValueError(f"unknown reduction operation: {operation!r}")


def values(rng: np.random.Generator, length: int) -> np.ndarray:
    """Draw bounded asymmetric values whose reductions are usually distinct."""

    if length < 4:
        raise ValueError("reduction bags require at least four values")
    middle = rng.uniform(-0.2, 0.6, size=length - 2)
    low = rng.uniform(-1.3, -0.7, size=1)
    high = rng.uniform(0.9, 1.7, size=1)
    return np.concatenate((middle, low, high))


def records(*, bags: int, items_per_group: int, seed: int) -> Iterator[dict]:
    """Expand interleaved grouped bags into every group-operation request."""
    rng = np.random.default_rng(seed)
    length = len(GROUPS) * items_per_group
    for bag in range(bags):
        bag_values = np.concatenate([values(rng, items_per_group) + rng.uniform(-1.5, 1.5) for _ in GROUPS])
        labels = np.repeat(np.asarray(GROUPS), items_per_group)
        order = rng.permutation(length)
        items = [{"group": str(labels[index]), "value": float(bag_values[index])} for index in order]
        for group in GROUPS:
            selected = bag_values[labels == group]
            for operation in OPERATIONS:
                yield {
                    "bag": bag,
                    "selected_group": group,
                    "operation": operation,
                    "items": items,
                    "answer": reduce(selected, operation),
                }


def scores(train: list[dict], test: list[dict], predicted: np.ndarray, keys: tuple[str, ...]) -> dict:
    """Normalize each request cell against that cell's training-target mean."""
    result = {}
    cells = sorted({tuple(row[key] for key in keys) for row in test})
    for cell in cells:
        mean = float(np.mean([row["answer"] for row in train if tuple(row[key] for key in keys) == cell]))
        indices = [i for i, row in enumerate(test) if tuple(row[key] for key in keys) == cell]
        actual = np.asarray([test[i]["answer"] for i in indices])
        error = float(np.sqrt(np.mean(np.square(predicted[indices] - actual))))
        baseline = float(np.sqrt(np.mean(np.square(mean - actual))))
        result["/".join(cell)] = {"rmse": error, "baseline_rmse": baseline, "nrmse": error / baseline}
    return result


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    inputs = [{key: value for key, value in row.items() if key != "answer"} for row in rows]
    output = pl.DataFrame(model.predict(inputs)).to_dicts()
    return np.asarray([row["predictions"]["answer"]["content"] for row in output])


class Item(rf.Branch):
    value = rf.Number
    group = rf.Category(p_unavailable=0.0)


class PredictionModel(rf.Model):
    items = Item(length=12, n_layers=2, reduction=None)
    answer = rf.Number(mask=True, objective="mse")
    operation = rf.Category(p_unavailable=0.0)
    selected_group = rf.Category(p_unavailable=0.0)


@experiment("P027", seed=16)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = PredictionModel(d_model=64, n_layers=2, n_heads=4, reduction=rf.Attention(n_outputs=12, n_layers=2))
    data = rf.SyntheticDataModule(
        train=partial(records, bags=512, items_per_group=4, seed=seed + 144),
        validate=partial(records, bags=64, items_per_group=4, seed=seed + 145),
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
        max_steps=1200 if steps is None else min(steps, 1200),
        max_epochs=-1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    train = list(records(bags=512, items_per_group=4, seed=seed + 144))
    test = list(records(bags=128, items_per_group=4, seed=seed + 146))
    intact_prediction = predict(model, test)
    intact = scores(train, test, intact_prediction, ("operation", "selected_group"))
    metrics = {f"intact/{cell}/{name}": value for cell, score in intact.items() for name, value in score.items()}
    checks = {
        f"{cell} finite calibration nRMSE <= 1.60": bool(np.isfinite(score["nrmse"])) and score["nrmse"] <= 1.60
        for cell, score in intact.items()
    }
    checks.update({f"{cell} nRMSE <= 0.30": score["nrmse"] <= 0.30 for cell, score in intact.items()})
    return metrics, checks


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    import matplotlib.pyplot as plt
    import numpy as np

    from evidence.diagnostics.data import human
    from evidence.diagnostics.plots import caption

    m = bundle.record["metrics"]
    identifier = bundle.manifest["id"]
    cells = []
    for key, value in m.items():
        parts = key.split("/")
        if parts[-1] != "nrmse":
            continue
        panel, row, column = ("Intact requests", parts[1], parts[2])
        cells.append((panel, row, column, float(value), False))
    if not cells:
        raise ValueError(f"{identifier}: no authored composition cells matched the captured record")
    panels = list(dict.fromkeys((item[0] for item in cells)))
    rows = list(dict.fromkeys((item[1] for item in cells)))
    columns = list(dict.fromkeys((item[2] for item in cells)))
    fig = plt.figure(figsize=(max(7.8, len(panels) * 4.5), 4.8), layout="constrained")
    grid = fig.add_gridspec(1, len(panels) + 1, width_ratios=[*[1] * len(panels), 0.045])
    maximum = max((value for _, _, _, value, _ in cells))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#8a9691")
    for index, panel in enumerate(panels):
        ax = fig.add_subplot(grid[0, index])
        matrix = np.full((len(rows), len(columns)), np.nan)
        flags = {}
        for p, row, column, value, flag in cells:
            if p == panel:
                matrix[rows.index(row), columns.index(column)] = value
                flags[rows.index(row), columns.index(column)] = flag
        image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(maximum, 0.1), aspect="auto")
        ax.set(
            xticks=range(len(columns)),
            xticklabels=[human(c) for c in columns],
            yticks=range(len(rows)),
            yticklabels=[human(r) for r in rows],
            title=caption(panel),
        )
        ax.tick_params(axis="x", labelsize=9, rotation=15)
        for i, j in np.ndindex(matrix.shape):
            value = matrix[i, j]
            text = "unrecorded" if np.isnan(value) else f"{value:.3f}" + (" *" if flags.get((i, j)) else "")
            ax.text(
                j, i, text, ha="center", va="center", fontsize=9, color="white" if value > maximum * 0.65 else "#243b35"
            )
    fig.colorbar(image, cax=fig.add_subplot(grid[0, -1]), label="Normalized RMSE (lower is better)")
    note = "All panels share the same error scale. Every displayed cell is a measured condition; gray cells are unrecorded, rather than zero."
    return (
        fig,
        note,
        {
            "metric_cells": [
                {"panel": p, "row": r, "column": c, "normalized_rmse": value, "withheld": flag}
                for p, r, c, value, flag in cells
            ]
        },
    )


if __name__ == "__main__":
    report(run)
