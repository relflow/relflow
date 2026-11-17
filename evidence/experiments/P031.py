# %% [markdown]
# ---
# title: Can an instruction change the calculation?
# categories:
# - Operation-conditioned reduction
# description: Ask for a list’s minimum, maximum, average, or total by changing a visible operation field.
# evidence-readout:
# - label: Total, instruction visible
#   metric:
#   - intact/sum/nrmse
#   format: error
# - label: Total, instruction hidden
#   metric:
#   - hidden/sum/nrmse
#   format: error
# evidence-group: Relational
# evidence-diagram:
#   caption: Amber cards are hidden prediction targets. Repeated collections keep their fields together.
#   alt: Model tree with request, items, value, answer, operation. Amber cards are hidden prediction targets. Repeated
#     collections keep their fields together.
# evidence-review:
#   question: Can an instruction change the calculation?
#   modeling: Give the same list a visible request for minimum, maximum, average, or total. Hide the requested result.
#     Compare total-prediction error with the request visible and hidden; without the request, different questions
#     about one list become indistinguishable.
#   comparison:
#     kind: Controlled comparison
#     metric: RMSE
#     conditions:
#     - label: Total, instruction hidden
#       path:
#       - hidden/sum/rmse
#       format: number
#     - label: Total, instruction visible
#       path:
#       - intact/sum/rmse
#       format: number
#     definition: Root mean squared prediction error, in the target’s original units. Zero is an exact prediction.
#     direction: Lower error is better
#   note: The chart shows the total task. Minimum, maximum, average, and item-order stability each have separate checks.
#   criteria:
#     operations: request
# evidence-plot:
#   target: answer
#   focus: Does a visible operation request determine the answer?
#   comparison: Compare requested operations, hidden requests, and complete-item permutations.
#   limit: With the request hidden, one fixed collection can correspond to several correct operation-specific answers.
#   renderer: plot
# evidence-cost:
#   baseline: 83
#   candidate: 82
# ---
#
# ## Example
#
# The same eight visible numbers can support four different questions. In the illustrative bag their total
# is `1.3`, their average is `0.1625`, their minimum is `-1.0`, and their maximum is `1.2`. The visible
# `operation: max` therefore selects the hidden answer `1.2`. Each generated bag produces all four requests,
# so its values alone cannot determine which answer is wanted; the instruction must also be used.
#
# ```yaml
# bag: 0
# items:
#   - {value: 0.1}
#   - {value: -0.1}
#   - {value: 0.4}
#   - {value: 0.2}
#   - {value: 0.5}
#   - {value: 0.0}
#   - {value: -1.0}
#   - {value: 1.2}
# operation: max
# answer: 1.2
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```typst
# #tree(node("request", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("value", type: "Number"),
#   )),
#   node("answer", kind: "target", type: "Number", width: 150pt,),
#   node("operation", type: "Category"),
# )))
# ```
#
# ## Comparison
#
# Reordering whole items leaves all four mathematical answers unchanged. Hiding `operation`, however, makes
# the four requests for a bag indistinguishable to the model while their target answers remain different.
# The control checks both that predictions collapse within each hidden-instruction bag and that this loss of
# information hurts several operations. Evaluation uses fresh bags, with an operation-specific constant
# baseline rather than one average across incompatible tasks.
#
# Hide the operation while keeping the different answers. Also reorder items without changing the requested calculation.
#
# ## Result
#
# The compact readout highlights the total, while the detailed metrics retain separate errors for sum, mean,
# minimum, and maximum. An error ratio below one improves on that operation's training-mean guess;
# permutation drift measures sensitivity to item order. Success requires both useful intact predictions and
# the expected information loss when the instruction disappears. Every training and evaluation bag has eight
# values, so this does not test extrapolation to longer lists.
#
# The operation matters: hiding it makes predictions collapse to the same answer. All lists have eight items, so new lengths remain untested.
#
# ## Method
#
# ### How it works
#
# The decoder must learn how the request changes the use of collection context.
# Permuting complete items should preserve a statistic; hiding the operation
# makes the four requests for a bag identical. Their predictions should then
# collapse to one value and lose accuracy on the contradictory targets.
#
# ### Remaining work
#
# Evaluate every operation at lengths 4–16, then characterize degradation through
# 32. If sum fails alone, a visible-count control can isolate lost cardinality.
# Repeat three core seeds and ten calibration seeds before fixing the supported
# range. Exact numerical reductions belong in preprocessing when exactness is
# the application contract.
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
# conditions = ["missing-inputs", "transformations"]
# ///

"""Learn four visibly requested reductions of the same eight-value bag.

Every bag produces sum, mean, minimum, and maximum requests. Permuting whole
items should preserve answers. Hiding the operation makes these requests
identical, exposing whether the requested operation drives the prediction.

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


def records(*, bags: int, length: int, seed: int) -> Iterator[dict]:
    """Expand each independently drawn bag into its four reduction requests."""
    rng = np.random.default_rng(seed)
    for bag in range(bags):
        bag_values = values(rng, length)
        order = rng.permutation(length)
        items = [{"value": float(bag_values[index])} for index in order]
        for operation in OPERATIONS:
            yield {"bag": bag, "operation": operation, "items": items, "answer": reduce(bag_values, operation)}


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


def permute_items(rows: list[dict], seed: int) -> list[dict]:
    """Reorder whole item records once per bag, preserving all four requests."""
    rng = np.random.default_rng(seed)
    reordered = {}
    result = []
    for row in rows:
        bag = row["bag"]
        if bag not in reordered:
            items = row["items"]
            reordered[bag] = [items[i] for i in rng.permutation(len(items))]
        result.append({**row, "items": reordered[bag]})
    return result


class Item(rf.Branch):
    value = rf.Number


class PredictionModel(rf.Model):
    items = Item(length=8, n_layers=2, reduction=rf.Attention(n_layers=2))
    answer = rf.Number(mask=True, objective="mse")
    operation = rf.Category(p_unavailable=0.0)


@experiment("P031", seed=15)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = PredictionModel(d_model=64, n_layers=2, n_heads=4, reduction=rf.Attention(n_layers=2))
    data = rf.SyntheticDataModule(
        train=partial(records, bags=768, length=8, seed=seed + 135),
        validate=partial(records, bags=96, length=8, seed=seed + 136),
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
        max_steps=800 if steps is None else min(steps, 800),
        max_epochs=-1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    train = list(records(bags=768, length=8, seed=seed + 135))
    test = list(records(bags=192, length=8, seed=seed + 137))
    intact_prediction = predict(model, test)
    intact = scores(train, test, intact_prediction, ("operation",))
    metrics = {f"intact/{cell}/{name}": value for cell, score in intact.items() for name, value in score.items()}
    permuted_prediction = predict(model, permute_items(test, seed + 138))
    permuted = scores(train, test, permuted_prediction, ("operation",))
    hidden_prediction = predict(model, [{**row, "operation": None} for row in test])
    hidden = scores(train, test, hidden_prediction, ("operation",))
    for label, result in (("permuted", permuted), ("hidden", hidden)):
        metrics.update(
            {f"{label}/{cell}/{name}": value for cell, score in result.items() for name, value in score.items()}
        )
    scale = float(np.std([row["answer"] for row in test]))
    delta = float(np.sqrt(np.mean(np.square(intact_prediction - permuted_prediction)))) / scale
    hidden_spread = float(np.max(np.ptp(hidden_prediction.reshape(-1, len(OPERATIONS)), axis=1)))
    metrics.update(permutation_delta=delta, hidden_request_spread=hidden_spread)
    checks = {f"{cell} intact nRMSE <= 0.20": score["nrmse"] <= 0.20 for cell, score in intact.items()}
    checks.update({f"{cell} permuted nRMSE <= 0.25": score["nrmse"] <= 0.25 for cell, score in permuted.items()})
    checks.update(
        {
            "Permutation changes predictions by <= 0.05 target SD": delta <= 0.05,
            "Hidden operations yield identical predictions within each bag": hidden_spread <= 1e-6,
            "At least three hidden operations have nRMSE >= 0.75": sum(
                score["nrmse"] >= 0.75 for score in hidden.values()
            )
            >= 3,
        }
    )
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
        panel, row, column = ("Request conditions", parts[1], parts[0])
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
