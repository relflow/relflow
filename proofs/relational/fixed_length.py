# %% [markdown]
# ---
# title: Can an instruction change the calculation?
# categories:
# - Operation-conditioned reduction
# proof-id: P031
# description: Ask for a list’s minimum, maximum, average, or total by changing a visible operation field.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: Total, instruction visible
#   metric:
#   - intact/sum/nrmse
#   format: error
# - label: Total, instruction hidden
#   metric:
#   - hidden/sum/nrmse
#   format: error
# ---
#
# ## Example
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
# ```{typst}
# //| label: fig-proof-fixed-length
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with request, items, value, answer, operation. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
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
# Hide the operation while keeping the different answers. Also reorder items without changing the requested calculation.
#
# ## Result
#
# {{< proof P031 status >}}
#
# The operation matters: hiding it makes predictions collapse to the same answer. All lists have eight items, so new lengths remain untested.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P031 evidence >}}
#
# ### Run this experiment
#
# {{< proof P031 script >}}
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
# ### Complete experiment code
#

# %%
"""Learn four visibly requested reductions of the same eight-value bag.

Every bag produces sum, mean, minimum, and maximum requests. Permuting whole
items should preserve answers. Hiding the operation makes these requests
identical, exposing whether the requested operation drives the prediction.

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

PROOF_ID = "P031"
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
    output = model.predict(inputs).to_pylist()
    return np.asarray([row["predictions"]["/answer"]["content"] for row in output])


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


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=64,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        batch_size=128,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        items=rf.Branch(length=8, n_layers=2, reduction=rf.Attention(n_layers=2), value=rf.Number),
        answer=rf.Number(mask=True, objective="mse"),
        operation=rf.Category(p_unavailable=0.0),
    )
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, bags=768, length=8, seed=seed + 135),
        validate=partial(records, bags=96, length=8, seed=seed + 136),
    )
    trainer = lit.Trainer(
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
    trainer.fit(model, datamodule=data)
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


if __name__ == "__main__":
    report(PROOF_ID, run, seed=15)

# %% [markdown]
# </details>
