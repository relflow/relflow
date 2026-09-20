# %% [markdown]
# ---
# title: Choose a Group and an Operation
# categories:
# - Category-conditioned reduction
# proof-id: P027
# description: Compose category filtering with a requested sum, mean, minimum, or maximum.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# One interleaved bag supports twelve requests: three groups combined with four
# operations. The model must use both root request fields to select the relevant
# items and infer the requested statistic.
#
# {{< proof P027 status >}}
#
# ## Insights
#
# **The model can answer requests for different groups and operations, but this combined task still needs
# checks that it uses both requests.**
# The same bag supplies different answers according to visible group and
# operation requests. Measuring every cell prevents an easy subset from hiding
# a failed operation or group.
#
# Item coordinates bind categories to values, retained item slots preserve their
# evidence, and request fields can condition the scalar decoder. The twelve
# root outputs are learned summaries, not twelve declared statistics. Label and
# request corruptions have not been applied to this exact route; neighboring
# proofs do not substitute for them. The evidence supports bounded conditional
# prediction, with broader relational composition still requiring validation.
#
# ## Setup

# %%
"""Compose a requested group filter with sum, mean, minimum, or maximum.

Each bag contains four items from each of three groups and produces all twelve
requests. Score every group/operation cell against its own training-mean
baseline so one easy operation cannot hide failure on another.

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

PROOF_ID = "P027"
OPERATIONS = ("sum", "mean", "min", "max")
GROUPS = ("A", "B", "C")

# %% [markdown]
# ## Examples
#
# ### Request group B's maximum
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
# The answer is masked training supervision and omitted at prediction. `bag` is
# evaluation metadata, excluded from the model. The same bag also supplies B's
# sum (0.6), mean (0.15), and minimum (−0.9) as separate requests.
#
# ### Request the mean instead
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
# operation: mean
# answer: 0.15
# ```
#
# The items and selected group stay fixed. Changing the operation changes the
# correct target from 1.3 to 0.15: B's four values sum to 0.6.
#
# ### Change both parts of the request
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
# selected_group: C
# operation: min
# answer: -1.2
# ```
#
# This is another correctly labeled request for the same bag. The test evaluates
# each group-operation combination; it does not yet apply a corruption control
# to this composed task.
#
# ## Synthetic data and controls


# %%
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
    output = model.predict(inputs).to_pylist()
    return np.asarray([row["predictions"]["/answer"]["content"] for row in output])


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-group-operation-composition
# //| fig-cap: "All twelve items remain available to a root with twelve learned summaries. Summary slots do not declare particular group-operation results."
# //| fig-alt: "Request has 12 grouped Number items, visible operation and selected-group Categories, and a masked Number answer. The branch keeps all tokens and the root learns twelve attention summaries."
# #tree(node("request", kind: "root", width: 150pt, body: [
#   - *Reduction:* `Attention`
#   - *Learned summaries:* 12
# ], children: (
#   node("items", kind: "branch", repeated: true, width: 150pt, body: [
#     - *Capacity:* 12 items
#     - *Reduction:* Keep all tokens
#   ], children: (
#     node("value", type: "Number"),
#     node("group", type: "Category"),
#   )),
#   node("answer", kind: "target", type: "Number", width: 150pt, body: [
#     - *Input:* always hidden
#   ]),
#   node("operation", type: "Category"),
#   node("selected_group", type: "Category"),
# )))
# ```
#
# The item branch uses `reduction=None`; the root uses
# `rf.Attention(n_outputs=12, n_layers=2)`. The twelve learned outputs do not
# declare twelve semantic group-operation slots.
#
# ## How it works
#
# Coordinate-local mixing binds groups to values, retained item slots preserve
# their associations, and visible request fields condition the decoder query.
# Independently generated bags are assigned to each split before expansion into
# requests, so related answers stay together. Measuring all twelve cells prevents
# a favorable average from hiding one failed group or operation.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=64,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_outputs=12, n_layers=2),
        batch_size=128,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        items=rf.Branch(
            length=12,
            n_layers=2,
            reduction=None,
            value=rf.Number,
            group=rf.Category(p_unavailable=0.0),
        ),
        answer=rf.Number(mask=True, objective="mse"),
        operation=rf.Category(p_unavailable=0.0),
        selected_group=rf.Category(p_unavailable=0.0),
    )
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, bags=512, items_per_group=4, seed=seed + 144),
        validate=partial(records, bags=64, items_per_group=4, seed=seed + 145),
    )
    trainer = lit.Trainer(
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
    trainer.fit(model, datamodule=data)
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


# %% [markdown]
# ## Evidence
#
# {{< proof P027 evidence >}}
#
# ## Remaining work
#
# Add label, selected-group, and operation corruptions to this composed case.
# Repeat three core seeds and ten calibration seeds, vary collection length, and
# test absent groups and missing values. Sweep reduction width independently of
# the number of requested cells. See the [filtered-mean control](filtered-mean.html).
#
# ## Reproduce
#
# {{< proof P027 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=16)
