# %% [markdown]
# ---
# title: Can it total weighted values for one group?
# categories:
# - Grouped weighted aggregation
# proof-id: P010
# description: Ask for a group’s total after multiplying each of its values by its own weight.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: Correct data
#   metric:
#   - intact
#   - nrmse
#   format: error
# - label: Swapped weights
#   metric:
#   - pairing
#   - nrmse
#   format: error
# - label: Changed groups
#   metric:
#   - labels
#   - nrmse
#   format: error
# ---
#
# ## Example
#
# ```yaml
# selected_group: A
# items:
#   - {group: B, value: 0.3, weight: 0.4}
#   - {group: A, value: 0.8, weight: 0.5}
#   - {group: C, value: -0.5, weight: 1.4}
#   - {group: A, value: -0.4, weight: 1.25}
#   - {group: B, value: -0.2, weight: 0.9}
#   - {group: C, value: 0.7, weight: 0.3}
# answer: -0.1
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-proof-selected-group-raw-weighted-sum
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with request, selected_group, items, group, value, weight, answer. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("request", kind: "root", width: 120pt, children: (
#   node("selected_group", type: "Category"),
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("group", type: "Category"),
#     node("value", type: "Number"),
#     node("weight", type: "Number"),
#   )),
#   node("answer", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Change the requested group. Then separately shuffle group labels and swap weights, retaining the original answers.
#
# ## Result
#
# {{< proof P010 status >}}
#
# The model uses both membership and value–weight pairing. The groups are familiar and equally sized; unfamiliar or missing groups are not tested.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P010 evidence >}}
#
# ### Run this experiment
#
# {{< proof P010 script >}}
#
# ### How it works
#
# Each bag contains two items from each of A, B, and C. The generator emits
# the same bag once per requested group, so the bag alone cannot determine the
# answer. `reduction=None` preserves the item branch’s encoded coordinates;
# root `rf.Attention` combines them with the visible `selected_group` request.
#
# Rotating item group labels tests group selection. Swapping weights within each
# group independently tests value–weight binding while preserving per-group
# marginals. Both corruptions keep original labels and must worsen error.
# Reordering complete items should preserve predictions. The
# [supplied-product control](selected-group-supplied-contribution-sum.html)
# separates multiplication from selection and reduction.
#
# ### Remaining work
#
# Repeat all gates across seeds, then vary group sizes and test an absent
# requested group. Unseen labels, simultaneous requests, weight scaling, and
# selected-group weighted means remain open. These scores are not a
# capacity-matched comparison with the supplied-product control.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""The natural schema should learn a selected group's weighted sum.

Learn filtering, coordinate binding, multiplication, and summation.

Run: uv run python proofs/run.py P010"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P010"
GROUPS = ("A", "B", "C")
ITEMS_PER_GROUP = 2
ITEMS = len(GROUPS) * ITEMS_PER_GROUP


def records(*, bags: int, seed: int) -> Iterator[dict]:
    """Draw interleaved groups and emit one request for every selected group."""
    rng = np.random.default_rng(seed)
    for bag in range(bags):
        items: list[dict[str, object]] = []
        answers: dict[str, float] = {}
        for group in GROUPS:
            values = rng.uniform(-1.0, 1.0, size=ITEMS_PER_GROUP)
            weights = rng.uniform(0.2, 1.8, size=ITEMS_PER_GROUP)
            contributions = values * weights
            answers[group] = float(contributions.sum())
            for value, weight, contribution in zip(values, weights, contributions, strict=True):
                item: dict[str, object] = {"group": group}
                item.update(value=float(value), weight=float(weight))
                items.append(item)
        interleaved = [items[index] for index in rng.permutation(len(items))]
        for selected_group in GROUPS:
            yield {
                "bag": bag,
                "selected_group": selected_group,
                "items": interleaved,
                "answer": answers[selected_group],
            }


def rotate_group_labels(observations: list[dict]) -> list[dict]:
    """Break group/item association while preserving all group marginals."""
    successor = {group: GROUPS[(index + 1) % len(GROUPS)] for index, group in enumerate(GROUPS)}
    rows = deepcopy(observations)
    for row in rows:
        row["items"] = [{**item, "group": successor[item["group"]]} for item in row["items"]]
    return rows


def swap_weights_within_groups(observations: list[dict]) -> list[dict]:
    """Break value/weight pairing without changing any per-group marginal."""
    rows = deepcopy(observations)
    cached: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        bag = int(row["bag"])
        if bag not in cached:
            items = [dict(item) for item in row["items"]]
            for group in GROUPS:
                indices = [index for index, item in enumerate(items) if item["group"] == group]
                if len(indices) != ITEMS_PER_GROUP:
                    raise ValueError(f"bag {bag} has {len(indices)} items in group {group!r}")
                weights = [items[index]["weight"] for index in indices]
                for index, weight in zip(indices, np.roll(weights, 1), strict=True):
                    items[index]["weight"] = float(weight)
            cached[bag] = items
        row["items"] = cached[bag]
    return rows


def permute_items(observations: list[dict], *, seed: int) -> list[dict]:
    """Jointly permute complete items while retaining targets."""
    rng = np.random.default_rng(seed)
    rows = deepcopy(observations)
    cached: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        bag = int(row["bag"])
        if bag not in cached:
            items = row["items"]
            cached[bag] = [items[index] for index in rng.permutation(len(items))]
        row["items"] = cached[bag]
    return rows


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"selected_group": row["selected_group"], "items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["/answer"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["answer"] for row in test], dtype=np.float64)
    baseline_rmse = rmse(actual, float(np.asarray([row["answer"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline_rmse, "nrmse": measured / baseline_rmse}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(records(bags=512, seed=seed + 1))
    test = list(records(bags=256, seed=seed + 3))
    model = rf.Model(
        d_model=48,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_outputs=ITEMS, n_layers=2),
        batch_size=96,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        items=rf.Branch(
            length=ITEMS,
            n_layers=2,
            reduction=None,
            group=rf.Category(p_unavailable=0.0),
            value=rf.Number,
            weight=rf.Number,
        ),
        selected_group=rf.Category(p_unavailable=0.0),
        answer=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        model=model,
        train=lambda: records(bags=512, seed=seed + 1),
        validate=lambda: records(bags=128, seed=seed + 2),
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
    intact = score(train=train, test=test, predicted=intact_prediction)
    labels = score(train=train, test=test, predicted=prediction(model, rotate_group_labels(test)))
    pairing = score(train=train, test=test, predicted=prediction(model, swap_weights_within_groups(test)))
    permuted_prediction = prediction(model, permute_items(test, seed=seed + 4))
    target_scale = float(np.std(np.asarray([row["answer"] for row in test], dtype=np.float64)))
    permutation_drift = rmse(intact_prediction, permuted_prediction) / target_scale
    metrics = {
        "intact": intact,
        "labels": labels,
        "pairing": pairing,
        "permutation_drift": permutation_drift,
        "steps": trainer.global_step,
    }
    checks = {
        "All measurements finite": bool(
            all(
                (
                    np.isfinite(value)
                    for value in (intact["nrmse"], labels["nrmse"], pairing["nrmse"], permutation_drift)
                )
            )
        ),
        "Raw grouped sum nRMSE below 0.50": bool(intact["nrmse"] < 0.5),
        "Rotated labels raise nRMSE by at least 0.20": bool(labels["nrmse"] >= intact["nrmse"] + 0.2),
        "Swapped weights raise nRMSE by at least 0.20": bool(pairing["nrmse"] >= intact["nrmse"] + 0.2),
        "Permutation drift below 0.10 target SD": bool(permutation_drift < 0.1),
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3605)

# %% [markdown]
# </details>
