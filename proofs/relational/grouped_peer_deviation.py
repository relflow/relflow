# %% [markdown]
# ---
# title: Compare Each Item with Its Group
# categories:
# - Peer-relative inference
# proof-id: P032
# description: Select same-group peers and route their mean back to each item's deviation
#   target.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# Each item should receive its value minus the mean of its own group. Groups
# are interleaved, their locations vary independently, and no input supplies
# their means. The model must use membership as well as raw values.
#
# {{< proof P032 status >}}
#
# ## Insights
#
# **The model uses an item’s group to compare its value with the right peers.** Rotating labels preserves
# value and label
# marginals while retaining original targets, and accuracy deteriorates.
# Independent group translations and whole-item permutations preserve the true
# relationship and pass their respective controls.
#
# The repeated decoder can combine retained collection evidence with its own
# aligned group and value. That provides a route for membership-dependent peer
# context without a supplied mean. However, every group has exactly two members,
# where deviation is half the difference between their values. This is narrower
# than arbitrary group averaging. Approximate invariance, broader cardinalities,
# and missing membership still need separate evidence.
#
# ## Setup

# %%
"""Infer each item's value minus the mean of its own interleaved group.

No peer mean is supplied. Preserved tokens and coordinate-aligned queries must
support peer selection and local subtraction. Corrupt labels, translate each
group, and permute whole items to distinguish these behaviors.

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

PROOF_ID = "P032"
ITEMS = 6
GROUPS = ("A", "B", "C")

# %% [markdown]
# ## Examples
#
# ### Use the mean of the matching group
#
# ```yaml
# items:
#   - {group: A, value: 2.0, deviation: -1.0}
#   - {group: B, value: 1.0, deviation: 1.0}
#   - {group: C, value: -3.0, deviation: -1.0}
#   - {group: A, value: 4.0, deviation: 1.0}
#   - {group: C, value: -1.0, deviation: 1.0}
#   - {group: B, value: -1.0, deviation: -1.0}
# ```
#
# The group means are A: 3, B: 0, and C: −2. Deviations are supervision hidden
# by `mask=True` and removed at prediction.
#
# ### Translate groups independently
#
# ```yaml
# items:
#   - {group: A, value: 3.0, deviation: -1.0}
#   - {group: B, value: 0.5, deviation: 1.0}
#   - {group: C, value: -1.0, deviation: -1.0}
#   - {group: A, value: 5.0, deviation: 1.0}
#   - {group: C, value: 1.0, deviation: 1.0}
#   - {group: B, value: -1.5, deviation: -1.0}
# ```
#
# A moves by +1, B by −0.5, and C by +2. Their means move by the same amounts,
# so every original deviation target remains correct. The test applies this
# kind of independent group translation.
#
# ### Rotate group labels while retaining targets
#
# ```yaml
# items:
#   - {group: B, value: 2.0, deviation: -1.0}
#   - {group: C, value: 1.0, deviation: 1.0}
#   - {group: A, value: -3.0, deviation: -1.0}
#   - {group: C, value: 4.0, deviation: 1.0}
#   - {group: B, value: -1.0, deviation: 1.0}
#   - {group: A, value: -1.0, deviation: -1.0}
# ```
#
# This applies the test's one-position label rotation to the first record.
# The retained targets intentionally describe its original grouping. Under the
# new labels, the first item's deviation would be 1.5 rather than −1. Error
# against the old targets should rise when prediction depends on membership.
#
# ## Synthetic data and controls


# %%
def records(*, rows: int, seed: int) -> Iterator[dict]:
    """Generate randomly interleaved, exactly centered two-member groups."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        items: list[dict[str, object]] = []
        for group in GROUPS:
            mean = float(rng.uniform(-3.5, 3.5))
            deviation = float(rng.uniform(0.2, 1.2))
            for signed in (-deviation, deviation):
                item: dict[str, object] = {"group": group, "value": mean + signed, "deviation": signed}
                items.append(item)
        order = rng.permutation(ITEMS)
        yield {"items": [items[index] for index in order]}


def targets(rows: list[dict]) -> np.ndarray:
    return np.asarray([item["deviation"] for row in rows for item in row["items"]])


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    inputs = [
        {"items": [{key: value for key, value in item.items() if key != "deviation"} for item in row["items"]]}
        for row in rows
    ]
    output = model.predict(inputs).to_pylist()
    return np.asarray(
        [value["content"] for row in output for value in row["predictions"]["collection/items/deviation"]]
    )


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def corrupt_groups(rows: list[dict]) -> list[dict]:
    """Rotate only labels, preserving all values, targets, and label counts."""
    result = []
    for row in rows:
        labels = [item["group"] for item in row["items"]]
        shifted = labels[1:] + labels[:1]
        result.append({"items": [{**item, "group": label} for item, label in zip(row["items"], shifted, strict=True)]})
    return result


def implied_deviation(rows: list[dict]) -> np.ndarray:
    """Measure how strongly changed peer memberships alter the correct answer."""
    values = []
    for row in rows:
        means = {
            group: float(np.mean([item["value"] for item in row["items"] if item["group"] == group]))
            for group in GROUPS
        }
        values.extend(item["value"] - means[item["group"]] for item in row["items"])
    return np.asarray(values)


def translate(rows: list[dict], seed: int) -> list[dict]:
    """Shift visible values while retaining the original relative targets."""
    rng = np.random.default_rng(seed)
    result = []
    for row in rows:
        offsets = {group: float(rng.uniform(-2.5, 2.5)) for group in GROUPS}
        result.append({"items": [{**item, "value": item["value"] + offsets[item["group"]]} for item in row["items"]]})
    return result


def permute_items(rows: list[dict], seed: int) -> tuple[list[dict], np.ndarray]:
    """Reorder complete records and retain indices for comparing predictions."""
    rng = np.random.default_rng(seed)
    result, flattened = [], []
    for index, row in enumerate(rows):
        order = rng.permutation(ITEMS)
        result.append({"items": [row["items"][i] for i in order]})
        flattened.extend(index * ITEMS + order)
    return result, np.asarray(flattened)


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-grouped-peer-deviation
# //| fig-cap: "Both branch and root keep all tokens. The group and value stay aligned with each masked deviation."
# //| fig-alt: "Collection contains six repeated items with value and group inputs and a masked Number deviation. The root keeps all tokens. The item branch also keeps all tokens."
# #tree(node("collection", kind: "root", width: 150pt, body: [
#   - *Reduction:* Keep all tokens
# ], children: (
#   node("items", kind: "branch", repeated: true, width: 150pt, body: [
#     - *Capacity:* 6 items
#     - *Reduction:* Keep all tokens
#   ], children: (
#     node("value", type: "Number"),
#     node("group", type: "Category"),
#     node("deviation", kind: "target", type: "Number", width: 150pt, body: [
#       - *Input:* always hidden
#     ]),
#   )),
# )))
# ```
#
# The branch and root use `reduction=None`, retaining the item evidence available
# to each coordinate-conditioned decoder query.
#
# ## How it works
#
# Same-coordinate encoding binds membership to value, and the repeated decoder
# can use its own group/value context to select peers. Rotating labels preserves
# all values, group counts, and original targets while breaking membership.
# The separately recomputed oracle measures intervention strength. Independently
# shifting each group's values preserves its deviations. Permuting complete items should
# permute the outputs with them.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        name="collection",
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
            group=rf.Category(size=len(GROUPS), p_unavailable=0.0),
            deviation=rf.Number(mask=True, objective="mse", n_linear=2),
        ),
    )
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, rows=2048, seed=seed + 1),
        validate=partial(records, rows=512, seed=seed + 2),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        max_steps=900 if steps is None else min(steps, 900),
        max_epochs=-1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=data)
    train = list(records(rows=2048, seed=seed + 1))
    test = list(records(rows=768, seed=seed + 3))
    actual = targets(test)
    baseline = rmse(actual, float(targets(train).mean()))
    predicted = predict(model, test)
    intact_error = rmse(actual, predicted)
    intact_nrmse = intact_error / baseline
    metrics = {"intact_rmse": intact_error, "baseline_rmse": baseline, "intact_nrmse": intact_nrmse}
    checks = {"Baseline RMSE > 0.000001": baseline > 1e-6}
    corrupted = corrupt_groups(test)
    corrupted_error = rmse(actual, predict(model, corrupted))
    corrupted_nrmse = corrupted_error / baseline
    oracle = rmse(implied_deviation(corrupted), actual) / baseline
    metrics.update(corrupted_rmse=corrupted_error, corrupted_nrmse=corrupted_nrmse, oracle_corruption_nrmse=oracle)
    checks["Label corruption changes the oracle by > 0.75 nRMSE"] = oracle > 0.75
    translated = translate(test, seed + 4)
    translated_error = rmse(targets(translated), predict(model, translated))
    translated_nrmse = translated_error / baseline
    permuted, order = permute_items(test, seed + 5)
    drift = rmse(predict(model, permuted), predicted[order]) / baseline
    metrics.update(translated_rmse=translated_error, translated_nrmse=translated_nrmse, permutation_drift=drift)
    checks.update(
        {
            "Training deviations are exactly centered": abs(float(targets(train).mean())) < 1e-12,
            "Translation preserves target shape": targets(translated).shape == actual.shape,
            "Translation preserves all targets": rmse(targets(translated), actual) == 0.0,
            "Permutation preserves the target multiset": sorted(targets(permuted).tolist()) == sorted(actual.tolist()),
            "Normalized permutation drift < 0.20": drift < 0.20,
        }
    )
    checks.update(
        {
            "Grouped deviation nRMSE < 0.40": intact_nrmse < 0.40,
            "Group-translated nRMSE < 0.50": translated_nrmse < 0.50,
            "Label corruption increases nRMSE by >= 0.25": corrupted_nrmse >= intact_nrmse + 0.25,
        }
    )
    checks["Finite calibration and model metrics"] = bool(np.isfinite(list(metrics.values())).all())
    return metrics, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P032 evidence >}}
#
# ## Remaining work
#
# Repeat three core seeds and ten calibration seeds. Vary group cardinalities,
# test missing membership and singleton groups, and extend targets to ranks or
# leave-one-out means. Current permutation gates allow approximate behavior.
# The [one-summary variant](single-query-preserves-group-membership.html) tests
# whether a narrower collection route retains enough context.
#
# ## Reproduce
#
# {{< proof P032 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=3720)
