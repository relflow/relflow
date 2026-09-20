# %% [markdown]
# ---
# title: Share One Summary Across Grouped Targets
# categories:
# - Peer-relative inference
# proof-id: P033
# description: Recover grouped deviations from one learned collection summary plus aligned
#   item context.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# This repeats grouped peer deviation with one learned branch output. The shared
# summary carries collection context while each target still receives its own
# visible group and value as query context.
#
# {{< proof P033 status >}}
#
# ## Insights
#
# **A single collection summary can support per-item predictions when the decoder also sees each item’s
# group and value.** The result is not
# evidence that one vector losslessly stores an arbitrary collection or that
# all useful information passes through that vector alone.
#
# Corrupting group labels while retaining original targets removes accuracy,
# supporting use of the membership relationship in this compressed route.
# Translation and permutation controls from the retained-token case have not
# been repeated here. This model also trains for more steps than that case, so
# the recorded accuracy comparison is not an efficiency comparison. Validate
# summary capacity for the actual target and collection sizes before generalizing
# from these three two-member groups.
#
# ## Setup

# %%
"""Infer grouped deviations using one learned collection summary.

Each repeated target still receives its aligned visible group and value as
query context. The shared summary supplies peer context; this is not a claim
that one vector losslessly preserves every collection. Rotated labels test
whether the model uses the original peer memberships.

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

PROOF_ID = "P033"
ITEMS = 6
GROUPS = ("A", "B", "C")

# %% [markdown]
# ## Examples
#
# ### Decode each group's original deviations
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
# Each deviation is its value minus the mean of values sharing its group. The
# target is masked by `mask=True` during training and omitted at prediction.
#
# ### Different values require different deviations
#
# ```yaml
# items:
#   - {group: A, value: -0.5, deviation: -0.5}
#   - {group: B, value: 2.5, deviation: 0.5}
#   - {group: C, value: -2.0, deviation: -0.5}
#   - {group: A, value: 0.5, deviation: 0.5}
#   - {group: C, value: -1.0, deviation: 0.5}
#   - {group: B, value: 1.5, deviation: -0.5}
# ```
#
# The means are now A: 0, B: 2, and C: −1.5. These are correctly labeled
# examples of the same task with a smaller within-group spread. The single
# summary must support these changed targets alongside each item's local context.
#
# ### Remove the original membership relationship
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
# The first example's labels rotate by one position, but values and deviation
# targets stay fixed. The targets therefore no longer describe the visible
# grouping. The retained corruption control checks that this change removes
# accuracy; it does not require reproducing those now-inconsistent targets.
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
    return np.asarray([value["content"] for row in output for value in row["predictions"]["/items/deviation"]])


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


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-single-query-preserves-group-membership
# //| fig-cap: "One learned item summary supplies shared context, while each target retains its aligned visible group and value. The root keeps its routed tokens."
# //| fig-alt: "Collection contains six repeated items with value and group inputs and a masked Number deviation. The root keeps all tokens. The item branch learns one attention summary; aligned visible siblings also condition each target."
# #tree(node("collection", kind: "root", width: 150pt, body: [
#   - *Reduction:* Keep all tokens
# ], children: (
#   node("items", kind: "branch", repeated: true, width: 150pt, body: [
#     - *Capacity:* 6 items
#     - *Reduction:* `Attention`
#     - *Learned summaries:* 1
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
# The branch uses `rf.Attention(n_outputs=1, n_layers=2)` and the root uses
# `reduction=None`. The tree matches the [retained-token case](grouped-peer-deviation.html);
# the reduction route is the difference.
#
# ## How it works
#
# Coordinate-local encoding first mixes each item's group and value. A learned
# summary can then retain peer context used by separate, coordinate-conditioned
# target queries. Label corruption changes peer membership while retaining the
# original targets and the value and label marginals. The resulting loss of
# accuracy tests whether that membership survived in the information available
# to the decoder.
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
            reduction=rf.Attention(n_outputs=1, n_layers=2),
            value=rf.Number,
            group=rf.Category(p_unavailable=0.0),
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
    checks.update(
        {
            "Single-summary grouped nRMSE <= 0.20": intact_nrmse <= 0.20,
            "Corrupted labels nRMSE >= 0.90": corrupted_nrmse >= 0.90,
            "Label corruption increases nRMSE by >= 0.75": corrupted_nrmse >= intact_nrmse + 0.75,
        }
    )
    checks["Finite calibration and model metrics"] = bool(np.isfinite(list(metrics.values())).all())
    return metrics, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P033 evidence >}}
#
# ## Remaining work
#
# Repeat seed calibration and add translation and complete-item permutation
# controls to this compressed case. Sweep group sizes, simultaneous targets,
# and summary width before generalizing its capacity. A smaller representation
# alone does not establish a speed or memory improvement.
#
# ## Reproduce
#
# {{< proof P033 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=3720)
