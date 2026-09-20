# %% [markdown]
# ---
# title: Average the Requested Group
# categories:
# - Category-conditioned reduction
# proof-id: P026
# description: Select one interleaved group before averaging its values.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# The model receives an interleaved collection and a requested group. Its answer
# should average only members of that group. This isolates category-based
# selection before adding a choice of mathematical operation.
#
# {{< proof P026 status >}}
#
# ## Insights
#
# **The model uses group membership to select which values contribute to its
# answer.** The same bag supplies separate requests for each group, while the
# operation stays fixed at mean. Item coordinates keep group and value together;
# the visible root request can condition how the decoder uses the learned
# collection summary.
#
# Shuffling only group labels preserves all values and category counts but makes
# the retained original answers inconsistent with the new membership. Increased
# error supports use of the group/value relationship. It does not establish an
# exact filtering algorithm or behavior for absent groups and unknown categories.
# This small task also shows that pass-through reduction is not required for
# every learned group selection.
#
# ## Setup

# %%
"""Select the mean of one requested group from interleaved items.

Every bag contains two values from each of three groups. The operation stays
fixed at mean, isolating group selection. Permuting only group labels destroys
the original group/value relationship without changing either marginal.

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

PROOF_ID = "P026"
OPERATIONS = ("sum", "mean", "min", "max")
GROUPS = ("A", "B", "C")

# %% [markdown]
# ## Examples
#
# ### Select group B
#
# ```yaml
# bag: 0
# items:
#   - {group: A, value: 1.0}
#   - {group: B, value: -0.4}
#   - {group: C, value: 0.5}
#   - {group: A, value: 1.2}
#   - {group: B, value: -0.2}
#   - {group: C, value: 0.7}
# selected_group: B
# operation: mean
# answer: -0.3
# ```
#
# The mean of B's values is −0.3. `answer` is masked supervision and omitted
# from prediction requests. `bag` identifies paired requests for evaluation;
# it is not a model field.
#
# ### Select another group in the same bag
#
# ```yaml
# bag: 0
# items:
#   - {group: A, value: 1.0}
#   - {group: B, value: -0.4}
#   - {group: C, value: 0.5}
#   - {group: A, value: 1.2}
#   - {group: B, value: -0.2}
#   - {group: C, value: 0.7}
# selected_group: A
# operation: mean
# answer: 1.1
# ```
#
# Only the requested group changes. Its correct answer is now the mean of 1.0
# and 1.2. Each training bag supplies a request for every group.
#
# ### Break group membership
#
# ```yaml
# bag: 0
# items:
#   - {group: B, value: 1.0}
#   - {group: A, value: -0.4}
#   - {group: C, value: 0.5}
#   - {group: B, value: 1.2}
#   - {group: A, value: -0.2}
#   - {group: C, value: 0.7}
# selected_group: B
# operation: mean
# answer: -0.3
# ```
#
# Permuting labels retains every value and label count. This control deliberately
# keeps the first record's answer, −0.3, although B's visible mean is now 1.1.
# Increased error against the retained target exposes reliance on membership.
#
# ## Synthetic data and controls


# %%
def records(*, bags: int, items_per_group: int, seed: int) -> Iterator[dict]:
    """Generate the smallest group-filtering rung with one fixed reduction."""
    rng = np.random.default_rng(seed)
    for bag in range(bags):
        labels = np.repeat(np.asarray(GROUPS), items_per_group)
        bag_values = np.concatenate(
            [rng.uniform(-1.5, 1.5) + rng.uniform(-0.15, 0.15, size=items_per_group) for _ in GROUPS]
        )
        order = rng.permutation(len(labels))
        items = [{"group": str(labels[index]), "value": float(bag_values[index])} for index in order]
        for group in GROUPS:
            yield {
                "bag": bag,
                "selected_group": group,
                "operation": "mean",
                "items": items,
                "answer": float(bag_values[labels == group].mean()),
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


def corrupt_labels(rows: list[dict], seed: int) -> list[dict]:
    """Permute group/value pairings once per bag; retain every original target."""
    rng = np.random.default_rng(seed)
    corrupted = {}
    result = []
    for row in rows:
        bag = row["bag"]
        if bag not in corrupted:
            items = row["items"]
            labels = rng.permutation([item["group"] for item in items]).tolist()
            corrupted[bag] = [{**item, "group": label} for item, label in zip(items, labels, strict=True)]
        result.append({**row, "items": corrupted[bag]})
    return result


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-filtered-mean
# //| fig-cap: "The six-item branch and root each learn one attention summary. Visible group and operation requests condition the masked answer."
# //| fig-alt: "Request has 6 grouped Number items, visible operation and selected-group Categories, and a masked Number answer. Branch and root each learn one attention summary."
# #tree(node("request", kind: "root", width: 150pt, body: [
#   - *Reduction:* `Attention`
#   - *Learned summaries:* 1
# ], children: (
#   node("items", kind: "branch", repeated: true, width: 150pt, body: [
#     - *Capacity:* 6 items
#     - *Reduction:* `Attention`
#     - *Learned summaries:* 1
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
# This rung uses attention reduction on both branch and root. `operation` remains
# a visible schema field but always equals `mean` in this experiment.
#
# ## How it works
#
# Shared item coordinates let encoding bind each category to its value. The
# visible root selection can condition which evidence the answer decoder uses.
# Each underlying bag produces three requests, one per group. Corrupting only
# item labels preserves their counts and all values while breaking membership.
# The control therefore tests selection rather than aggregate statistics alone.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=64,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        batch_size=128,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        items=rf.Branch(
            length=6,
            n_layers=2,
            reduction=rf.Attention(n_layers=2),
            value=rf.Number,
            group=rf.Category(p_unavailable=0.0),
        ),
        answer=rf.Number(mask=True, objective="mse"),
        operation=rf.Category(p_unavailable=0.0),
        selected_group=rf.Category(p_unavailable=0.0),
    )
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, bags=512, items_per_group=2, seed=seed + 147),
        validate=partial(records, bags=64, items_per_group=2, seed=seed + 148),
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
    train = list(records(bags=512, items_per_group=2, seed=seed + 147))
    test = list(records(bags=128, items_per_group=2, seed=seed + 149))
    intact_prediction = predict(model, test)
    intact = scores(train, test, intact_prediction, ("selected_group",))
    metrics = {f"intact/{cell}/{name}": value for cell, score in intact.items() for name, value in score.items()}
    corrupted = scores(train, test, predict(model, corrupt_labels(test, seed + 150)), ("selected_group",))
    metrics.update(
        {f"corrupted/{cell}/{name}": value for cell, score in corrupted.items() for name, value in score.items()}
    )
    checks = {f"Group {cell} nRMSE <= 0.55": score["nrmse"] <= 0.55 for cell, score in intact.items()}
    checks.update(
        {
            f"Group {cell} corruption increases nRMSE by >= 0.25": corrupted[cell]["nrmse"] >= score["nrmse"] + 0.25
            for cell, score in intact.items()
        }
    )
    return metrics, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P026 evidence >}}
#
# ## Remaining work
#
# Repeat across three core seeds and ten calibration seeds. Add missing values,
# absent groups, and variable group sizes. The next rung
# [combines group selection with four operations](group-operation-composition.html).
# For an exact business calculation, compute the filtered mean in preprocessing.
#
# ## Reproduce
#
# {{< proof P026 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=16)
