# %% [markdown]
# ---
# title: The Collection Overlap Boundary
# categories:
# - Collection overlap
# proof-id: P028
# description: Document the current failure to infer overlap between two natural sibling
#   collections of unseen identities.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# Two collections overlap if any identity appears on both sides. The natural
# sibling-branch model currently remains near chance on this task, even when
# every identity token is retained.
#
# {{< proof P028 status >}}
#
# ## Insights
#
# **Retaining every identity token does not make the current model reliably
# detect overlap between two collections.** The scalar target must discover a
# match anywhere across both sides, which is a different demand from a repeated
# target carrying its own visible query identity. The passing flat equality
# control shows that unseen Hash comparison itself can work.
#
# This proof deliberately passes when predictions remain near chance and barely
# respond to removing the shared member. Low drift under renaming and permutation
# is not useful invariance by itself: an uninformative predictor can also stay
# stable. The result describes the tested architecture and budget, not an
# impossibility theorem. Compute overlap directly when an application requires
# it today.
#
# ## Setup

# %%
"""Measure the unresolved overlap boundary between two sibling collections.

Positive rows contain exactly one shared identity; negative rows contain none.
Fresh names, balanced labels, varying lengths, and shuffled order remove easy
shortcuts. The recorded gates describe the current expected chance behavior,
with renaming/permutation and overlap-removal controls.

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

PROOF_ID = "P028"
MIN_LENGTH = 2
MAX_LENGTH = 5

# %% [markdown]
# ## Examples
#
# ### One identity is shared
#
# ```yaml
# left:
#   - {entity_id: fox}
#   - {entity_id: owl}
#   - {entity_id: lynx}
# right:
#   - {entity_id: yak}
#   - {entity_id: owl}
# has_overlap: true
# ```
#
# The shared identity is `owl`. `has_overlap` is supervision hidden by
# `mask=True`. The example states the true answer, not a successful prediction.
#
# ### No identity is shared
#
# ```yaml
# left:
#   - {entity_id: fox}
#   - {entity_id: owl}
#   - {entity_id: lynx}
# right:
#   - {entity_id: yak}
#   - {entity_id: eel}
# has_overlap: false
# ```
#
# The right side now shares no identity with the left, so the correct label is
# false. The unchanged side lengths cannot distinguish this case from the first.
#
# ### Remove an overlap but retain its old label
#
# ```yaml
# left:
#   - {entity_id: fox}
#   - {entity_id: owl}
#   - {entity_id: lynx}
# right:
#   - {entity_id: yak}
#   - {entity_id: broken-000000-1}
# has_overlap: true
# ```
#
# The corruption replaces the first record's shared right member with a fresh
# identity. Its retained label is intentionally true even though actual overlap
# is now false. A successful detector should lower its positive probability;
# the current limitation proof instead records almost no response.
#
# ## Synthetic data and controls


# %%
def records(*, rows: int, seed: int, namespace: str) -> Iterator[dict]:
    """Generate balanced, variable-width sibling collections with zero/one overlap."""
    if rows <= 0 or rows % 2:
        raise ValueError(f"collection rows must be a positive even integer, got {rows}")
    rng = np.random.default_rng(seed)
    labels = np.tile(np.asarray([False, True]), rows // 2)
    rng.shuffle(labels)
    for row_index, has_overlap in enumerate(labels):
        left_length = int(rng.integers(MIN_LENGTH, MAX_LENGTH + 1))
        right_length = int(rng.integers(MIN_LENGTH, MAX_LENGTH + 1))
        left_ids = [f"{namespace}-{row_index:06d}-left-{index}" for index in range(left_length)]
        right_ids = [f"{namespace}-{row_index:06d}-right-{index}" for index in range(right_length)]
        if has_overlap:
            right_ids[int(rng.integers(0, right_length))] = left_ids[int(rng.integers(0, left_length))]
        rng.shuffle(left_ids)
        rng.shuffle(right_ids)
        yield {
            "left": [{"entity_id": value} for value in left_ids],
            "right": [{"entity_id": value} for value in right_ids],
            "has_overlap": bool(has_overlap),
        }


def auc(target: np.ndarray, predicted: np.ndarray) -> float:
    """Compute binary ROC AUC from all positive/negative score pairs."""

    positive = predicted[target]
    negative = predicted[~target]
    if not len(positive) or not len(negative):
        raise ValueError("AUC requires at least one positive and one negative example")
    comparisons = positive[:, None] - negative[None, :]
    return float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())


def rename_and_permute(rows: list[dict], *, seed: int) -> Iterator[dict]:
    """Apply an equality-preserving unseen-ID bijection and reorder both sides."""
    rng = np.random.default_rng(seed)
    for row_index, row in enumerate(rows):
        left_ids = [item["entity_id"] for item in row["left"]]
        right_ids = [item["entity_id"] for item in row["right"]]
        identities = list(dict.fromkeys([*left_ids, *right_ids]))
        replacements = [f"renamed-{seed}-{row_index:06d}-{index}" for index in range(len(identities))]
        rng.shuffle(replacements)
        mapping = dict(zip(identities, replacements, strict=True))
        renamed_left = [mapping[value] for value in left_ids]
        renamed_right = [mapping[value] for value in right_ids]
        rng.shuffle(renamed_left)
        rng.shuffle(renamed_right)
        yield {
            "left": [{"entity_id": value} for value in renamed_left],
            "right": [{"entity_id": value} for value in renamed_right],
            "has_overlap": row["has_overlap"],
        }


def break_overlaps(rows: list[dict]) -> Iterator[dict]:
    """Remove the shared right member while retaining labels and all controls."""
    for row_index, row in enumerate(rows):
        left_ids = [item["entity_id"] for item in row["left"]]
        left_set = set(left_ids)
        replacements = {
            value: f"broken-{row_index:06d}-{index}"
            for index, value in enumerate((item["entity_id"] for item in row["right"]))
            if value in left_set
        }
        right_ids = [replacements.get(item["entity_id"], item["entity_id"]) for item in row["right"]]
        yield {
            "left": row["left"],
            "right": [{"entity_id": value} for value in right_ids],
            "has_overlap": row["has_overlap"],
        }


def overlap_truth(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [
            bool({item["entity_id"] for item in row["left"]} & {item["entity_id"] for item in row["right"]})
            for row in rows
        ]
    )


def probabilities(model: rf.Model, rows: list[dict]) -> np.ndarray:
    output = model.predict(rows).to_pylist()
    return np.asarray([row["predictions"]["overlap/has_overlap"]["content"]["probability"] for row in output])


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-direct-sibling-hash-overlap
# //| fig-cap: "Both sibling collections and the root keep all tokens, but the masked scalar target still does not reliably learn collection overlap in this proof."
# //| fig-alt: "Overlap has left and right Hash-identity branches, each with capacity five, and a masked Boolean target. Both branches and root keep all tokens; this remains the recorded unsupported collection-comparison route."
# #tree(node("overlap", kind: "root", width: 150pt, body: [
#   - *Reduction:* Keep all tokens
# ], children: (
#   node("left", kind: "branch", repeated: true, width: 150pt, body: [
#     - *Capacity:* 5 items
#     - *Reduction:* Keep all tokens
#   ], children: (
#     node("entity_id", type: "Hash"),
#   )),
#   node("right", kind: "branch", repeated: true, width: 150pt, body: [
#     - *Capacity:* 5 items
#     - *Reduction:* Keep all tokens
#   ], children: (
#     node("entity_id", type: "Hash"),
#   )),
#   node("has_overlap", kind: "target", type: "Boolean", width: 150pt, body: [
#     - *Input:* always hidden
#   ]),
# )))
# ```
#
# The root and both branches use `reduction=None`. Retaining tokens by itself
# does not guarantee that a scalar decoder learns every cross-collection match.
#
# ## How it works
#
# Positive rows share exactly one identity; negative rows share none. Labels are
# balanced, identities are fresh per row and split, and independent side lengths
# range from two to five. Order, length, and vocabulary frequency cannot identify
# the label. Renaming identities consistently and permuting each side preserves
# the relation; removing the shared right member destroys it.
#
# The [flat equality control](flat-unseen-hash-equality-control.html) succeeds,
# localizing the observed gap beyond the primitive unseen-identity comparison.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        name="overlap",
        d_model=48,
        n_layers=3,
        n_heads=4,
        reduction=None,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.003),
        left=rf.Branch(length=MAX_LENGTH, n_layers=1, reduction=None, entity_id=rf.Hash(n_hashes=4, n_bands=8)),
        right=rf.Branch(length=MAX_LENGTH, n_layers=1, reduction=None, entity_id=rf.Hash(n_hashes=4, n_bands=8)),
        has_overlap=rf.Boolean(mask=True),
    )
    # Restart the training RNG independently of parameter initialization.
    lit.seed_everything(seed, workers=True)
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, rows=3072, seed=seed + 1, namespace="train"),
        validate=partial(records, rows=768, seed=seed + 2, namespace="validate"),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        max_steps=700 if steps is None else min(steps, 700),
        max_epochs=-1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        check_val_every_n_epoch=5,
    )
    trainer.fit(model, datamodule=data)
    test = list(records(rows=1024, seed=seed + 3, namespace="test"))
    target = np.asarray([row["has_overlap"] for row in test], dtype=np.bool_)
    intact = probabilities(model, test)
    invariant = list(rename_and_permute(test, seed=seed + 4))
    broken = list(break_overlaps(test))
    invariant_probability = probabilities(model, invariant)
    broken_probability = probabilities(model, broken)
    metrics = {
        "intact_auc": auc(target, intact),
        "invariant_auc": auc(target, invariant_probability),
        "broken_auc": auc(target, broken_probability),
        "invariant_drift": float(np.mean(np.abs(intact - invariant_probability))),
        "positive_break_drop": float(np.mean(intact[target] - broken_probability[target])),
    }
    checks = {
        "Labels are exactly balanced": float(target.mean()) == 0.5,
        "Labels match actual overlap": bool(np.array_equal(target, overlap_truth(test))),
        "Renaming and permutation preserve overlap": bool(np.array_equal(target, overlap_truth(invariant))),
        "Intervention removes all overlaps": not bool(overlap_truth(broken).any()),
        "Finite metrics": bool(np.isfinite(list(metrics.values())).all()),
        "Intact AUC remains in the calibrated chance band": 0.35 <= metrics["intact_auc"] <= 0.65,
        "Renamed and permuted AUC remains in the chance band": 0.35 <= metrics["invariant_auc"] <= 0.65,
        "Broken AUC remains in the chance band": 0.35 <= metrics["broken_auc"] <= 0.65,
        "Invariant probability drift <= 0.15": metrics["invariant_drift"] <= 0.15,
        "Absolute positive break response <= 0.10": abs(metrics["positive_break_drop"]) <= 0.10,
    }
    # These are generator invariants, separate from the model's expected limitation.
    for split, rows in (
        ("train", list(records(rows=3072, seed=seed + 1, namespace="train"))),
        ("validate", list(records(rows=768, seed=seed + 2, namespace="validate"))),
        ("test", test),
        ("invariant", invariant),
    ):
        labels = np.asarray([row["has_overlap"] for row in rows])
        checks[f"{split} generator has balanced truthful labels"] = float(labels.mean()) == 0.5 and bool(
            np.array_equal(labels, overlap_truth(rows))
        )
        checks[f"{split} sides have two to five unique members"] = all(
            MIN_LENGTH <= len(row[side]) <= MAX_LENGTH
            and len(row[side]) == len({item["entity_id"] for item in row[side]})
            for row in rows
            for side in ("left", "right")
        )
        checks[f"{split} positive rows have exactly one overlap"] = all(
            len({item["entity_id"] for item in row["left"]} & {item["entity_id"] for item in row["right"]})
            == int(row["has_overlap"])
            for row in rows
        )
    return metrics, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P028 evidence >}}
#
# ## Remaining work
#
# Repeat across more seeds and larger training budgets, then evaluate bounded
# set-comparison designs with accuracy, intervention, and invariance gates.
# This result characterizes the tested schema and budget, not an impossibility
# result for transformers. Compute deterministic overlap in a preprocessor or
# the application when it is needed today.
#
# ## Reproduce
#
# {{< proof P028 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=3711)
