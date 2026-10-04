# %% [markdown]
# ---
# title: Can it compare two unfamiliar IDs directly?
# categories:
# - Collection overlap
# proof-id: P029
# description: Ask whether two individual IDs match, as a simpler comparison beside the two-list overlap
#   test.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: Original pairs
#   metric:
#   - intact_auc
#   format: auc
# - label: Matches removed
#   metric:
#   - broken_auc
#   format: auc
# ---
#
# ## Example
#
# ```yaml
# left_id: owl
# right_id: owl
# equal: true
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-proof-flat-unseen-hash-equality-control
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with pair, left_id, right_id, equal. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("pair", kind: "root", width: 120pt, children: (
#   node("left_id", type: "Hash"),
#   node("right_id", type: "Hash"),
#   node("equal", kind: "target", type: "Boolean", width: 150pt,),
# )))
# ```
#
# ## Comparison
#
# Turn matching pairs into nonmatching pairs while retaining the original answers.
#
# ## Result
#
# {{< proof P029 status >}}
#
# Direct ID matching works, so the harder overlap failure cannot be blamed solely on the Hash representation. This does not establish comparisons between all pairs in two lists.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P029 evidence >}}
#
# ### Run this experiment
#
# {{< proof P029 script >}}
#
# ### How it works
#
# Hash preserves equality within an encoded batch, allowing attention and the
# Boolean decoder to learn the relationship between two visible identities.
# The labels are balanced and each row uses fresh keys. The intervention makes
# all formerly equal pairs unequal while keeping their original labels, so
# positive probabilities should drop and ranking against those labels should
# return to chance.
#
# ### Remaining work
#
# Repeat the primitive control alongside the collection proof across the
# promotion seed matrix. Retain the broken-equality intervention when changing
# Hash configuration or testing more difficult schemas, so a collection failure
# can be distinguished from a failure of scalar identity comparison.
#
# ### Complete experiment code
#

# %%
"""Learn equality between two scalar Hash fields with unseen identities.

This control removes collection routing from the overlap problem. Break every
positive equality while retaining its label to verify that strong accuracy
comes from equality rather than memorizing names or label proportions.

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

PROOF_ID = "P029"
MIN_LENGTH = 2
MAX_LENGTH = 5


def records(*, rows: int, seed: int, namespace: str, break_equal: bool = False) -> Iterator[dict]:
    """Generate the unseen-Hash equality control used to calibrate the primitive."""
    if rows <= 0 or rows % 2:
        raise ValueError(f"pair rows must be a positive even integer, got {rows}")
    rng = np.random.default_rng(seed)
    labels = np.tile(np.asarray([False, True]), rows // 2)
    rng.shuffle(labels)
    for row_index, equal in enumerate(labels):
        left = f"{namespace}-{row_index:06d}-left"
        right = left if equal and (not break_equal) else f"{namespace}-{row_index:06d}-right"
        yield {"left_id": left, "right_id": right, "equal": bool(equal)}


def auc(target: np.ndarray, predicted: np.ndarray) -> float:
    """Compute binary ROC AUC from all positive/negative score pairs."""

    positive = predicted[target]
    negative = predicted[~target]
    if not len(positive) or not len(negative):
        raise ValueError("AUC requires at least one positive and one negative example")
    comparisons = positive[:, None] - negative[None, :]
    return float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())


def probabilities(model: rf.Model, rows: list[dict]) -> np.ndarray:
    output = model.predict(rows).to_pylist()
    return np.asarray([row["predictions"]["/equal"]["content"]["probability"] for row in output])


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=48,
        n_layers=2,
        n_heads=4,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.003),
        left_id=rf.Hash(n_hashes=4, n_bands=8),
        right_id=rf.Hash(n_hashes=4, n_bands=8),
        equal=rf.Boolean(mask=True),
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
        max_steps=600 if steps is None else min(steps, 600),
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
    target = np.asarray([row["equal"] for row in test], dtype=np.bool_)
    intact = probabilities(model, test)
    broken = list(records(rows=1024, seed=seed + 3, namespace="test", break_equal=True))
    broken_probability = probabilities(model, broken)
    metrics = {
        "intact_auc": auc(target, intact),
        "broken_auc": auc(target, broken_probability),
        "positive_probability_drop": float(np.mean(intact[target] - broken_probability[target])),
    }
    checks = {
        "Finite metrics": bool(np.isfinite(list(metrics.values())).all()),
        "Unseen equality AUC >= 0.95": metrics["intact_auc"] >= 0.95,
        "All-unequal control AUC is in [0.35, 0.65]": 0.35 <= metrics["broken_auc"] <= 0.65,
        "Breaking equality lowers positive probability by >= 0.25": metrics["positive_probability_drop"] >= 0.25,
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3701)

# %% [markdown]
# </details>
