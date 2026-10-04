# %% [markdown]
# ---
# title: Can it recognize matching IDs it has never seen?
# categories:
# - Identity
# proof-id: P021
# description: Ask whether two IDs are equal, using IDs absent from training.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: Hash inputs
#   metric:
#   - hash_auc
#   format: auc
# - label: Unknown Category inputs
#   metric:
#   - category_oov_auc
#   format: auc
# - label: Shuffled answers
#   metric:
#   - shuffled_auc
#   format: auc
# ---
#
# ## Example
#
# ```yaml
# left_id: test-17
# right_id: test-17
# equal: true
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-proof-identity-unseen-identity-equality
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with identity, left_id, right_id, equal. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("identity", kind: "root", children: (
#   node("left_id", type: "Hash", width: 150pt,),
#   node("right_id", type: "Hash", width: 150pt,),
#   node("equal", kind: "target", type: "Boolean",),
# )))
# ```
#
# ## Comparison
#
# Compare Hash inputs with Category inputs that treat new IDs as unknown. Shuffle answers as a second check.
#
# ## Result
#
# {{< proof P021 status >}}
#
# Hash preserves a useful matching signal for new IDs. This test compares two IDs within one record; larger lookups and joins require their own evidence.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P021 evidence >}}
#
# ### Run this experiment
#
# {{< proof P021 script >}}
#
# ### How it works
#
# [Hash](../../data-types/hash.qmd) preserves equality across fields within an
# encoded batch. Its representation does not require a persistent entry for
# each identifier, so new strings can still supply the comparison signal.
#
# The test trains both a Hash model and a matched Category model. Train,
# validation, and test use disjoint `train-`, `validate-`, and `test-` namespaces.
# The Category control encounters unknown content in both fields at test time.
# A second control evaluates the trained Hash model after shuffling the test
# labels, preserving identities while breaking their association with the answer.
#
# The protocol uses 4,096 training, 1,024 validation, and 4,096 test pairs, with
# 20 deterministic epochs. Equal and unequal pairs are balanced.
#
# ### Remaining work
#
# Apply the same unseen-identity renaming to both fields and check whether
# predictions remain stable. Repeat the three evaluations over three core
# seeds and at least ten calibration seeds, keeping the Category boundary.
#
# ### Complete experiment code
#

# %%
"""P021: infer equality between identities absent from the training vocabulary.

Hash is compared with shuffled held-out labels and a matched Category model.
Train, validation, and test identity namespaces are disjoint.
"""

from collections.abc import Callable, Iterator
from functools import partial
from typing import Literal

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P021"


def records(*, rows: int, seed: int, namespace: str, shuffle_targets: bool = False) -> Iterator[dict]:
    """Yield balanced equal/unequal pairs from one identity namespace."""
    rng = np.random.default_rng(seed)
    target = np.tile(np.array([False, True]), (rows + 1) // 2)[:rows]
    rng.shuffle(target)
    left = rng.integers(0, rows * 2, size=rows)
    other = rng.integers(0, rows * 2 - 1, size=rows)
    other += other >= left
    right = np.where(target, left, other)
    labels = target.copy()
    if shuffle_targets:
        rng.shuffle(labels)
    for index in range(rows):
        yield {
            "left_id": f"{namespace}-{left[index]}",
            "right_id": f"{namespace}-{right[index]}",
            "equal": bool(labels[index]),
        }


def fit(*, identity: Literal["hash", "category"], seed: int, steps: int | None, accelerator: str) -> rf.Model:
    """Train one representation on the same identity pairs."""
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=48,
        n_layers=2,
        n_heads=4,
        batch_size=128,
        left_id=rf.Hash(n_hashes=4) if identity == "hash" else rf.Category(p_unavailable=0.0),
        right_id=rf.Hash(n_hashes=4) if identity == "hash" else rf.Category(p_unavailable=0.0),
        equal=rf.Boolean(mask=True),
    )
    model.optimizer = lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3)
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, rows=4096, seed=seed + 1, namespace="train"),
        validate=partial(records, rows=1024, seed=seed + 2, namespace="validate"),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        max_epochs=20,
        max_steps=steps if steps is not None else -1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model=model, datamodule=data)
    return model


def score(model: rf.Model, records: Callable[[], Iterator[dict]], accelerator: str) -> float:
    """Evaluate Boolean AUC without updating the vocabulary."""
    data = rf.SyntheticDataModule(model=model, test=records)
    trainer = lit.Trainer(
        accelerator=accelerator,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    metrics = trainer.test(model=model, datamodule=data, verbose=False)[0]
    return float(metrics[".equal/test.auc.content"])


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    test = partial(records, rows=4096, seed=seed + 3, namespace="test")
    control = partial(records, rows=4096, seed=seed + 3, namespace="test", shuffle_targets=True)
    hash_model = fit(identity="hash", seed=seed, steps=steps, accelerator=accelerator)
    category_model = fit(identity="category", seed=seed, steps=steps, accelerator=accelerator)
    equality_auc = score(hash_model, test, accelerator)
    control_auc = score(hash_model, control, accelerator)
    category_oov_auc = score(category_model, test, accelerator)
    gap = equality_auc - control_auc
    return {
        "hash_auc": equality_auc,
        "shuffled_auc": control_auc,
        "category_oov_auc": category_oov_auc,
        "auc_gap": gap,
    }, {
        "Unseen Hash equality AUC is at least 0.95": equality_auc >= 0.95,
        "Shuffled labels remain between 0.42 and 0.58 AUC": 0.42 <= control_auc <= 0.58,
        "Category OOV AUC is at most 0.65": category_oov_auc <= 0.65,
        "Hash exceeds shuffled labels by at least 0.35 AUC": gap >= 0.35,
    }


if __name__ == "__main__":
    report(PROOF_ID, run, seed=29)

# %% [markdown]
# </details>
