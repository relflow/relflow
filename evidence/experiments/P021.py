# %% [markdown]
# ---
# title: Can it recognize matching IDs it has never seen?
# description: Ask whether two IDs are equal, using IDs absent from training.
# evidence-readout:
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
# evidence-group: Identity
# evidence-review:
#   question: Can relflow recognize matching IDs relflow has never seen?
#   modeling: 'Give the model two identifiers and hide a Boolean target indicating whether they match. Training and
#     test use disjoint identifier names. Compare Hash with Category on the same equality task: new Category labels
#     share the unknown representation, whereas Hash retains equality within an encoded batch. A shuffled-label control
#     checks for an accidental signal.'
#   comparison:
#     kind: Controlled comparison
#     metric: AUC
#     conditions:
#     - label: Unknown Category inputs
#       path:
#       - category_oov_auc
#       format: auc
#     - label: Hash inputs
#       path:
#       - hash_auc
#       format: auc
#     definition: 'Ranking score: 0.5 is chance and 1 is perfect separation. This is not the percentage of correct
#       predictions.'
#     direction: Higher is better
#     domain:
#     - 0
#     - 1
#   note: ''
# evidence-plot:
#   target: equal
#   focus: Can equality transfer to identities absent from training?
#   comparison: Compare Hash, unknown Category labels, and shuffled-label training.
#   limit: The task tests equality within a record, not learning a persistent identity vocabulary.
#   renderer: plot
# evidence-highlight:
#   order: 1
#   title: Match IDs absent from training
#   result:
#     metric:
#     - hash_auc
#     text: '**{value:.4f} AUC** with `Hash`'
#     minimum: 0
#     maximum: 1
#   control:
#     metric:
#     - category_oov_auc
#     text: '{value:.4f} with unknown `Category` labels'
#     minimum: 0
#     maximum: 1
# evidence-cost:
#   baseline: 62
#   candidate: 62
# ---
#
# The task is equality, not remembering what a particular ID means. The model receives two strings
# and predicts whether they are the same. Test identities use a separate namespace from training, so
# a memorized table of training names cannot provide the answers.
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
# The illustrated pair test-17 and test-17 has answer true. Replacing only the right-hand ID with
# test-18 makes the answer false; replacing both with a different shared test ID should still mean
# equality. The experiment scores new equal and unequal pairs, rather than assigning a permanent
# yes/no meaning to any one string.
#
# ## Comparison
#
# Compare Hash inputs with Category inputs that treat new IDs as unknown. Shuffle answers as a
# second check.
#
# A matched Category model uses a learned vocabulary, so both unfamiliar test strings become unknown
# content; their original equality is no longer represented that way. Hash retains a within-batch
# matching signal for new strings. A second control leaves the Hash model and its test pairs
# unchanged but shuffles the answers, breaking the relationship that accurate equality predictions
# would otherwise exploit.
#
# ## Result
#
# Hash preserves a useful matching signal for new IDs. This test compares two IDs within one record;
# larger lookups and joins require their own evidence.
#
# AUC describes ranking of equal versus unequal pairs: 0.5 is chance and 1 is perfect ranking.
# Strong Hash AUC together with chance-like Category and shuffled-label controls supports the tested
# unseen-identity comparison. It is not a guarantee of collision-free hashing, probability
# calibration, or reliable lookup across arbitrarily large memories.
#
# ## Method
#
# ### How it works
#
# [Hash](/data-types/hash/) preserves equality across fields within an
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

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.10",
#     "lightning",
#     "numpy",
#     "pyyaml>=6",
#     "relflow>=0.5.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "structure"
# types = ["Category", "Hash"]
# conditions = ["unseen-identities", "changed-associations", "representations"]
# ///

"""P021: infer equality between identities absent from the training vocabulary.

Hash is compared with shuffled held-out labels and a matched Category model.
Train, validation, and test identity namespaces are disjoint.
"""

import sys
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Literal

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import experiment, report


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

    class PredictionModel(rf.Model):
        left_id = rf.Hash(n_hashes=4) if identity == "hash" else rf.Category(p_unavailable=0.0)
        right_id = rf.Hash(n_hashes=4) if identity == "hash" else rf.Category(p_unavailable=0.0)
        equal = rf.Boolean(mask=True)

    model = PredictionModel(d_model=48, n_layers=2, n_heads=4)
    data = rf.SyntheticDataModule(
        train=partial(records, rows=4096, seed=seed + 1, namespace="train"),
        validate=partial(records, rows=1024, seed=seed + 2, namespace="validate"),
        seed=seed,
    )
    model.fit(
        data,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.003),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
        accelerator=accelerator,
        max_epochs=20,
        max_steps=steps if steps is not None else -1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    return model


def score(model: rf.Model, records: Callable[[], Iterator[dict]], accelerator: str) -> float:
    """Evaluate Boolean AUC without updating the vocabulary."""
    data = rf.SyntheticDataModule(test=records)
    metrics = model.test(
        data,
        accelerator=accelerator,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        verbose=False,
        batch_size=128,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
    )[0]
    return float(metrics[".equal/test.auc.content"])


@experiment("P021", seed=29)
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


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    import matplotlib.pyplot as plt

    from evidence.diagnostics.plots import dots

    m = bundle.record["metrics"]
    labels = ["Hash, new identities", "Category, unavailable identities", "Shuffled-label training"]
    values = [m["hash_auc"], m["category_oov_auc"], m["shuffled_auc"]]
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    dots(
        ax, labels, {"Held-out AUC": values}, xlabel="Ranking AUC (higher is better)", reference=(0.5, "Chance ranking")
    )
    ax.set_xlim(-0.02, 1.03)
    return (
        fig,
        "These are the proof's independent held-out ranking scores. AUC is not accuracy, and no unobserved ROC curve is drawn from these summary values.",
        {},
    )


if __name__ == "__main__":
    report(run)
