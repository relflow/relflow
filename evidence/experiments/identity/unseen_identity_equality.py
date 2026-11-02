# %% [markdown]
# ---
# title: Can it recognize matching IDs it has never seen?
# categories:
# - Identity
# evidence-id: P021
# description: Ask whether two IDs are equal, using IDs absent from training.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
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
# evidence-result:
#   accelerator: cpu
#   checks:
#     Category OOV AUC is at most 0.65: true
#     Hash exceeds shuffled labels by at least 0.35 AUC: true
#     Shuffled labels remain between 0.42 and 0.58 AUC: true
#     Unseen Hash equality AUC is at least 0.95: true
#   duration_seconds: 48.223
#   metrics:
#     auc_gap: 0.5059829950332642
#     category_oov_auc: 0.5
#     hash_auc: 0.9991424083709717
#     shuffled_auc: 0.4931594133377075
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: bd5c3c04edc0f5240a4677fd609bd2bdc3d48b661521032d09c403bd5c81c983
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: ee0b824921cb
#     packages:
#       lightning: 2.6.4
#       numpy: 2.4.6
#       polars: 2.0.0
#       pyarrow: 24.0.0
#       pydantic: 2.13.4
#       relflow: 0.4.1
#       tensordict: 0.12.4
#       torch: 2.12.0+cpu
#       torchmetrics: 1.9.0
#     platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.41
#     python: 3.12.16
#     relflow_path: /tmp/runtime/lib/python3.12/site-packages/relflow
#     relflow_revision: 41d6f5fb74c0ba9f0be104c896913a55adb125f5
#     relflow_source_sha256: 00fdf397aeefb3a4e705d431850e0a6294eef142989513fddcb903bb1aefa873
#     script_sha256: e328ce70449017c8b1c856b23b05eeaf71a2d16a6b9eb78b9debd81880af3e86
#     source_sha256: 04da6df5c0f6576898a0f86828cadf0b1a93337980af7730fa63c979a48ed9d2
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 29
#   started_at: '2026-11-01T01:36:14.970454+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41261482839-1
#   candidate:
#     revision: 41d6f5fb74c0ba9f0be104c896913a55adb125f5
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   baseline:
#     revision: 1add6d8919e686dc63b7fa6c4cdbdc437be2b570
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#   manifest_sha256: f4b3bfe706e43e38785ab5bed94c3f51eb44fe7314d0a1b2f0dd3c6f3f7bef92
#   passed: true
#   github_run:
#     event: pull_request
#     head_branch: dev/evidence-in-docs
#     head_sha: 8bb7764d8186c743eb554cb4a91dedb205ad4cb9
#     html_url: https://github.com/relflow/relflow/actions/runs/41261482839
#     pull_requests:
#     - base_sha: 1add6d8919e686dc63b7fa6c4cdbdc437be2b570
#       head_sha: 8bb7764d8186c743eb554cb4a91dedb205ad4cb9
#       number: 64
#     run_attempt: 1
#     run_id: 41261482839
#   release: https://github.com/relflow/docs/releases/tag/ci-41261482839-attempt-1
#   reports_sha256: bc67a1cb3725cbf9c5a8deb65844297d70f6d2122cd3b7de5c1d15f56603864f
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
# //| label: fig-evidence-identity-unseen-identity-equality
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
# {{< evidence P021 status >}}
#
# Hash preserves a useful matching signal for new IDs. This test compares two IDs within one record; larger lookups and joins require their own evidence.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P021 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P021 script >}}
#
# ### How it works
#
# [Hash](../../datatypes/hash.qmd) preserves equality across fields within an
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
import relflow as rf
import torch
from reporting import report

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
        left_id=rf.Hash(n_hashes=4) if identity == "hash" else rf.Category(p_unavailable=0.0),
        right_id=rf.Hash(n_hashes=4) if identity == "hash" else rf.Category(p_unavailable=0.0),
        equal=rf.Boolean(mask=True),
    )
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
