# %% [markdown]
# ---
# title: Can it compare two unfamiliar IDs directly?
# categories:
# - Collection overlap
# evidence-id: P029
# description: Ask whether two individual IDs match, as a simpler comparison beside the two-list overlap test.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Original pairs
#   metric:
#   - intact_auc
#   format: auc
# - label: Matches removed
#   metric:
#   - broken_auc
#   format: auc
# evidence-group: Relational
# evidence-result:
#   accelerator: cpu
#   checks:
#     All-unequal control AUC is in [0.35, 0.65]: true
#     Breaking equality lowers positive probability by >= 0.25: true
#     Finite metrics: true
#     Unseen equality AUC >= 0.95: true
#   duration_seconds: 21.705
#   metrics:
#     broken_auc: 0.507781982421875
#     intact_auc: 0.9984359741210938
#     positive_probability_drop: 0.9882744003023731
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 030e418ee02008cf8752161f2ea51fb6a31198466bbd33c68c922a578d941dd3
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 4e6ab7d60670
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
#     script_sha256: 322cb66e466990d280328beee433a335945d00698c138889781858ee61ca989c
#     source_sha256: 4f303b1d19bc8756d543c352afc8e8c7053f04586be82c5d3768f098444dcb6b
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 3701
#   started_at: '2026-11-01T01:39:20.879939+00:00'
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
# left_id: owl
# right_id: owl
# equal: true
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-flat-unseen-hash-equality-control
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
# {{< evidence P029 status >}}
#
# Direct ID matching works, so the harder overlap failure cannot be blamed solely on the Hash representation. This does not establish comparisons between all pairs in two lists.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P029 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P029 script >}}
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
# Repeat the primitive control alongside the collection evidence across the
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
import relflow as rf
import torch
from reporting import report

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
    return np.asarray([row["predictions"]["equal"]["content"]["probability"] for row in output])


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=48,
        n_layers=2,
        n_heads=4,
        left_id=rf.Hash(n_hashes=4, n_bands=8),
        right_id=rf.Hash(n_hashes=4, n_bands=8),
        equal=rf.Boolean(mask=True),
    )
    # Restart the training RNG independently of parameter initialization.
    lit.seed_everything(seed, workers=True)
    data = rf.SyntheticDataModule(
        train=partial(records, rows=3072, seed=seed + 1, namespace="train"),
        validate=partial(records, rows=768, seed=seed + 2, namespace="validate"),
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
        max_steps=600 if steps is None else min(steps, 600),
        max_epochs=-1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        check_val_every_n_epoch=5,
    )
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
