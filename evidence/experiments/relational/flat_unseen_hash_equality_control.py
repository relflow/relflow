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
#   duration_seconds: 21.231
#   metrics:
#     broken_auc: 0.507781982421875
#     intact_auc: 0.9984359741210938
#     positive_probability_drop: 0.9882744003023731
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 030e418ee02008cf8752161f2ea51fb6a31198466bbd33c68c922a578d941dd3
#     cuda: null
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: 2bc9b39e2a6c
#     packages:
#       lightning: 2.6.4
#       numpy: 2.4.6
#       polars: 2.0.0
#       pyarrow: 24.0.0
#       pydantic: 2.13.4
#       relflow: 0.4.2
#       tensordict: 0.12.4
#       torch: 2.12.0+cpu
#       torchmetrics: 1.9.0
#     platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.41
#     python: 3.12.16
#     relflow_path: /tmp/runtime/lib/python3.12/site-packages/relflow
#     relflow_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     relflow_source_sha256: 77953506049ed76c92adc5f32a9bcc32432a788068a4e366b2f78bcf709f29a2
#     script_sha256: 5213225c00df348dbaa3e7d148d323df03bf3e5daf31b9443eff1a56224ccc50
#     source_sha256: c21bad02148849dfa1fddd82ce31d042059b1c6d1ea406f3fb5769123f323ea0
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 3701
#   started_at: '2026-11-02T17:49:26.601550+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41494093900-1
#   candidate:
#     payload_sha256: 2169c968a7733085889a6eed6e895bcb2590059bcc46243d49a61ffacc42f67f
#     revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   baseline:
#     payload_sha256: 2a6af4bd3869faac23f6b7327f2c59e8f5b367580b2f2fa609710958775c08ed
#     revision: 7dec5e860dc389c37c3642d8904867249e831773
#     wheel_sha256: d3da50b65e0c0c7b98059f095b82bf6bf5d0ea0c5ccd0f302015118075f1dd85
#   evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#   executor_revision: 92b3c887eabc598ee2fc6c9dba4e05b9166cb8ba
#   manifest_sha256: 4f0926f3fc366fd8d14bf728b2d06f4a475d61852679176704816c110ae51cfd
#   passed: true
#   github_run:
#     event: push
#     head_branch: main
#     head_sha: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     html_url: https://github.com/relflow/relflow/actions/runs/41494093900
#     path: .github/workflows/ci.yml
#     run_attempt: 1
#     run_id: 41494093900
#     workflow_id: 403231172
#   release: https://github.com/relflow/relflow/releases/tag/evidence-ci-41494093900-attempt-1
#   reports_sha256: b1bfee5622ffbd389b1777443dc05e6d8f08dccff9e2540a601f0c2608948594
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
