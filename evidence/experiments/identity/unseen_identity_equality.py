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
#   duration_seconds: 49.56
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
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: 32ed41a4aa03
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
#     script_sha256: dfb999d1c302b7a503c7e69352cc75b1d6a5715cbd1729d44f42c37c7b7722d0
#     source_sha256: 9f4d2fe47ea363da6fd353e03ac2099579a8a467024cb3048cf6c90b4384cca1
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 29
#   started_at: '2026-11-02T17:48:15.372536+00:00'
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
