# %% [markdown]
# ---
# title: Can it find a shared ID across two lists?
# categories:
# - Collection overlap
# evidence-id: P028
# description: Predict whether two lists have any ID in common, using IDs absent from training.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Original lists
#   metric:
#   - intact_auc
#   format: auc
# - label: Match removed
#   metric:
#   - broken_auc
#   format: auc
# evidence-group: Relational
# evidence-result:
#   accelerator: cpu
#   checks:
#     Absolute positive break response <= 0.10: true
#     Broken AUC remains in the chance band: true
#     Finite metrics: true
#     Intact AUC remains in the calibrated chance band: true
#     Intervention removes all overlaps: true
#     Invariant probability drift <= 0.15: true
#     Labels are exactly balanced: true
#     Labels match actual overlap: true
#     Renamed and permuted AUC remains in the chance band: true
#     Renaming and permutation preserve overlap: true
#     invariant generator has balanced truthful labels: true
#     invariant positive rows have exactly one overlap: true
#     invariant sides have two to five unique members: true
#     test generator has balanced truthful labels: true
#     test positive rows have exactly one overlap: true
#     test sides have two to five unique members: true
#     train generator has balanced truthful labels: true
#     train positive rows have exactly one overlap: true
#     train sides have two to five unique members: true
#     validate generator has balanced truthful labels: true
#     validate positive rows have exactly one overlap: true
#     validate sides have two to five unique members: true
#   duration_seconds: 48.883
#   metrics:
#     broken_auc: 0.47284698486328125
#     intact_auc: 0.5601234436035156
#     invariant_auc: 0.6170501708984375
#     invariant_drift: 0.12359561448101886
#     positive_break_drop: 0.03347706462955102
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 688951bebf95cf9abf09ab239d7d47691aa6b15b77becac2d1a2d2ea5482871c
#     cuda: null
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: 9dbc7400c1d5
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
#     script_sha256: 56c2a3a69c0ec3b67021a3f410b4debeaeedda158f478ddc2ca57818a8e181e4
#     source_sha256: 9be47a571e265e687aa5792c31ca0015105433b78e5cb6fb826d3eebe31da4ae
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 3711
#   started_at: '2026-11-02T17:49:11.317662+00:00'
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
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-direct-sibling-hash-overlap
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with overlap, left, entity_id, right, entity_id, has_overlap. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("overlap", kind: "root", width: 120pt, children: (
#   node("left", kind: "branch", repeated: true, width: 120pt, children: (
#     node("entity_id", type: "Hash"),
#   )),
#   node("right", kind: "branch", repeated: true, width: 120pt, children: (
#     node("entity_id", type: "Hash"),
#   )),
#   node("has_overlap", kind: "target", type: "Boolean", width: 150pt,),
# )))
# ```
#
# ## Comparison
#
# Rename and reorder IDs without changing overlap, then remove the shared member. A useful predictor should notice the removal.
#
# ## Result
#
# {{< evidence P028 status >}}
#
# This model stays near chance and barely responds to removing the match. The result describes this setup and budget, rather than proving that overlap can never be learned.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P028 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P028 script >}}
#
# ### How it works
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
# ### Remaining work
#
# Repeat across more seeds and larger training budgets, then evaluate bounded
# set-comparison designs with accuracy, intervention, and invariance gates.
# This result characterizes the tested schema and budget, not an impossibility
# result for transformers. Compute deterministic overlap in a preprocessor or
# the application when it is needed today.
#
# ### Complete experiment code
#

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
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P028"
MIN_LENGTH = 2
MAX_LENGTH = 5


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
            for index, value in enumerate(item["entity_id"] for item in row["right"])
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
    return np.asarray([row["predictions"]["has_overlap"]["content"]["probability"] for row in output])


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=48,
        n_layers=3,
        n_heads=4,
        reduction=None,
        left=rf.Branch(length=MAX_LENGTH, n_layers=1, reduction=None, entity_id=rf.Hash(n_hashes=4, n_bands=8)),
        right=rf.Branch(length=MAX_LENGTH, n_layers=1, reduction=None, entity_id=rf.Hash(n_hashes=4, n_bands=8)),
        has_overlap=rf.Boolean(mask=True),
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
        max_steps=700 if steps is None else min(steps, 700),
        max_epochs=-1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        check_val_every_n_epoch=5,
    )
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


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3711)

# %% [markdown]
# </details>
