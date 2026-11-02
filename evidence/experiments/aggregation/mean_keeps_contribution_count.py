# %% [markdown]
# ---
# title: Can an average reveal how many items there were?
# categories:
# - Weighted aggregation
# evidence-id: P013
# description: Use an average-only branch, with item attention disabled, to predict both an average and a total; the
#   branch's count lane carries the multiplicity a plain average erases.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Average prediction
#   metric:
#   - mean_score
#   - nrmse
#   format: error
# - label: Total prediction
#   metric:
#   - sum_score
#   - nrmse
#   format: error
# - label: 'Predicted total, six copies (correct: 4.5)'
#   metric:
#   - paired_sum
#   - 1
#   format: number
# evidence-group: Aggregation
# evidence-result:
#   accelerator: cpu
#   checks:
#     Mean contribution nRMSE below 0.25: true
#     Paired totals land within 0.5 of their correct values: true
#     Six copies predict a larger total than one copy: true
#     Weighted sum nRMSE below 0.35: true
#   duration_seconds: 9.147
#   metrics:
#     mean_score:
#       baseline_rmse: 0.711858212210264
#       nrmse: 0.00990314706249706
#       rmse: 0.007049636563164485
#     paired_sum:
#     - 0.8220831942603826
#     - 4.48253251018186
#     paired_sum_error:
#     - 0.07208319426038257
#     - -0.017467489818139725
#     steps: 300
#     sum_score:
#       baseline_rmse: 2.727324603677681
#       nrmse: 0.030006765498606366
#       rmse: 0.08183818982113572
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: eac04a1e9b8232276dd1f367cb08c39a629d2ea1db661dad31912c5819c182a0
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 06d4dc34fc44
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
#     script_sha256: 2e32d6eb1d73a2c44c9085263b56407b19957b84c251cdde26d60347941911c8
#     source_sha256: 21ee9e11d4138e15fc7eff57e616a10ab3b21e513202142e82e01f456d4a9b68
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 3309
#   started_at: '2026-11-01T01:34:17.505359+00:00'
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
# items:
#   - {contribution: 0.75}
#   - {contribution: 0.75}
#   - {contribution: 0.75}
#   - {contribution: 0.75}
#   - {contribution: 0.75}
#   - {contribution: 0.75}
# mean_contribution: 0.75
# weighted_sum: 4.5
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-mean-keeps-contribution-count
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, items, contribution, mean_contribution, weighted_sum. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("contribution", type: "Number"),
#   )),
#   node("mean_contribution", width: 150pt, kind: "target", type: "Number"),
#   node("weighted_sum", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Repeat the same contribution different numbers of times. The average stays fixed while the correct total changes.
#
# ## Result
#
# {{< evidence P013 status >}}
#
# The model learns both the average and a total that scales with the count. Mean's count lane preserves the multiplicity its plain average would erase.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P013 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P013 script >}}
#
# ### How it works
#
# The item branch uses `attention=None` and `rf.Mean()`. Every item in a
# record repeats the same randomly drawn contribution, so averaging its
# identical encoded tokens gives the same content for one repetition or six.
# Mean adds a learned direction scaled by the number of present items over the
# branch capacity, and that lane is the only difference the parent sees between
# the two. The total head learns to multiply the shared contribution by it.
#
# The model should learn the contribution average and the total. A separate
# probe compares total predictions for one and six copies of 0.75, whose
# correct totals are 0.75 and 4.5: the six-copy prediction must exceed the
# one-copy prediction, and each must land within 0.5 of its correct value.
#
# ### Remaining work
#
# Check the count lane across seeds and at lengths the training data never
# showed. Count-preserving routes without the lane are covered separately by
# [Attention sums](attention-sum-unseen-lengths.html) and
# [visible count with Mean](visible-count-sum.html).
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""Mean keeps the count a weighted sum needs when item cardinality varies.

Learn the raw average and the total while a no-attention Mean carries
multiplicity through its count lane.

Run: uv run python evidence/experiments/run.py P013"""

from __future__ import annotations

from collections.abc import Iterator

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P013"
ITEMS = 6


def repeated_contribution_records(*, rows: int, seed: int) -> Iterator[dict]:
    """Draw variable counts of one repeated contribution with its average and total."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        count = int(rng.integers(1, ITEMS + 1))
        contribution = float(rng.uniform(-1.25, 1.25))
        yield {
            "items": [{"contribution": contribution}] * count,
            "mean_contribution": contribution,
            "weighted_sum": count * contribution,
        }


def prediction(model: rf.Model, observations: list[dict], target: str) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row[target]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray, target: str) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row[target] for row in test], dtype=np.float64)
    baseline_rmse = rmse(actual, float(np.asarray([row[target] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline_rmse, "nrmse": measured / baseline_rmse}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(repeated_contribution_records(rows=1024, seed=seed + 1))
    test = list(repeated_contribution_records(rows=512, seed=seed + 3))
    model = rf.Model(
        d_model=24,
        n_layers=1,
        n_heads=4,
        items=rf.Branch(length=ITEMS, attention=None, reduction=rf.Mean(), contribution=rf.Number),
        mean_contribution=rf.Number(mask=True, objective="mse"),
        weighted_sum=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        train=lambda: repeated_contribution_records(rows=1024, seed=seed + 1),
        validate=lambda: repeated_contribution_records(rows=256, seed=seed + 2),
        seed=seed,
    )
    model.fit(
        datamodule,
        batch_size=64,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.003),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=300 if steps is None else min(steps, 300),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer

    # Evaluate held-out answers, then probe one versus six copies of the same contribution.
    mean_score = score(
        train=train, test=test, predicted=prediction(model, test, "mean_contribution"), target="mean_contribution"
    )
    sum_score = score(train=train, test=test, predicted=prediction(model, test, "weighted_sum"), target="weighted_sum")
    paired = [{"items": [{"contribution": 0.75}]}, {"items": [{"contribution": 0.75}] * ITEMS}]
    paired_sum = prediction(model, paired, "weighted_sum")
    correct = np.asarray([0.75, 0.75 * ITEMS])
    metrics = {
        "mean_score": mean_score,
        "sum_score": sum_score,
        "paired_sum": paired_sum.tolist(),
        "paired_sum_error": (paired_sum - correct).tolist(),
        "steps": trainer.global_step,
    }
    checks = {
        "Mean contribution nRMSE below 0.25": bool(mean_score["nrmse"] < 0.25),
        "Weighted sum nRMSE below 0.35": bool(sum_score["nrmse"] < 0.35),
        "Six copies predict a larger total than one copy": bool(paired_sum[1] > paired_sum[0] + 1.0),
        "Paired totals land within 0.5 of their correct values": bool(np.all(np.abs(paired_sum - correct) < 0.5)),
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3309)

# %% [markdown]
# </details>
