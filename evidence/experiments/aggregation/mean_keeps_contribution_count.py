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
#   duration_seconds: 8.443
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
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: f31c3e535af7
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
#     script_sha256: 6e4046295cf5eca597377c400f85f79b45d094eac1bc3e732d1932d46c9279b5
#     source_sha256: fdbe548e4941ae995607df83c74d11fbf0864dc10d7021572993360074d2ada6
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 3309
#   started_at: '2026-11-02T17:47:28.159100+00:00'
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
