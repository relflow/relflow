# %% [markdown]
# ---
# title: Does the count help with longer lists?
# categories:
# - Cardinality generalization
# evidence-id: P005
# description: Give the model both a repeated amount and its count, then test larger counts absent from training.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Larger counts
#   metric:
#   - unseen_score
#   - nrmse
#   format: error
# evidence-group: Aggregation
# evidence-result:
#   accelerator: cpu
#   checks:
#     All measurements finite: true
#     Unseen nRMSE below 0.40 and duplication error below 0.20: true
#   duration_seconds: 18.91
#   metrics:
#     duplication_error: 0.1253142277019298
#     steps: 700
#     unseen_score:
#       baseline_rmse: 4.8922481485064635
#       nrmse: 0.18511797591621246
#       rmse: 0.9056430749313544
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 1d018126581bd037925a6ef6fc3463b402afcab3bf39e6579e037982acc8e424
#     cuda: null
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: d39434311c7f
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
#     script_sha256: 615291891125e30bbef697803b9e60a29e7c96db189e436d227a51cfce27a053
#     source_sha256: cb81323a852a1489bb787df15cd5a4bbb5817273477323ab81cfb8ba7a197687
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 3621
#   started_at: '2026-11-02T17:47:00.804619+00:00'
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
# item_count: 8
# items:
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
# total: 4.0
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-visible-count-unseen-lengths
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, item_count, items, amount, total. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("item_count", type: "Number"),
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("amount", type: "Number"),
#   )),
#   node("total", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Repeat the list and double the visible count. The expected total doubles too.
#
# ## Result
#
# {{< evidence P005 status >}}
#
# The model extends the amount-times-count relationship to these larger counts. The count is supplied, so this does not show that it learned to count items.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P005 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P005 script >}}
#
# ### How it works
#
# The schema matches the [in-range count control](visible-count-sum.html).
# A Mean item branch has no item attention; its equal-valued bags supply content
# but lose multiplicity. The root receives `item_count` as an ordinary Number.
#
# Training counts are one through six. The test uses seven through ten, so
# accuracy measures count extrapolation. A separate complete-bag duplication
# intervention also doubles the visible count and requires approximately doubled
# predictions. It checks whether the learned relationship respects scaling,
# within a fixed branch capacity of twelve.
#
# ### Remaining work
#
# Repeat across seeds and broaden the input distributions beyond repeated
# values. This evidence establishes a route with an explicitly supplied count;
# [Attention without a count field](attention-sum-unseen-lengths.html) covers the
# separate structural-cardinality claim.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""The explicit-count control should extrapolate more cleanly than hidden count.

Require count-aware sum accuracy and duplication scaling out of range.

Run: uv run python evidence/experiments/run.py P005"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P005"
TRAIN_MAX = 6
CAPACITY = 12


def random_records(*, rows: int, seed: int, minimum: int = 1, maximum: int = TRAIN_MAX) -> Iterator[dict]:
    """Draw variable-length numerical bags and their mean and sum."""
    if not 1 <= minimum <= maximum <= CAPACITY:
        raise ValueError(f"length range must satisfy 1 <= minimum <= maximum <= {CAPACITY}")
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        length = int(rng.integers(minimum, maximum + 1))
        values = np.repeat(rng.uniform(-1.0, 1.0), length)
        row: dict[str, object] = {
            "items": [{"amount": float(value)} for value in values],
            "mean_amount": float(values.mean()),
            "total": float(values.sum()),
        }
        row["item_count"] = length
        yield row


def duplicate(observations: list[dict]) -> list[dict]:
    """Duplicate every complete bag and update its algebraic targets."""
    rows = deepcopy(observations)
    if any(len(row["items"]) * 2 > CAPACITY for row in rows):
        raise ValueError(f"duplicated collection exceeds configured capacity {CAPACITY}")
    for row in rows:
        row["items"] = [*row["items"], *row["items"]]
        row["total"] *= 2.0
        row["item_count"] *= 2
    return rows


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"], "item_count": row["item_count"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["total"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["total"] for row in test], dtype=np.float64)
    baseline = rmse(actual, float(np.asarray([row["total"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline, "nrmse": measured / baseline}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(random_records(rows=1024, seed=seed + 1))
    unseen = list(random_records(rows=512, seed=seed + 3, minimum=7, maximum=10))
    short = list(random_records(rows=256, seed=seed + 4, maximum=5))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        items=rf.Branch(length=CAPACITY, attention=None, n_layers=2, reduction=rf.Mean(), amount=rf.Number),
        item_count=rf.Number,
        total=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        train=lambda: random_records(rows=1024, seed=seed + 1),
        validate=lambda: random_records(rows=256, seed=seed + 2),
        seed=seed,
    )
    model.fit(
        datamodule,
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=700 if steps is None else min(steps, 700),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer

    # Evaluate held-out answers and retain their original labels in corruption controls.
    unseen_score = score(train=train, test=unseen, predicted=prediction(model, unseen))
    short_prediction = prediction(model, short)
    duplicated_prediction = prediction(model, duplicate(short))
    duplication_error = rmse(2.0 * short_prediction, duplicated_prediction) / float(
        np.std(np.asarray([row["total"] for row in unseen], dtype=np.float64))
    )
    metrics = {"unseen_score": unseen_score, "duplication_error": duplication_error, "steps": trainer.global_step}
    checks = {
        "All measurements finite": bool(np.isfinite([unseen_score["nrmse"], duplication_error]).all()),
        "Unseen nRMSE below 0.40 and duplication error below 0.20": bool(
            unseen_score["nrmse"] < 0.4 and duplication_error < 0.2
        ),
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3621)

# %% [markdown]
# </details>
