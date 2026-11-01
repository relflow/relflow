# %% [markdown]
# ---
# title: Can a visible count help recover a total?
# categories:
# - Cardinality generalization
# evidence-id: P004
# description: Every item has the same amount. Give the model the item count as an extra input and ask for the total.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Total prediction
#   metric:
#   - measured
#   - nrmse
#   format: error
# evidence-group: Aggregation
# evidence-result:
#   accelerator: cpu
#   checks:
#     Matched-count RMSE below 0.35: true
#     Matched-count prediction gap above 2.5: true
#     Visible-count nRMSE below 0.25: true
#   duration_seconds: 18.586
#   metrics:
#     measured:
#       baseline_rmse: 2.2334354369152405
#       nrmse: 0.04427298902836138
#       rmse: 0.09888086259410195
#     probe_error: 0.07098732934383868
#     probe_prediction:
#     - 0.6470540289043993
#     - 4.114705650834975
#     probe_target:
#     - 0.7
#     - 4.199999999999999
#     steps: 700
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 2ac166aaba807dccf0b3f017c6f7a5e2ced6e581375b2ddc804fcca939cf575c
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 99b6d3a3b85e
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
#     script_sha256: 04d00dc364065aceee7e621577d92da561975de5d4c98f0c958bee454ef4af41
#     source_sha256: 627f0a4b21b2e35c57a6ba9785ed3c64b7dcbbad7088f0d442e2098ac65b3002
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 3605
#   started_at: '2026-11-01T01:32:43.063452+00:00'
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
# item_count: 6
# items:
#   - {amount: 0.7}
#   - {amount: 0.7}
#   - {amount: 0.7}
#   - {amount: 0.7}
#   - {amount: 0.7}
#   - {amount: 0.7}
# total: 4.2
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-visible-count-sum
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
# Keep the amount fixed while changing the count. The answers must differ even though an average-only branch sees the same value.
#
# ## Result
#
# {{< evidence P004 status >}}
#
# The extra count supplies information the average loses. This test uses repeated equal amounts, rather than arbitrary mixed lists.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P004 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P004 script >}}
#
# ### How it works
#
# Every item in a record repeats one random amount. The branch uses
# `attention=None` and `rf.Mean()`, so its summary does not change with repetition
# count. A visible root `item_count` supplies the missing factor, and learned root
# attention combines both inputs to predict their product.
#
# Train and test lengths are one through six. Matched probes repeat 0.7 once or
# six times, requiring totals 0.7 and 4.2. Their predictions must separate by more
# than 2.5 and stay close to those targets. This isolates learning from a supplied
# count; [the count-free Attention evidence](attention-sum-in-range.html) tests
# structural recovery of multiplicity.
#
# ### Remaining work
#
# Repeat across seeds and test nonidentical bags, missing items, and nested
# collections. [Unseen visible counts](visible-count-unseen-lengths.html) have a
# separate evidence; this case checks only the trained count range.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""A visible item count diagnoses cardinality lost by Mean.

Learn value times visible cardinality and distinguish matched probes.

Run: uv run python evidence/experiments/run.py P004"""

from __future__ import annotations

from collections.abc import Iterator

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P004"
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


def equal_value_probes(*, value: float, lengths: tuple[int, ...]) -> list[dict]:
    """Create matched bags that differ only in repetition count."""
    if not lengths or min(lengths) < 1 or max(lengths) > CAPACITY:
        raise ValueError(f"probe lengths must be within 1..{CAPACITY}, got {lengths!r}")
    rows: list[dict[str, object]] = []
    for length in lengths:
        row: dict[str, object] = {"items": [{"amount": value}] * length, "mean_amount": value, "total": length * value}
        row["item_count"] = length
        rows.append(row)
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
    test = list(random_records(rows=512, seed=seed + 3))
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
    predicted = prediction(model, test)
    measured = score(train=train, test=test, predicted=predicted)
    probes = equal_value_probes(value=0.7, lengths=(1, TRAIN_MAX))
    probe_prediction = prediction(model, probes)
    probe_target = np.asarray([row["total"] for row in probes], dtype=np.float64)
    probe_error = rmse(probe_target, probe_prediction)
    metrics = {
        "measured": measured,
        "probe_error": probe_error,
        "probe_target": probe_target.tolist(),
        "probe_prediction": probe_prediction.tolist(),
        "steps": trainer.global_step,
    }
    checks = {
        "Visible-count nRMSE below 0.25": bool(measured["nrmse"] < 0.25),
        "Matched-count prediction gap above 2.5": bool(probe_prediction[1] - probe_prediction[0] > 2.5),
        "Matched-count RMSE below 0.35": bool(probe_error < 0.35),
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3605)

# %% [markdown]
# </details>
