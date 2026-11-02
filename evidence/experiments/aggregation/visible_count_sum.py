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
#   duration_seconds: 18.965
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
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: cb094a7cf603
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
#     script_sha256: ea101b0033419a990c69b277aeb35a3d96336f504f7c256e302a4e91e64979c4
#     source_sha256: 682c49137398a6dd7755a00807cf571c474cea5e040be709f69e877df68f5ec9
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 3605
#   started_at: '2026-11-02T17:47:00.817449+00:00'
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
