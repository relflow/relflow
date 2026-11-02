# %% [markdown]
# ---
# title: Can it add a list of numbers?
# categories:
# - Cardinality generalization
# evidence-id: P001
# description: Predict the total of a list whose length varies within the range used in training.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: New lists
#   metric:
#   - measured
#   - nrmse
#   format: error
# evidence-group: Aggregation
# evidence-result:
#   accelerator: cpu
#   checks:
#     Permutation drift below 0.12 target SD: true
#     Seen-length nRMSE below 0.35: true
#   duration_seconds: 38.717
#   metrics:
#     measured:
#       baseline_rmse: 1.0961391615079292
#       nrmse: 0.012362172408772186
#       rmse: 0.013550661298568002
#     permutation_drift: 0.005140111435372105
#     steps: 1100
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 18ae1262c3a2941ba68d45c1fb83871e4e7a44e6c07da1855773d90fa8e6a3ed
#     cuda: null
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: 95e14d2adeb1
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
#     script_sha256: 5e95ac6e2b0a75400839574f76c790393836f8afaa03149f84b197fa15a6558c
#     source_sha256: 801ac3c8282bd8e8d7ea9c3d2fa5c8ab1f44db2da682b6d6ca17b9564d9305a4
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 3609
#   started_at: '2026-11-02T17:47:00.789810+00:00'
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
#   - {amount: 0.5}
#   - {amount: -0.2}
#   - {amount: 0.9}
# total: 1.2
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-attention-sum-in-range
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, items, amount, total. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("amount", type: "Number"),
#   )),
#   node("total", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Test on fresh lists, then reorder the same items. Reordering should preserve the total.
#
# ## Result
#
# {{< evidence P001 status >}}
#
# The model learns an approximate total in this range. Longer lists need a separate test.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P001 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P001 script >}}
#
# ### How it works
#
# Training, validation, and test records contain one through six independent
# amounts drawn from −1 to 1. Both the repeated item branch and root use learned
# `rf.Attention` reductions. There is no visible count input.
#
# Attention’s reduction retains aggregate evidence and present-token count as
# well as its normalized learned summary. These representations offer a route to
# predicting sums with changing length. Independent values prevent count alone
# from solving the task; a complete-item permutation checks approximate order
# stability. [Unseen-length tests](attention-sum-unseen-lengths.html) separately
# check whether the learned behavior extends beyond interpolation.
#
# ### Remaining work
#
# Repeat across seeds and change capacities, widths, and nested placement.
# This in-range result alone establishes neither arbitrary-length generalization
# nor an exact arithmetic sum; empty and missing-item cases need separate checks.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""Learned Attention can interpolate a sum over seen cardinalities.

Check ordinary in-range accuracy and approximate order invariance.

Run: uv run python evidence/experiments/run.py P001"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P001"
TRAIN_MAX = 6
CAPACITY = 12


def random_records(*, rows: int, seed: int, minimum: int = 1, maximum: int = TRAIN_MAX) -> Iterator[dict]:
    """Draw variable-length numerical bags and their mean and sum."""
    if not 1 <= minimum <= maximum <= CAPACITY:
        raise ValueError(f"length range must satisfy 1 <= minimum <= maximum <= {CAPACITY}")
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        length = int(rng.integers(minimum, maximum + 1))
        values = rng.uniform(-1.0, 1.0, size=length)
        row: dict[str, object] = {
            "items": [{"amount": float(value)} for value in values],
            "mean_amount": float(values.mean()),
            "total": float(values.sum()),
        }
        yield row


def permute(observations: list[dict], *, seed: int) -> list[dict]:
    """Jointly permute complete items without changing any target."""
    rng = np.random.default_rng(seed)
    rows = deepcopy(observations)
    for row in rows:
        items = row["items"]
        row["items"] = [items[index] for index in rng.permutation(len(items))]
    return rows


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
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
    train = list(random_records(rows=1536, seed=seed + 1))
    test = list(random_records(rows=768, seed=seed + 3))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        items=rf.Branch(
            length=CAPACITY, attention="mha", n_layers=2, reduction=rf.Attention(n_layers=2), amount=rf.Number
        ),
        total=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        train=lambda: random_records(rows=1536, seed=seed + 1),
        validate=lambda: random_records(rows=384, seed=seed + 2),
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
        max_steps=1100 if steps is None else min(steps, 1100),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer

    # Evaluate held-out answers and retain their original labels in corruption controls.
    intact_prediction = prediction(model, test)
    measured = score(train=train, test=test, predicted=intact_prediction)
    permuted_prediction = prediction(model, permute(test, seed=seed + 5))
    target_scale = float(np.asarray([row["total"] for row in test], dtype=np.float64).std())
    permutation_drift = rmse(intact_prediction, permuted_prediction) / target_scale
    metrics = {"measured": measured, "permutation_drift": permutation_drift, "steps": trainer.global_step}
    checks = {
        "Seen-length nRMSE below 0.35": bool(measured["nrmse"] < 0.35),
        "Permutation drift below 0.12 target SD": bool(permutation_drift < 0.12),
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3609)

# %% [markdown]
# </details>
