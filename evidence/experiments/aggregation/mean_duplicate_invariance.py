# %% [markdown]
# ---
# title: Does repeating a list preserve its average?
# categories:
# - Cardinality generalization
# evidence-id: P003
# description: Predict the average amount, using a branch that averages its encoded items without item attention;
#   the branch's count lane sees the repetition, the average must not.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Average prediction
#   metric:
#   - measured
#   - nrmse
#   format: error
# - label: Change after repetition
#   metric:
#   - duplication_drift
#   format: number
# evidence-group: Aggregation
# evidence-result:
#   accelerator: cpu
#   checks:
#     Duplication drift below 0.05 target SD: true
#     Mean nRMSE below 0.25: true
#     Permutation drift below 1e-6 target SD: true
#   duration_seconds: 8.13
#   metrics:
#     duplication_drift: 0.0011719380509312303
#     measured:
#       baseline_rmse: 0.4464820375017577
#       nrmse: 0.00950241596273492
#       rmse: 0.004242658040231113
#     permutation_drift: 5.3915851615687914e-08
#     steps: 350
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 058eff83d0803189ee390e1fc806f493a5380f2ec8bf8bd7b2a8c132428fb15d
#     cuda: null
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: f9374915423f
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
#     script_sha256: 52781cf50d9509c33adcb45ab5dc1184026751ae3d3a77c6ba60907a40c13726
#     source_sha256: 0d80863d6b519c3d8333ba0ee011ed4b8d8fa2ed1ee666f2bb46eee6e166e944
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 3600
#   started_at: '2026-11-02T17:47:00.770514+00:00'
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
#   - {amount: 0.2}
#   - {amount: 0.8}
# mean_amount: 0.5
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-mean-duplicate-invariance
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, items, amount, mean_amount. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("amount", type: "Number"),
#   )),
#   node("mean_amount", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Reorder the list and repeat every item. Both changes should leave its average unchanged.
#
# ## Result
#
# {{< evidence P003 status >}}
#
# This setup learns the average and keeps it under repetition to within a thousandth of the target spread. The summary itself is no longer identical: Mean carries a count lane, so the same summary can also tell a total apart.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P003 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P003 script >}}
#
# ### How it works
#
# The item branch uses `attention=None` and `rf.Mean()`. It averages encoded
# Number tokens rather than directly averaging raw amounts, and adds a learned
# direction scaled by the number of present items over the branch capacity. A
# learned downstream path must still map that representation to the
# arithmetic-mean target.
#
# Without preceding item attention, permuting tokens leaves the summary
# unchanged exactly, and the test checks that invariance at numerical precision.
# Duplicating the complete bag leaves the averaged content unchanged but doubles
# the count lane, so the summary differs; the mean readout learns to ignore that
# lane, and the test bounds the remaining drift at five percent of the target
# spread. The same lane is what lets a total be recovered from this route, which
# the [contribution count](mean-keeps-contribution-count.html) and
# [cardinality](../structure/mean-keeps-cardinality.html) evidences measure.
#
# ### Remaining work
#
# Repeat across seeds and test missing values, empty collections, and nested
# placement. The invariance applies to this no-item-attention path; it does not
# imply that every model using Mean is insensitive to multiplicity.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""Mean keeps its average under repetition with item attention disabled.

Learn the average, retain it exactly under permutation, and keep it within
a small tolerance when the whole bag is duplicated and only the count lane moves.

Run: uv run python evidence/experiments/run.py P003"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P003"
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


def duplicate(observations: list[dict]) -> list[dict]:
    """Duplicate every complete bag and update its algebraic targets."""
    rows = deepcopy(observations)
    if any(len(row["items"]) * 2 > CAPACITY for row in rows):
        raise ValueError(f"duplicated collection exceeds configured capacity {CAPACITY}")
    for row in rows:
        row["items"] = [*row["items"], *row["items"]]
        row["total"] *= 2.0
    return rows


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
    return np.asarray([row["mean_amount"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["mean_amount"] for row in test], dtype=np.float64)
    baseline = rmse(actual, float(np.asarray([row["mean_amount"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline, "nrmse": measured / baseline}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(random_records(rows=768, seed=seed + 1))
    test = list(random_records(rows=384, seed=seed + 3, maximum=3))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        items=rf.Branch(length=CAPACITY, attention=None, n_layers=2, reduction=rf.Mean(), amount=rf.Number),
        mean_amount=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        train=lambda: random_records(rows=768, seed=seed + 1),
        validate=lambda: random_records(rows=192, seed=seed + 2),
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
        max_steps=350 if steps is None else min(steps, 350),
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
    permutation_prediction = prediction(model, permute(test, seed=seed + 4))
    duplication_prediction = prediction(model, duplicate(test))
    scale = float(np.std(np.asarray([row["mean_amount"] for row in test], dtype=np.float64)))
    permutation_drift = rmse(intact_prediction, permutation_prediction) / scale
    duplication_drift = rmse(intact_prediction, duplication_prediction) / scale
    metrics = {
        "measured": measured,
        "permutation_drift": permutation_drift,
        "duplication_drift": duplication_drift,
        "steps": trainer.global_step,
    }
    checks = {
        "Mean nRMSE below 0.25": bool(measured["nrmse"] < 0.25),
        "Permutation drift below 1e-6 target SD": bool(permutation_drift < 1e-06),
        "Duplication drift below 0.05 target SD": bool(duplication_drift < 0.05),
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3600)

# %% [markdown]
# </details>
