# %% [markdown]
# ---
# title: Can it add longer lists than it saw in training?
# categories:
# - Cardinality generalization
# evidence-id: P002
# description: Train on short lists, then ask for totals of longer lists.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Familiar lengths
#   metric:
#   - in_range_score
#   - nrmse
#   format: error
# - label: Longer lists
#   metric:
#   - unseen_score
#   - nrmse
#   format: error
# evidence-group: Aggregation
# evidence-result:
#   accelerator: cpu
#   checks:
#     All measurements finite: true
#     Seen nRMSE < 0.35, unseen nRMSE < 0.30, duplication and cardinality errors < 0.20: true
#   duration_seconds: 38.747
#   metrics:
#     duplication_error: 0.07085658887561218
#     in_range_score:
#       baseline_rmse: 1.0401230906449561
#       nrmse: 0.015460484500793705
#       rmse: 0.01608080692183399
#     probe_error: 0.04737908791419197
#     probe_prediction:
#     - 0.663047156025836
#     - 1.9400064310296836
#     - 3.7542805325034276
#     probe_target:
#     - 0.65
#     - 1.9500000000000002
#     - 3.9000000000000004
#     steps: 1100
#     unseen_score:
#       baseline_rmse: 1.7874220810107455
#       nrmse: 0.07348960132621424
#       rmse: 0.1313569361351519
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 6024779b451c8cf541076491e00245c747b33d5a6f49f81c94308defa4787d1d
#     cuda: null
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: ef37cbc5d837
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
#     script_sha256: 6359aa52144d44f2e156545c4cc0c3e23469ab74054dd0de43f0cd0248c67161
#     source_sha256: a655e13bd4ba724bb25074851fe58e02b5109eef8b55460f24e02c8ef2d2da02
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 3615
#   started_at: '2026-11-02T17:47:00.768062+00:00'
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
# //| label: fig-evidence-attention-sum-unseen-lengths
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
# Compare familiar and unfamiliar lengths. Also repeat a whole list: its total should double.
#
# ## Result
#
# {{< evidence P002 status >}}
#
# The learned total extends to the tested longer lists. This does not establish exact addition or unlimited length.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P002 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P002 script >}}
#
# ### How it works
#
# The schema matches the [in-range sum](attention-sum-in-range.html):
# learned `rf.Attention` reductions receive raw amounts without a supplied count.
# Training lengths are one through six, while the unseen test set uses seven
# through ten. Branch capacity is twelve, allowing these inputs without overflow.
#
# A second intervention duplicates complete bags of at most five items and checks
# that predictions approximately double. Equal-value probes at lengths one,
# three, and six isolate sensitivity to multiplicity. The capacity bound permits
# these shapes; the accuracy and intervention gates establish the learned behavior.
#
# ### Remaining work
#
# Repeat the gates across seeds, capacities, and nested branches. Heavy tails,
# high duplication, empty collections, missing values, and overflow semantics need
# separate coverage. The doubling gate allows approximation error; it is not an
# exact-sum contract.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""Characterize sum extrapolation beyond every trained collection length.

Require unseen-length accuracy and exact complete-bag scaling together.

Run: uv run python evidence/experiments/run.py P002"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P002"
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


def equal_value_probes(*, value: float, lengths: tuple[int, ...]) -> list[dict]:
    """Create matched bags that differ only in repetition count."""
    if not lengths or min(lengths) < 1 or max(lengths) > CAPACITY:
        raise ValueError(f"probe lengths must be within 1..{CAPACITY}, got {lengths!r}")
    rows: list[dict[str, object]] = []
    for length in lengths:
        row: dict[str, object] = {"items": [{"amount": value}] * length, "mean_amount": value, "total": length * value}
        rows.append(row)
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
    in_range = list(random_records(rows=512, seed=seed + 3))
    unseen = list(random_records(rows=512, seed=seed + 4, minimum=7, maximum=10))
    short = list(random_records(rows=256, seed=seed + 5, maximum=5))
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
    in_range_score = score(train=train, test=in_range, predicted=prediction(model, in_range))
    unseen_score = score(train=train, test=unseen, predicted=prediction(model, unseen))
    short_prediction = prediction(model, short)
    duplicated_prediction = prediction(model, duplicate(short))
    duplication_error = rmse(2.0 * short_prediction, duplicated_prediction) / float(
        np.std(np.asarray([row["total"] for row in unseen], dtype=np.float64))
    )
    probes = equal_value_probes(value=0.65, lengths=(1, 3, 6))
    probe_prediction = prediction(model, probes)
    probe_target = np.asarray([row["total"] for row in probes], dtype=np.float64)
    probe_error = rmse(probe_target, probe_prediction) / float(
        np.std(np.asarray([row["total"] for row in unseen], dtype=np.float64))
    )
    metrics = {
        "in_range_score": in_range_score,
        "unseen_score": unseen_score,
        "duplication_error": duplication_error,
        "probe_error": probe_error,
        "probe_target": probe_target.tolist(),
        "probe_prediction": probe_prediction.tolist(),
        "steps": trainer.global_step,
    }
    checks = {
        "All measurements finite": bool(
            np.isfinite([in_range_score["nrmse"], unseen_score["nrmse"], duplication_error, probe_error]).all()
        ),
        "Seen nRMSE < 0.35, unseen nRMSE < 0.30, duplication and cardinality errors < 0.20": bool(
            in_range_score["nrmse"] < 0.35
            and unseen_score["nrmse"] < 0.3
            and (duplication_error < 0.2)
            and (probe_error < 0.2)
        ),
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3615)

# %% [markdown]
# </details>
