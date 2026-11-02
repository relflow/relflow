# %% [markdown]
# ---
# title: Can it total contributions already calculated?
# categories:
# - Weighted aggregation
# evidence-id: P015
# description: Supply each item’s contribution to a total, instead of its separate value and weight.
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
#     Contribution sum nRMSE below 0.25: true
#   duration_seconds: 19.503
#   metrics:
#     measured:
#       baseline_rmse: 1.2867287797077822
#       nrmse: 0.015952727029177247
#       rmse: 0.020526832983264593
#     steps: 500
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 76e651117ca6992e288755665c557d45ae788877a12103d0380fa51fa661920d
#     cuda: null
#     evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     gpu: null
#     host: 31ee505ac853
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
#     relflow_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     relflow_source_sha256: 77953506049ed76c92adc5f32a9bcc32432a788068a4e366b2f78bcf709f29a2
#     script_sha256: d272015c8f24e16400de36426ac86dced6798ce7d18f9d95b9f05f2867d320b7
#     source_sha256: 4721078d1289a5f22e7690ad5fd2d4c11f470021134306202228049b324da560
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   seed: 3300
#   started_at: '2026-11-02T04:07:41.456773+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41386396488-1
#   candidate:
#     revision: 7dec5e860dc389c37c3642d8904867249e831773
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   baseline:
#     revision: 7dec5e860dc389c37c3642d8904867249e831773
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#   executor_revision: 349b2202e6390a5fe0cbdf09a0c5972e495351ed
#   manifest_sha256: 1f096d9b4dcdf1b2235963715e8b41c325d2a989f2cf15da7c154ddfb9bfb08d
#   passed: true
#   github_run:
#     event: workflow_dispatch
#     head_branch: main
#     head_sha: 7dec5e860dc389c37c3642d8904867249e831773
#     html_url: https://github.com/relflow/relflow/actions/runs/41386396488
#     path: .github/workflows/ci.yml
#     run_attempt: 1
#     run_id: 41386396488
#     workflow_id: 403231172
#   release: https://github.com/relflow/relflow/releases/tag/evidence-ci-41386396488-attempt-1
#   reports_sha256: fca05e4cdb79410b5090075bab03e4fc6d4f2fbc70e798f208976f4f1d8a7ff9
# ---
#
# ## Example
#
# ```yaml
# items:
#   - {contribution: 0.4}
#   - {contribution: -0.4}
#   - {contribution: 0.3}
#   - {contribution: 0.1}
#   - {contribution: -0.2}
#   - {contribution: 0.3}
# weighted_sum: 0.5
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-supplied-contribution-sum
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, items, contribution, weighted_sum. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("contribution", type: "Number"),
#   )),
#   node("weighted_sum", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Compare predictions with the correct totals on new lists of the same length.
#
# ## Result
#
# {{< evidence P015 status >}}
#
# The simpler summation task works. Fixed length means averaging and a fixed multiplier could also solve it; varying lengths are not tested.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P015 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P015 script >}}
#
# ### How it works
#
# The generator computes each `contribution = value * weight` before the
# record reaches the model. Learned `rf.Attention` reductions on the item branch
# and root combine six contributions into the masked sum target.
#
# This is a diagnostic control for the
# [raw-pair evidence](raw-value-weight-sum.html). Success establishes fixed-length
# summation of supplied products. It does not establish that the model learned
# multiplication or same-item value–weight binding.
#
# ### Remaining work
#
# Repeat the control across seeds and test variable lengths and complete-item
# duplication. Keep the raw-pair evidence alongside it so that supplying products does
# not conceal a missing learned interaction.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""A supplied item-local product isolates fixed-width sum reduction.

Attention learns a fixed-width sum after products are supplied.

Run: uv run python evidence/experiments/run.py P015"""

from __future__ import annotations

from collections.abc import Iterator

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P015"
ITEMS = 6


def weighted_records(*, rows: int, length: int, seed: int) -> Iterator[dict]:
    """Draw random value/weight pairs and expose raw fields or their products."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        values = rng.uniform(-1.0, 1.0, size=length)
        weights = rng.uniform(0.15, 1.5, size=length)
        contributions = values * weights
        items = [{"contribution": float(contribution)} for contribution in contributions]
        yield {
            "items": items,
            "weighted_sum": float(contributions.sum()),
            "weighted_mean": float(contributions.sum() / weights.sum()),
        }


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["weighted_sum"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["weighted_sum"] for row in test], dtype=np.float64)
    baseline_rmse = rmse(actual, float(np.asarray([row["weighted_sum"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline_rmse, "nrmse": measured / baseline_rmse}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(weighted_records(rows=768, length=ITEMS, seed=seed + 1))
    test = list(weighted_records(rows=384, length=ITEMS, seed=seed + 3))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        items=rf.Branch(length=ITEMS, n_layers=2, reduction=rf.Attention(n_layers=2), contribution=rf.Number),
        weighted_sum=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        train=lambda: weighted_records(rows=768, length=ITEMS, seed=seed + 1),
        validate=lambda: weighted_records(rows=192, length=ITEMS, seed=seed + 2),
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
        max_steps=500 if steps is None else min(steps, 500),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer

    # Evaluate held-out answers and retain their original labels in corruption controls.
    measured = score(train=train, test=test, predicted=prediction(model, test))
    metrics = {"measured": measured, "steps": trainer.global_step}
    checks = {"Contribution sum nRMSE below 0.25": bool(measured["nrmse"] < 0.25)}
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3300)

# %% [markdown]
# </details>
