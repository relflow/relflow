# %% [markdown]
# ---
# title: Can it subtract an average supplied as input?
# categories:
# - Peer-relative inference
# evidence-id: P034
# description: Give each item both its value and its peer average, then predict their difference.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Difference prediction
#   metric:
#   - intact_nrmse
#   format: error
# evidence-group: Relational
# evidence-result:
#   accelerator: cpu
#   checks:
#     Baseline RMSE > 0.000001: true
#     Finite calibration and model metrics: true
#     Supplied-mean subtraction nRMSE < 0.25: true
#   duration_seconds: 100.85
#   metrics:
#     baseline_rmse: 0.7110739767923704
#     intact_nrmse: 0.02569433225137723
#     intact_rmse: 0.018270571015011266
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 6fc72e3f92b33400c643808017b5b4a54b3dd443c5167e31c0aca80b5bb95e53
#     cuda: null
#     evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     gpu: null
#     host: 79561056ee96
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
#     script_sha256: 026c13a3c6d750042e5c0c25261975412fe03bf2ab3442ac035739da7c665eab
#     source_sha256: 838c467b48cf5a4e28367cd8c88d23c478519ad87d62a553c9266fdbe376aae1
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   seed: 3700
#   started_at: '2026-11-02T04:15:58.802269+00:00'
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
#   - {value: 1.5, peer_mean: 2.5, deviation: -1.0}
#   - {value: 1.9, peer_mean: 2.5, deviation: -0.6}
#   - {value: 2.3, peer_mean: 2.5, deviation: -0.2}
#   - {value: 2.7, peer_mean: 2.5, deviation: 0.2}
#   - {value: 3.1, peer_mean: 2.5, deviation: 0.6}
#   - {value: 3.5, peer_mean: 2.5, deviation: 1.0}
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-supplied-peer-mean-control
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with collection, items, value, peer_mean, deviation. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("collection", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("value", type: "Number"),
#     node("peer_mean", type: "Number"),
#     node("deviation", kind: "target", type: "Number", width: 150pt,),
#   )),
# )))
# ```
#
# ## Comparison
#
# Compare predictions with the correct differences on fresh records.
#
# ## Result
#
# {{< evidence P034 status >}}
#
# Local subtraction works when both numbers are supplied. This does not show that the model can calculate the average or choose the right peers.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P034 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P034 script >}}
#
# ### How it works
#
# The repeated decoder can use visible siblings at the same coordinate. Since
# both operands are already supplied, a failure would point toward local
# arithmetic or repeated output routing before collection aggregation is tested.
# The generator varies the collection's location independently of its centered
# deviations, so a constant or raw-value shortcut is inadequate.
#
# ### Remaining work
#
# Repeat three core seeds and ten calibration seeds, then test missing operands
# and variable lengths. The [raw ungrouped case](ungrouped-peer-deviation.html)
# removes the supplied mean to test learned aggregation.
#
# ### Complete experiment code
#

# %%
"""Subtract a supplied peer mean and write each result to its item coordinate.

The mean is deliberately provided beside every value. This diagnostic removes
aggregation and peer selection, isolating local subtraction and repeated
prediction writeback.

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

PROOF_ID = "P034"
ITEMS = 6


def records(*, rows: int, seed: int) -> Iterator[dict]:
    """Generate collections with independent locations and centered residuals."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        basis = rng.normal(size=ITEMS)
        basis -= basis.mean()
        basis /= np.sqrt(np.mean(np.square(basis)))
        scale = float(rng.uniform(0.25, 1.1))
        mean = float(rng.uniform(-3.5, 3.5))
        deviations = scale * basis
        values = mean + deviations
        items: list[dict[str, float]] = []
        for value, deviation in zip(values, deviations, strict=True):
            item = {"value": float(value), "deviation": float(deviation)}
            item["peer_mean"] = mean
            items.append(item)
        yield {"items": items}


def targets(rows: list[dict]) -> np.ndarray:
    return np.asarray([item["deviation"] for row in rows for item in row["items"]])


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    inputs = [
        {"items": [{key: value for key, value in item.items() if key != "deviation"} for item in row["items"]]}
        for row in rows
    ]
    output = model.predict(inputs).to_pylist()
    return np.asarray([item["deviation"]["content"] for row in output for item in row["predictions"]["items"]])


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=48,
        n_layers=3,
        n_heads=4,
        reduction=None,
        items=rf.Branch(
            length=ITEMS,
            overflow="error",
            n_layers=2,
            reduction=None,
            value=rf.Number,
            peer_mean=rf.Number,
            deviation=rf.Number(mask=True, objective="mse", n_linear=2),
        ),
    )
    data = rf.SyntheticDataModule(
        train=partial(records, rows=1024, seed=seed + 1),
        validate=partial(records, rows=256, seed=seed + 2),
        seed=seed,
    )
    model.fit(
        data,
        batch_size=128,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
        accelerator=accelerator,
        max_steps=600 if steps is None else min(steps, 600),
        max_epochs=-1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    train = list(records(rows=1024, seed=seed + 1))
    test = list(records(rows=512, seed=seed + 3))
    actual = targets(test)
    baseline = rmse(actual, float(targets(train).mean()))
    predicted = predict(model, test)
    intact_error = rmse(actual, predicted)
    intact_nrmse = intact_error / baseline
    metrics = {"intact_rmse": intact_error, "baseline_rmse": baseline, "intact_nrmse": intact_nrmse}
    checks = {"Baseline RMSE > 0.000001": baseline > 1e-6}
    checks["Supplied-mean subtraction nRMSE < 0.25"] = intact_nrmse < 0.25
    checks["Finite calibration and model metrics"] = bool(np.isfinite(list(metrics.values())).all())
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3700)

# %% [markdown]
# </details>
