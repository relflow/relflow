# %% [markdown]
# ---
# title: Can it add six numbers?
# categories:
# - Hierarchical statistics
# evidence-id: P039
# description: Predict the total of six independently chosen amounts.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Total prediction
#   metric:
#   - test_nrmse
#   format: error
# evidence-group: Structure
# evidence-result:
#   accelerator: cpu
#   checks:
#     Test normalized RMSE is below 0.25: true
#   duration_seconds: 15.392
#   metrics:
#     baseline_rmse: 1.395327666226233
#     steps: 400
#     target_std: 1.3947314823294805
#     test_nrmse: 0.013546748643388798
#     test_rmse: 0.01890215316953308
#     validate_rmse: 0.02033274876095201
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 7afe6db365453b1f98f918eec33372cac240249aa2f4085850cdec3a6db2ef3f
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 290507ebded3
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
#     script_sha256: 54ff39581aa94ecd4a096edc94231a009873cf8824a78b565c6f7233d5e86523
#     source_sha256: 4c2356652a0e5cf3fbd3d080c3e54ca599b7648ce95247f4766f3f2b25656167
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 2600
#   started_at: '2026-11-01T01:45:27.943358+00:00'
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
# transactions:
#   - amount: 0.8
#   - amount: -0.4
#   - amount: 0.2
#   - amount: 0.5
#   - amount: -0.5
#   - amount: 0.9
# global_total: 1.5
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-structure-attention-learns-fixed-width-sum
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, transactions, amount, global_total. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("transactions", kind: "branch", repeated: true, width: 120pt, children: (
#     node("amount", type: "Number"),
#   )),
#   node("global_total", kind: "target", type: "Number",),
# )))
# ```
#
# ## Comparison
#
# Compare test error with a simple predictor that always returns the training-set average total.
#
# ## Result
#
# {{< evidence P039 status >}}
#
# The model learns this small sum. This baseline does not test item reordering, varying lengths, deeper nesting, or exact addition.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P039 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P039 script >}}
#
# ### How it works
#
# Both the transaction branch and the root use `rf.Attention` reduction.
# The model must transform the visible amounts into a summary from which a
# numerical decoder can approximate their sum. Fixed length makes this a
# simpler control than inferring totals through several variable-size groups.
#
# Training, validation, and test draw 512, 128, and 256 independent observations.
# Each amount is sampled uniformly from −1 to 1. After 400 deterministic
# steps, the model predicts totals from the transaction inputs alone.
#
# ### Remaining work
#
# Add the nested, regrouping-invariant `global_total` case: the present control
# has only one collection level. The family also calls for a
# `largest_session_average` target, broader value patterns, multiple Attention
# output counts, and nested capacity ranges, followed by repeated-seed gates.
#
# The [nested-cardinality evidence](attention-preserves-nested-cardinality.html)
# tests a statistic whose answer changes when those same amounts are regrouped.
# Use [preprocessing](../../guides/preprocessing.qmd) when a sum must be exact.
#
# ### Complete experiment code
#

# %%
"""P039: learn a fixed-width sum through Attention reduction.

All six independent values must contribute. Test RMSE is compared with the
constant training-mean predictor; learned sums remain approximations.
"""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P039"


def records(*, rows: int, seed: int) -> Iterator[dict]:
    """Yield six transactions and their total."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        amounts = rng.uniform(-1.0, 1.0, size=6)
        yield {
            "transactions": [{"amount": float(amount)} for amount in amounts],
            "global_total": float(amounts.sum()),
        }


def prediction(model: rf.Model, rows: list[dict], source: str, target: str) -> np.ndarray:
    """Predict one hidden scalar from its visible repeated context."""
    inputs = [{source: row[source]} for row in rows]
    output = model.predict(inputs).to_pylist()
    return np.asarray([row["predictions"][target]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    data_seed = (seed - 2600) % (2**32)
    train = partial(records, rows=512, seed=data_seed + 11)
    validate = partial(records, rows=128, seed=data_seed + 22)
    test = list(records(rows=256, seed=data_seed + 33))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        transactions=rf.Branch(
            length=6,
            n_layers=2,
            reduction=rf.Attention(n_layers=2),
            amount=rf.Number,
        ),
        global_total=rf.Number(mask=True, objective="mse"),
    )
    data = rf.SyntheticDataModule(train=train, validate=validate, seed=seed)
    model.fit(
        data,
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
        accelerator=accelerator,
        max_epochs=-1,
        max_steps=steps if steps is not None else 400,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer = model.trainer

    train_target = np.asarray([row["global_total"] for row in train()])
    validate_rows = list(validate())
    validate_target = np.asarray([row["global_total"] for row in validate_rows])
    test_target = np.asarray([row["global_total"] for row in test])
    baseline = rmse(test_target, float(train_target.mean()))
    validate_rmse = rmse(validate_target, prediction(model, validate_rows, "transactions", "global_total"))
    test_rmse = rmse(test_target, prediction(model, test, "transactions", "global_total"))
    nrmse = test_rmse / baseline
    return {
        "steps": trainer.global_step,
        "baseline_rmse": baseline,
        "validate_rmse": validate_rmse,
        "test_rmse": test_rmse,
        "test_nrmse": nrmse,
        "target_std": float(test_target.std()),
    }, {"Test normalized RMSE is below 0.25": nrmse < 0.25}


if __name__ == "__main__":
    report(PROOF_ID, run, seed=2600)

# %% [markdown]
# </details>
