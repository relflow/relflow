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
#   duration_seconds: 16.216
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
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: 7c190352b4f1
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
#     script_sha256: b023c626ab3e0f0e1db82ea6f3399bf7222538c00260adb2f1aaa067bc2a3da6
#     source_sha256: 409b3c1bc23b25d681fd05078babec3a2f1e1b6204d78a9507e68b6dc7c3a9de
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 2600
#   started_at: '2026-11-02T17:52:29.993474+00:00'
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
