# %% [markdown]
# ---
# title: Can an average lose the total?
# categories:
# - Hierarchical statistics
# evidence-id: P041
# description: Predict both average and total after an average-only branch with item attention disabled; the branch's
#   count lane keeps the total recoverable.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Average prediction
#   metric:
#   - mean_nrmse
#   format: error
# - label: Total prediction
#   metric:
#   - total_nrmse
#   format: error
# - label: 'Predicted total, six copies (correct: 7.5)'
#   metric:
#   - paired_total_predictions
#   - 1
#   format: number
# evidence-group: Structure
# evidence-result:
#   accelerator: cpu
#   checks:
#     Mean normalized RMSE is below 0.25: true
#     Paired totals land within 0.5 of their correct values: true
#     Six equal values predict at least four times the one-value total: true
#     Total normalized RMSE is below 0.35: true
#   duration_seconds: 8.299
#   metrics:
#     mean_baseline_rmse: 0.2930153390602116
#     mean_nrmse: 0.007615575141951133
#     mean_rmse: 0.00223148033235733
#     paired_total_error:
#     - -0.0286342794738097
#     - -0.05227024370365818
#     paired_total_predictions:
#     - 1.2213657205261903
#     - 7.447729756296342
#     steps: 300
#     total_baseline_rmse: 2.006717852522177
#     total_nrmse: 0.017467889694615556
#     total_rmse: 0.035053126096073195
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 9cbf7ca86ebf1cca056ae1ec0a6fdfba5585ea0683be658221358c0760281f70
#     cuda: null
#     evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     gpu: null
#     host: b0a1e48d0342
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
#     script_sha256: b97ae0a19e5c0838f91ec690379609a47e16198705e203e470a9556118ff0364
#     source_sha256: 1c8e30e74a9376644c7bc909ea243e32fb87e41e5ba94de7c50998570420d6ab
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   seed: 2599
#   started_at: '2026-11-02T04:18:16.300622+00:00'
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
# transactions:
#   - amount: 1.25
# mean_amount: 1.25
# global_total: 1.25
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-structure-mean-keeps-cardinality
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, transactions, amount, mean_amount, global_total. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", children: (
#   node("transactions", kind: "branch", repeated: true, width: 120pt, children: (
#     node("amount", type: "Number"),
#   )),
#   node("mean_amount", kind: "target", type: "Number",),
#   node("global_total", kind: "target", type: "Number",),
# )))
# ```
#
# ## Comparison
#
# Compare one amount with six copies of the same amount. Their averages match, but their totals differ.
#
# ## Result
#
# {{< evidence P041 status >}}
#
# The summary keeps the count: one and six copies produce different totals that track the correct values, while the average stays shared.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P041 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P041 script >}}
#
# ### How it works
#
# The transaction branch uses `attention=None, reduction=rf.Mean()`. Its
# identical encoded amounts are averaged without preceding branch attention
# introducing position information, and Mean adds a learned direction scaled
# by the number of present items over the branch capacity. The parent sees the
# same averaged content for one and six repeats and reads their counts from
# that lane, so it can recover their different totals.
#
# The synthetic training process varies the number of repeated equal amounts
# from one to six and varies the shared amount across observations. After 300
# deterministic steps, evaluation checks the average and the total on 512
# held-out rows and compares predictions for the explicit one-versus-six pair.
# Training and validation contain 1,024 and 256 independently generated rows.
#
# ### Remaining work
#
# Broaden the value patterns and nested partitions. The family still needs a
# nested `global_total` task and a `largest_session_average` target, then three
# paired core seeds and at least ten calibration seeds for the learned gates.
#
# Use [preprocessing](../../guides/preprocessing.qmd) for exact business
# arithmetic. For learned hierarchy compression, the
# [nested Attention case](attention-preserves-nested-cardinality.html) tests a
# route that retains cardinality information through attention instead.
#
# ### Complete experiment code
#

# %%
"""P041: learn an average and a total from repeated equal values through Mean's count lane.

With branch attention disabled, Mean averages one and six identical inputs to
the same content and tells them apart only through its count lane. A
downstream sum must read that lane to recover their different totals.
"""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P041"


def records(*, rows: int, seed: int) -> Iterator[dict]:
    """Yield variable-length bags of equal values with mean and sum targets."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        count = int(rng.integers(1, 7))
        amount = float(rng.uniform(0.5, 1.5))
        yield {
            "transactions": [{"amount": amount} for _ in range(count)],
            "mean_amount": amount,
            "global_total": count * amount,
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
    data_seed = (seed - 2599) % (2**32)
    train = partial(records, rows=1024, seed=data_seed + 1)
    validate = partial(records, rows=256, seed=data_seed + 2)
    test = list(records(rows=512, seed=data_seed + 3))
    model = rf.Model(
        d_model=24,
        n_layers=1,
        n_heads=4,
        transactions=rf.Branch(length=6, attention=None, reduction=rf.Mean(), amount=rf.Number),
        mean_amount=rf.Number(mask=True, objective="mse"),
        global_total=rf.Number(mask=True, objective="mse"),
    )
    data = rf.SyntheticDataModule(train=train, validate=validate, seed=seed)
    model.fit(
        data,
        batch_size=64,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.003),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
        accelerator=accelerator,
        max_epochs=-1,
        max_steps=steps if steps is not None else 300,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer = model.trainer

    training = list(train())
    scores = {}
    for target in ("mean_amount", "global_total"):
        actual = np.asarray([row[target] for row in test])
        baseline = rmse(actual, float(np.mean([row[target] for row in training])))
        measured = rmse(actual, prediction(model, test, "transactions", target))
        scores[target] = {"baseline_rmse": baseline, "rmse": measured, "nrmse": measured / baseline}
    paired = [
        {"transactions": [{"amount": 1.25}]},
        {"transactions": [{"amount": 1.25} for _ in range(6)]},
    ]
    paired_total = prediction(model, paired, "transactions", "global_total")
    correct = np.asarray([1.25, 7.5])
    return {
        "steps": trainer.global_step,
        "mean_baseline_rmse": scores["mean_amount"]["baseline_rmse"],
        "mean_rmse": scores["mean_amount"]["rmse"],
        "mean_nrmse": scores["mean_amount"]["nrmse"],
        "total_baseline_rmse": scores["global_total"]["baseline_rmse"],
        "total_rmse": scores["global_total"]["rmse"],
        "total_nrmse": scores["global_total"]["nrmse"],
        "paired_total_predictions": paired_total.tolist(),
        "paired_total_error": (paired_total - correct).tolist(),
    }, {
        "Mean normalized RMSE is below 0.25": scores["mean_amount"]["nrmse"] < 0.25,
        "Total normalized RMSE is below 0.35": scores["global_total"]["nrmse"] < 0.35,
        "Six equal values predict at least four times the one-value total": bool(
            paired_total[1] > 4.0 * paired_total[0] > 0.0
        ),
        "Paired totals land within 0.5 of their correct values": bool(np.all(np.abs(paired_total - correct) < 0.5)),
    }


if __name__ == "__main__":
    report(PROOF_ID, run, seed=2599)

# %% [markdown]
# </details>
