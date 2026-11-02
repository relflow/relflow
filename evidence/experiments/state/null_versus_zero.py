# %% [markdown]
# ---
# title: Can it distinguish missing values from zero?
# categories:
# - Value state
# evidence-id: P038
# description: Predict a label from whether a number is missing, even though every present number is zero.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Nulls preserved
#   metric:
#   - state_accuracy
#   format: percent
# - label: Nulls filled with zero
#   metric:
#   - prefilled_accuracy
#   format: percent
# evidence-group: State
# evidence-result:
#   accelerator: cpu
#   checks:
#     Independent validity AUC remains between 0.42 and 0.58: true
#     Independent validity accuracy remains between 0.45 and 0.55: true
#     Null state reaches 0.98 accuracy: true
#     Null state reaches 0.99 AUC: true
#     Null-to-zero AUC remains between 0.42 and 0.58: true
#     Null-to-zero accuracy remains between 0.45 and 0.55: true
#     State exceeds its control by at least 0.40 AUC: true
#   duration_seconds: 5.048
#   metrics:
#     auc_gap: 0.49365234375
#     independent_accuracy: 0.5
#     independent_auc: 0.50634765625
#     prefilled_accuracy: 0.5
#     prefilled_auc: 0.5
#     state_accuracy: 1.0
#     state_auc: 1.0
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: af827d6177c6448f76ecf7d5b24778519a941b4e55ef909adccec516663b6fa1
#     cuda: null
#     evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     gpu: null
#     host: a8bd784b034f
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
#     script_sha256: 5d16666beeb56e941170df51ea4ea3d17869b8d226ff30c86e59fbc62111450c
#     source_sha256: 742733c386b9eedd039790212757024194edab5613fd5f1e2b0318ed435efb36
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   seed: 11
#   started_at: '2026-11-02T04:17:45.513290+00:00'
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
# measurement: null
# target: true
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-state-null-versus-zero
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with state, measurement, target. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("state", kind: "root", children: (
#   node("measurement", type: "Number", width: 150pt,),
#   node("target", kind: "target", type: "Boolean",),
# )))
# ```
#
# ## Comparison
#
# Compare informative missingness with unrelated missingness. Then replace all nulls with zeros while retaining the original answers.
#
# ## Result
#
# {{< evidence P038 status >}}
#
# The model can use the difference between missing and zero. Filling nulls with zeros erases that signal; whether missingness helps in real data depends on the application.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P038 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P038 script >}}
#
# ### How it works
#
# relflow represents a Number's state separately from its continuous content.
# The model can therefore distinguish an Arrow null from a valued zero without
# any variation among the actual numbers.
#
# Two controls probe this explanation. A separate training run makes validity
# independent of the label. An inference intervention replaces every null with
# zero in the signal model's test set, while retaining the original labels.
# Both should remove the useful information.
#
# The paired runs use the `xs` preset with 1,024 training, 512 validation, and
# 2,048 test rows from independent streams, with eight deterministic epochs.
#
# ### Remaining work
#
# Measure all three variants over three paired core seeds and at least ten
# calibration seeds. Retain both controls when checking stability and selecting
# the final chance bands.
#
# For the representation contract, see [Data Types](../../datatypes/index.qmd).
#
# ### Complete experiment code
#

# %%
"""P038: distinguish an explicit null from a real numeric zero.

Validity is the only predictive feature. Independent validity and replacing
all nulls with zeros each remove that signal while retaining the targets.
"""

from collections.abc import Callable, Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P038"


def records(*, rows: int, seed: int, signal: bool, fill_nulls: bool = False) -> Iterator[dict]:
    """Yield zero-valued measurements with predictive or independent nulls."""
    rng = np.random.default_rng(seed)
    target = np.tile(np.array([False, True]), (rows + 1) // 2)[:rows]
    rng.shuffle(target)
    missing = target if signal else rng.integers(0, 2, size=rows).astype(bool)
    for index in range(rows):
        yield {
            "measurement": None if missing[index] and not fill_nulls else 0.0,
            "target": bool(target[index]),
        }


def score(model: rf.Model, records: Callable[[], Iterator[dict]], accelerator: str) -> tuple[float, float]:
    """Measure held-out AUC and accuracy at the fixed 0.5 threshold."""
    data = rf.SyntheticDataModule(test=records)
    metrics = model.test(
        data,
        accelerator=accelerator,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        verbose=False,
        batch_size=128,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
    )[0]
    return float(metrics[".target/test.auc.content"]), float(metrics[".target/test.accuracy@0.5.content"])


def fit(*, signal: bool, seed: int, steps: int | None, accelerator: str) -> rf.Model:
    lit.seed_everything(seed, workers=True)
    model = rf.Model.xs(
        measurement=rf.Number,
        target=rf.Boolean(mask=True),
    )
    data = rf.SyntheticDataModule(
        train=partial(records, rows=1024, seed=seed + 1, signal=signal),
        validate=partial(records, rows=512, seed=seed + 2, signal=signal),
        seed=seed,
    )
    model.fit(
        data,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.005),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
        accelerator=accelerator,
        max_epochs=8,
        max_steps=steps if steps is not None else -1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    return model


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    signal_model = fit(signal=True, seed=seed, steps=steps, accelerator=accelerator)
    control_model = fit(signal=False, seed=seed, steps=steps, accelerator=accelerator)
    state_auc, state_accuracy = score(
        signal_model, partial(records, rows=2048, seed=seed + 3, signal=True), accelerator
    )
    control_auc, control_accuracy = score(
        control_model, partial(records, rows=2048, seed=seed + 3, signal=False), accelerator
    )
    prefilled_auc, prefilled_accuracy = score(
        signal_model, partial(records, rows=2048, seed=seed + 3, signal=True, fill_nulls=True), accelerator
    )
    return {
        "state_auc": state_auc,
        "state_accuracy": state_accuracy,
        "independent_auc": control_auc,
        "independent_accuracy": control_accuracy,
        "prefilled_auc": prefilled_auc,
        "prefilled_accuracy": prefilled_accuracy,
        "auc_gap": state_auc - control_auc,
    }, {
        "Null state reaches 0.99 AUC": state_auc >= 0.99,
        "Null state reaches 0.98 accuracy": state_accuracy >= 0.98,
        "Independent validity AUC remains between 0.42 and 0.58": 0.42 <= control_auc <= 0.58,
        "Independent validity accuracy remains between 0.45 and 0.55": 0.45 <= control_accuracy <= 0.55,
        "Null-to-zero AUC remains between 0.42 and 0.58": 0.42 <= prefilled_auc <= 0.58,
        "Null-to-zero accuracy remains between 0.45 and 0.55": 0.45 <= prefilled_accuracy <= 0.55,
        "State exceeds its control by at least 0.40 AUC": state_auc - control_auc >= 0.40,
    }


if __name__ == "__main__":
    report(PROOF_ID, run, seed=11)

# %% [markdown]
# </details>
