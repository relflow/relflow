# %% [markdown]
# ---
# title: Can it distinguish missing values from zero?
# categories:
# - Value state
# proof-id: P038
# description: Predict a label from whether a number is missing, even though every present number is zero.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: Nulls preserved
#   metric:
#   - state_accuracy
#   format: percent
# - label: Nulls filled with zero
#   metric:
#   - prefilled_accuracy
#   format: percent
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
# //| label: fig-proof-state-null-versus-zero
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
# {{< proof P038 status >}}
#
# The model can use the difference between missing and zero. Filling nulls with zeros erases that signal; whether missingness helps in real data depends on the application.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P038 evidence >}}
#
# ### Run this experiment
#
# {{< proof P038 script >}}
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
# For the representation contract, see [Data Types](../../core-concepts/data-types.qmd).
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
import torch
from reporting import report

import relflow as rf

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
    data = rf.SyntheticDataModule(model=model, test=records)
    trainer = lit.Trainer(
        accelerator=accelerator,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    metrics = trainer.test(model=model, datamodule=data, verbose=False)[0]
    return float(metrics[".target/test.auc.content"]), float(metrics[".target/test.accuracy@0.5.content"])


def fit(*, signal: bool, seed: int, steps: int | None, accelerator: str) -> rf.Model:
    lit.seed_everything(seed, workers=True)
    model = rf.Model.xs(
        batch_size=128,
        measurement=rf.Number,
        target=rf.Boolean(mask=True),
    )
    model.optimizer = lambda module: torch.optim.AdamW(module.parameters(), lr=5e-3)
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, rows=1024, seed=seed + 1, signal=signal),
        validate=partial(records, rows=512, seed=seed + 2, signal=signal),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        max_epochs=8,
        max_steps=steps if steps is not None else -1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model=model, datamodule=data)
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
