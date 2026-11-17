# %% [markdown]
# ---
# title: Can it distinguish missing values from zero?
# categories:
# - Value state
# description: Predict a label from whether a number is missing, even though every present number is zero.
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
# evidence-diagram:
#   caption: Amber cards are hidden prediction targets. Other fields provide the input.
#   alt: Model tree with state, measurement, target. Amber cards are hidden prediction targets. Other fields provide
#     the input.
# evidence-review:
#   question: Can relflow distinguish missing values from zero?
#   modeling: Generate records in which a measurement is either null or the number zero, and hide the Boolean answer
#     identifying which case occurred. Compare preserving these states with replacing nulls by zero. Filling nulls
#     removes the information needed to distinguish the two cases; the test measures whether the model uses the preserved
#     distinction.
#   comparison:
#     kind: Controlled comparison
#     metric: Accuracy
#     conditions:
#     - label: Nulls filled with zero
#       path:
#       - prefilled_accuracy
#       format: percent
#     - label: Nulls preserved
#       path:
#       - state_accuracy
#       format: percent
#     definition: Fraction of predictions matching the expected answer, over the population named in each condition.
#     direction: Higher is better
#     domain:
#     - 0
#     - 1
#   note: ''
# evidence-plot:
#   target: target
#   focus: Does the model distinguish a missing value from a real zero?
#   comparison: Compare preserved state, nulls filled with zero, and an independent control.
#   limit: Filling null with zero removes information before the model sees the record.
#   renderer: plot
# evidence-highlight:
#   order: 3
#   title: Distinguish nulls from zero
#   result:
#     metric:
#     - state_accuracy
#     text: '**{value:.4g}% accuracy** with value states preserved'
#     scale: 100
#     minimum: 0
#     maximum: 1
#   control:
#     metric:
#     - prefilled_accuracy
#     text: '{value:.4g}% after filling nulls with zero'
#     scale: 100
#     minimum: 0
#     maximum: 1
# evidence-cost:
#   baseline: 11
#   candidate: 11
# ---
#
# ## Example
#
# The only useful distinction is whether `measurement` is present. In the illustrated signal case, a null
# measurement means the hidden label is true; a present numeric `0.0` means false. Every present number is
# zero, so comparing numeric magnitudes cannot help. This is a controlled way to ask whether the model can
# use missingness itself, rather than silently treating an explicit null as an ordinary zero.
#
# ```yaml
# measurement: null
# target: true
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```typst
# #tree(node("state", kind: "root", children: (
#   node("measurement", type: "Number", width: 150pt,),
#   node("target", kind: "target", type: "Boolean",),
# )))
# ```
#
# ## Comparison
#
# The signal model learns from data where nullness follows the label. A separately trained control receives
# the same balanced label task but independently chosen nullness, removing that relationship. A second
# intervention takes the signal model and fills all nulls with zero at evaluation time. Its positive and
# negative examples then look identical. These two controls distinguish learning a genuine validity pattern
# from guessing labels or exploiting a useful numeric value.
#
# Compare informative missingness with unrelated missingness. Then replace all nulls with zeros while retaining the original answers.
#
# ## Result
#
# Accuracy counts correct labels at the fixed 0.5 probability threshold; AUC in the details measures ranking
# independently of that threshold. High signal scores alongside chance-level independent and filled-input
# scores show why preserving the distinction matters here. Missingness is deliberately informative in this
# generator. The result does not imply that absent values are informative in every application, or that a
# model should prefer them to correctly measured data.
#
# The model can use the difference between missing and zero. Filling nulls with zeros erases that signal; whether missingness helps in real data depends on the application.
#
# ## Method
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
# The paired runs use the frozen Evidence control policy with 1,024 training, 512 validation, and
# 2,048 test rows from independent streams, with eight deterministic epochs.
#
# ### Remaining work
#
# Measure all three variants over three paired core seeds and at least ten
# calibration seeds. Retain both controls when checking stability and selecting
# the final chance bands.
#
# For the representation contract, see [data types](/concepts/data-types/).
#

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.10",
#     "lightning",
#     "numpy",
#     "pyyaml>=6",
#     "relflow>=0.5.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "reliability"
# types = ["Number"]
# conditions = ["missing-inputs", "value-states"]
# ///

"""P038: distinguish an explicit null from a real numeric zero.

Validity is the only predictive feature. Independent validity and replacing
all nulls with zeros each remove that signal while retaining the targets.
"""

import sys
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from policies import CONTROL
from reporting import experiment, report


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


class Model(rf.Model):
    measurement = rf.Number
    target = rf.Boolean(mask=True)


def fit(*, signal: bool, seed: int, steps: int | None, accelerator: str) -> rf.Model:
    lit.seed_everything(seed, workers=True)

    model = Model(preset=CONTROL.name)
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


@experiment("P038", seed=11)
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


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    from evidence.diagnostics.plots import axes_pair, dots

    m = bundle.record["metrics"]
    fig, axes = axes_pair()
    labels = ["Null state preserved", "Null filled with zero", "Independent labels"]
    dots(
        axes[0],
        labels,
        {"Ranking AUC": [m["state_auc"], m["prefilled_auc"], m["independent_auc"]]},
        xlabel="AUC (higher is better)",
        reference=(0.5, "Chance ranking"),
    )
    dots(
        axes[1],
        labels,
        {"Accuracy": [m["state_accuracy"], m["prefilled_accuracy"], m["independent_accuracy"]]},
        xlabel="Accuracy at the declared decision rule",
        reference=(0.5, "Balanced chance"),
        percent=True,
    )
    for ax in axes:
        ax.set_xlim(-0.02, 1.03)
    return (
        fig,
        "The same three information conditions are shown on separate ranking and accuracy axes. Filling missing values with zero removes the state distinction before learning.",
        {},
    )


if __name__ == "__main__":
    report(run)
