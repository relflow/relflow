# %% [markdown]
# ---
# title: Can it distinguish an unknown set from an empty set?
# categories:
# - Vocabulary
# description: Predict whether a set is nonempty, including sets whose member names were absent from training.
# evidence-readout:
# - label: Unknown-input error, fallback trained
#   metric:
#   - augmented_unknown
#   format: number
# - label: Unknown-input error, fallback untrained
#   metric:
#   - control_unknown
#   format: number
# evidence-group: Vocabulary
# evidence-diagram:
#   caption: Amber cards are hidden prediction targets. Other fields provide the input.
#   alt: Model tree with record, tags, nonempty. Amber cards are hidden prediction targets. Other fields provide the
#     input.
# evidence-review:
#   question: Can relflow distinguish an unknown set from an empty set?
#   modeling: Predict whether a set is nonempty. Train one model with input members sometimes made unavailable and
#     another without that augmentation. Evaluate both on empty sets and sets containing only names absent from training,
#     using the same hidden answers.
#   comparison:
#     kind: Controlled comparison
#     metric: RMSE
#     conditions:
#     - label: Unknown-input error, fallback untrained
#       path:
#       - control_unknown
#       format: number
#     - label: Unknown-input error, fallback trained
#       path:
#       - augmented_unknown
#       format: number
#     definition: Root mean squared prediction error, in the target’s original units. Zero is an exact prediction.
#     direction: Lower error is better
#   note: The primary comparison is RMSE on unseen-member inputs. The task is nonemptiness, not recovering the identities
#     of unseen members.
# evidence-plot:
#   target: nonempty
#   focus: Can training exposure teach a useful unavailable-set fallback?
#   comparison: Compare exposed and unexposed fallback arms on known and unknown members.
#   limit: Identity and duplicate-member interventions are checked separately from fallback prediction error.
#   renderer: plot
# evidence-cost:
#   baseline: 25
#   candidate: 25
# ---
#
# The prediction target is deliberately simple: does the input contain at least one member? Individual
# tag names have no special meanings in this rule. That makes it possible to test whether a learned
# unavailable-membership representation preserves the fact that something was present even when none
# of its names are in the vocabulary.
#
# ## Example
#
# ```yaml
# tags: [tag-12]
# nonempty: 1.0
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```typst
# #tree(node("record", kind: "root", children: (
#   node("tags", type: "Set", detail: "Known or unavailable membership"),
#   node("nonempty", kind: "target", type: "Number"),
# )))
# ```
#
# The illustrated one-member set has target 1. An empty list has target 0, and changing `tag-12` to a
# completely new name still leaves the reference answer at 1. Repeating that new member also leaves
# the answer unchanged. These are illustrative examples of the generator rule, rather than a request
# to recover an unseen tag's individual effect.
#
# ## Comparison
#
# Compare training with input members sometimes made unavailable against training that never exposes this condition. Repeat unknown members to check duplicates.
#
# Both models learn from familiar empty and nonempty sets. Only the augmented arm sometimes loses the
# input members' familiar content during training, while the nonempty target stays positive. It
# therefore gets examples of the fallback condition before evaluation. The control arm can solve the
# known-name task without learning how to interpret that unfamiliar condition.
#
# ## Result
#
# The trained fallback distinguishes unknown nonempty input from empty input. It preserves the presence of members, rather than learning their individual meanings.
#
# RMSE is measured directly on the zero/one target. The constant 0.5 forecast has error 0.5, giving a
# useful reference for judging transfer to new names. The comparison tests empty versus nonempty
# information, not counting members, identifying them, or reconstructing their meanings.
# Frozen-vocabulary checks ensure that the unknown-name test is not converted into another training
# phase.
#
# ## Method
#
# ### Training and controls
#
# Each arm trains for 400 AdamW updates on 8,192 rows, with 512 independent
# validation rows. Evaluate 2,048 known and disjoint-label rows. The constant
# 0.5 predictor has RMSE 0.5. Both arms must learn familiar sets (RMSE < 0.10).
# Only the augmented arm must transfer to unknown sets (RMSE < 0.10); its
# gain over the untrained-fallback control must exceed 0.40 RMSE. Duplicate
# and unknown-spelling changes must leave predictions unchanged.
#

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.10",
#     "lightning",
#     "numpy",
#     "polars",
#     "pyarrow",
#     "pyyaml>=6",
#     "relflow>=0.5.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "reliability"
# types = ["Set"]
# conditions = ["unseen-values", "missing-inputs", "value-states"]
# ///

"""P064: input corruption teaches an unavailable Set representation."""

import sys
from collections.abc import Iterator
from functools import partial
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
from reporting import experiment, report

BUDGET = 400


def records(*, rows: int, seed: int, novel: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for label, nonempty in zip(rng.integers(32, size=rows), rng.random(rows) < 0.5, strict=True):
        yield {"tags": [f"{'novel' if novel else 'tag'}-{label}"] if nonempty else [], "nonempty": float(nonempty)}


def build(p_unavailable: float) -> rf.Model:
    class PredictionModel(rf.Model):
        tags = rf.Set(p_unavailable=p_unavailable)
        nonempty = rf.Number(mask=True, objective="mse")

    model = PredictionModel(d_model=32, n_layers=1, n_heads=4, dropout=0.0)
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [
            row["nonempty"]["content"]
            for row in pl.DataFrame(model.predict(pa.Table.from_pylist(rows)))["predictions"].to_list()
        ]
    )


@experiment("P064", seed=7641)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    metrics, checks = {"constant_rmse": 0.5}, {}
    for probability in (0.0, 0.5):
        lit.seed_everything(seed, workers=True)
        model = build(probability)
        data = rf.SyntheticDataModule(
            train=partial(records, rows=8192, seed=seed + 1),
            validate=partial(records, rows=512, seed=seed + 2),
            seed=seed,
        )
        model.fit(
            data,
            batch_size=64,
            learning_rate=0.002,
            early_stopping=False,
            rollback=False,
            precision="32-true",
            strategy="auto",
            accelerator=accelerator,
            devices=1,
            max_epochs=-1,
            max_steps=BUDGET if steps is None else min(steps, BUDGET),
            deterministic=True,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        vocabulary = rf.Set.vocabulary(model, "/tags")
        arm = "augmented" if probability else "control"
        for novel in (False, True):
            rows = list(records(rows=2048, seed=seed + 3, novel=novel))
            expected = np.asarray([row["nonempty"] for row in rows])
            values = predict(model, rows)
            key = f"{arm}_{'unknown' if novel else 'known'}"
            metrics[key] = float(np.sqrt(np.mean((values - expected) ** 2)))
            if not novel or probability:
                checks[f"{key}: RMSE below 0.10"] = metrics[key] < 0.10
            if novel:
                changed = [{**row, "tags": ["another", "another"] if row["tags"] else []} for row in rows]
                drift = float(np.max(np.abs(values - predict(model, changed))))
                metrics[f"{arm}_duplicate_and_identity_drift"] = drift
                checks[f"{arm}: unknown spelling and duplicates are invariant"] = drift < 1e-5
        checks[f"{arm}: unknown evaluation leaves vocabulary frozen"] = rf.Set.vocabulary(model, "/tags") == vocabulary
    checks["Trained fallback improves unknown-input RMSE by more than 0.40"] = (
        metrics["control_unknown"] - metrics["augmented_unknown"] > 0.40
    )
    return metrics, checks


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    from evidence.diagnostics.plots import axes_pair, dots

    m = bundle.record["metrics"]
    fig, axes = axes_pair()
    labels = ["Fallback exposed during training", "Fallback unexposed"]
    dots(
        axes[0],
        labels,
        {
            "Known members": [m["augmented_known"], m["control_known"]],
            "Unknown members": [m["augmented_unknown"], m["control_unknown"]],
        },
        xlabel="RMSE against the generator's answer",
        reference=(m["constant_rmse"], "Constant-guess RMSE"),
    )
    dots(
        axes[1],
        labels,
        {
            "Duplicate/identity drift": [
                m["augmented_duplicate_and_identity_drift"],
                m["control_duplicate_and_identity_drift"],
            ]
        },
        xlabel="Maximum output change under the declared invariant",
    )
    return (
        fig,
        "Prediction error tests whether fallback behavior is useful. The separate drift panel tests whether renaming unknown members or duplicating them changes that behavior.",
        {},
    )


if __name__ == "__main__":
    report(run)
