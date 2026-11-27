# %% [markdown]
# ---
# title: Can it tell a recorded zero from an absent or invalid reading?
# description: Predict three labels from a numeric zero, an Arrow null, or a valid floating-point NaN, preserving
#   their distinct states.
# evidence-readout:
# - label: Intact accuracy
#   metric:
#   - arms
#   - intact
#   - balanced
#   - accuracy
#   format: percent
# - label: Coalesced accuracy
#   metric:
#   - arms
#   - coalesced
#   - balanced
#   - accuracy
#   format: percent
# - label: Filled accuracy
#   metric:
#   - arms
#   - filled
#   - balanced
#   - accuracy
#   format: percent
# evidence-group: State
# evidence-review:
#   question: Can relflow tell a recorded zero from an absent or invalid reading?
#   modeling: Generate three balanced classes identified solely by a numeric zero, null, or valid NaN. Hide the class
#     label. Compare an intact model with separately trained controls that merge NaN into null or replace every input
#     with zero.
#   comparison:
#     metric: Accuracy
#     kind: Controlled comparison
#     conditions:
#     - label: Every input filled with zero
#       path:
#       - arms
#       - filled
#       - balanced
#       - accuracy
#       format: percent
#     - label: NaN merged into null
#       path:
#       - arms
#       - coalesced
#       - balanced
#       - accuracy
#       format: percent
#     - label: All three states preserved
#       path:
#       - arms
#       - intact
#       - balanced
#       - accuracy
#       format: percent
#     definition: Fraction of predictions matching the expected answer, over the population named in each condition.
#     direction: Higher is better
#     domain:
#     - 0
#     - 1
#   note: The controls intentionally remove information. Separate recall checks repeat the intact test under changed
#     class proportions so majority accuracy cannot hide a failed state.
# evidence-plot:
#   target: target
#   focus: Are zero, null, and invalid numbers distinguishable?
#   comparison: Use per-state recall and confusion matrices beside the input-normalization controls.
#   limit: A macro score cannot hide the collapse of one state into another.
#   labels:
#     test: Balanced state population
#     shifted: Shifted state frequencies
#   mode: classification
# ---
#
# Predict three labels from a numeric zero, an Arrow null, or a valid floating-point NaN, preserving
# their distinct states.
#
# ## Example
#
# ```yaml
# measurement: .nan
# target: invalid
# ```
#
# Here the instrument supplied a value, but the value is NaN. That is different from a missing
# instrument reading, and different again from a valid zero. The target label is hidden during
# prediction.
#
# The typed Arrow construction explicitly retains NaN as a valid floating-point entry. A data adapter
# that converts NaN into a null would erase the distinction before the model could learn it.
#
# ## Comparison
#
# Compare intact states with separately trained controls that coalesce NaN into null or fill every input
# with zero. With balanced labels, their best possible accuracies are two thirds and one third
# respectively. Evaluate the intact model again after changing class proportions.
#
# ## Result
#
# The numerical checks below are provisional, predeclared hypotheses, not measured successes. No full
# learning result is claimed until a matching run is recorded. Smoke runs establish execution only.
#
# ## Method
#
# ### Population, controls, and scoring
#
# Use 3,072 balanced training and test rows plus 768 validation rows. Each of three matched models
# receives 400 updates. A second independent test population has class proportions 10%, 20%, and 70%;
# per-state recall prevents its majority class from concealing failures. The full-information recall
# gates are provisional. Existing tensorization unit tests establish the codec contract; this experiment
# asks whether end-to-end learning actually uses the distinction.
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
# types = ["Number"]
# conditions = ["missing-inputs", "value-states"]
# ///

"""P088: learn the three distinct states zero, null, and valid NaN."""

import sys
from collections.abc import Iterator
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
from policies import CONTROL
from reporting import experiment, report
from visibility import TARGET

BUDGET = 400
TRAIN_ROWS = 3072
TEST_ROWS = 3072
LABELS = ("zero", "absent", "invalid")


def records(*, rows: int, seed: int, shifted: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    states = rng.choice(3, rows, p=(0.1, 0.2, 0.7)) if shifted else np.resize(np.arange(3), rows)
    rng.shuffle(states)
    for state in states:
        yield {"measurement": (0.0, None, float("nan"))[int(state)], "target": LABELS[int(state)]}


def table(rows: list[dict], *, arm: str = "intact", targets: bool = True) -> pa.Table:
    values = []
    for row in rows:
        value = row["measurement"]
        if arm == "coalesced" and value is not None and np.isnan(value):
            value = None
        if arm == "filled":
            value = 0.0
        values.append(value)
    columns = {"measurement": pa.array(values, type=pa.float64(), from_pandas=False)}
    if targets:
        columns["target"] = pa.array([row["target"] for row in rows], type=pa.string())
    return pa.table(columns)


def prediction(model: rf.Model, rows: list[dict], arm: str) -> np.ndarray:
    output = pl.DataFrame(model.predict(table(rows, arm=arm, targets=False), batch_size=256))["predictions"].to_list()
    return np.asarray([row["target"]["content"]["value"] for row in output])


def score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    confusion = [[int(np.sum((actual == left) & (predicted == right))) for right in LABELS] for left in LABELS]
    recalls = {name: float(np.mean(predicted[actual == name] == name)) for name in LABELS}
    return {
        "accuracy": float(np.mean(predicted == actual)),
        "macro_recall": float(np.mean(list(recalls.values()))),
        "recall": recalls,
        "confusion": confusion,
    }


def fit(model: rf.Model, train: pa.Table, validate: pa.Table, seed: int, steps: int | None, accelerator: str) -> int:
    """Fit the declared fixed-data protocol without checkpoint selection."""
    data = rf.ArrowDataModule(train=train, validate=validate, num_workers=0, persistent_workers=False, pin_memory=False)
    model.fit(
        data,
        batch_size=64,
        learning_rate=0.003,
        weight_decay=0.01,
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
    model.eval()
    return int(model.trainer.global_step)


class Model(rf.Model):
    measurement = rf.Number
    target = rf.Enum(values=LABELS, **TARGET)


@experiment("P088", seed=8801)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train = list(records(rows=TRAIN_ROWS, seed=seed + 1))
    validation = list(records(rows=768, seed=seed + 2))
    test = list(records(rows=TEST_ROWS, seed=seed + 3))
    shifted = list(records(rows=TEST_ROWS, seed=seed + 4, shifted=True))
    metrics = {"budget": BUDGET, "arms": {}}
    checks = {}
    for arm in ("intact", "coalesced", "filled"):
        lit.seed_everything(seed, workers=True)

        model = Model(preset=CONTROL.name)
        updates = fit(model, table(train, arm=arm), table(validation, arm=arm), seed, steps, accelerator)
        metrics["arms"][arm] = {
            "steps": updates,
            "balanced": score(np.asarray([row["target"] for row in test]), prediction(model, test, arm)),
            "shifted": score(np.asarray([row["target"] for row in shifted]), prediction(model, shifted, arm)),
        }
        checks[f"{arm}: full budget retained"] = updates == BUDGET
    full = metrics["arms"]["intact"]
    for name in LABELS:
        checks[f"Intact balanced {name} recall at least 0.95"] = full["balanced"]["recall"][name] >= 0.95
        checks[f"Intact shifted {name} recall at least 0.95"] = full["shifted"]["recall"][name] >= 0.95
    checks["Coalescing cannot exceed its two-thirds information ceiling"] = (
        metrics["arms"]["coalesced"]["balanced"]["accuracy"] <= 2 / 3 + 0.03
    )
    checks["Filling cannot exceed its one-third information ceiling"] = (
        metrics["arms"]["filled"]["balanced"]["accuracy"] <= 1 / 3 + 0.03
    )
    typed = table(test)["measurement"].combine_chunks()
    checks["Arrow keeps NaN valid and null absent"] = (
        typed.null_count == TEST_ROWS // 3 and int(np.sum(np.isnan(typed.fill_null(0).to_numpy()))) == TEST_ROWS // 3
    )
    return metrics, checks


if __name__ == "__main__":
    report(run)
