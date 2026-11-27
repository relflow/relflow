# %% [markdown]
# ---
# title: Do nested Duration sources preserve learnable elapsed-time signals?
# description: Compare ancestor-based Duration inputs and previous-record gaps with prepared numeric values and
#   independently shuffled controls.
# evidence-readout:
# - label: Duration held-out normalized RMSE
#   metric:
#   - arms
#   - duration
#   - overall
#   - nrmse
#   format: number
# - label: Prepared Number held-out normalized RMSE
#   metric:
#   - arms
#   - number
#   - overall
#   - nrmse
#   format: number
# - label: Shuffled predecessor gaps
#   metric:
#   - arms
#   - shuffled_gap
#   - overall
#   - nrmse
#   format: number
# evidence-group: Temporal
# evidence-review:
#   question: Do nested Duration sources preserve learnable elapsed-time signals?
#   modeling: Retain the final event of each account history and predict a hidden target from age and the previous-event
#     gap. Compare Duration with the same prepared Number values and separately shuffled training inputs.
#   comparison:
#     metric: Normalized RMSE
#     kind: Controlled comparison
#     conditions:
#     - label: Shuffled predecessor gaps
#       path:
#       - arms
#       - shuffled_gap
#       - overall
#       - nrmse
#       format: number
#     - label: Shuffled ancestor ages
#       path:
#       - arms
#       - shuffled_age
#       - overall
#       - nrmse
#       format: number
#     - label: Prepared numeric values
#       path:
#       - arms
#       - number
#       - overall
#       - nrmse
#       format: number
#     - label: Nested Duration sources
#       path:
#       - arms
#       - duration
#       - overall
#       - nrmse
#       format: number
#     definition: Root mean squared error divided by the error of the training-mean predictor on the same split.
#     direction: Lower error is better
#   note: The predecessor comes from original source order before cropping. This synthetic task does not establish
#     general temporal reasoning.
# evidence-plot:
#   mode: summary
#   target: target
#   focus: Do ancestor-based Duration inputs and previous-record gaps retain the right values after cropping?
#   comparison: Compare Duration with prepared Number inputs and separate shuffled-age and shuffled-gap controls.
#   limit: The task uses synthetic histories and one retained event.
# ---
#
# Each account contains an origin timestamp and an ordered event history. Only the final
# event is retained by the model. Its hidden target depends on both time since the origin
# and time since the previous event. Histories of length one have no predecessor; for those
# records the generating rule uses only the account age.
#
# ## Example
#
# ```yaml
# origin: "2026-01-01T00:00:00Z"
# events:
# - at: "2026-01-01T03:00:00Z"
# - at: "2026-01-01T07:00:00Z"
#   target: 15.0
# ```
#
# The last event is seven hours after its ancestor origin and four hours after its
# predecessor. Its noiseless target is 7 + 2 * 4 = 15. Cropping to one retained event
# must not erase the predecessor. The target is omitted from prediction inputs.
#
#
# ## Comparison
#
# Train four models on identical generated observations and budgets. The Duration arm
# reads its age from an ancestor and its gap with `delta=True`; Number receives the exact
# precomputed values and equivalent nulls. Two controls shuffle only one
# numeric training feature, preserving its marginal distribution and missingness.
# Validation and test records remain unshuffled. Evaluate independently generated test
# cases, including no predecessor, negative gaps, and histories cropped from length > 2.
# Jointly translating all timestamps by 371 days should leave Duration predictions intact.
#
# ## Result
#
#
# Checks are declared before execution. A low error alone does not establish broad
# temporal reasoning or performance on real histories.
#
#
#
#
#
# ## Method
#
# ### Population, controls, and scoring
#
# Generate independent train, validation, and test observations with separate seeds.
# Ages range from -12 to 12 hours and predecessor gaps from -6 to 6 hours in quarter-hour
# increments. A quarter of histories have one event; the rest have two to five events.
# The target is age + 2 * gap + Gaussian noise with standard deviation 0.1; a missing
# predecessor contributes zero. The branch retains the last event only. All timestamps are
# UTC and their absolute origins vary independently. Each arm trains for 300 updates with
# the same frozen architecture, model seed, optimizer, precision, and sample counts.
# Report RMSE, MAE, normalized RMSE against the training-mean predictor, subgroup counts
# and errors, aligned predictions, and the translation drift.
#
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "lightning",
#     "numpy",
#     "polars",
#     "pyarrow",
#     "pyyaml>=6",
#     "relflow>=0.6.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "representation"
# types = ["Duration", "Number"]
# conditions = ["changed-associations", "missing-inputs", "transformations", "varying-collections", "representations"]
# ///

"""P113: compare nested Duration features with matched values and independent controls."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
from policies import CONTROL
from reporting import experiment, report

BUDGET = 300
TRAIN_ROWS = 2048
VALIDATION_ROWS = 512
TEST_ROWS = 1024
ANCHOR = datetime(2026, 1, 1, tzinfo=timezone.utc)
ARMS = ("duration", "number", "shuffled_gap", "shuffled_age")


class DurationEvent(rf.Branch):
    age = rf.Duration(query=("../origin", "at"), granularity="hours")
    gap = rf.Duration(query="at", delta=True, granularity="hours")
    target = rf.Number(predict=True, objective="mse")


class NumericEvent(rf.Branch):
    age = rf.Number(query="prepared_age")
    gap = rf.Number(query="prepared_gap")
    target = rf.Number(predict=True, objective="mse")


class DurationModel(rf.Model):
    events = DurationEvent(length=1, overflow="tail")


class NumericModel(rf.Model):
    events = NumericEvent(length=1, overflow="tail")


def records(*, rows: int, seed: int) -> Iterator[dict]:
    """Generate paired elapsed values and histories without using a model's prediction."""
    rng = np.random.default_rng(seed)
    for index in range(rows):
        age = int(rng.integers(-48, 49)) / 4
        has_previous = bool(rng.random() >= 0.25)
        gap = int(rng.integers(-24, 25)) / 4 if has_previous else 0.0
        length = int(rng.integers(2, 6)) if has_previous else 1
        origin = ANCHOR + timedelta(days=int(rng.integers(-500, 501)))
        end = origin + timedelta(hours=age)
        previous = end - timedelta(hours=gap)
        events = [
            {
                "at": previous - timedelta(hours=length - position - 2),
                "prepared_age": None,
                "prepared_gap": None,
                "target": None,
            }
            for position in range(length - 1)
        ]
        target = age + 2 * gap + float(rng.normal(0, 0.1))
        events.append(
            {
                "at": end,
                "prepared_age": age,
                "prepared_gap": gap if has_previous else None,
                "target": target,
            }
        )
        yield {"case": f"{seed}:{index}", "origin": origin, "events": events, "age": age, "gap": gap, "target": target}


def frame(rows: list[dict], *, targets: bool = True, shuffle: str | None = None, seed: int = 0) -> pl.DataFrame:
    """Hide answers for inference and shuffle only the selected training feature."""
    indices = np.arange(len(rows))
    rng = np.random.default_rng(seed)
    if shuffle == "age":
        age_donors = rng.permutation(indices)
    else:
        age_donors = indices
    gap_donors = indices.copy()
    if shuffle == "gap":
        eligible = np.asarray([index for index, row in enumerate(rows) if len(row["events"]) > 1], dtype=np.int64)
        gap_donors[eligible] = rng.permutation(eligible)
    observations = []
    for index, row in enumerate(rows):
        events = [{key: value for key, value in item.items() if targets or key != "target"} for item in row["events"]]
        events[-1]["prepared_age"] = rows[int(age_donors[index])]["age"]
        if len(events) > 1:
            events[-1]["prepared_gap"] = rows[int(gap_donors[index])]["gap"]
        observations.append({"origin": row["origin"], "events": events})
    return pl.DataFrame(observations, infer_schema_length=None)


def build(arm: str) -> rf.Model:
    if arm == "duration":
        return DurationModel(preset=CONTROL.name)
    return NumericModel(preset=CONTROL.name)


def prediction(model: rf.Model, rows: list[dict]) -> np.ndarray:
    output = model.predict(frame(rows, targets=False), batch_size=256)["predictions"].to_list()
    return np.asarray([row["events"][0]["target"]["content"] for row in output], dtype=np.float64)


def metrics(actual: np.ndarray, predicted: np.ndarray, mean: float, selected: np.ndarray) -> dict:
    expected, output = actual[selected], predicted[selected]
    error = output - expected
    baseline = float(np.sqrt(np.mean((expected - mean) ** 2)))
    rmse = float(np.sqrt(np.mean(error**2)))
    return {"rows": int(len(expected)), "rmse": rmse, "mae": float(np.mean(np.abs(error))), "nrmse": rmse / baseline}


def fit(
    model: rf.Model, train: pl.DataFrame, validate: pl.DataFrame, seed: int, steps: int | None, accelerator: str
) -> int:
    data = rf.PolarsDataModule(
        train=train, validate=validate, seed=seed, num_workers=0, persistent_workers=False, pin_memory=False
    )
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


@experiment("P113", seed=11301)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train = list(records(rows=TRAIN_ROWS, seed=seed + 1))
    validate = list(records(rows=VALIDATION_ROWS, seed=seed + 2))
    test = list(records(rows=TEST_ROWS, seed=seed + 3))
    actual = np.asarray([row["target"] for row in test])
    mean = float(np.mean([row["target"] for row in train]))
    has_previous = np.asarray([len(row["events"]) > 1 for row in test])
    gaps = np.asarray([row["gap"] for row in test])
    groups = {
        "overall": np.ones(len(test), dtype=bool),
        "with_predecessor": has_previous,
        "without_predecessor": ~has_previous,
        "negative_gap": has_previous & (gaps < 0),
        "nonnegative_gap": has_previous & (gaps >= 0),
        "long_history": np.asarray([len(row["events"]) > 2 for row in test]),
    }
    output = {
        "budget": BUDGET,
        "rows": {"train": len(train), "validation": len(validate), "test": len(test)},
        "train_mean": mean,
        "arms": {},
        "test": {
            "case": [row["case"] for row in test],
            "target": actual,
            "age": [row["age"] for row in test],
            "gap": gaps,
            "history_length": [len(row["events"]) for row in test],
        },
    }
    checks = {}
    predictions = {}
    for arm in ARMS:
        lit.seed_everything(seed, workers=True)
        model = build(arm)
        shuffle = arm.removeprefix("shuffled_") if arm.startswith("shuffled_") else None
        updates = fit(model, frame(train, shuffle=shuffle, seed=seed + 4), frame(validate), seed, steps, accelerator)
        predicted = prediction(model, test)
        if not np.isfinite(predicted).all():
            raise ValueError(f"{arm} produced nonfinite held-out predictions")
        predictions[arm] = predicted
        measured = {name: metrics(actual, predicted, mean, selection) for name, selection in groups.items()}
        measured.update(steps=updates, finite=bool(np.isfinite(predicted).all()), predictions=predicted)
        if arm == "duration":
            translated = [
                {
                    **row,
                    "origin": row["origin"] + timedelta(days=371),
                    "events": [{**item, "at": item["at"] + timedelta(days=371)} for item in row["events"]],
                }
                for row in test
            ]
            translated_prediction = prediction(model, translated)
            if not np.isfinite(translated_prediction).all():
                raise ValueError("duration produced nonfinite translated predictions")
            measured["translation_max_drift"] = float(np.max(np.abs(predicted - translated_prediction)))
        output["arms"][arm] = measured
        checks[f"{arm}: full budget retained"] = updates == BUDGET
        checks[f"{arm}: finite predictions"] = measured["finite"]
    duration, number = output["arms"]["duration"], output["arms"]["number"]
    output["duration_number_prediction_rmse"] = float(
        np.sqrt(np.mean((predictions["duration"] - predictions["number"]) ** 2))
    )
    checks["Duration held-out nRMSE below 0.35"] = duration["overall"]["nrmse"] < 0.35
    checks["Duration stays within 0.03 nRMSE of prepared Number"] = (
        abs(duration["overall"]["nrmse"] - number["overall"]["nrmse"]) < 0.03
    )
    checks["Joint timestamp translation preserves predictions"] = duration["translation_max_drift"] < 1e-5
    for name in ("gap", "age"):
        control = output["arms"][f"shuffled_{name}"]["overall"]["nrmse"]
        checks[f"shuffled {name}: nRMSE remains above 0.50"] = control > 0.50
        checks[f"Duration improves on shuffled {name} by 0.25 nRMSE"] = control - duration["overall"]["nrmse"] > 0.25
    return output, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
