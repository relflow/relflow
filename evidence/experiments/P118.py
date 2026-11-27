# %% [markdown]
# ---
# title: Can masked delta targets learn aligned record intervals?
# description: Predict retained record intervals from other fields while hiding or poisoning the source timestamps.
# evidence-readout:
# - label: Shuffled clue
#   metric:
#   - arms
#   - shuffled
#   - overall
#   - nrmse
#   format: number
# - label: Prepared Number
#   metric:
#   - arms
#   - number
#   - overall
#   - nrmse
#   format: number
# - label: Duration delta
#   metric:
#   - arms
#   - duration
#   - overall
#   - nrmse
#   format: number
# evidence-group: Temporal
# evidence-review:
#   question: Can masked delta targets learn aligned record intervals?
#   modeling: Train masked delta and prepared Number targets on independently generated histories with signed intervals
#     and a visible predecessor flag. Evaluate retained records, poisoned target timestamps, and a shuffled-clue control.
#   comparison:
#     metric: Normalized RMSE
#     kind: Controlled comparison
#     conditions:
#     - label: Shuffled clue
#       path:
#       - arms
#       - shuffled
#       - overall
#       - nrmse
#       format: number
#     - label: Prepared Number
#       path:
#       - arms
#       - number
#       - overall
#       - nrmse
#       format: number
#     - label: Duration delta
#       path:
#       - arms
#       - duration
#       - overall
#       - nrmse
#       format: number
#     definition: Root mean squared error on valued intervals divided by the training-mean predictor error on those
#       same positions.
#     direction: Lower error is better
#   note: The first record has a null interval and the predecessor is selected before cropping.
# evidence-plot:
#   mode: summary
#   target: target
#   focus: Can masked delta targets learn aligned record intervals?
#   comparison: Predict retained record intervals from other fields while hiding or poisoning the source timestamps.
#   limit: The first record has a null interval and the predecessor is selected before cropping.
# ---
#
# For the second event the previous-event interval is five seconds. The timestamp is a target source, not a visible model input, and is omitted when predicting.
#
# ## Example
#
# ```yaml
# events:
# - {hint: 0.0, has_previous: false, at: "2026-01-01T00:00:00Z"}
# - {hint: 2.5, has_previous: true, at: "2026-01-01T00:00:05Z"}
# ```
#
# ## Comparison
#
# Train masked delta and prepared Number targets on identical histories. Evaluate retained intervals, source poisoning and a shuffled-clue control.
# Independent splits, matched seeds, equal update budgets, and a negative control are retained.
#
# ## Result
#
# Measurements and checks are recorded independently; one synthetic task does not establish general performance.
#
# ## Method
#
# Each arm uses the same declared budget, optimizer, model/data seeds, and independent held-out observations.
# Per-arm metrics retain subgroup results and aligned predictions.
#
# Generate 2,048 training, 512 validation and 1,024 test histories of length 2–6.
# Noninitial signed integer gaps range from -8 to 8 seconds; their clue is gap / 2
# with Gaussian noise of standard deviation 0.15. A visible has_previous flag supplies
# the target state, not its value. Retain up to three tail records and run 500 updates.
# Shuffle clues only among retained, noninitial records. Scores exclude null targets;
# state accuracy, signed subgroups, source poisoning and alignment are recorded separately.
# Poisoning is checked against both numeric content and state probabilities.
#
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["lightning", "numpy", "polars", "pyarrow", "pyyaml>=6", "relflow>=0.5.0", "torch"]
#
# [tool.relflow.evidence]
# capability = "representation"
# types = ["Duration", "Number"]
# conditions = ["varying-collections", "hidden-values", "changed-associations", "objectives"]
# ///

"""P118: learn masked per-record Duration deltas without target timestamps at inference."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
from policies import CONTROL
from reporting import experiment, report
from visibility import TARGET

BUDGET = 500
TRAIN_ROWS = 2048
VALIDATION_ROWS = 512
TEST_ROWS = 1024
CAP = 3
ANCHOR = datetime(2026, 1, 1, tzinfo=timezone.utc)
ARMS = ("duration", "number", "shuffled")


class DurationEvent(rf.Branch):
    hint = rf.Number
    has_previous = rf.Boolean
    gap = rf.Duration(query="at", delta=True, granularity="seconds", **TARGET)


class NumericEvent(rf.Branch):
    hint = rf.Number
    has_previous = rf.Boolean
    gap = rf.Number(query="prepared_gap", **TARGET)


class DurationModel(rf.Model):
    events = DurationEvent(length=CAP, overflow="tail")


class NumericModel(rf.Model):
    events = NumericEvent(length=CAP, overflow="tail")


def records(*, rows: int, seed: int):
    rng = np.random.default_rng(seed)
    for index in range(rows):
        length = int(rng.integers(2, 7))
        now = ANCHOR + timedelta(days=int(rng.integers(-300, 301)))
        events = [{"hint": float(rng.normal()), "has_previous": False, "at": now, "prepared_gap": None}]
        for _ in range(length - 1):
            gap = int(rng.integers(-8, 9))
            now = now + timedelta(seconds=gap)
            events.append(
                {
                    "hint": float(gap / 2 + rng.normal(0, 0.15)),
                    "has_previous": True,
                    "at": now,
                    "prepared_gap": float(gap),
                }
            )
        yield {"case": f"{seed}:{index}", "events": events}


def frame(
    rows: list[dict], *, targets: bool = True, shuffle: bool = False, seed: int = 0, poison: bool = False
) -> pl.DataFrame:
    events = [[dict(event) for event in row["events"]] for row in rows]
    if shuffle:
        locations = [
            (row, index)
            for row, items in enumerate(events)
            for index, item in enumerate(items)
            if index >= max(0, len(items) - CAP) and item["has_previous"]
        ]
        hints = [events[row][index]["hint"] for row, index in locations]
        donors = np.random.default_rng(seed).permutation(len(hints))
        for (row, index), donor in zip(locations, donors, strict=True):
            events[row][index]["hint"] = hints[int(donor)]
    for items in events:
        for index, item in enumerate(items):
            if not targets:
                item.pop("at")
                item.pop("prepared_gap")
                if poison:
                    item["at"] = ANCHOR + timedelta(days=1000 + index * 17)
    return pl.DataFrame({"events": events}, infer_schema_length=None)


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


def prediction(model: rf.Model, rows: list[dict], *, poison: bool = False) -> tuple[np.ndarray, list[bool], np.ndarray]:
    result = model.predict(frame(rows, targets=False, poison=poison), batch_size=128)["predictions"].to_list()
    items = [item["gap"] for row in result for item in row["events"]]
    return (
        np.asarray([item["content"] for item in items], dtype=np.float64),
        [item["inferred"] for item in items],
        np.asarray(
            [[item["state"][state] for state in ("valued", "null", "padded", "masked", "other")] for item in items]
        ),
    )


def score(actual: np.ndarray, predicted: np.ndarray, mean: float, selected: np.ndarray) -> dict:
    expected, value = actual[selected], predicted[selected]
    rmse = float(np.sqrt(np.mean((value - expected) ** 2)))
    baseline = float(np.sqrt(np.mean((expected - mean) ** 2)))
    return {
        "rows": int(len(expected)),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(value - expected))),
        "nrmse": rmse / baseline,
    }


@experiment("P118", seed=11801)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train = list(records(rows=TRAIN_ROWS, seed=seed + 1))
    validate = list(records(rows=VALIDATION_ROWS, seed=seed + 2))
    test = list(records(rows=TEST_ROWS, seed=seed + 3))
    retained = [event for row in test for event in row["events"][-CAP:]]
    training = [
        event["prepared_gap"] for row in train for event in row["events"][-CAP:] if event["prepared_gap"] is not None
    ]
    valid = np.asarray([event["prepared_gap"] is not None for event in retained])
    actual = np.asarray([event["prepared_gap"] if event["prepared_gap"] is not None else 0.0 for event in retained])
    mean = float(np.mean(training))
    groups = {"overall": valid, "negative": valid & (actual < 0), "nonnegative": valid & (actual >= 0)}
    metrics = {
        "budget": BUDGET,
        "rows": {"train": len(train), "validation": len(validate), "test": len(test)},
        "train_mean": mean,
        "test": {
            "case": [row["case"] for row in test],
            "length": [len(row["events"][-CAP:]) for row in test],
            "target": actual,
            "valid": valid,
        },
        "arms": {},
    }
    checks = {}
    for arm in ARMS:
        lit.seed_everything(seed, workers=True)
        model = (DurationModel if arm == "duration" else NumericModel)(preset=CONTROL.name)
        updates = fit(
            model, frame(train, shuffle=arm == "shuffled", seed=seed + 4), frame(validate), seed, steps, accelerator
        )
        predicted, inferred, probabilities = prediction(model, test)
        states = probabilities.argmax(axis=1)
        expected_states = np.where(valid, 0, 1)
        if not np.isfinite(predicted).all() or len(predicted) != len(actual):
            raise ValueError(f"{arm} produced nonfinite or misaligned output")
        measured = {name: score(actual, predicted, mean, selected) for name, selected in groups.items()}
        measured.update(
            steps=updates,
            inferred_all=all(inferred),
            state_accuracy=float(np.mean(states == expected_states)),
            null_state_accuracy=float(np.mean(states[~valid] == 1)),
            valued_state_accuracy=float(np.mean(states[valid] == 0)),
            state_probabilities=probabilities,
            predictions=predicted,
        )
        if arm == "duration":
            poisoned, poison_inferred, poison_probabilities = prediction(model, test, poison=True)
            measured["poison_max_drift"] = float(np.max(np.abs(predicted - poisoned)))
            measured["poison_state_max_drift"] = float(np.max(np.abs(probabilities - poison_probabilities)))
            measured["poison_inferred_all"] = all(poison_inferred)
            checks["Poisoned target timestamps do not affect predictions"] = (
                measured["poison_max_drift"] < 1e-5
                and measured["poison_state_max_drift"] < 1e-5
                and all(poison_inferred)
            )
            checks["Duration null state accuracy at least 0.5"] = measured["null_state_accuracy"] >= 0.5
        metrics["arms"][arm] = measured
        checks[f"{arm}: full budget retained"] = updates == BUDGET
        checks[f"{arm}: finite inferred output"] = all(inferred)
        checks[f"{arm}: state accuracy at least 0.8"] = measured["state_accuracy"] >= 0.8
    checks["Duration delta held-out nRMSE below 0.8"] = metrics["arms"]["duration"]["overall"]["nrmse"] < 0.8
    checks["Duration delta stays near prepared Number"] = (
        abs(metrics["arms"]["duration"]["overall"]["nrmse"] - metrics["arms"]["number"]["overall"]["nrmse"]) < 0.1
    )
    checks["Shuffled clues retain nRMSE above 0.8"] = metrics["arms"]["shuffled"]["overall"]["nrmse"] > 0.8
    return metrics, checks


if __name__ == "__main__":
    report(run)
