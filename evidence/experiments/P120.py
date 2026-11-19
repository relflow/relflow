# %% [markdown]
# ---
# title: Do delta inputs preserve useful signals across retained records?
# description: Compare Duration and prepared Number gaps across multiple retained branch records against an alignment-corruption
#   control.
# evidence-readout:
# - label: Shuffled gaps
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
#   question: Do delta inputs preserve useful signals across retained records?
#   modeling: Predict each retained event target from its own prior gap and sibling value. Compare Duration with aligned
#     Numbers and an independently shuffled gap using matched histories, seeds, and budgets.
#   comparison:
#     metric: Normalized RMSE
#     kind: Controlled comparison
#     conditions:
#     - label: Shuffled gaps
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
#     definition: Root mean squared error on retained event targets divided by the training-mean predictor error on
#       those positions.
#     direction: Lower error is better
#   note: A shuffled gap breaks alignment while preserving the retained gap distribution. Histories are synthetic.
# evidence-plot:
#   mode: summary
#   target: target
#   focus: Do delta inputs preserve useful signals across retained records?
#   comparison: Compare Duration and prepared Number gaps across multiple retained branch records against an alignment-corruption
#     control.
#   limit: A shuffled gap breaks alignment while preserving the retained gap distribution. Histories are synthetic.
# ---
#
# Each target depends on its event’s predecessor gap and sibling value. In the example 4 + 2 × 2 = 8; the target is hidden at inference.
#
# ## Example
#
# ```yaml
# events:
# - {at: "2026-01-01T00:00:00Z", value: 1.0}
# - {at: "2026-01-01T00:00:04Z", value: 2.0, target: 8.0}
# ```
#
# ## Comparison
#
# Predict each retained event target from its own prior gap and sibling value. Compare Duration with aligned Numbers and an independently shuffled gap.
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
# Content and predicted value states are recorded separately; all test targets are valued.
#
# Generate 2,048 training, 512 validation and 1,024 test histories of length 2–6.
# Each noninitial gap is an integer in [-8, 8] seconds; each sibling value is uniform
# on [-4, 4], and the target is gap + 2 × value with Gaussian noise 0.1. The first
# gap is null and contributes zero to the rule. Retain three tail records and train
# 500 updates per arm. Shuffle gap values among retained noninitial training events
# while preserving the target, value and missingness. Score each retained event.
#
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["lightning", "numpy", "polars", "pyarrow", "pyyaml>=6", "relflow>=0.5.0", "torch"]
#
# [tool.relflow.evidence]
# capability = "representation"
# types = ["Duration", "Number"]
# conditions = ["varying-collections", "changed-associations", "representations"]
# ///

"""P120: learn from multiple aligned previous-event gaps and sibling values."""

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

BUDGET = 500
TRAIN_ROWS = 2048
VALIDATION_ROWS = 512
TEST_ROWS = 1024
CAP = 3
ANCHOR = datetime(2026, 1, 1, tzinfo=timezone.utc)
ARMS = ("duration", "number", "shuffled")


class DurationEvent(rf.Branch):
    gap = rf.Duration(query="at", delta=True, granularity="seconds")
    value = rf.Number
    target = rf.Number(mask=True, objective="mse")


class NumericEvent(rf.Branch):
    gap = rf.Number(query="prepared_gap")
    value = rf.Number
    target = rf.Number(mask=True, objective="mse")


class DurationModel(rf.Model):
    events = DurationEvent(length=CAP, overflow="tail")


class NumericModel(rf.Model):
    events = NumericEvent(length=CAP, overflow="tail")


def records(*, rows: int, seed: int):
    rng = np.random.default_rng(seed)
    for index in range(rows):
        length = int(rng.integers(2, 7))
        now = ANCHOR + timedelta(days=int(rng.integers(-300, 301)))
        events = []
        for position in range(length):
            gap = None if position == 0 else int(rng.integers(-8, 9))
            if gap is not None:
                now += timedelta(seconds=gap)
            value = float(rng.uniform(-4, 4))
            target = (gap or 0) + 2 * value + float(rng.normal(0, 0.1))
            events.append({"at": now, "prepared_gap": gap, "value": value, "target": target})
        yield {"case": f"{seed}:{index}", "events": events}


def frame(rows: list[dict], *, targets: bool = True, shuffle: bool = False, seed: int = 0) -> pl.DataFrame:
    events = [[dict(event) for event in row["events"]] for row in rows]
    if shuffle:
        locations = [
            (row, index)
            for row, items in enumerate(events)
            for index, item in enumerate(items)
            if index >= max(0, len(items) - CAP) and item["prepared_gap"] is not None
        ]
        gaps = [events[row][index]["prepared_gap"] for row, index in locations]
        donors = np.random.default_rng(seed).permutation(len(gaps))
        for (row, index), donor in zip(locations, donors, strict=True):
            events[row][index]["prepared_gap"] = gaps[int(donor)]
    if not targets:
        for items in events:
            for item in items:
                item.pop("target")
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


@experiment("P120", seed=12001)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train = list(records(rows=TRAIN_ROWS, seed=seed + 1))
    validate = list(records(rows=VALIDATION_ROWS, seed=seed + 2))
    test = list(records(rows=TEST_ROWS, seed=seed + 3))
    retained = [event for row in test for event in row["events"][-CAP:]]
    actual = np.asarray([event["target"] for event in retained])
    mean = float(np.mean([event["target"] for row in train for event in row["events"][-CAP:]]))
    predecessor = np.asarray([event["prepared_gap"] is not None for event in retained])
    cropped = np.asarray([len(row["events"]) > CAP for row in test for _ in row["events"][-CAP:]])
    groups = {
        "overall": np.ones(len(retained), dtype=bool),
        "with_predecessor": predecessor,
        "without_predecessor": ~predecessor,
        "cropped": cropped,
    }
    metrics = {
        "budget": BUDGET,
        "rows": {"train": len(train), "validation": len(validate), "test": len(test)},
        "train_mean": mean,
        "test": {
            "case": [row["case"] for row in test],
            "length": [len(row["events"][-CAP:]) for row in test],
            "target": actual,
            "has_predecessor": predecessor,
        },
        "arms": {},
    }
    checks = {}
    predictions = {}
    for arm in ARMS:
        lit.seed_everything(seed, workers=True)
        model = (DurationModel if arm == "duration" else NumericModel)(preset=CONTROL.name)
        updates = fit(
            model, frame(train, shuffle=arm == "shuffled", seed=seed + 4), frame(validate), seed, steps, accelerator
        )
        output = model.predict(frame(test, targets=False), batch_size=128)["predictions"].to_list()
        items = [item["target"] for row in output for item in row["events"]]
        predicted = np.asarray([item["content"] for item in items], dtype=np.float64)
        inferred = all(item["inferred"] for item in items)
        states = [max(item["state"], key=item["state"].get) for item in items]
        if len(predicted) != len(actual) or not np.isfinite(predicted).all():
            raise ValueError(f"{arm} produced nonfinite or misaligned output")
        predictions[arm] = predicted
        measured = {name: score(actual, predicted, mean, selected) for name, selected in groups.items()}
        measured.update(
            steps=updates,
            inferred=inferred,
            state_accuracy=float(np.mean(np.asarray(states) == "valued")),
            states=states,
            predictions=predicted,
        )
        metrics["arms"][arm] = measured
        checks[f"{arm}: full budget retained"] = updates == BUDGET
        checks[f"{arm}: finite inferred output"] = inferred
        checks[f"{arm}: valued state accuracy at least 0.95"] = measured["state_accuracy"] >= 0.95
    metrics["duration_number_prediction_rmse"] = float(
        np.sqrt(np.mean((predictions["duration"] - predictions["number"]) ** 2))
    )
    checks["Duration held-out nRMSE below 0.8"] = metrics["arms"]["duration"]["overall"]["nrmse"] < 0.8
    checks["Duration stays near prepared Number"] = (
        abs(metrics["arms"]["duration"]["overall"]["nrmse"] - metrics["arms"]["number"]["overall"]["nrmse"]) < 0.1
    )
    checks["Shuffled gaps retain nRMSE above 0.55"] = metrics["arms"]["shuffled"]["overall"]["nrmse"] > 0.55
    checks["Duration improves on shuffled gaps"] = (
        metrics["arms"]["duration"]["overall"]["nrmse"] + 0.1 < metrics["arms"]["shuffled"]["overall"]["nrmse"]
    )
    return metrics, checks


if __name__ == "__main__":
    report(run)
