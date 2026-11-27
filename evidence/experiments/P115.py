# %% [markdown]
# ---
# title: Can Duration targets predict held-out elapsed time without future timestamps?
# description: Compare masked Duration targets with prepared Number targets while withholding their future source
#   timestamps.
# evidence-readout:
# - label: Two-source Duration target
#   metric:
#   - arms
#   - duration
#   - overall
#   - nrmse
#   format: number
# - label: Prepared Number target
#   metric:
#   - arms
#   - number
#   - overall
#   - nrmse
#   format: number
# - label: Shuffled training signal
#   metric:
#   - arms
#   - shuffled
#   - overall
#   - nrmse
#   format: number
# evidence-group: Temporal
# evidence-review:
#   question: Can Duration targets predict held-out elapsed time without future timestamps?
#   modeling: Predict a hidden signed elapsed time from a numeric clue. Compare two-query and configured-start Duration
#     targets with a prepared Number target and shuffled-clue control, with target timestamps omitted during prediction.
#   comparison:
#     metric: Normalized RMSE
#     kind: Controlled comparison
#     conditions:
#     - label: Shuffled numeric clue
#       path:
#       - arms
#       - shuffled
#       - overall
#       - nrmse
#       format: number
#     - label: Prepared Number target
#       path:
#       - arms
#       - number
#       - overall
#       - nrmse
#       format: number
#     - label: Duration target
#       path:
#       - arms
#       - duration
#       - overall
#       - nrmse
#       format: number
#     definition: Root mean squared error divided by the error of the training-mean predictor on the same split.
#     direction: Lower error is better
#   note: Poisoned timestamps test target masking separately. The signal is synthetic.
# evidence-plot:
#   mode: summary
#   target: target
#   focus: Can a Duration target be predicted without future timestamp sources?
#   comparison: Compare two-query and fixed-start Duration with Number and a shuffled-signal control.
#   limit: Prediction hides target endpoints; the signal and relationship are synthetic.
# ---
#
# A numeric clue predicts a signed future elapsed time. The model learns from a hidden
# Duration target calculated from two timestamps, or from an equivalent Number target.
# Prediction records contain the clue and omit both target timestamps.
#
# ## Example
#
# ```yaml
# signal: 2.0
# started_at: "2026-01-01T08:00:00Z"
# ended_at: "2026-01-01T11:30:00Z"
# ```
#
# The timestamps are 3.5 hours apart. Their relationship to the signal is used during
# training; only `signal` is supplied when predicting a new observation. Generated
# targets add independent noise and can also be negative.
#
#
# ## Comparison
#
# Compare two-query, translated two-query, and configured-start Duration targets with
# the same numeric targets. A Number control shuffles only the training clues. All arms
# use identical underlying observations, architecture, model and data seeds, optimizer,
# and update budget. Test timestamps are absent from normal prediction inputs. Supplying
# valid but unrelated target timestamps at prediction tests whether the hidden target
# leaks through its source fields.
#
# ## Result
#
#
# These predeclared synthetic checks measure held-out target prediction and masking.
# They do not establish performance for arbitrary timestamp distributions or production data.
#
#
#
#
#
# ## Method
#
# ### Population, controls, and scoring
#
# Generate 4,096 training, 512 validation, and 2,048 test observations with independent
# seeds. The clue is uniform on [-10, 10], and the elapsed target is 1.75 times the clue
# plus Gaussian noise with standard deviation 0.1 hours. Dynamic start instants span
# 1,001 days. Each arm trains for 300 updates. Report signed-subgroup RMSE and MAE,
# normalized RMSE against the training-mean predictor, paired predictions, whether all
# outputs are inferred, and drift when valid target timestamps are poisoned.
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
# conditions = ["changed-associations", "hidden-values", "transformations", "representations"]
# ///

"""P115: predict signed Duration targets while hiding their timestamp sources."""

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
TRAIN_ROWS = 4096
VALIDATION_ROWS = 512
TEST_ROWS = 2048
ANCHOR = datetime(2026, 1, 1, tzinfo=timezone.utc)
ARMS = ("duration", "translated", "static", "number", "shuffled")


class DurationTarget(rf.Model):
    signal = rf.Number
    target = rf.Duration(query=("started_at", "ended_at"), granularity="hours", predict=True, objective="mse")


class StaticTarget(rf.Model):
    signal = rf.Number
    target = rf.Duration(query="ended_at", start=ANCHOR, granularity="hours", predict=True, objective="mse")


class NumericTarget(rf.Model):
    signal = rf.Number
    target = rf.Number(predict=True, objective="mse")


def records(*, rows: int, seed: int) -> Iterator[dict]:
    """Generate a clue, exact timestamp-derived target, and independently varying origin."""
    rng = np.random.default_rng(seed)
    for index in range(rows):
        signal = float(rng.uniform(-10, 10))
        duration = 1.75 * signal + float(rng.normal(0, 0.1))
        start = ANCHOR + timedelta(days=int(rng.integers(-500, 501)))
        end = start + timedelta(hours=duration)
        actual = (end - start).total_seconds() / 3600
        yield {
            "case": f"{seed}:{index}",
            "signal": signal,
            "started_at": start,
            "ended_at": end,
            "fixed_end": ANCHOR + timedelta(hours=actual),
            "target": actual,
        }


def frame(rows: list[dict], *, arm: str, targets: bool = True, shuffle: bool = False, seed: int = 0) -> pl.DataFrame:
    """Present only the matching target sources; prediction receives no answers."""
    indices = np.arange(len(rows))
    if shuffle:
        indices = np.random.default_rng(seed).permutation(indices)
    columns = {"signal": [rows[int(index)]["signal"] for index in indices]}
    if targets:
        if arm in ("duration", "translated"):
            delta = timedelta(days=371) if arm == "translated" else timedelta()
            columns["started_at"] = [row["started_at"] + delta for row in rows]
            columns["ended_at"] = [row["ended_at"] + delta for row in rows]
        elif arm == "static":
            columns["ended_at"] = [row["fixed_end"] for row in rows]
        else:
            columns["target"] = [row["target"] for row in rows]
    return pl.DataFrame(columns)


def build(arm: str) -> rf.Model:
    if arm in ("duration", "translated"):
        return DurationTarget(preset=CONTROL.name)
    if arm == "static":
        return StaticTarget(preset=CONTROL.name)
    if arm in ("number", "shuffled"):
        return NumericTarget(preset=CONTROL.name)
    raise ValueError(f"unknown Duration target arm {arm!r}")


def prediction(model: rf.Model, inputs: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    result = model.predict(inputs, batch_size=256)["predictions"].struct.field("target")
    return result.struct.field("content").to_numpy(), result.struct.field("inferred").to_numpy()


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


@experiment("P115", seed=11501)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train = list(records(rows=TRAIN_ROWS, seed=seed + 1))
    validate = list(records(rows=VALIDATION_ROWS, seed=seed + 2))
    test = list(records(rows=TEST_ROWS, seed=seed + 3))
    actual = np.asarray([row["target"] for row in test])
    mean = float(np.mean([row["target"] for row in train]))
    groups = {
        "overall": np.ones(len(test), dtype=bool),
        "negative": actual < 0,
        "nonnegative": actual >= 0,
    }
    output = {
        "budget": BUDGET,
        "rows": {"train": len(train), "validation": len(validate), "test": len(test)},
        "train_mean": mean,
        "arms": {},
        "test": {"case": [row["case"] for row in test], "signal": [row["signal"] for row in test], "target": actual},
    }
    checks = {}
    predictions = {}
    for arm in ARMS:
        lit.seed_everything(seed, workers=True)
        model = build(arm)
        updates = fit(
            model,
            frame(train, arm=arm, shuffle=arm == "shuffled", seed=seed + 4),
            frame(validate, arm=arm),
            seed,
            steps,
            accelerator,
        )
        inputs = frame(test, arm=arm, targets=False)
        predicted, inferred = prediction(model, inputs)
        if not np.isfinite(predicted).all():
            raise ValueError(f"{arm} produced nonfinite held-out predictions")
        predictions[arm] = predicted
        measured = {name: metrics(actual, predicted, mean, selection) for name, selection in groups.items()}
        measured.update(steps=updates, finite=True, inferred_all=bool(inferred.all()), predictions=predicted)
        if arm in ("duration", "translated", "static"):
            poison = inputs.with_columns(
                pl.lit(ANCHOR + timedelta(days=1000)).alias("ended_at"),
                pl.lit(ANCHOR - timedelta(days=1000)).alias("started_at"),
            )
            poisoned, poison_inferred = prediction(model, poison)
            if not np.isfinite(poisoned).all():
                raise ValueError(f"{arm} produced nonfinite poisoned-target predictions")
            measured["poison_max_drift"] = float(np.max(np.abs(predicted - poisoned)))
            measured["poison_inferred_all"] = bool(poison_inferred.all())
        output["arms"][arm] = measured
        checks[f"{arm}: full budget retained"] = updates == BUDGET
        checks[f"{arm}: finite and inferred predictions"] = measured["finite"] and measured["inferred_all"]
        if arm != "shuffled":
            checks[f"{arm}: held-out nRMSE below 0.25"] = measured["overall"]["nrmse"] < 0.25
        if arm in ("duration", "translated", "static"):
            checks[f"{arm}: poisoned target endpoints do not change predictions"] = (
                measured["poison_max_drift"] < 1e-5 and measured["poison_inferred_all"]
            )
    duration = output["arms"]["duration"]["overall"]["nrmse"]
    number = output["arms"]["number"]["overall"]["nrmse"]
    output["duration_number_prediction_rmse"] = float(
        np.sqrt(np.mean((predictions["duration"] - predictions["number"]) ** 2))
    )
    checks["Duration target stays within 0.03 nRMSE of Number"] = abs(duration - number) < 0.03
    checks["Translated timestamps preserve Duration target predictions"] = (
        float(np.max(np.abs(predictions["duration"] - predictions["translated"]))) < 1e-5
    )
    checks["Shuffled signal retains nRMSE above 0.85"] = output["arms"]["shuffled"]["overall"]["nrmse"] > 0.85
    checks["Duration improves on shuffled signal by 0.50 nRMSE"] = (
        output["arms"]["shuffled"]["overall"]["nrmse"] - duration > 0.50
    )
    return output, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
