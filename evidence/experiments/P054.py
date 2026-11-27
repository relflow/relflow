# %% [markdown]
# ---
# title: Can it predict the next value without seeing the future?
# description: Expose the beginning of a simple numeric sequence, hide its remaining events, and predict the next
#   value.
# evidence-readout:
# - label: Next value after two events
#   metric:
#   - prefixes
#   - '2'
#   - nrmse
#   format: error
# - label: Short rollout
#   metric:
#   - rollout
#   - nrmse
#   format: error
# evidence-group: Masking
# evidence-review:
#   question: Can relflow predict the next value without seeing the future?
#   modeling: Generate straight-line sequences with a fresh starting value and increment for every record. Show only
#     the first two to four values and hide the future. For the primary comparison, roll predictions forward from
#     two true starting values and compare with always repeating the last observed value.
#   comparison:
#     metric: RMSE
#     kind: Baseline reference
#     conditions:
#     - label: Repeat the last observed value
#       path:
#       - rollout
#       - persistence_rmse
#       format: number
#     - label: Model rollout
#       path:
#       - rollout
#       - rmse
#       format: number
#     definition: Root mean squared prediction error, in the target’s original units. Zero is an exact prediction.
#     direction: Lower error is better
#   note: This is a short synthetic linear rollout, not a general time-series forecast. Separate tests score each
#     prefix length and verify that hidden future values cannot leak.
# evidence-plot:
#   target: events/value
#   focus: Can visible prefixes support the next hidden value and a rollout?
#   comparison: Separate one-step prefix predictions, swapped prefixes, and multi-step generated continuations.
#   limit: Rollout error and one-step normalized error have different scales and are shown separately.
#   renderer: plot
# evidence-cost:
#   baseline: 40
#   candidate: 40
# ---
#
# ## Example
#
# Each five-event sequence follows one constant increment. The visible prefix `0.2, 0.5` suggests an
# increment of `0.3`, giving the next hidden value `0.8`, then `1.1` and `1.4`. These are illustrative
# arithmetic targets, not measured predictions. `future` excludes the suffix from model input, while `next`
# requests exactly one coordinate for reconstruction. Training varies the starting value, increment, and cut
# after two, three, or four visible events.
#
# ```yaml
# events:
#   - {value: 0.2, future: false, next: false}
#   - {value: 0.5, future: false, next: false}
#   - {value: 0.8, future: true, next: true}
#   - {value: 1.1, future: true, next: false}
#   - {value: 1.4, future: true, next: false}
# ```
#
# Events marked future are hidden. next selects the single value to predict.
#
# ## Comparison
#
# Replacing all stored future values with large unrelated numbers tests whether hidden suffixes can leak
# into the answer. Swapping the visible prefix with another record's prefix should instead change the
# prediction and worsen agreement with the original target. The separate rollout starts with two true
# events, then feeds each generated value back as visible context for the next step. It never supplies the
# true held-out suffix to correct earlier prediction errors.
#
# Change hidden future values to check they cannot influence prediction. Then feed predicted values back for a short rollout.
#
# ## Result
#
# Single-step errors use a constant training-target guess for the corresponding sequence position. Rollout
# error summarizes the three generated suffix values and is also compared with persistence, which repeats
# the last observed value. These measurements distinguish predicting one next step from remaining accurate
# after earlier generated errors enter the context. The sequences are short, noiseless arithmetic
# progressions; the result does not establish open-ended sequence modeling, language generation, or
# forecasting with changing dynamics.
#
# Next-value prediction and the short rollout work on these sequences. This does not establish language generation or general forecasting.
#
# ## Method
#
# ### Data and model
#
# A whole sequence draws `offset ~ Uniform(-1, 1)` and
# `increment ~ Uniform(-0.4, 0.4)`, then `value[t] = offset + t*increment`.
# Neither latent parameter is exposed. Choose a prefix length uniformly from
# 2, 3, and 4. A preprocessor retains each prefix value as context and creates
# a supervised answer slot only for the first future value. Whole sequences are independently generated for
# training (4,096), validation (512), and testing (1,024).
#
# ### Training and controls
#
# Fit for 800 AdamW updates at learning rate 0.002. Score one-step predictions
# separately for all three prefix lengths, requiring nRMSE below 0.35. A rollout
# from the first two true values must achieve suffix nRMSE below 0.45 and beat
# persistence (repeating the second value). Baselines use training means at each
# position, not held-out statistics. Permuting entire visible prefixes between
# records retains the original labels and must give nRMSE above 0.85.
#
# Poison all suffix values, including the selected target: drift must remain
# below 1e-5. Training objectives and inference flags must select exactly one
# coordinate, while encoder presence selects exactly the prefix. The supplied
# correct suffix is used only as an evaluation label, never in the rollout.
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
# capability = "training"
# types = []
# conditions = ["changed-associations", "hidden-values", "objectives"]
# ///

"""P054: leakage-free prefix conditioning and free-running arithmetic-sequence rollout."""

import sys
from collections.abc import Iterator
from functools import partial
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
from reporting import experiment, report
from visibility import TARGET

LENGTH = 5
BUDGET = 800


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        offset = float(rng.uniform(-1.0, 1.0))
        increment = float(rng.uniform(-0.4, 0.4))
        cut = int(rng.integers(2, LENGTH))
        yield {
            "events": [
                {"value": offset + index * increment, "future": index >= cut, "next": index == cut}
                for index in range(LENGTH)
            ]
        }


@rf.preprocess
def prepare(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep observed values and requested answers in their original row slots."""
    row = pl.element()
    value = pl.struct(value=row.struct.field("value"))
    context = pl.when(~row.struct.field("future")).then(value)
    answer = pl.when(row.struct.field("next")).then(value)
    slots = pl.struct(context=context, answers=pl.concat_list(answer))
    return frame.with_columns(pl.col("events").list.eval(slots))


class Event(rf.Branch):
    value = rf.Number(query="context.value")
    answers = rf.Branch(length=1, reduction=None, value=rf.Number(objective="mse", **TARGET))


class PredictionModel(rf.Model):
    events = Event(length=LENGTH, reduction=None)


def build() -> rf.Model:
    return PredictionModel(d_model=32, n_layers=2, n_heads=4, dropout=0.0)


def prediction(model: rf.Model, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    output = pl.DataFrame(model.predict(pa.Table.from_pylist(rows), preprocess=prepare))["predictions"].to_list()
    content = np.asarray(
        [
            [item["answers"][0]["value"]["content"] if item["answers"][0] else 0.0 for item in row["events"]]
            for row in output
        ],
        dtype=np.float64,
    ).reshape(len(rows), LENGTH)
    inferred = np.asarray(
        [
            [bool(item["answers"][0] and item["answers"][0]["value"]["inferred"]) for item in row["events"]]
            for row in output
        ],
        dtype=bool,
    ).reshape(len(rows), LENGTH)
    return content, inferred


def rollout(model: rf.Model, rows: list[dict]) -> np.ndarray:
    """Expose generated values only; never teacher-force the held-out suffix."""
    values = np.zeros((len(rows), LENGTH), dtype=np.float64)
    values[:, :2] = [[event["value"] for event in row["events"][:2]] for row in rows]
    for cut in range(2, LENGTH):
        inputs = [
            {
                "events": [
                    {"value": float(value), "future": index >= cut, "next": index == cut}
                    for index, value in enumerate(sequence)
                ]
            }
            for sequence in values
        ]
        content, _ = prediction(model, inputs)
        values[:, cut] = content[:, cut]
    return values


@experiment("P054", seed=5401)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    data = rf.SyntheticDataModule(
        preprocessor=prepare,
        train=partial(records, rows=4096, seed=seed + 1),
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
    trainer = model.trainer
    model.eval()
    train = list(records(rows=4096, seed=seed + 1))
    test = list(records(rows=1024, seed=seed + 3))
    actual = np.asarray([[event["value"] for event in row["events"]] for row in test])
    mean = np.asarray([[event["value"] for event in row["events"]] for row in train]).mean(axis=0)
    order = np.random.default_rng(seed + 4).permutation(len(test))
    metrics = {"steps": trainer.global_step, "prefixes": {}}
    checks = {}
    # A separate model audits target selection without fitting held-out moments.
    probe = build()
    for cut in range(2, LENGTH):
        selected = [
            {
                "events": [
                    {**event, "future": index >= cut, "next": index == cut} for index, event in enumerate(row["events"])
                ]
            }
            for row in test
        ]
        content, inferred = prediction(model, selected)
        poisoned = [
            {
                "events": [
                    {**event, "value": 1000.0 + index} if index >= cut else event
                    for index, event in enumerate(row["events"])
                ]
            }
            for row in selected
        ]
        shuffled = [
            {
                "events": [
                    {**event, "value": float(actual[other, index])} if index < cut else event
                    for index, event in enumerate(row["events"])
                ]
            }
            for row, other in zip(selected, order, strict=True)
        ]
        replaced, _ = prediction(model, poisoned)
        broken, _ = prediction(model, shuffled)
        baseline = float(np.sqrt(np.mean((actual[:, cut] - mean[cut]) ** 2)))
        measured = {
            "nrmse": float(np.sqrt(np.mean((content[:, cut] - actual[:, cut]) ** 2))) / baseline,
            "baseline_rmse": baseline,
            "shuffled_nrmse": float(np.sqrt(np.mean((broken[:, cut] - actual[:, cut]) ** 2))) / baseline,
            "future_value_drift": float(np.max(np.abs(content[:, cut] - replaced[:, cut]))),
        }
        metrics["prefixes"][cut] = measured
        checks[f"prefix {cut}: next-value nRMSE below 0.35"] = measured["nrmse"] < 0.35
        checks[f"prefix {cut}: shuffled prefix nRMSE above 0.85"] = measured["shuffled_nrmse"] > 0.85
        checks[f"prefix {cut}: future cannot affect prediction"] = measured["future_value_drift"] < 1e-5
        fields = probe.encode(pa.Table.from_pylist(selected), preprocess=prepare, strata="train")
        expected = np.broadcast_to(np.arange(LENGTH) == cut, inferred.shape)
        visible = np.broadcast_to(np.arange(LENGTH) < cut, inferred.shape)
        checks[f"prefix {cut}: exactly the next coordinate is requested"] = bool(np.array_equal(inferred, expected))
        checks[f"prefix {cut}: exactly the next coordinate is trained"] = bool(
            np.array_equal(fields["/events/answers/value"].trainable.cpu().numpy().reshape(expected.shape), expected)
        )
        checks[f"prefix {cut}: only prefix values are encoder inputs"] = bool(
            np.array_equal(fields["/events/value"].present.cpu().numpy().reshape(visible.shape), visible)
            and not fields["/events/answers/value"].present.any()
        )
    generated = rollout(model, test)
    baseline = float(np.sqrt(np.mean((actual[:, 2:] - mean[2:]) ** 2)))
    rmse = float(np.sqrt(np.mean((generated[:, 2:] - actual[:, 2:]) ** 2)))
    persistence = float(np.sqrt(np.mean((actual[:, 1:2] - actual[:, 2:]) ** 2)))
    metrics["rollout"] = {
        "rmse": rmse,
        "baseline_rmse": baseline,
        "nrmse": rmse / baseline,
        "persistence_rmse": persistence,
    }
    checks["Free-running suffix nRMSE below 0.45"] = metrics["rollout"]["nrmse"] < 0.45
    checks["Rollout beats persistence"] = rmse < persistence
    return metrics, checks


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    from evidence.diagnostics.plots import axes_pair, dots

    m = bundle.record["metrics"]
    fig, axes = axes_pair(width=10, height=4.5)
    keys = sorted(m["prefixes"], key=int)
    dots(
        axes[0],
        [f"{key} visible events" for key in keys],
        {
            "Actual prefix": [m["prefixes"][key]["nrmse"] for key in keys],
            "Swapped prefix": [m["prefixes"][key]["shuffled_nrmse"] for key in keys],
        },
        xlabel="One-step normalized RMSE",
    )
    dots(
        axes[1],
        ["Autoregressive rollout", "Last-value persistence"],
        {"Source-unit RMSE": [m["rollout"]["rmse"], m["rollout"]["persistence_rmse"]]},
        xlabel="Multi-step rollout RMSE in source units",
    )
    return (
        fig,
        "One-step predictions and repeated rollout are different tests and use separate scales. Only evaluated prefix lengths are shown; future-value poisoning remains an independent masking check.",
        {},
    )


if __name__ == "__main__":
    report(run)
