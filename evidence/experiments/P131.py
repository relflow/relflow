# %% [markdown]
# ---
# title: Can one model learn supervised and reconstruction tasks on different fields?
# description: Fit two independent calibrations together and change the source of each answer separately.
# evidence-group: Masking
# evidence-readout:
# - {label: Supervised error, metric: [supervised_error], format: error}
# - {label: Reconstruction error, metric: [reconstruction_error], format: error}
# evidence-review:
#   question: Does a model learn both tasks from their respective inputs without reading either hidden answer?
#   modeling: Each observation has two independent reference readings. A supervised result, always hidden, follows the primary reference, while a meter reading follows the secondary reference and is reconstructed only where a Boolean review flag selects it.
#   comparison:
#     kind: Controlled comparison
#     metric: Normalized RMSE
#     conditions:
#     - {label: Correct references, path: [aligned_error], format: number}
#     - {label: Reassigned reference, path: [control_error], format: number}
#     definition: Errors are divided by the corresponding training target standard deviation. Summaries take the worst aligned and best reassigned result across both tasks and all seeds.
#     direction: Lower error indicates better recovery of the held-out answer.
#   note: This measures two disjoint numeric targets with independent linear signals.
# evidence-plot:
#   mode: summary
#   focus: Do both learned outputs depend on their own reference signal?
#   comparison: Reassign the source of one answer at a time in the same held-out observations.
#   limit: Each calibration has one informative scalar input.
# ---
#
# ## Example
#
# A primary reference of 0.2 gives a supervised result of 0.6. A secondary
# reference of -0.3 gives a meter reading of 0.8. Marking the meter for review
# requests its reconstruction alongside the supervised result.
#
# ```yaml
# primary: 0.2
# secondary: -0.3
# settled: 0.6
# meter: 0.8
# review: true
# ```
#
# ## Comparison
#
# Rotate just the primary reference, then just the secondary reference.
# Score the affected task against its original answers. Repeat the aligned
# request with both hidden answers overwritten. On a mixed set of requests,
# audit which rows receive inferred answers for each task.
#
# ## Result
#
# Both tasks must achieve normalized error below 0.30 on every seed.
# Reassigning the corresponding reference must raise each error above 0.80.
# Replacing the answers must preserve both predictions.
#
# ## Method
#
# For each of three seeds, draw 1,024 training, 192 validation and 512 held-out
# records independently. Flag half the records for reconstruction in expectation.
# Train a width-32 model for 320 AdamW steps with learning rate 0.002 and batches
# of 32. Request every meter reading for the calibration and leakage comparisons.
#
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["lightning", "matplotlib", "numpy", "polars", "pyyaml>=6", "relflow>=0.5.0", "torch"]
#
# [tool.relflow.evidence]
# capability = "training"
# types = ["Number"]
# conditions = ["hidden-values", "changed-associations", "objectives"]
# ///

"""Measure two task contracts with independent signals and matched controls."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
from policies import CONTROL
from reporting import experiment, report
from visibility import TARGET, reconstruct


class Calibration(rf.Model):
    primary = rf.Number
    secondary = rf.Number
    settled = rf.Number(objective="mse", **TARGET)
    meter = rf.Number(objective="mse", **reconstruct("review"))


def observations(seed, count):
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(-1.0, 1.0, (2, count))
    for first, second, review in zip(a, b, rng.random(count) < 0.5, strict=True):
        yield {
            "primary": first,
            "secondary": second,
            "settled": first * 3,
            "meter": 0.5 - second,
            "review": bool(review),
        }


def readout(model, rows):
    results = model.predict(rows, batch_size=128)["predictions"].struct
    return np.column_stack([results.field(name).struct.field("content").to_numpy() for name in ("settled", "meter")])


def trial(seed, steps, accelerator):
    lit.seed_everything(seed, workers=True)
    train, val, test = (
        pl.DataFrame(observations(seed + offset, count)) for offset, count in ((1, 1024), (2, 192), (3, 512))
    )
    model = Calibration(preset=CONTROL.name, d_model=32)
    data = rf.PolarsDataModule(
        train=train, validate=val, seed=seed, num_workers=0, persistent_workers=False, pin_memory=False
    )
    model.fit(
        data,
        max_steps=320 if steps is None else min(steps, 320),
        max_epochs=-1,
        batch_size=32,
        learning_rate=0.002,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        accelerator=accelerator,
        strategy="auto",
        devices=1,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    routing = model.encode(test, strata="predict")
    flags = test["review"].to_numpy()
    routed = bool(
        routing["/settled"].inferred.all() and np.array_equal(routing["/meter"].inferred.flatten().cpu().numpy(), flags)
    )
    expected = test.select("settled", "meter").to_numpy()
    scale = train.select("settled", "meter").to_numpy().std(axis=0)
    requests = test.with_columns(review=pl.lit(True))
    aligned = readout(model, requests)
    poisoned = readout(model, requests.with_columns(settled=pl.lit(1234.0), meter=pl.lit(-1234.0)))
    errors = np.sqrt(np.mean((aligned - expected) ** 2, axis=0)) / scale
    controls = []
    for i, name in enumerate(("primary", "secondary")):
        reassigned = requests.with_columns(pl.Series(name, np.roll(requests[name].to_numpy(), 1)))
        estimate = readout(model, reassigned)[:, i]
        controls.append(float(np.sqrt(np.mean((estimate - expected[:, i]) ** 2)) / scale[i]))
    return {
        "seed": seed,
        "steps": model.trainer.global_step,
        "supervised": float(errors[0]),
        "reconstructed": float(errors[1]),
        "controls": controls,
        "drift": float(np.abs(aligned - poisoned).max()),
        "routing": routed,
    }


@experiment("P131", seed=13001)
def run(seed, steps, accelerator):
    trials = [trial(seed + 10 * offset, steps, accelerator) for offset in range(3)]
    supervised = max(t["supervised"] for t in trials)
    reconstructed = max(t["reconstructed"] for t in trials)
    metrics = {
        "trials": trials,
        "supervised_error": supervised,
        "reconstruction_error": reconstructed,
        "aligned_error": max(supervised, reconstructed),
        "control_error": min(min(t["controls"]) for t in trials),
    }
    return metrics, {
        "Both disjoint tasks learn their calibration": metrics["aligned_error"] < 0.30,
        "Each task needs its own reference": metrics["control_error"] > 0.80,
        "Supplying different hidden answers preserves both predictions": all(t["drift"] < 1e-6 for t in trials),
        "Only flagged readings request reconstruction": all(t["routing"] for t in trials),
    }


if __name__ == "__main__":
    report(run)
