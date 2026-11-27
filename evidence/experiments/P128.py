# %% [markdown]
# ---
# title: Can a child branch learn all its targets when one answer is intermittently absent?
# description: Predict paired calibration results from a reference reading and measure the effect of dropping one target on individual rows.
# evidence-group: Masking
# evidence-readout:
# - {label: Complete target, metric: [complete_error], format: error}
# - {label: Intermittent target, metric: [intermittent_error], format: error}
# evidence-review:
#   question: Can branch supervision recover two numeric answers when one leaf loses half its training observations?
#   modeling: Each observation has one reference reading and up to four replicate slots. A child branch predicts two deterministic calibration results per real slot. Leaf ablation drops one result independently per replicate during training.
#   comparison:
#     kind: Controlled comparison
#     metric: Normalized RMSE
#     conditions:
#     - {label: Correct reference, path: [intermittent_error], format: number}
#     - {label: Reassigned reference, path: [control_error], format: number}
#     definition: RMSE divided by the corresponding training target standard deviation. The summary uses the worst correct-reference score and best control score across seeds.
#     direction: Lower error indicates more accurate calibration.
#   note: Replicates share a deterministic answer for a given reference reading.
# evidence-plot:
#   mode: summary
#   focus: Does the intermittent target remain predictable from its reference reading?
#   comparison: Compare its held-out error with the reference intact and reassigned.
#   limit: The synthetic calibration functions are linear.
# ---
#
# ## Example
#
# A reference reading of 0.4 produces a doubled reading of 0.8 and an inverted
# reading of -0.4 at each replicate. A null replicate has neither answer.
#
# ```yaml
# reference: 0.4
# replicates: [{doubled: 0.8, inverted: -0.4}, null]
# ```
#
# ## Comparison
#
# Request predictions using empty records where answers belong. Repeat with
# deliberately incorrect answers in those slots, then reassign the reference
# reading while keeping the original answers for scoring.
#
# ## Result
#
# Both targets must achieve normalized RMSE below 0.30 on each seed. Reassigning
# the reference must raise the intermittent target's error above 0.80. Incorrect
# supplied answers must leave predictions unchanged, including the null slots.
#
# ## Method
#
# Draw independent uniform references for 1,024 training, 192 validation and
# 512 test observations. Each contains one to four slots with a 10% chance
# of a null record per slot. Train for 320 AdamW steps at learning rate 0.002,
# batch size 32, on each of three seeds. Check the training target counts
# separately on a fresh model to keep test values out of the fitted statistics.
#
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["lightning", "numpy", "polars", "pyyaml>=6", "relflow>=0.6.0", "torch"]
#
# [tool.relflow.evidence]
# capability = "reliability"
# types = ["Number"]
# conditions = ["missing-inputs", "hidden-values", "objectives"]
# ///

"""Score supervised replicate targets with independently ablated observations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
from policies import CONTROL
from reporting import experiment, report


class Model(rf.Model):
    reference = rf.Number
    replicates = rf.Branch(
        length=4, predict=True, doubled=rf.Number(objective="mse"), inverted=rf.Number(objective="mse", ablate=0.5)
    )


def records(seed, count):
    rng = np.random.default_rng(seed)
    for x in rng.uniform(-1, 1, size=count):
        yield {
            "reference": x,
            "replicates": [
                None if rng.random() < 0.1 else {"doubled": 2 * x, "inverted": -x} for _ in range(rng.integers(1, 5))
            ],
        }


def targets(rows):
    return np.asarray([[v["doubled"], v["inverted"]] for r in rows for v in r["replicates"] if v is not None])


def readout(model, rows, references, *, poison=False):
    # An unmodeled field carries record validity when both target columns are omitted.
    slot = {"doubled": 99.0, "inverted": -99.0} if poison else {"slot": True}
    slots = [[None if v is None else slot for v in row["replicates"]] for row in rows]
    source = pl.DataFrame({"reference": references, "replicates": slots})
    results = model.predict(source, batch_size=64)["predictions"].struct.field("replicates").to_list()
    aligned = [[v is None for v in row] for row in results] == [[v is None for v in row] for row in slots]
    values = [[v["doubled"]["content"], v["inverted"]["content"]] for row in results for v in row if v is not None]
    return np.asarray(values), aligned


def trial(seed, steps, accelerator):
    lit.seed_everything(seed, workers=True)
    train, val, test = list(records(seed + 1, 1024)), list(records(seed + 2, 192)), list(records(seed + 3, 512))
    model = Model(preset=CONTROL.name, d_model=32)
    data = rf.PolarsDataModule(
        train=pl.DataFrame(train),
        validate=pl.DataFrame(val),
        seed=seed,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
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
    references = [r["reference"] for r in test]
    expected, scale = targets(test), targets(train).std(axis=0)
    fitted, aligned = readout(model, test, references)
    poisoned, aligned_poison = readout(model, test, references, poison=True)
    control, aligned_control = readout(model, test, np.roll(references, 1))
    error = np.sqrt(np.mean((fitted - expected) ** 2, axis=0)) / scale
    audit = Model(preset=CONTROL.name, d_model=32)
    counts = {}
    for split in ("train", "validate", "test"):
        batch = audit.encode(pl.DataFrame(test), strata=split, seed=seed)
        counts[split] = [int(batch[f"/replicates/{key}"].trainable.sum()) for key in ("doubled", "inverted")]
    return {
        "seed": seed,
        "steps": model.trainer.global_step,
        "counts": counts,
        "slots": len(expected),
        "aligned": aligned and aligned_poison and aligned_control,
        "drift": float(np.abs(fitted - poisoned).max()),
        "complete": float(error[0]),
        "intermittent": float(error[1]),
        "control": float(np.sqrt(np.mean((control[:, 1] - expected[:, 1]) ** 2)) / scale[1]),
    }


@experiment("P128", seed=12701)
def run(seed, steps, accelerator):
    trials = [trial(seed + i * 10, steps, accelerator) for i in range(3)]
    metrics = {
        "trials": trials,
        "complete_error": max(t["complete"] for t in trials),
        "intermittent_error": max(t["intermittent"] for t in trials),
        "control_error": min(t["control"] for t in trials),
    }
    return metrics, {
        "Both child targets learn their calibration functions": max(
            metrics["complete_error"], metrics["intermittent_error"]
        )
        < 0.30,
        "Reassigning references breaks prediction": metrics["control_error"] > 0.80,
        "Poisoned answers preserve every prediction and slot": all(t["aligned"] and t["drift"] < 1e-6 for t in trials),
        "Only the chosen leaf loses training targets": all(
            t["counts"]["train"][0] == t["slots"]
            and 0.4 < t["counts"]["train"][1] / t["slots"] < 0.6
            and t["counts"]["validate"] == t["counts"]["test"] == [t["slots"]] * 2
            for t in trials
        ),
    }


if __name__ == "__main__":
    report(run)
