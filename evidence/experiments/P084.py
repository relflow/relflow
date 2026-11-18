# %% [markdown]
# ---
# title: Does recent history survive a real collection cap?
# description: An event branch retains four recent amounts from an over-cap history and predicts their total.
# evidence-readout:
# - label: Tail model
#   metric:
#   - tail
#   - familiar
#   - normalized_rmse
#   format: number
# - label: Head control
#   metric:
#   - head
#   - familiar
#   - normalized_rmse
#   format: number
# evidence-group: Temporal
# evidence-review:
#   question: Does recent history survive a real collection cap?
#   modeling: Generate histories of at least eight independent amounts and hide the total of the last four. Train
#     matched models that retain either the first four events or the last four under a collection cap. Compare their
#     error on the same full test histories.
#   comparison:
#     metric: Normalized RMSE
#     kind: Controlled comparison
#     conditions:
#     - label: Retain the first four events
#       path:
#       - head
#       - familiar
#       - normalized_rmse
#       format: number
#     - label: Retain the last four events
#       path:
#       - tail
#       - familiar
#       - normalized_rmse
#       format: number
#     definition: Root mean squared error divided by the reference scale defined in this experiment. Zero is exact;
#       one equals that reference scale.
#     direction: Lower error is better
#   note: First and last windows are disjoint. The head model cannot see any target-defining amount; prefix-invariance
#     checks confirm that discarded records cannot affect the tail model.
# evidence-plot:
#   target: answer
#   focus: Which retained events determine a recent-history prediction?
#   comparison: Compare crop policies and interventions inside or outside the retained history.
#   limit: An event outside the retained window supplies no model input, even when present in the source record.
#   renderer: plot
# ---
#
# The event list is ordered from oldest to newest. Only the last four amounts determine the hidden outcome.
# In the illustration, the older large amounts are outside the retained tail, so the answer is
# 0.2-0.1+0.5+0.4=1.0. The example uses conspicuous old values to explain cropping; measured training
# amounts are all between minus one and one.
#
# ## Example
#
# ```yaml
# events:
#   - {amount: 100.0}
#   - {amount: 200.0}
#   - {amount: 300.0}
#   - {amount: 400.0}
#   - {amount: 0.2}
#   - {amount: -0.1}
#   - {amount: 0.5}
#   - {amount: 0.4}
# answer: 1.0
# ```
#
# ## Comparison
#
# Train tail-cropped and head-cropped models on exactly the same full histories and targets. At least eight
# events ensure the first and last four are disjoint and independent. For the fitted tail model, supplying
# just the retained tail, or replacing and lengthening the discarded prefix, should preserve predictions.
# Changing retained amounts should produce the newly computed answer.
#
# ## Result
#
# These checks define a new experiment. A full recorded run is required before claiming that the model
# meets them. Lower normalized error is better; a value of one matches the stated constant reference. The
# numerical gates are provisional and must be calibrated on separate validation seeds rather than adjusted
# after seeing acceptance-test outcomes.
#
# ## Method
#
# ### Process, controls, and held-out evaluation
#
# Independent streams provide 4096 training, 512 validation, and 1024 test histories per condition.
# Training source lengths are 8-12; a separate test uses 13-16. Both models have capacity 4, identical
# initialization, 500 AdamW updates, batch 64, learning rate 0.002, weight decay 0.01, FP32, and no
# checkpoint selection. The head arm cannot observe any target-defining amount. Discarded-prefix
# interventions replace old amounts by values above 1000 and prepend three extra old records; the retained
# suffix remains byte-for-byte equivalent. The retained intervention applies 0.5*x+0.2 to those four
# amounts and recomputes the target. Tests report quality and structural invariants separately. This is a
# tail-retention experiment; random-window augmentation requires a different, window-observable label.
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
# capability = "structure"
# types = []
# conditions = ["transformations", "varying-collections"]
# ///

"""P084: learn the retained tail while discarded history remains causally irrelevant."""

import sys
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
import torch
from policies import CONTROL
from reporting import experiment, report

BUDGET = 500
TRAIN_ROWS = 4096
TEST_ROWS = 1024
CAPACITY = 4


def records(*, rows: int, seed: int, long: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        length = int(rng.integers(13, 17) if long else rng.integers(8, 13))
        amounts = rng.uniform(-1.0, 1.0, length)
        yield {"events": [{"amount": float(value)} for value in amounts], "answer": float(amounts[-CAPACITY:].sum())}


class Event(rf.Branch):
    amount = rf.Number


def build(overflow: str) -> rf.Model:

    class Model(rf.Model):
        events = Event(length=CAPACITY, overflow=overflow, reduction=rf.Attention())
        answer = rf.Number(mask=True, objective="mse")

    model = Model(preset=CONTROL.name)
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    output = pl.DataFrame(
        model.predict(pa.Table.from_pylist([{"events": row["events"]} for row in rows]), batch_size=128)
    )["predictions"].to_list()
    return np.asarray([row["answer"]["content"] for row in output])


def score(predicted: np.ndarray, rows: list[dict], mean: float) -> dict:
    target = np.asarray([row["answer"] for row in rows])
    rmse = float(np.sqrt(np.mean((predicted - target) ** 2)))
    return {"rmse": rmse, "normalized_rmse": rmse / float(np.sqrt(np.mean((target - mean) ** 2)))}


def intervene(rows: list[dict], mode: str) -> list[dict]:
    """Change discarded or retained records, with corresponding explicit target semantics."""
    result = []
    for row in rows:
        tail = row["events"][-CAPACITY:]
        if mode == "trim":
            events = tail
        elif mode == "discarded":
            events = [{"amount": 1000.0 + index} for index in range(len(row["events"]) - CAPACITY + 3)] + tail
        else:
            events = row["events"][:-CAPACITY] + [{"amount": 0.5 * item["amount"] + 0.2} for item in tail]
        answer = sum(item["amount"] for item in events[-CAPACITY:])
        result.append({"events": events, "answer": answer})
    return result


def fit(model: rf.Model, train: list[dict], validate: list[dict], *, seed: int, budget: int, accelerator: str) -> int:
    """Own the fixed optimizer budget and independent synthetic splits."""
    data = rf.ArrowDataModule(
        train=pa.Table.from_pylist(train),
        validate=pa.Table.from_pylist(validate),
        seed=seed,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    model.fit(
        data,
        batch_size=64,
        learning_rate=0.002,
        weight_decay=0.01,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=budget,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    model.eval()
    return int(model.trainer.global_step)


@experiment("P084", seed=8401)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train = list(records(rows=TRAIN_ROWS, seed=seed * 100 + 1))
    validate = list(records(rows=512, seed=seed * 100 + 2))
    tests = {
        "familiar": list(records(rows=TEST_ROWS, seed=seed * 100 + 3)),
        "longer_source": list(records(rows=TEST_ROWS, seed=seed * 100 + 4, long=True)),
    }
    mean = float(np.mean([row["answer"] for row in train]))
    metrics, checks = {}, {}
    for overflow in ("tail", "head"):
        lit.seed_everything(seed, workers=True)
        model = build(overflow)
        completed = fit(
            model,
            train,
            validate,
            seed=seed,
            budget=BUDGET if steps is None else min(steps, BUDGET),
            accelerator=accelerator,
        )
        before = deepcopy(model.state_dict())
        arm = {"steps": completed}
        checks[f"{overflow}: full budget completed"] = completed == BUDGET
        for name, rows in tests.items():
            predicted = predict(model, rows)
            arm[name] = score(predicted, rows, mean)
            checks[f"{overflow}/{name}: finite predictions"] = bool(np.isfinite(predicted).all())
            if overflow == "tail":
                for mode in ("trim", "discarded", "retained"):
                    changed = intervene(rows, mode)
                    output = predict(model, changed)
                    measured = score(output, changed, mean)
                    measured["maximum_prediction_drift"] = float(np.max(np.abs(output - predicted)))
                    arm[name][mode] = measured
                    if mode in ("trim", "discarded"):
                        checks[f"{name}/{mode}: prediction drift below 0.00001"] = (
                            measured["maximum_prediction_drift"] < 1e-5
                        )
                    else:
                        checks[f"{name}: changed retained tail remains below 0.45 normalized RMSE"] = (
                            measured["normalized_rmse"] < 0.45
                        )
        checks[f"{overflow}: evaluation preserves full state"] = all(
            (
                torch.equal(value, model.state_dict()[name])
                if isinstance(value, torch.Tensor)
                else value == model.state_dict()[name]
            )
            for name, value in before.items()
        )
        metrics[overflow] = arm
    for name in tests:
        checks[f"{name}: tail normalized RMSE below 0.35"] = metrics["tail"][name]["normalized_rmse"] < 0.35
        checks[f"{name}: head control normalized RMSE above 0.80"] = metrics["head"][name]["normalized_rmse"] > 0.80
    return metrics, checks


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    from evidence.diagnostics.data import human
    from evidence.diagnostics.plots import axes_pair, dots

    m = bundle.record["metrics"]
    populations = ["familiar", "longer_source"]
    fig, axes = axes_pair(width=10, height=4.8)
    dots(
        axes[0],
        ["Familiar source lengths", "Longer source lengths"],
        {
            "Keep recent tail": [m["tail"][p]["normalized_rmse"] for p in populations],
            "Keep oldest head": [m["head"][p]["normalized_rmse"] for p in populations],
            "Tail after retained values change": [m["tail"][p]["retained"]["normalized_rmse"] for p in populations],
        },
        xlabel="Test RMSE / constant-guess RMSE",
        reference=(1, "Constant-guess error"),
    )
    labels = [
        ("Familiar source" if p == "familiar" else "Longer source") + " · " + human(mode)
        for p in populations
        for mode in ["trim", "discarded", "retained"]
    ]
    dots(
        axes[1],
        labels,
        {
            "Paired output drift": [
                m["tail"][p][mode]["maximum_prediction_drift"]
                for p in populations
                for mode in ["trim", "discarded", "retained"]
            ]
        },
        xlabel="Maximum absolute prediction change",
        reference=(1e-05, "Declared invariance tolerance"),
    )
    axes[1].set_xscale("symlog", linthresh=1e-06)
    maximum = max(
        (
            m["tail"][p][mode]["maximum_prediction_drift"]
            for p in populations
            for mode in ["trim", "discarded", "retained"]
        )
    )
    axes[1].set_xlim(-2.5e-07, maximum * 3)
    axes[1].set_xticks([0, 1e-05, 0.001, 0.1, 1], labels=["0", "10⁻⁵", "10⁻³", "0.1", "1"])
    axes[1].minorticks_off()
    return (
        fig,
        "Tail and head policies use the same fixed training budget and independent test populations. Trim removes discarded events; discarded changes only events outside the retained tail. Retained changes the tail and recomputes its answer. The drift axis is linear around zero and logarithmic beyond 0.000001; exact zero is retained.",
        {},
    )


if __name__ == "__main__":
    report(run)
