# %% [markdown]
# ---
# title: Does recent history survive a real collection cap?
# categories:
# - Temporal
# description: An event branch retains four recent amounts from an over-cap history and predicts their total.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
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
# ```{typst}
# //| label: fig-evidence-p084
# //| fig-cap: "Visible inputs and the hidden prediction target for this controlled task."
# //| fig-alt: "An event branch retains four recent amounts from an over-cap history and predicts their total."
# #tree(node("history", kind: "root", children: (node("events", kind: "branch", detail: "Retain last four", children: (node("amount", type: "Number"),)), node("answer", kind: "target", type: "Number"),)))
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
# {{< evidence P084 status >}}
#
# These checks define a new experiment. A full recorded run is required before claiming that the model
# meets them. Lower normalized error is better; a value of one matches the stated constant reference. The
# numerical gates are provisional and must be calibrated on separate validation seeds rather than adjusted
# after seeing acceptance-test outcomes.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P084 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P084 script >}}
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
# ### Complete experiment code

# %%
"""P084: learn the retained tail while discarded history remains causally irrelevant."""

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
import torch
from relflow.presets.base import resolve
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


def build(overflow: str) -> rf.Model:
    # Freeze the experimental architecture independently of SDK preset changes.
    control_preset = rf.presets.Preset(
        name="xs",
        d_model=64,
        root=rf.presets.BranchDefaults(n_layers=1, n_heads=2, reduction=rf.Attention()),
        branch=rf.presets.BranchDefaults(n_layers=1, n_heads=2, reduction=rf.Attention()),
        leaf=rf.presets.LeafDefaults(n_heads=2),
    )
    model = rf.Model(
        schema=resolve(
            control_preset,
            events=rf.Branch(length=CAPACITY, overflow=overflow, reduction=rf.Attention(), amount=rf.Number),
            answer=rf.Number(mask=True, objective="mse"),
        )
    )
    model.preset = control_preset
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    output = model.predict(pa.Table.from_pylist([{"events": row["events"]} for row in rows]), batch_size=128)[
        "predictions"
    ].to_pylist()
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


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
