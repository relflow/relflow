# %% [markdown]
# ---
# title: Can an input be disabled and restored?
# categories:
# - Mutation ablation
# proof-id: P049
# description: Train on y = a + b, temporarily disable b, then enable it again.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: Original model
#   metric:
#   - source
#   - nrmse
#   format: error
# - label: Best prediction using only a
#   metric:
#   - controls
#   - only_a_oracle
#   - nrmse
#   format: error
# ---
#
# ## Example
#
# ```yaml
# a: 0.5
# b: 0.75
# y: 1.25
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-proof-mutations-deactivate-input
# //| fig-alt: "Model tree with record, a, b, y. Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("record", kind: "root", children: (
#   node("a", type: "Number", body: [Always active]),
#   node("b", type: "Number", body: [Active → inactive → active]),
#   node("y", type: "Number", kind: "target", body: [Always hidden]),
# )))
# ```
#
# ## Comparison
#
# Check repeated edits, temporary overrides, and a saved inactive model. No training occurs while b is disabled.
#
# ## Result
#
# {{< proof P049 status >}}
#
# Disabling b removes useful information; enabling it restores the trained predictions. This tests an input field, rather than disabling a learned output decoder.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P049 evidence >}}
#
# ### Run this experiment
#
# {{< proof P049 script >}}
#
# ### Remaining work
#
# Three GPU seeds support this case. Calibrate numerical gates on ten independent
# seeds. Deactivating output heads and branches that produce context requires
# separate experiments. Training inside an override does not have checkpoint
# rollback semantics and is not covered here.
#
# ### Complete experiment code
#

# %%
"""P049: remove and restore an informative input without further training."""

from collections.abc import Iterator
from copy import deepcopy
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P049"


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        a, b = map(float, rng.uniform(-1, 1, size=2))
        yield {"a": a, "b": b, "y": a + b}


def prediction(model: rf.Model, rows: list[dict], *, include_b: bool = True, corrupt: bool = False) -> np.ndarray:
    inputs = [{"a": row["a"], **({"b": row["b"]} if include_b else {})} for row in rows]
    if corrupt:
        for row in inputs:
            row["y"] = 1000.0
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["/y"]["content"] for row in output], dtype=np.float64)


def errors(actual: np.ndarray, predicted: np.ndarray, baseline: float) -> dict:
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    return {"rmse": rmse, "baseline_rmse": baseline, "nrmse": rmse / baseline}


def equal(first, second) -> bool:
    """Compare tensors and extension-owned state, including normalization."""
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return (
            isinstance(second, dict)
            and first.keys() == second.keys()
            and all(equal(value, second[name]) for name, value in first.items())
        )
    if isinstance(first, (tuple, list)):
        return (
            type(first) is type(second)
            and len(first) == len(second)
            and all(equal(a, b) for a, b in zip(first, second, strict=True))
        )
    return type(first) is type(second) and first == second


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    budget = 512 if steps is None else min(steps, 512)
    train = list(records(rows=2048, seed=seed + 1))
    test = list(records(rows=2048, seed=seed + 3))
    actual = np.asarray([row["y"] for row in test])
    baseline = float(np.sqrt(np.mean((actual - np.mean([row["y"] for row in train])) ** 2)))
    tolerance = 1e-6 * float(np.std([row["y"] for row in train]))
    oracle = errors(actual, np.asarray([row["a"] for row in test]), baseline)
    rng = np.random.default_rng(seed + 4)
    order = rng.permutation(len(test))
    shuffled = [{**row, "b": test[index]["b"]} for row, index in zip(test, order, strict=True)]
    changed = [{**row, "b": -row["b"]} for row in test]
    model = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        batch_size=128,
        a=rf.Number,
        b=rf.Number,
        y=rf.Number(mask=True, objective="mse"),
    )
    model.optimizer = rf.adamw(learning_rate=3e-3, fused=False)
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, rows=2048, seed=seed + 1),
        validate=partial(records, rows=512, seed=seed + 2),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=budget,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=data)
    model.eval()
    reference = prediction(model, test)
    source = errors(actual, reference, baseline)
    corruption = errors(actual, prediction(model, shuffled), baseline)
    zero_filled = errors(actual, prediction(model, [{**row, "b": 0.0} for row in test]), baseline)
    schema = deepcopy(model.schema.model_dump())
    state = deepcopy(model.state_dict())
    selected = rf.where("address") == "/b"
    learned = source["nrmse"] < 0.25
    checks = {
        "Source learns both-input relationship below 0.25 nRMSE": learned,
        "Shuffling b destroys useful signal": corruption["nrmse"] > source["nrmse"] + 0.35
        and corruption["nrmse"] >= 0.9 * oracle["nrmse"],
        "Source hidden target values cannot affect predictions": bool(
            np.allclose(reference, prediction(model, test, corrupt=True), rtol=1e-5, atol=tolerance)
        ),
    }
    edits = []

    def observe(label: str, *, inactive: bool) -> np.ndarray:
        """Evaluate the appropriate information-loss or restoration invariant."""
        values = prediction(model, test)
        measured = errors(actual, values, baseline)
        same_state = equal(state, model.state_dict())
        checks[f"{label}: all learned state survives"] = same_state
        if inactive:
            omitted = prediction(model, test, include_b=False)
            flipped = prediction(model, changed)
            checks[f"{label}: input is inactive"] = not model.schema.requests["/b"].active
            checks[f"{label}: b values cannot affect predictions"] = bool(
                np.allclose(values, omitted, rtol=1e-5, atol=tolerance)
                and np.allclose(values, flipped, rtol=1e-5, atol=tolerance)
            )
            checks[f"{label}: removing b loses information"] = (
                measured["nrmse"] > source["nrmse"] + 0.35 and measured["nrmse"] >= 0.9 * oracle["nrmse"]
            )
        else:
            checks[f"{label}: original schema is restored"] = model.schema.model_dump() == schema
            checks[f"{label}: trained predictions are restored"] = bool(
                np.allclose(reference, values, rtol=1e-5, atol=tolerance)
            )
        edits.append(
            {
                "edit": label,
                "inactive": inactive,
                "scores": measured,
                "max_source_prediction_drift": float(np.max(np.abs(values - reference))),
                "learned_state_preserved": same_state,
            }
        )
        return values

    with TemporaryDirectory(prefix="relflow-mutation-") as directory:
        checkpoint = Path(directory) / "source.ckpt"
        model.save(checkpoint)
        unchanged = rf.Model.load(checkpoint).to(model.device).eval()
        checks["Unchanged checkpoint preserves predictions"] = bool(
            np.allclose(reference, prediction(unchanged, test), rtol=1e-5, atol=tolerance)
        )
        for cycle in range(3):
            model.update(selected, active=False)
            inactive_values = observe(f"cycle {cycle + 1} deactivate", inactive=True)
            if cycle == 0:
                path = Path(directory) / "inactive.ckpt"
                model.save(path)
                loaded = rf.Model.load(path).to(model.device).eval()
                checks["Inactive checkpoint preserves schema and state"] = (
                    loaded.schema.model_dump() == model.schema.model_dump() and equal(state, loaded.state_dict())
                )
                checks["Inactive checkpoint preserves predictions"] = bool(
                    np.allclose(inactive_values, prediction(loaded, test), rtol=1e-5, atol=tolerance)
                )
                loaded.update(selected, active=True)
                checks["Loaded inactive input can restore its trained function"] = bool(
                    loaded.schema.model_dump() == schema
                    and equal(state, loaded.state_dict())
                    and np.allclose(reference, prediction(loaded, test), rtol=1e-5, atol=tolerance)
                )
            model.update(selected, active=True)
            observe(f"cycle {cycle + 1} reactivate", inactive=False)
        with model.override(selected, active=False):
            observe("temporary override", inactive=True)
        observe("normal override exit", inactive=False)
        marker = RuntimeError("intentional override exit")
        try:
            with model.override(selected, active=False):
                observe("override before exception", inactive=True)
                raise marker
        except RuntimeError as error:
            if error is not marker:
                raise
        observe("exceptional override exit", inactive=False)
    return {
        "source": source,
        "source_steps": trainer.global_step,
        "source_prerequisite_met": learned,
        "downstream_interpretable": learned,
        "controls": {"shuffled_b": corruption, "zero_filled_b": zero_filled, "only_a_oracle": oracle},
        "edits": edits,
        "test_rows": len(test),
        "optimizer_policy": "Fresh AdamW for source; no fitting inside updates or overrides",
        "mutation": "update /b active=False/True three times; override with normal and exceptional exits",
    }, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=4901)

# %% [markdown]
# </details>
