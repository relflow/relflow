# %% [markdown]
# ---
# title: Do harmless schema edits preserve predictions?
# categories:
# - Mutation retention
# proof-id: P046
# description: Train a useful predictor, then make edits that should not alter its visible information or
#   learned state.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: Trained model
#   metric:
#   - source
#   - nrmse
#   format: error
# - label: Completely reset
#   metric:
#   - controls
#   - complete_reset
#   - nrmse
#   format: error
# ---
#
# ## Example
#
# ```yaml
# x: 0.5
# code: code-0
# y: -0.25
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-proof-mutations-neutral-edits
# //| fig-alt: "Model tree with record, x, code, y, unused. Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number", body: [Source key becomes *renamed_x*]),
#   node("code", type: "Category"),
#   node("y", type: "Number", kind: "target", body: [Always hidden]),
#   node("unused", type: "Number", body: [Added inactive, then deleted]),
# )))
# ```
#
# ## Comparison
#
# Repeat metadata changes, inactive-field additions and deletions, equivalent input-key changes, and save/load. Compare with a complete reset.
#
# ## Result
#
# {{< proof P046 status >}}
#
# These edits preserve predictions; a reset loses the learned relationship. Changes to visible context or model size need separate checks.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P046 evidence >}}
#
# ### Run this experiment
#
# {{< proof P046 script >}}
#
# ### Remaining work
#
# Three GPU seeds support these edits. Calibrate on ten separate seeds before
# freezing numerical gates. Test inactive nested branches separately.
# Renaming schema addresses and resizing learned tensors require separate
# experiments; matching physical input values does not make those neutral edits.
#
# ### Complete experiment code
#

# %%
"""P046: preserve a trained function across neutral public schema edits."""

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

PROOF_ID = "P046"
OFFSETS = (-0.75, -0.25, 0.25, 0.75)


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        x = float(rng.uniform(-1, 1))
        code = int(rng.integers(len(OFFSETS)))
        yield {"x": x, "code": f"code-{code}", "y": x + OFFSETS[code]}


def prediction(model: rf.Model, rows: list[dict], *, key: str = "x") -> np.ndarray:
    output = model.predict([{key: row["x"], "code": row["code"]} for row in rows])["predictions"].to_pylist()
    return np.asarray([row["/y"]["content"] for row in output], dtype=np.float64)


def equal(first, second) -> bool:
    """Compare learned tensors and extension-owned nested state exactly."""
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


def errors(actual: np.ndarray, predicted: np.ndarray, baseline: float) -> dict:
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    return {"rmse": rmse, "baseline_rmse": baseline, "nrmse": rmse / baseline}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    budget = 512 if steps is None else min(steps, 512)
    train = list(records(rows=2048, seed=seed + 1))
    test = list(records(rows=1024, seed=seed + 3))
    actual = np.asarray([row["y"] for row in test])
    baseline = float(np.sqrt(np.mean((actual - np.mean([row["y"] for row in train])) ** 2)))
    scale = float(np.std([row["y"] for row in train]))
    model = rf.Model.xs(
        batch_size=128,
        x=rf.Number,
        code=rf.Category(p_unavailable=0.0),
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
    state = deepcopy(model.state_dict())
    vocabulary = model.nodes["/code"].embedder.vocab.snapshot()
    root = rf.where("address") == "/"
    source = rf.where("address") == "/x"
    measured = errors(actual, reference, baseline)
    edits = []
    checks = {"Source learns the relationship below 0.25 nRMSE": measured["nrmse"] < 0.25}

    def observe(label: str, *, key: str = "x") -> None:
        result = prediction(model, test, key=key)
        current = model.state_dict()
        missing = [name for name, value in state.items() if name not in current or not equal(value, current[name])]
        preserved = bool(np.allclose(reference, result, rtol=1e-5, atol=1e-6 * scale))
        same_vocabulary = model.nodes["/code"].embedder.vocab.snapshot() == vocabulary
        edits.append(
            {
                "edit": label,
                "max_prediction_drift": float(np.max(np.abs(reference - result))),
                "nrmse": errors(actual, result, baseline)["nrmse"],
                "changed_or_missing_state_entries": missing,
                "vocabulary_preserved": same_vocabulary,
            }
        )
        checks[f"{label}: predictions survive"] = preserved
        checks[f"{label}: all original state and vocabulary survive"] = not missing and same_vocabulary

    for cycle in range(3):
        model.update(source, description=f"Equivalent input, cycle {cycle + 1}")
        observe(f"cycle {cycle + 1} metadata")
        model.extend(root, unused=rf.Number(active=False))
        observe(f"cycle {cycle + 1} inactive extension")
        model.delete(rf.where("address") == "/unused")
        observe(f"cycle {cycle + 1} inactive deletion")
    model.update(source, query="renamed_x")
    observe("equivalent source rebind", key="renamed_x")
    model.update(source, query=None)
    observe("source rebind restoration")
    rejected = False
    try:
        model.extend(root, x=rf.Number)
    except ValueError:
        rejected = True
    checks["Duplicate-name edit is rejected"] = rejected
    observe("rejected duplicate extension")

    with TemporaryDirectory(prefix="relflow-mutation-") as directory:
        path = Path(directory) / "edited.ckpt"
        model.save(path)
        loaded = rf.Model.load(path).to(model.device).eval()
        restored = prediction(loaded, test)
        checks["Edited checkpoint preserves schema"] = loaded.schema.model_dump() == model.schema.model_dump()
        checks["Edited checkpoint preserves learned state"] = equal(model.state_dict(), loaded.state_dict())
        checks["Edited checkpoint preserves predictions"] = bool(
            np.allclose(reference, restored, rtol=1e-5, atol=1e-6 * scale)
        )
        lit.seed_everything(seed + 100, workers=True)
        loaded.reset(root, descendants=True)
        reset = errors(actual, prediction(loaded.eval(), test), baseline)
    checks["Complete reset loses the learned relationship"] = (
        reset["nrmse"] > 0.8 and reset["nrmse"] > 3 * measured["nrmse"]
    )
    return {
        "source": measured,
        "source_prerequisite_met": measured["nrmse"] < 0.25,
        "downstream_interpretable": measured["nrmse"] < 0.25,
        "source_steps": trainer.global_step,
        "edits": edits,
        "controls": {"complete_reset": reset},
        "source_vocabulary": vocabulary,
        "test_rows": len(test),
        "optimizer_policy": "Fresh AdamW for source training; no post-edit fitting",
    }, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=4601)

# %% [markdown]
# </details>
