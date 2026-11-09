# %% [markdown]
# ---
# title: What happens when a new prediction target is added?
# categories:
# - Schema mutation
# description: Add hidden target w to a model already trained to predict u and v from a and b.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: New target before training
#   metric:
#   - arms
#   - extended
#   - before
#   - w
#   - nrmse
#   format: error
# - label: New target after training
#   metric:
#   - arms
#   - extended
#   - after
#   - w
#   - nrmse
#   format: error
# evidence-group: Mutations
# ---
#
# Start with a model that already answers two questions about the same pair of
# numbers. Then add a third prediction target to its schema. This experiment
# asks two separate questions: does the edit preserve the old answers before
# further training, and can the extended model learn the new answer afterward?
#
# Adding a target changes what we ask the model to predict. The visible inputs
# stay the same; the new target is not an extra clue supplied at prediction time.
#
# ## Example
#
# ```yaml
# a: 0.5
# b: 0.25
# u: 1.0
# v: 0.75
# w: 1.75
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# Here `a` and `b` are the two visible numbers. The original targets are
# `u = a + 2b` and `v = 2a - b`. The added target is `w = 3a + b`:
#
# | Question | Worked answer for a = 0.5 and b = 0.25 |
# | --- | --- |
# | Original target u | 0.5 + 2 × 0.25 = 1.0 |
# | Original target v | 2 × 0.5 - 0.25 = 0.75 |
# | Added target w | 3 × 0.5 + 0.25 = 1.75 |
#
# The formulas define the generated training answers. The neural model learns
# to approximate them from examples. Although `w` also equals `u + v`, those
# two answers remain hidden from the input encoder, so the new task still has
# to be learned from `a` and `b`.
#
# ```{typst}
# //| label: fig-evidence-mutations-add-target
# //| fig-alt: "Model tree with record, a, b, u, v, w. Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("record", kind: "root", children: (
#   node("a", type: "Number"),
#   node("b", type: "Number"),
#   node("u", type: "Number", kind: "target", body: [Existing, always hidden]),
#   node("v", type: "Number", kind: "target", body: [Existing, always hidden]),
#   node("w", type: "Number", kind: "target", body: [Added, always hidden]),
# )))
# ```
#
# ## Comparison
#
# First train the original model on `u` and `v` for 512 updates and save it.
# Compare four subsequent paths using fresh examples from the same generator:
#
# | Path | Starting point | What happens next |
# | --- | --- | --- |
# | Extended model | The trained two-target checkpoint | Add w, then train all three targets for 256 updates. |
# | Continued original | The same checkpoint | Keep the original schema and train u and v for another 256 updates. |
# | Fresh, matched adaptation | A new three-target model | Train from scratch for 256 updates. |
# | Fresh, matched total updates | A new three-target model | Train from scratch for 768 updates, matching the source-plus-adaptation update count. |
#
# Check old predictions immediately after adding `w`, before any additional
# fitting. This isolates the effect of the schema edit. Then check all three
# targets after fitting to assess learning. Retaining the old parameter tensors
# and retaining the old predictions are separate checks: architecture-dependent
# computation can change even when the stored tensors do not.
#
# In the plot, step zero is the start of adaptation. The extended model already
# had its 512-update source training; the fresh comparison starts untrained.
# Each plotted point measures the new target on fixed validation examples.
# Normalized RMSE divides prediction error by a constant-guess baseline: 1
# matches that guess, 0.25 is one quarter of its error, and lower is better.
#
# ## Result
#
# {{< evidence P047 status >}}
#
# Read the preservation checks and the learning curve as separate results.
# The checks tell you whether the edit preserved existing state and predictions;
# the curve shows how the added task is learned during further training. Low
# final error alone does not establish that the edit was initially neutral.
#
# Old targets remain supervised throughout adaptation. This tests extending a
# model while continuing to rehearse its original tasks. It does not establish
# retention when old labels are unavailable, an advantage over every fresh
# model, or transfer to a different data distribution.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P047 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P047 script >}}
#
# ### Remaining work
#
# The selected record describes its own seed, model revision, and device;
# the measured checks above determine its outcome. Test repeatability across
# independently declared seeds before generalizing that outcome, and calibrate
# the provisional 0.25 nRMSE gate on a separate seed panel.
# Adding visible context, targets in repeated branches, and
# adaptation without old labels remain separate claims. These linear targets
# measure a simple useful extension, not transfer to a new data distribution.
# The pooling-scale diagnostic helps investigate preservation failures: should
# permanently hidden outputs change the capacity used to scale visible evidence?
#
# ### Complete experiment code
#

# %%
"""P047: add a hidden regression target to a trained model and adapt."""

from collections.abc import Iterator
from copy import deepcopy
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
import torch
from reporting import experiment, report

TARGETS = ("u", "v", "w")


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        a, b = map(float, rng.uniform(-1, 1, size=2))
        yield {"a": a, "b": b, "u": a + 2 * b, "v": 2 * a - b, "w": 3 * a + b}


def build(*, extended: bool = False) -> rf.Model:
    targets = TARGETS if extended else TARGETS[:2]
    return rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        fields={
            "a": rf.Number,
            "b": rf.Number,
            **{name: rf.Number(mask=True, objective="mse") for name in targets},
        },
    )


def prediction(model: rf.Model, rows: list[dict], targets: tuple[str, ...], *, corrupt: bool = False) -> dict:
    inputs = [{"a": row["a"], "b": row["b"]} for row in rows]
    if corrupt:
        for row in inputs:
            row.update({name: 1000.0 for name in TARGETS})
    output = pa.table(model.predict(inputs))["predictions"].to_pylist()
    return {name: np.asarray([row[name]["content"] for row in output], dtype=np.float64) for name in targets}


def scores(rows: list[dict], predicted: dict, means: dict) -> dict:
    measured = {}
    for name, values in predicted.items():
        actual = np.asarray([row[name] for row in rows])
        baseline = float(np.sqrt(np.mean((actual - means[name]) ** 2)))
        rmse = float(np.sqrt(np.mean((actual - values) ** 2)))
        measured[name] = {"rmse": rmse, "baseline_rmse": baseline, "nrmse": rmse / baseline}
    return measured


def equal(first, second) -> bool:
    """Compare all tensor and extension-owned state, including normalizers."""
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


class Curve(lit.Callback):
    """Observe fixed validation checkpoints without restarting the optimizer."""

    def __init__(self, rows: list[dict], means: dict, targets: tuple[str, ...], budget: int):
        self.rows, self.means, self.targets = rows, means, targets
        self.checkpoints = {32, 128, budget}
        self.measurements = []

    def on_train_start(self, trainer, pl_module):
        self.measurements.append(
            {
                "step": 0,
                "scores": scores(self.rows, prediction(pl_module, self.rows, self.targets), self.means),
            }
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step in self.checkpoints:
            self.measurements.append(
                {
                    "step": trainer.global_step,
                    "scores": scores(self.rows, prediction(pl_module, self.rows, self.targets), self.means),
                }
            )


def fit(model: rf.Model, *, seed: int, split: int, budget: int, accelerator: str, targets: tuple[str, ...]) -> dict:
    lit.seed_everything(seed, workers=True)
    training = list(records(rows=2048, seed=split))
    means = {name: float(np.mean([row[name] for row in training])) for name in TARGETS}
    curve = Curve(list(records(rows=512, seed=split + 1)), means, targets, budget)
    data = rf.SyntheticDataModule(
        train=partial(records, rows=2048, seed=split),
        validate=partial(records, rows=512, seed=split + 1),
        seed=seed,
    )
    model.fit(
        data,
        batch_size=128,
        learning_rate=0.003,
        fused=False,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=budget,
        callbacks=[curve],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer
    model.eval()
    optimized = {id(p) for group in trainer.optimizers[0].param_groups for p in group["params"]}
    current = {id(p) for p in model.parameters() if p.requires_grad}
    return {
        "steps": trainer.global_step,
        "optimizer_covers_current_parameters": current == optimized,
        "validation_curve": curve.measurements,
    }


@experiment("P047", seed=4701)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    source_budget = 512 if steps is None else min(steps, 512)
    adapt_budget = 256 if steps is None else min(steps, 256)
    full_budget = 768 if steps is None else min(steps, 768)
    lit.seed_everything(seed, workers=True)
    train = list(records(rows=2048, seed=seed + 1))
    test = list(records(rows=1024, seed=seed + 3))
    means = {name: float(np.mean([row[name] for row in train])) for name in TARGETS}
    scales = {name: float(np.std([row[name] for row in train])) for name in TARGETS}
    source = build()
    source_fit = fit(
        source, seed=seed, split=seed + 1, budget=source_budget, accelerator=accelerator, targets=TARGETS[:2]
    )
    reference = prediction(source, test, TARGETS[:2])
    initial = scores(test, reference, means)
    learned = all(value["nrmse"] < 0.25 for value in initial.values())
    checks = {
        "Source learns both original tasks below 0.25 nRMSE": learned,
        "Source optimizer covers current parameters": source_fit["optimizer_covers_current_parameters"],
    }
    source_state = deepcopy(source.state_dict())
    arms = {}

    with TemporaryDirectory(prefix="relflow-mutation-") as directory:
        checkpoint = Path(directory) / "source.ckpt"
        source.save(checkpoint)
        for index, arm in enumerate(("extended", "continuation", "scratch_adaptation", "scratch_total")):
            lit.seed_everything(seed + 100 + index, workers=True)
            model = (
                build(extended=True).to(source.device)
                if arm.startswith("scratch")
                else rf.Model.load(checkpoint).to(source.device)
            )
            targets = TARGETS[:2] if arm == "continuation" else TARGETS
            if arm == "extended":
                model.extend(rf.where("address") == "/", w=rf.Number(mask=True, objective="mse"))
                extended_schema = model.schema.model_dump()
                retained = model.state_dict()
                changes = [
                    name
                    for name, value in source_state.items()
                    if name not in retained or not equal(value, retained[name])
                ]
                checks["Extension preserves every existing state entry"] = not changes
                head = {name: p.detach().clone() for name, p in model.nodes["/w"].named_parameters()}
            elif arm.startswith("scratch"):
                checks[f"{arm}: schema and field order match extension"] = model.schema.model_dump() == extended_schema
            model.eval()
            before = prediction(model, test, targets)
            drift = {name: float(np.max(np.abs(before[name] - reference[name]))) for name in TARGETS[:2]}
            if arm == "extended":
                checks["Extension immediately preserves old predictions"] = all(
                    np.allclose(reference[name], before[name], rtol=1e-5, atol=1e-6 * scales[name])
                    for name in TARGETS[:2]
                )
                # Isolate a schema-derived scale that is absent from state_dict.
                # Restore the edited scale before any fitting or checkpointing.
                pool = model.nodes["/"].encoder.pool
                capacity = pool.mass_capacity
                source_capacity = source.nodes["/"].encoder.pool.mass_capacity
                try:
                    pool.mass_capacity = source_capacity
                    fixed_scale = prediction(model, test, TARGETS)
                finally:
                    pool.mass_capacity = capacity
                scale_control = {
                    "source_capacity": source_capacity,
                    "extended_capacity": capacity,
                    "old_prediction_drift_with_source_capacity": {
                        name: float(np.max(np.abs(fixed_scale[name] - reference[name]))) for name in TARGETS[:2]
                    },
                }
                checks["Diagnostic source pooling scale restores old predictions"] = all(
                    np.allclose(reference[name], fixed_scale[name], rtol=1e-5, atol=1e-6 * scales[name])
                    for name in TARGETS[:2]
                )
            fitting = fit(
                model,
                seed=seed + 200,
                split=seed + 11,
                budget=full_budget if arm == "scratch_total" else adapt_budget,
                accelerator=accelerator,
                targets=targets,
            )
            after = prediction(model, test, targets)
            final = scores(test, after, means)
            checks[f"{arm}: fresh optimizer covers current parameters"] = fitting["optimizer_covers_current_parameters"]
            arms[arm] = {
                "before": scores(test, before, means),
                "after": final,
                "initial_old_prediction_drift": drift,
                **fitting,
            }
            if arm in ("extended", "scratch_total"):
                checks[f"{arm}: all final targets below 0.25 nRMSE"] = all(
                    value["nrmse"] < 0.25 for value in final.values()
                )
            if arm == "extended":
                arms[arm]["changed_existing_state_entries"] = changes
                arms[arm]["pooling_scale_control"] = scale_control
                changed = [name for name, p in model.nodes["/w"].named_parameters() if not torch.equal(head[name], p)]
                arms[arm]["new_head_updated_parameters"] = changed
                checks["Added head parameters actually learn"] = bool(changed)
                corrupted = prediction(model, test, TARGETS, corrupt=True)
                checks["Hidden target placeholders cannot affect predictions"] = all(
                    np.allclose(after[name], corrupted[name], rtol=1e-5, atol=1e-6 * scales[name]) for name in TARGETS
                )
                path = Path(directory) / "extended.ckpt"
                model.save(path)
                loaded = rf.Model.load(path).to(model.device).eval()
                restored = prediction(loaded, test, TARGETS)
                checks["Extended checkpoint preserves schema and learned state"] = (
                    loaded.schema.model_dump() == model.schema.model_dump()
                    and equal(model.state_dict(), loaded.state_dict())
                )
                checks["Extended checkpoint preserves predictions"] = all(
                    np.allclose(after[name], restored[name], rtol=1e-5, atol=1e-6 * scales[name]) for name in TARGETS
                )
    return {
        "source": initial,
        "source_fit": source_fit,
        "source_prerequisite_met": learned,
        "downstream_interpretable": learned,
        "arms": arms,
        "test_rows": len(test),
        "baseline": "Source-training mean for test comparisons; phase-training mean for validation curves",
        "optimizer_policy": "New AdamW factory and Trainer for every fit; all available labels rehearsed",
        "mutation": "extend record with hidden Number /w",
    }, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
