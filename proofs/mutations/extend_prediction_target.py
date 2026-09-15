# %% [markdown]
# ---
# title: Learn an Added Target Without Losing Existing Tasks
# categories:
# - Mutation adaptation
# proof-id: P047
# description: Extend a trained model with a hidden numerical target, then compare adaptation with continuation and fresh final-schema controls.
# execute: {enabled: false, eval: false}
# code-fold: true
# ---
#
# A useful schema extension must participate in learning. This experiment
# starts with two learned outputs, adds a third, and measures all three before
# and after adaptation with a fresh optimizer.
#
# {{< proof P047 status >}}
#
# ## Insights
#
# **The new hidden target learns, but extension shifts the old predictions
# before training.** Existing state entries survive exactly. The root pool's
# capacity nevertheless grows from four to five, changing the scale of its
# additive evidence. A diagnostic that restores only the old capacity restores
# the old predictions. The strict immediate-preservation gate remains failed.
#
# After adaptation, all three targets meet the learning gate in the recorded
# runs. Both old targets remain supervised: this is retention with rehearsal,
# not protection against forgetting without labels.
#
# Separate controls continue the unchanged source model and train the final
# schema from scratch. The experiment reports their errors without requiring
# transfer to beat scratch training. Mutation correctness does not imply a
# pretraining advantage.
#
# ## Setup

# %%
"""P047: add a hidden regression target to a trained model and adapt."""

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

PROOF_ID = "P047"
TARGETS = ("u", "v", "w")

# %% [markdown]
# ## Examples
#
# Independently draw `a` and `b` uniformly from [−1, 1]. The source learns
# `u = a + 2b` and `v = 2a − b`. The extension adds `w = 3a + b`.
# All three answers are hidden from the encoder.
# The new task is the sum of the original tasks; their labels are never inputs.
#
# ```yaml
# a: 0.5
# b: 0.25
# u: 1.0
# v: 0.75
# w: 1.75
# ```
#
# Changing only `b` affects all outputs, with a different coefficient:
#
# ```yaml
# a: 0.5
# b: -0.25
# u: 0.0
# v: 1.25
# w: 1.25
# ```
#
# Neither input alone determines a target:
#
# ```yaml
# a: -0.5
# b: 0.25
# u: 0.0
# v: -1.25
# w: -1.25
# ```
#
# ## Data, models, and controls


# %%
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
        batch_size=128,
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
    output = model.predict(inputs)["predictions"].to_pylist()
    return {
        name: np.asarray([row[f"record/{name}"]["content"] for row in output], dtype=np.float64) for name in targets
    }


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
    model.optimizer = rf.adamw(learning_rate=3e-3, fused=False)
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, rows=2048, seed=split),
        validate=partial(records, rows=512, seed=split + 1),
        seed=seed,
    )
    trainer = lit.Trainer(
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
    trainer.fit(model, datamodule=data)
    model.eval()
    optimized = {id(p) for group in trainer.optimizers[0].param_groups for p in group["params"]}
    current = {id(p) for p in model.parameters() if p.requires_grad}
    return {
        "steps": trainer.global_step,
        "optimizer_covers_current_parameters": current == optimized,
        "validation_curve": curve.measurements,
    }


# %% [markdown]
# ## Before and after extension
#
# ```{typst}
# //| label: fig-proof-mutations-add-target
# //| fig-alt: "Record reads a and b to predict hidden u and v. Extending the root adds hidden Number w while retaining the original inputs and targets."
# //| fig-cap: "The extension adds one supervised output; visible information stays fixed."
# #tree(node("record", kind: "root", children: (
#   node("a", type: "Number"),
#   node("b", type: "Number"),
#   node("u", type: "Number", kind: "target", body: [Existing, always hidden]),
#   node("v", type: "Number", kind: "target", body: [Existing, always hidden]),
#   node("w", type: "Number", kind: "target", body: [Added, always hidden]),
# )))
# ```
#
# Source training uses 2,048 rows and 512 updates. Adaptation uses a separate
# 2,048-row split and 256 updates. Each phase has 512 independent validation
# rows. All arms share a fixed, independent 1,024-row test panel.
#
# The two scratch controls use the adaptation split for 256 or 768 updates.
# The latter matches the warm model's total update count, not its exposure to
# two distinct training sets. Scratch heads are independently initialized.
# Validation curves record steps 0, 32, 128, and the final update; no test
# observation selects a checkpoint, training budget, or acceptance threshold.
# Test nRMSE divides RMSE by the error of predicting the source-training target
# mean. Validation curves use each phase's training mean. The provisional
# learning gate is 0.25. Strict retention uses `rtol=1e-5` and `atol=1e-6` times
# source-training target SD.
#
# A diagnostic temporarily holds the root pool's internal `mass_capacity` at
# its source value. It isolates the effect of the scaling change and is restored
# before adaptation or saving. It is not a proposed public mutation workflow.
#
# ## Training, extension, and evaluation


# %%
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
                model.extend(rf.where("address") == "record", w=rf.Number(mask=True, objective="mse"))
                extended_schema = model.schema.model_dump()
                retained = model.state_dict()
                changes = [
                    name
                    for name, value in source_state.items()
                    if name not in retained or not equal(value, retained[name])
                ]
                checks["Extension preserves every existing state entry"] = not changes
                head = {name: p.detach().clone() for name, p in model.nodes["record/w"].named_parameters()}
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
                pool = model.nodes["record"].encoder.pool
                capacity = pool.mass_capacity
                source_capacity = source.nodes["record"].encoder.pool.mass_capacity
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
                changed = [
                    name for name, p in model.nodes["record/w"].named_parameters() if not torch.equal(head[name], p)
                ]
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
        "mutation": "extend record with hidden Number record/w",
    }, checks


# %% [markdown]
# ## Evidence
#
# Early pilots used a different field order in the scratch controls. Their
# measurements remain in the raw history under the earlier code fingerprints.
# The current experiment explicitly checks that both scratch schemas, including
# field order, match the extended model before interpreting their results.
#
# {{< proof P047 evidence >}}
#
# ## Remaining work
#
# Three GPU seeds reproduce the retention failure and successful adaptation.
# Calibrate the provisional 0.25 nRMSE gate on ten separate seeds.
# Adding visible context, targets in repeated branches, and
# adaptation without old labels remain separate claims. These linear targets
# measure a simple useful extension, not transfer to a new data distribution.
# The failed preservation gate identifies a further design question: should
# permanently hidden outputs change the capacity used to scale visible evidence?
#
# ## Reproduce
#
# {{< proof P047 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=4701)
