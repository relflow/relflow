# %% [markdown]
# ---
# title: Reset One Learned Target and Teach It Again
# categories:
# - Mutation reset
# proof-id: P048
# description: Reset one hidden regression head, verify immediate loss is localized, and compare relearning against unchanged and complete-reset controls.
# execute: {enabled: false, eval: false}
# code-fold: true
# ---
#
# Reset should erase selected learned state. This experiment asks whether the
# loss is visible in predictions, whether an unselected hidden target survives,
# and whether the selected target can learn again.
#
# {{< proof P048 status >}}
#
# ## Insights
#
# **Reset erases one hidden head's learned prediction while its sibling
# retains it.** The full runs show localized changes to state and held-out
# behavior, followed by successful relearning of the selected target.
#
# Both targets remain supervised during adaptation. An unchanged continuation
# and a complete reset use the same adaptation examples and update budget.
# Their scores provide context without assuming selective reset must learn
# faster. Branch resets can affect shared context and need a separate proof.
#
# ## Setup

# %%
"""P048: localize loss of learned behavior after reset and demonstrate relearning."""

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

PROOF_ID = "P048"
TARGETS = ("u", "v")

# %% [markdown]
# ## Examples
#
# Draw independent `a` and `b` uniformly from [−1, 1]. Both targets are hidden:
# `u = a + 2b`, and `v = 2a − b`.
#
# ```yaml
# a: 0.5
# b: 0.25
# u: 1.0
# v: 0.75
# ```
#
# Holding `a` fixed while changing `b` changes both answers:
#
# ```yaml
# a: 0.5
# b: -0.25
# u: 0.0
# v: 1.25
# ```
#
# Holding `b` fixed does not determine either answer:
#
# ```yaml
# a: -0.5
# b: 0.25
# u: 0.0
# v: -1.25
# ```
#
# ## Data and comparisons


# %%
def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        a, b = map(float, rng.uniform(-1, 1, size=2))
        yield {"a": a, "b": b, "u": a + 2 * b, "v": 2 * a - b}


def prediction(model: rf.Model, rows: list[dict], *, corrupt: bool = False) -> dict:
    inputs = [{"a": row["a"], "b": row["b"]} for row in rows]
    if corrupt:
        for row in inputs:
            row.update({name: 1000.0 for name in TARGETS})
    output = model.predict(inputs)["predictions"].to_pylist()
    return {
        name: np.asarray([row[f"record/{name}"]["content"] for row in output], dtype=np.float64) for name in TARGETS
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
    """Compare tensor and extension-owned state, including normalization."""
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

    def __init__(self, rows: list[dict], means: dict, budget: int):
        self.rows, self.means = rows, means
        self.checkpoints = {32, 128, budget}
        self.measurements = []

    def on_train_start(self, trainer, pl_module):
        self.measurements.append(
            {
                "step": 0,
                "scores": scores(self.rows, prediction(pl_module, self.rows), self.means),
            }
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step in self.checkpoints:
            self.measurements.append(
                {
                    "step": trainer.global_step,
                    "scores": scores(self.rows, prediction(pl_module, self.rows), self.means),
                }
            )


def fit(model: rf.Model, *, seed: int, split: int, budget: int, accelerator: str) -> dict:
    lit.seed_everything(seed, workers=True)
    training = list(records(rows=2048, seed=split))
    means = {name: float(np.mean([row[name] for row in training])) for name in TARGETS}
    curve = Curve(list(records(rows=512, seed=split + 1)), means, budget)
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
# ## Scope of the reset
#
# ```{typst}
# //| label: fig-proof-mutations-reset-target
# //| fig-alt: "Record reads a and b to predict hidden u and v. Only the runtime node for u is reset. The root, inputs, and v retain their trained state."
# //| fig-cap: "Reset replaces the selected target's runtime state; its schema and task stay fixed."
# #tree(node("record", kind: "root", children: (
#   node("a", type: "Number"),
#   node("b", type: "Number"),
#   node("u", type: "Number", kind: "target", body: [Reset, then relearned]),
#   node("v", type: "Number", kind: "target", body: [Unselected, always hidden]),
# )))
# ```
#
# Source training uses 2,048 records and 512 updates. Adaptation uses 2,048
# independently drawn records and 256 updates. The phases each have 512
# validation records. A separate, fixed panel of 1,024 records measures
# immediate loss, retention, and final skill for every arm.
#
# Validation curves record steps 0, 32, 128, and the final update. All arms
# restart AdamW and rehearse both targets. No test result selects a checkpoint.
# Test nRMSE divides RMSE by the error of predicting the source-training target
# mean. Validation curves use each phase's training mean. The provisional
# learning gate is 0.25; loss of skill requires nRMSE above 0.8 and three times
# source error. Strict retention uses `rtol=1e-5` and `atol=1e-6` times training SD.
#
# ## Training, reset, and relearning


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    source_budget = 512 if steps is None else min(steps, 512)
    adapt_budget = 256 if steps is None else min(steps, 256)
    lit.seed_everything(seed, workers=True)
    training = list(records(rows=2048, seed=seed + 1))
    test = list(records(rows=1024, seed=seed + 3))
    means = {name: float(np.mean([row[name] for row in training])) for name in TARGETS}
    scales = {name: float(np.std([row[name] for row in training])) for name in TARGETS}
    source = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        batch_size=128,
        a=rf.Number,
        b=rf.Number,
        u=rf.Number(mask=True, objective="mse"),
        v=rf.Number(mask=True, objective="mse"),
    )
    source_fit = fit(source, seed=seed, split=seed + 1, budget=source_budget, accelerator=accelerator)
    reference = prediction(source, test)
    initial = scores(test, reference, means)
    learned = all(value["nrmse"] < 0.25 for value in initial.values())
    checks = {"Source learns both targets below 0.25 nRMSE": learned}
    state = deepcopy(source.state_dict())
    arms = {}

    with TemporaryDirectory(prefix="relflow-mutation-") as directory:
        checkpoint = Path(directory) / "source.ckpt"
        source.save(checkpoint)
        for index, arm in enumerate(("selective_reset", "continuation", "complete_reset")):
            lit.seed_everything(seed + 100 + index, workers=True)
            model = rf.Model.load(checkpoint).to(source.device)
            if arm == "selective_reset":
                model.reset(rf.where("address") == "record/u")
            elif arm == "complete_reset":
                model.reset(rf.where("address") == "record", descendants=True)
            model.eval()
            current = model.state_dict()
            changes = [name for name, value in state.items() if name not in current or not equal(value, current[name])]
            before = prediction(model, test)
            immediate = scores(test, before, means)
            if arm == "selective_reset":
                checks["Reset changes selected learned state"] = any(
                    name.startswith("nodes.record/u.") for name in changes
                )
                checks["Reset preserves every unselected state entry"] = all(
                    name.startswith("nodes.record/u.") for name in changes
                )
                checks["Reset preserves the schema"] = model.schema.model_dump() == source.schema.model_dump()
                checks["Selected target immediately loses its skill"] = (
                    immediate["u"]["nrmse"] > 0.8 and immediate["u"]["nrmse"] > 3 * initial["u"]["nrmse"]
                )
                checks["Unselected target immediately preserves predictions"] = bool(
                    np.allclose(reference["v"], before["v"], rtol=1e-5, atol=1e-6 * scales["v"])
                )
                head = {name: p.detach().clone() for name, p in model.nodes["record/u"].named_parameters()}
            if arm == "complete_reset":
                checks["Complete reset loses both learned tasks"] = all(
                    immediate[name]["nrmse"] > 0.8 and immediate[name]["nrmse"] > 3 * initial[name]["nrmse"]
                    for name in TARGETS
                )
            fitting = fit(model, seed=seed + 200, split=seed + 11, budget=adapt_budget, accelerator=accelerator)
            after = prediction(model, test)
            final = scores(test, after, means)
            arms[arm] = {
                "before": immediate,
                "after": final,
                "changed_state_entries": changes,
                "initial_prediction_drift": {
                    name: float(np.max(np.abs(reference[name] - before[name]))) for name in TARGETS
                },
                **fitting,
            }
            checks[f"{arm}: fresh optimizer covers current parameters"] = fitting["optimizer_covers_current_parameters"]
            checks[f"{arm}: both final targets below 0.25 nRMSE"] = all(
                value["nrmse"] < 0.25 for value in final.values()
            )
            if arm == "selective_reset":
                changed = [
                    name for name, p in model.nodes["record/u"].named_parameters() if not torch.equal(head[name], p)
                ]
                arms[arm]["reset_head_updated_parameters"] = changed
                checks["Reset head parameters actually relearn"] = bool(changed)
                corrupted = prediction(model, test, corrupt=True)
                checks["Hidden target placeholders cannot affect predictions"] = all(
                    np.allclose(after[name], corrupted[name], rtol=1e-5, atol=1e-6 * scales[name]) for name in TARGETS
                )
                path = Path(directory) / "relearned.ckpt"
                model.save(path)
                loaded = rf.Model.load(path).to(model.device).eval()
                restored = prediction(loaded, test)
                checks["Relearned checkpoint preserves schema and state"] = (
                    loaded.schema.model_dump() == model.schema.model_dump()
                    and equal(model.state_dict(), loaded.state_dict())
                )
                checks["Relearned checkpoint preserves predictions"] = all(
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
        "optimizer_policy": "New AdamW factory and Trainer for every fit; both hidden labels rehearsed",
        "mutation": "reset record/u; complete-reset control resets record with descendants=True",
    }, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P048 evidence >}}
#
# ## Remaining work
#
# Three GPU seeds support this case; calibrate gates on ten separate seeds. Resetting a
# context-producing branch, resetting descendants, and relearning without
# sibling labels require their own experiments. Localized loss here depends
# on the selected field being hidden from the shared context.
#
# ## Reproduce
#
# {{< proof P048 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=4801)
