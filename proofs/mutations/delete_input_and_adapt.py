# %% [markdown]
# ---
# title: Delete a Learned Input and Adapt to the Remaining Information
# categories:
# - Mutation ablation
# proof-id: P050
# description: Delete an informative Number input, compare adaptation with inactive, scratch, and unchanged controls, and distinguish re-adding a node from checkpoint restoration.
# execute: {enabled: false, eval: false}
# code-fold: true
# ---
#
# Deleting an informative node should remove its state and its contribution.
# Further training can exploit the remaining input, but cannot reconstruct
# independent information that the schema no longer receives.
#
# {{< proof P050 status >}}
#
# ## Insights
#
# **After deleting b from a learned y = a + b task, adaptation approaches
# the conditional mean a.** The full runs approach the measured information
# limit, with similar results from inactive and scratch controls. An unchanged
# continuation retains substantially better accuracy because it still sees b.
#
# Deactivation and scratch models provide controls with the same information.
# Every arm receives the same additional training examples and update budget.
# Re-adding the deleted name is checked separately for fresh node state;
# loading the source checkpoint is the control for restoring the trained model.
#
# ## Setup

# %%
"""P050: delete an informative input and learn within the reduced information limit."""

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

PROOF_ID = "P050"

# %% [markdown]
# ## Examples
#
# Independently draw `a` and `b` uniformly from [−1, 1]. The target is always
# `y = a + b`, including after `b` is removed from the model schema.
#
# ```yaml
# a: 0.5
# b: 0.75
# y: 1.25
# ```
#
# The same `a` can accompany another correct target:
#
# ```yaml
# a: 0.5
# b: -0.75
# y: -0.25
# ```
#
# Deletion leaves enough information for a conditional mean that varies with
# `a`, rather than a constant prediction:
#
# ```yaml
# a: -0.5
# b: 0.75
# y: 0.25
# ```
#
# The removed source key remains in the synthetic rows as a control. Its
# presence or value must have no effect once the schema stops reading it.
#
# ## Data, fitting, and comparisons


# %%
def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        a, b = map(float, rng.uniform(-1, 1, size=2))
        yield {"a": a, "b": b, "y": a + b}


def build(*, restricted: bool = False) -> rf.Model:
    return rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        batch_size=128,
        fields={"a": rf.Number, **({} if restricted else {"b": rf.Number}), "y": rf.Number(mask=True, objective="mse")},
    )


def prediction(model: rf.Model, rows: list[dict], *, include_b: bool = True, corrupt: bool = False) -> np.ndarray:
    inputs = [{"a": row["a"], **({"b": row["b"]} if include_b else {})} for row in rows]
    if corrupt:
        for row in inputs:
            row["y"] = 1000.0
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["/y"]["content"] for row in output], dtype=np.float64)


def scores(rows: list[dict], predicted: np.ndarray, mean: float) -> dict:
    actual = np.asarray([row["y"] for row in rows])
    conditional = np.asarray([row["a"] for row in rows])
    baseline = float(np.sqrt(np.mean((actual - mean) ** 2)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    distance = float(np.sqrt(np.mean((conditional - predicted) ** 2)))
    return {
        "rmse": rmse,
        "baseline_rmse": baseline,
        "nrmse": rmse / baseline,
        "only_a_distance_nrmse": distance / baseline,
    }


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


class Curve(lit.Callback):
    """Measure validation progress at fixed steps without restarting AdamW."""

    def __init__(self, rows: list[dict], mean: float, budget: int):
        self.rows, self.mean = rows, mean
        self.checkpoints = {32, 128, budget}
        self.measurements = []

    def on_train_start(self, trainer, pl_module):
        self.measurements.append({"step": 0, "scores": scores(self.rows, prediction(pl_module, self.rows), self.mean)})

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step in self.checkpoints:
            self.measurements.append(
                {"step": trainer.global_step, "scores": scores(self.rows, prediction(pl_module, self.rows), self.mean)}
            )


def fit(model: rf.Model, *, seed: int, split: int, rows: int, budget: int, accelerator: str) -> dict:
    lit.seed_everything(seed, workers=True)
    mean = float(np.mean([row["y"] for row in records(rows=rows, seed=split)]))
    curve = Curve(list(records(rows=512, seed=split + 1)), mean, budget)
    model.optimizer = rf.adamw(learning_rate=3e-3, fused=False)
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, rows=rows, seed=split),
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
# ## Before and after deletion
#
# ```{typst}
# //| label: fig-proof-mutations-delete-input
# //| fig-alt: "The source record reads a and b to predict hidden y. Deleting b leaves Number a and hidden Number y; the training labels still depend on both original values."
# //| fig-cap: "The target stays fixed while the schema loses access to b."
# #tree(node("record", kind: "root", children: (
#   node("a", type: "Number", body: [Retained input]),
#   node("b", type: "Number", body: [Deleted before adaptation]),
#   node("y", type: "Number", kind: "target", body: [Always hidden; still a + b]),
# )))
# ```
#
# Source training uses 2,048 rows and 512 updates. All adaptation arms use the
# same independent 4,096-row split, batch sequence, and 256-update budget.
# Each phase has 512 validation records; all arms share 2,048 independent test
# records. Curves record validation steps 0, 32, 128, and the final update.
# No test result selects a checkpoint or budget.
#
# Test nRMSE divides prediction RMSE by the error of the source-training mean.
# Validation curves use each phase's training mean. The full-information gate
# is below 0.25. The best prediction from the remaining input is `a`: relative
# to a baseline using the population mean, its nRMSE is `1/sqrt(2)`. Its actual
# held-out error is also recorded.
#
# Immediately after deletion, nRMSE must increase by more than 0.35 and reach
# at least 90% of the oracle error. After adaptation, restricted models must
# finish between oracle nRMSE − 0.05 and oracle nRMSE + 0.10, with prediction
# distance from `a` below 0.20 baseline RMSE. This excludes a constant predictor
# and a claim to recover the missing independent `b`. These are provisional
# gates, with tolerance for finite test samples.
#
# Deletion and inactivity both change visible context and pooling capacity;
# equality with the source is not expected. Their immediate predictions should
# agree with each other for this schema. Checkpoint and omission comparisons
# use `rtol=1e-5` and `atol=1e-6` times source-training target SD.
#
# ## Training, deletion, and adaptation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    source_budget = 512 if steps is None else min(steps, 512)
    adapt_budget = 256 if steps is None else min(steps, 256)
    lit.seed_everything(seed, workers=True)
    train = list(records(rows=2048, seed=seed + 1))
    test = list(records(rows=2048, seed=seed + 3))
    mean = float(np.mean([row["y"] for row in train]))
    tolerance = 1e-6 * float(np.std([row["y"] for row in train]))
    oracle = scores(test, np.asarray([row["a"] for row in test]), mean)
    source = build()
    source_fit = fit(source, seed=seed, split=seed + 1, rows=2048, budget=source_budget, accelerator=accelerator)
    reference = prediction(source, test)
    initial = scores(test, reference, mean)
    learned = initial["nrmse"] < 0.25
    state = deepcopy(source.state_dict())
    selected_state = deepcopy(source.nodes["/b"].state_dict())
    normalization = rf.Number.normalization(source, "/b")
    checks = {
        "Source learns both-input relationship below 0.25 nRMSE": learned,
        "Source optimizer covers current parameters": source_fit["optimizer_covers_current_parameters"],
    }
    arms = {}
    selected = rf.where("address") == "/b"
    flipped = [{**row, "b": -row["b"]} for row in test]

    with TemporaryDirectory(prefix="relflow-mutation-") as directory:
        checkpoint = Path(directory) / "source.ckpt"
        source.save(checkpoint)
        for index, arm in enumerate(("deleted", "inactive", "scratch_restricted", "continuation")):
            lit.seed_everything(seed + 100 + index, workers=True)
            model = (
                build(restricted=True).to(source.device)
                if arm == "scratch_restricted"
                else rf.Model.load(checkpoint).to(source.device)
            )
            if arm == "deleted":
                model.delete(selected)
                deleted_schema = model.schema.model_dump()
                current = model.state_dict()
                removed = [name for name in state if name.startswith("nodes./b.")]
                changes = [
                    name
                    for name, value in state.items()
                    if not name.startswith("nodes./b.") and (name not in current or not equal(value, current[name]))
                ]
                checks["Deletion removes b from schema and runtime"] = (
                    "/b" not in model.schema.requests and "/b" not in model.nodes
                )
                checks["Deletion removes the selected state entries"] = bool(removed) and all(
                    name not in current for name in removed
                )
                checks["Deletion preserves all unselected state entries"] = not changes
            elif arm == "inactive":
                model.update(selected, active=False)
                checks["Inactive control preserves all original state"] = equal(state, model.state_dict())
            elif arm == "scratch_restricted":
                checks["Scratch schema and field order match deletion"] = model.schema.model_dump() == deleted_schema
            model.eval()
            before = prediction(model, test)
            immediate = scores(test, before, mean)
            if arm == "deleted":
                deleted_prediction = before.copy()
                checks["Deletion immediately loses the selected information"] = (
                    immediate["nrmse"] > initial["nrmse"] + 0.35 and immediate["nrmse"] >= 0.9 * oracle["nrmse"]
                )
            elif arm == "inactive":
                checks["Deletion and deactivation initially give the same predictions"] = bool(
                    np.allclose(deleted_prediction, before, rtol=1e-5, atol=tolerance)
                )
            elif arm == "continuation":
                checks["Unchanged checkpoint restores source predictions"] = bool(
                    np.allclose(reference, before, rtol=1e-5, atol=tolerance)
                )
            if arm != "continuation":
                checks[f"{arm} before: removed input values cannot affect predictions"] = bool(
                    np.allclose(before, prediction(model, test, include_b=False), rtol=1e-5, atol=tolerance)
                    and np.allclose(before, prediction(model, flipped), rtol=1e-5, atol=tolerance)
                )
            parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
            fitting = fit(
                model, seed=seed + 200, split=seed + 11, rows=4096, budget=adapt_budget, accelerator=accelerator
            )
            after = prediction(model, test)
            final = scores(test, after, mean)
            updated = [name for name, p in model.named_parameters() if not torch.equal(parameters[name], p)]
            arms[arm] = {
                "before": immediate,
                "after": final,
                **fitting,
                "updated_parameter_count": len(updated),
                "pooling_capacity": model.nodes["/"].encoder.pool.mass_capacity,
            }
            checks[f"{arm}: optimizer covers current parameters"] = fitting["optimizer_covers_current_parameters"]
            checks[f"{arm}: parameters actually learn"] = bool(updated)
            checks[f"{arm}: hidden target values cannot affect predictions"] = bool(
                np.allclose(after, prediction(model, test, corrupt=True), rtol=1e-5, atol=tolerance)
            )
            if arm == "continuation":
                checks["Continuation retains full-information accuracy below 0.25 nRMSE"] = final["nrmse"] < 0.25
            else:
                checks[f"{arm}: adapts near the remaining-information limit"] = (
                    oracle["nrmse"] - 0.05 <= final["nrmse"] <= oracle["nrmse"] + 0.10
                    and final["only_a_distance_nrmse"] < 0.20
                )
                checks[f"{arm} after: removed input values cannot affect predictions"] = bool(
                    np.allclose(after, prediction(model, test, include_b=False), rtol=1e-5, atol=tolerance)
                    and np.allclose(after, prediction(model, flipped), rtol=1e-5, atol=tolerance)
                )
            if arm == "deleted":
                arms[arm]["removed_state_entries"] = removed
                arms[arm]["changed_unselected_state_entries"] = changes
                path = Path(directory) / "deleted.ckpt"
                model.save(path)
                loaded = rf.Model.load(path).to(model.device).eval()
                checks["Deleted checkpoint preserves schema and learned state"] = (
                    loaded.schema.model_dump() == model.schema.model_dump()
                    and equal(model.state_dict(), loaded.state_dict())
                    and "/b" not in loaded.nodes
                )
                checks["Deleted checkpoint preserves adapted predictions"] = bool(
                    np.allclose(after, prediction(loaded, test), rtol=1e-5, atol=tolerance)
                )

        readded = rf.Model.load(checkpoint).to(source.device).eval()
        readded.delete(selected)
        lit.seed_everything(seed + 300, workers=True)
        readded.extend(rf.where("address") == "/", b=rf.Number)
        fresh_normalization = rf.Number.normalization(readded, "/b")
        checks["Re-adding the same name does not restore its learned state"] = not equal(
            selected_state, readded.nodes["/b"].state_dict()
        )
        fresh = build().to(source.device)
        checks["Re-added input has fresh normalization"] = (
            fresh_normalization == rf.Number.normalization(fresh, "/b") and fresh_normalization != normalization
        )
        readdition = {
            "scores": scores(test, prediction(readded, test), mean),
            "source_normalization": normalization,
            "fresh_normalization": fresh_normalization,
            "source_field_order": [str(node.address) for node in source.schema.fields.fields],
            "readded_field_order": [str(node.address) for node in readded.schema.fields.fields],
        }
    return {
        "source": initial,
        "source_fit": source_fit,
        "source_prerequisite_met": learned,
        "downstream_interpretable": learned,
        "only_a_oracle": oracle,
        "arms": arms,
        "readdition": readdition,
        "test_rows": len(test),
        "optimizer_policy": "New AdamW factory and Trainer per phase; matched adaptation data and updates",
        "mutation": "delete /b; inactive control updates active=False; separate source fork deletes and re-adds b",
    }, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P050 evidence >}}
#
# ## Remaining work
#
# Three GPU seeds support this case. Calibrate gates on ten independent seeds.
# Deleting a branch subtree or a learned output requires separate experiments.
# Re-adding `b` appends it after `y`, and also initializes fresh state; its
# prediction difference cannot be attributed solely to either change.
# Relearning after re-addition and transfer advantages are not established.
#
# ## Reproduce
#
# {{< proof P050 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=5001)
