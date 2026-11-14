# %% [markdown]
# ---
# title: Can Group recover nonlinear identity regimes?
# categories:
# - Group
# description: Recover five nonlinear response regimes for repeated identities and compare held-out predictions with controls.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Stable-group assignment agreement
#   metric: [group_counts, "5", stable, adjusted_rand]
#   format: number
# - label: Held-out normalized error
#   metric: [group_counts, "5", stable, normalized_rmse]
#   format: number
# evidence-group: Representation
# ---
#
# Five hidden nonlinear response regimes are shared by 100 repeated identities.
# Group has five fixed slots; recovered memberships and held-out predictions
# test whether these slots learn the generating partition.
#
# ## Example
#
# ```yaml
# entity: entity-0001
# x: 0.5
# y: 5.84
# ```
#
# The answer is illustrative and hidden during prediction.
#
# ```{typst}
# //| label: fig-proof-group-hidden-regression-regimes
# //| fig-cap: "Visible identity and context predict a hidden numerical response."
# //| fig-alt: "A Group identity and Number context predict a hidden Number."
# #tree(node("event", kind: "root", children: (node("entity", type: "Group"), node("x", type: "Number"), node("y", kind: "target", type: "Number"))))
# ```
#
# ## Comparison
#
# Compare stable identities with an independently shuffled identity control,
# and test the stable model with shuffled identities at prediction time.
#
# ## Result
#
# {{< evidence P108 status >}}
#
# New Group measurements do not replace the historical Cluster proof.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P108 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P108 script >}}
#
# ### Method and predeclared checks
#
# Balanced identities share five sine-based curves with intercepts -10, -5, 0,
# 5, and 10, and independent normal noise of standard deviation 0.1. Train,
# validation, and test use independent draws and the same identity partition.
# Each arm uses 4,800 AdamW updates, batch size 128, learning rate 0.003, and
# EMA 0.99. Require final assignment ARI at least 0.95, held-out normalized RMSE
# at most 0.35, control ARI at most 0.15, and control and shuffled-test normalized
# RMSE at least 0.85. Shuffling must increase error by at least 0.40. Require
# finite metrics, assignments for all 100 known identities, and unchanged
# assignments during evaluation. The default seed 10801 and confirmations
# 10802 and 10803 are declared before execution. Unseen identities are outside
# this question.
#
# ### Complete experiment code
#
# %%
"""P108: require near-perfect partition recovery and held-out skill from Group."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
from relflow.presets.base import resolve
from reporting import experiment, report

BUDGET = 4800
EMA_DECAY = 0.99
GROUP_COUNTS = (5,)
BATCH_SIZE = 128
TRAIN_ROWS = 4096
TEST_ROWS = 2048
ENTITY_COUNT = 100
REGIME_COUNT = 5
INTERCEPTS = np.asarray((-10.0, -5.0, 0.0, 5.0, 10.0))
CURVATURE = np.asarray((-1.0, 1.0, 0.0, 1.0, -1.0))


def partition(seed: int) -> np.ndarray:
    """A balanced random mapping known only to the generator and evaluator."""
    groups = np.arange(ENTITY_COUNT) % REGIME_COUNT
    return np.random.default_rng(seed).permutation(groups)


def records(*, rows: int, seed: int, partition_seed: int, no_regime: bool = False) -> Iterator[dict]:
    """Preserve numeric records exactly when shuffling the stable identity link."""
    rng = np.random.default_rng(seed)
    entities = np.resize(np.arange(ENTITY_COUNT), rows)
    rng.shuffle(entities)
    groups = partition(partition_seed)[entities]
    x = rng.uniform(-1.0, 1.0, rows)
    y = INTERCEPTS[groups] + CURVATURE[groups] * np.sin(2 * x) + rng.normal(0.0, 0.1, rows)
    if no_regime:
        entities = entities[rng.permutation(rows)]
    for entity, value, target in zip(entities, x, y, strict=True):
        yield {"entity": f"entity-{entity:04d}", "x": float(value), "y": float(target)}


def adjusted_rand(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Adjusted Rand agreement, invariant to the arbitrary group labels."""
    _, left = np.unique(actual, return_inverse=True)
    _, right = np.unique(predicted, return_inverse=True)
    cells = np.zeros((left.max() + 1, right.max() + 1), dtype=np.int64)
    np.add.at(cells, (left, right), 1)
    joint = float(np.sum(cells * (cells - 1) / 2))
    row = cells.sum(axis=1)
    column = cells.sum(axis=0)
    left_pairs = float(np.sum(row * (row - 1) / 2))
    right_pairs = float(np.sum(column * (column - 1) / 2))
    all_pairs = len(actual) * (len(actual) - 1) / 2
    expected = left_pairs * right_pairs / all_pairs
    maximum = (left_pairs + right_pairs) / 2
    return 1.0 if maximum == expected else (joint - expected) / (maximum - expected)


class AssignmentTrajectory(lit.Callback):
    def __init__(self, partition_seed: int, validation_rows: list[dict]) -> None:
        self.partition_seed = partition_seed
        self.validation_rows = validation_rows
        self.rows: list[dict] = []

    def on_validation_epoch_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        assignments = rf.Group.assignments(pl_module, rf.Address("entity"))
        assigned = np.asarray(
            [assignments.get(f"entity-{index:04d}", {}).get("group", -1) for index in range(ENTITY_COUNT)]
        )
        present = assigned >= 0
        validation_target = np.asarray([row["y"] for row in self.validation_rows])
        validation_prediction = predict(pl_module, self.validation_rows)
        validation_rmse = float(np.sqrt(np.mean(np.square(validation_target - validation_prediction))))
        self.rows.append(
            {
                "epoch": trainer.current_epoch,
                "step": trainer.global_step,
                "adjusted_rand": adjusted_rand(partition(self.partition_seed)[present], assigned[present])
                if present.sum() == ENTITY_COUNT
                else None,
                "validation_rmse": validation_rmse,
                "assigned_entities": int(present.sum()),
            }
        )


def build(n_groups: int) -> rf.Model:
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
            entity=rf.Group(n_groups=n_groups, p_unavailable=0.0),
            x=rf.Number,
            y=rf.Number(mask=True, objective="mse"),
        )
    )
    model.preset = control_preset
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{"entity": row["entity"], "x": row["x"]} for row in rows]
    output = pl.DataFrame(model.predict(pa.Table.from_pylist(visible)))["predictions"].to_list()
    return np.asarray([row["y"]["content"] for row in output])


def measure(model: rf.Model, rows: list[dict], training_mean: float, partition_seed: int) -> dict:
    target = np.asarray([row["y"] for row in rows])
    predicted = predict(model, rows)
    baseline = float(np.sqrt(np.mean((target - training_mean) ** 2)))
    rmse = float(np.sqrt(np.mean((target - predicted) ** 2)))
    assignments = rf.Group.assignments(model, rf.Address("entity"))
    groups = partition(partition_seed)
    # Partial-budget smoke runs may not yet encounter every ID; absence remains
    # an explicit failed coverage gate rather than silently dropping entities.
    assigned = np.asarray(
        [assignments.get(f"entity-{index:04d}", {}).get("group", -1) for index in range(ENTITY_COUNT)]
    )
    return {
        "rmse": rmse,
        "baseline_rmse": baseline,
        "normalized_rmse": rmse / baseline,
        "adjusted_rand": adjusted_rand(groups, assigned),
        "assigned_entities": int((assigned >= 0).sum()),
        "occupied_groups": int(len(np.unique(assigned[assigned >= 0]))),
        "assignments": assigned.tolist(),
        "expected_partition": groups.tolist(),
    }


@experiment("P108", seed=10801)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    metrics: dict[str, dict] = {"group_counts": {}}
    checks = {}
    partition_seed = seed * 100
    for n_groups in GROUP_COUNTS:
        group_key = str(n_groups)
        metrics["group_counts"][group_key] = {}
        for arm, no_regime in (("stable", False), ("no_regime", True)):
            lit.seed_everything(seed, workers=True)
            model = build(n_groups)
            ema = rf.EMA(decay=EMA_DECAY)
            options = {"partition_seed": partition_seed, "no_regime": no_regime}
            validation_rows = list(records(rows=1024, seed=seed * 100 + 2, **options))
            trajectory = AssignmentTrajectory(partition_seed, validation_rows)
            data = rf.SyntheticDataModule(
                train=partial(records, rows=TRAIN_ROWS, seed=seed * 100 + 1, **options),
                validate=partial(records, rows=1024, seed=seed * 100 + 2, **options),
                seed=seed,
            )
            model.fit(
                data,
                batch_size=BATCH_SIZE,
                learning_rate=0.003,
                early_stopping=False,
                rollback=False,
                precision="32-true",
                strategy="auto",
                accelerator=accelerator,
                devices=1,
                max_steps=BUDGET if steps is None else min(steps, BUDGET),
                max_epochs=-1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                deterministic=True,
                num_sanity_val_steps=0,
                callbacks=[ema, trajectory],
            )
            trainer = model.trainer
            model.eval()
            state = rf.Group.assignments(model, rf.Address("entity"))
            training_mean = float(
                np.mean([row["y"] for row in records(rows=TRAIN_ROWS, seed=seed * 100 + 1, **options)])
            )
            test = list(records(rows=TEST_ROWS, seed=seed * 100 + 3, **options))
            measured = measure(model, test, training_mean, partition_seed)
            measured.update(
                steps=trainer.global_step,
                training_mean=training_mean,
                noise_rmse=0.1,
                ema_decay=EMA_DECAY,
                group_count=n_groups,
                ari_by_step=trajectory.rows,
            )
            if arm == "stable":
                shuffled = list(
                    records(rows=TEST_ROWS, seed=seed * 100 + 3, partition_seed=partition_seed, no_regime=True)
                )
                measured["shuffled"] = measure(model, shuffled, training_mean, partition_seed)
            metrics["group_counts"][group_key][arm] = measured
            label = f"{n_groups} groups / {arm}"
            checks[f"{label}: finite prediction and partition scores"] = bool(
                np.isfinite([measured["rmse"], measured["normalized_rmse"], measured["adjusted_rand"]]).all()
            )
            checks[f"{label}: all 100 known identities have assignments"] = (
                measured["assigned_entities"] == ENTITY_COUNT
            )
            checks[f"{label}: evaluation preserves assignment state"] = state == rf.Group.assignments(
                model, rf.Address("entity")
            )
        stable = metrics["group_counts"][group_key]["stable"]
        control = metrics["group_counts"][group_key]["no_regime"]
        checks.update(
            {
                f"{n_groups} groups: stable regimes recover ARI at least 0.95": stable["adjusted_rand"] >= 0.95,
                f"{n_groups} groups: stable regimes predict with normalized RMSE at most 0.35": (
                    stable["normalized_rmse"] <= 0.35
                ),
                f"{n_groups} groups: no-regime control has ARI at most 0.15": control["adjusted_rand"] <= 0.15,
                f"{n_groups} groups: no-regime control has normalized RMSE at least 0.85": (
                    control["normalized_rmse"] >= 0.85
                ),
                f"{n_groups} groups: shuffled test IDs give normalized RMSE at least 0.85": (
                    stable["shuffled"]["normalized_rmse"] >= 0.85
                ),
                f"{n_groups} groups: shuffling increases normalized RMSE by at least 0.40": (
                    stable["shuffled"]["normalized_rmse"] - stable["normalized_rmse"] >= 0.40
                ),
            }
        )
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
