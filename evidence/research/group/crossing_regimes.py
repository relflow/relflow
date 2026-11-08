# %% [markdown]
# ---
# title: Can Group recover crossing behavioral rules?
# categories:
# - Group
# description: Repeated identities follow four numerical rules that cross at the same input value.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-requires:
# - A relflow wheel exporting the unreleased Group datatype.
# evidence-readout:
# - label: Stable identities — ARI
#   metric: [group_counts, '4', stable, adjusted_rand]
#   format: number
# - label: Stable identities — prediction error
#   metric: [group_counts, '4', stable, normalized_rmse]
#   format: error
# - label: Shuffled training identities — ARI
#   metric: [group_counts, '4', no_regime, adjusted_rand]
#   format: number
# evidence-group: Representation
# ---
#
# A model sees an opaque entity ID and a numeric setting, and predicts a response.
# Each of 64 known identities follows one of four hidden rules. Unlike an easy
# separation by intercept alone, all four rules meet at `x = -0.1`. Can the
# model learn which identities respond alike and predict new observations?
#
# ## Example
#
# ```yaml
# entity: entity-0001
# x: 0.5
# y: -0.60
# ```
#
# For the default generator seed, this identity follows the rule `y = -0.1 - x`
# before noise. At `x = 0.5`, its noiseless answer is -0.60. The model sees
# `entity` and `x`; `y` is a hidden target. The generating rule is known only to
# the evaluator. The illustrated answer excludes the independently sampled noise.
#
# ```{typst}
# //| label: fig-evidence-p101
# //| fig-cap: "An identity and setting predict a response under a hidden rule."
# //| fig-alt: "Model tree with visible Group entity and Number x, and a hidden Number target y."
# #tree(node("observation", kind: "root", children: (
#   node("entity", type: "Group", detail: "Repeated identity"),
#   node("x", type: "Number"),
#   node("y", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Compare stable identity-to-rule links with a no-regime control that shuffles
# identities across otherwise unchanged numerical observations. Also shuffle
# identities only at test time for the model trained on stable links.
#
# Assignment agreement uses adjusted Rand index (ARI): 1 means the partition
# matches regardless of group numbering, and 0 is chance-level agreement.
# Normalized RMSE compares prediction error with always guessing the training
# mean: 1 matches that guess, and lower is better. Both held-out observations
# and the training data contain the same identities; unseen identities are a
# separate question.
#
# ## Result
#
# {{< evidence P101 status >}}
#
# The gates below are provisional hypotheses, not measured successes. This
# research study requires a wheel exporting `Group`; no historical Cluster
# measurement establishes this result. A smoke run does not establish learning.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P101 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P101 script >}}
#
# ### Population, controls, and criteria
#
# The four rules are `y = -0.3 - 3x`, `-0.1 - x`, `0.1 + x`, and `0.3 + 3x`,
# each with independent Gaussian noise of standard deviation 0.1 and `x`
# uniformly sampled from -1 to 1. Sixteen identities follow each rule. The
# conditionally balanced rules have mean zero for every `x` when identity is
# unavailable, so the no-regime comparison remains meaningful.
#
# Each arm uses 4,096 train, 1,024 validation, and 2,048 independent test rows;
# four fixed group slots; 4,800 AdamW updates; batch size 128; learning rate
# 0.003; and EMA 0.99. The same data and initialization seeds are used across
# arms. Require stable ARI ≥ 0.95 and normalized RMSE ≤ 0.35, no-regime ARI
# ≤ 0.15 and normalized RMSE ≥ 0.85, and shuffled-test normalized RMSE ≥ 0.85
# and at least 0.40 worse than the intact test. Require all identities and
# unchanged evaluation assignments. These gates are intentionally not relaxed
# if the harder task fails.
#
# This single seed is exploratory. Unbalanced groups, new identities, a
# parameter-matched Category baseline, and multiple independent seeds require
# separate evidence.
#
# ### Complete experiment code
#
# %%
"""P101: test Group recovery when hidden response rules cross."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import experiment, report

BUDGET = 4800
EMA_DECAY = 0.99
GROUP_COUNTS = (4,)
BATCH_SIZE = 128
TRAIN_ROWS = 4096
TEST_ROWS = 2048
ENTITY_COUNT = 64
REGIME_COUNT = 4
INTERCEPTS = np.asarray((-0.3, -0.1, 0.1, 0.3))
SLOPES = np.asarray((-3.0, -1.0, 1.0, 3.0))


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
    y = INTERCEPTS[groups] + SLOPES[groups] * x + rng.normal(0.0, 0.1, rows)
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
    model = rf.Model.xs(
        entity=rf.Group(n_groups=n_groups, p_unavailable=0.0),
        x=rf.Number,
        y=rf.Number(mask=True, objective="mse"),
    )
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{"entity": row["entity"], "x": row["x"]} for row in rows]
    output = model.predict(pa.Table.from_pylist(visible))["predictions"].to_pylist()
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


@experiment("P101", seed=10101)
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
            checks[f"{label}: all 64 known identities have assignments"] = measured["assigned_entities"] == ENTITY_COUNT
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
