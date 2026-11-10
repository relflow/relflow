# %% [markdown]
# ---
# title: Can Group recover known hidden regimes?
# categories:
# - Group
# description: Repeated identities follow four hidden numerical rules. Test whether four fixed Group slots recover
#   the groups while predicting new observations.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Stable groups — ARI
#   metric:
#   - group_counts
#   - '4'
#   - stable
#   - adjusted_rand
#   format: number
# - label: No persistent groups — ARI
#   metric:
#   - group_counts
#   - '4'
#   - no_regime
#   - adjusted_rand
#   format: number
# - label: Stable groups — prediction error
#   metric:
#   - group_counts
#   - '4'
#   - stable
#   - normalized_rmse
#   format: error
# evidence-group: Representation
# ---
#
# The model sees an opaque identity and a numeric input `x`, then predicts `y`.
# Across repeated observations, each of 64 identities follows one of four hidden
# curves. The group labels and curve parameters are never supplied to the model;
# training targets provide the behavioral evidence. Can `Group` recover which
# identities behave alike, as well as predict their next measurement?
#
# Think of the identities as 64 named devices, each measured at different
# settings. Some devices respond similarly and should end up in the same
# learned group. The device names carry no clues about that behavior. These
# measurements are generated from known rules, so we can compare the learned
# groups with the groups that actually produced the data.
#
# ## Example
#
# ```yaml
# entity: entity-0001
# x: 0.5
# y: -6.27
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# In this illustrative record, the model receives the device name and the
# setting `x = 0.5`; it must predict the response `y`. Repeated training records
# let it learn how that identity behaves. A different device following the same
# rule should receive a similar grouping assignment even though its name differs.
#
# The four generator rules give these responses before the small random noise:
#
# | Hidden rule | Response at x = -0.5 | Response at x = 0.5 |
# | --- | ---: | ---: |
# | -6 - 0.6x | -5.7 | -6.3 |
# | -2 + 0.6x | -2.3 | -1.7 |
# | 2 - 0.6x | 2.3 | 1.7 |
# | 6 + 0.6x | 5.7 | 6.3 |
#
# These are worked examples of the data-generating rules. The model is never
# given this table or a device's true group. It learns from individual labeled
# measurements, and the evaluator keeps the true grouping for scoring.
#
# ```{typst}
# //| label: fig-proof-group-regime-recovery
# //| fig-cap: "The entity identity is visible input; y is the hidden prediction target."
# //| fig-alt: "Model tree with event, entity, x, y. The entity identity is visible input; y is the hidden prediction target."
# #tree(node("event", kind: "root", children: (
#   node("entity", type: "Group", detail: "Repeated identity; fixed group slots"),
#   node("x", type: "Number"),
#   node("y", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Compare four fixed Group slots with a no-regime control on the same hidden regimes.
#
# | Condition | What an identity tells the model | Expected behavior |
# | --- | --- | --- |
# | Stable groups | Each identity always follows the same hidden curve. | Recover the generating groups and predict new measurements accurately. |
# | No persistent groups | Identities are shuffled across observations; the numeric records and identity counts stay the same. | No agreement with the original grouping, and prediction error near a constant guess. |
#
# For the stable condition, imagine that the same device repeatedly follows
# the first row of the table. Its name becomes a useful clue about its next
# response. In the shuffled condition, that name might accompany the first
# rule on one record and the fourth rule on the next. The device still appears
# just as often, but remembering its name no longer identifies a response rule.
# In the detailed measurements, `stable` and `no_regime` name these two conditions.
#
# Both arms use the same four-slot model, initialization, and 4,800-update budget.
# Train, validation, and test contain the same 64 identities with independent
# numeric inputs and noise. This tests new observations of known identities;
# it does not test previously unseen identities.
#
# **Grouping:** the adjusted Rand index (ARI) compares which identities share
# a group, regardless of the names or numbers assigned to groups. An ARI of
# 1 means the partitions match, 0 means chance-level agreement, and negative
# values mean less agreement than expected by chance. The stable arm must reach
# at least 0.95; the no-group control must remain at or below 0.15.
# If two devices truly share a rule, putting them in groups called 2 and 5 is
# a disagreement; renaming a correctly recovered group from 2 to 5 is harmless.
# ARI scores that distinction, so merely occupying all four slots is
# insufficient.
#
# **Prediction:** normalized RMSE divides test prediction error by the error of
# always guessing the training mean. A value of 1 matches that constant guess;
# lower is better. The stable arm must reach at most 0.35, while the no-group
# control must remain at or above 0.85. Separately, shuffling IDs only at test
# time checks whether the trained stable model depends on the correct identity.
#
# ## Result
#
# {{< evidence P111 status >}}
#
# Archived results use the former Cluster contract and do not validate this Group-based source.
# Current runs compare assignment recovery and held-out prediction with four fixed Group slots.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P111 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P111 script >}}
#
# ### Process, observability, and split unit
#
# Randomly assign 64 opaque identities to four balanced groups. Each identity
# follows `y = intercept[group] + slope[group] * x + Normal(0, 0.1)`, with
# `x ~ Uniform(-1, 1)`. Intercepts are `(-6, -2, 2, 6)` and slopes are
# `(-0.6, 0.6, -0.6, 0.6)`. The partition stays fixed across independent train,
# validation and test observations. Neither the group nor its parameters are
# schema fields. Repeated IDs supply behavioral evidence through the hidden
# numeric target, and assignment angles learn through its downstream gradient.
#
# The split unit is the observation of a known identity. This tests whether
# repeated evidence reveals stable groups, not unseen-identity generalization.
# Every split contains the same 64 identities with fresh x and noise samples.
#
# In the no-regime control, an independently permuted identity can accompany
# any regime on each observation. Every numeric pair and the complete identity
# frequency distribution are retained, but an ID no longer identifies a curve.
# The conditional mean of the population remains zero for every x because the
# four intercepts and slopes each average to zero.
#
# ### Model and predeclared gates
#
# Run one capacity of four with the same stable-regime and no-regime data seeds.
# Each model trains for at most 4,800 AdamW updates, batch 128, learning rate
# 0.003, with EMA decay 0.99. Each arm receives 4,096 train, 1,024 validation,
# and 2,048 test rows. This tests recovery with a fixed capacity matching the
# number of regimes, not adaptive discovery of that number.
#
# The fixed gates require stable-arm ARI at least 0.95 and normalized RMSE at
# most 0.35. The no-regime control must have ARI at most 0.15 and normalized
# RMSE at least 0.85. Shuffling stable test IDs must give normalized RMSE at
# least 0.85 and increase it by at least 0.40. Require finite metrics, full
# identity exposure, and frozen evaluation assignments.
#
# ### Remaining work
#
# Preserve the near-perfect ARI gate across three seeds; any failure remains a
# failed recovery result rather than a reason to lower the threshold. Diagnose
# optimization behavior with training and validation data before a promotion
# panel.
#
# This task has strongly separated intercepts; crossing curves, unbalanced
# groups, unseen identities, and a benefit over a parameter-matched Category
# model remain separate questions.
#
# ### Complete experiment code
#

# %%
"""P111: require near-perfect partition recovery and held-out skill from Group."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import polars as pl
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
INTERCEPTS = np.asarray((-6.0, -2.0, 2.0, 6.0))
SLOPES = np.asarray((-0.6, 0.6, -0.6, 0.6))


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


@experiment("P111", seed=11101)
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
