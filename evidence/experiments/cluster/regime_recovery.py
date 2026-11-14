# %% [markdown]
# ---
# title: Can it discover hidden groups with different behaviors?
# categories:
# - Cluster
# description: Predict new measurements for repeated identities whose behavior follows four hidden curves, and test
#   whether Cluster recovers the groups behind those curves.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Stable groups — ARI
#   metric:
#   - stable
#   - adjusted_rand
#   format: number
# - label: No persistent groups — ARI
#   metric:
#   - no_regime
#   - adjusted_rand
#   format: number
# - label: Stable groups — prediction error
#   metric:
#   - stable
#   - normalized_rmse
#   format: error
# evidence-group: Representation
# ---
#
# The model sees an opaque identity and a numeric input `x`, then predicts `y`.
# Across repeated observations, each of 64 identities follows one of four hidden
# curves. The group labels and curve parameters are never supplied to the model;
# training targets provide the behavioral evidence. Can `Cluster` recover which
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
# //| label: fig-evidence-cluster-regime-recovery
# //| fig-cap: "Amber cards are hidden prediction targets. Identity is sometimes reconstructed; y is always hidden."
# //| fig-alt: "Model tree with event, entity, x, y. Amber cards are hidden prediction targets. Identity is sometimes reconstructed; y is always hidden."
# #tree(node("event", kind: "root", children: (
#   node("entity", type: "Cluster", detail: "Repeated identity; 10% training mask"),
#   node("x", type: "Number"),
#   node("y", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
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
# Both arms use the same model, initialization, and 600-update training budget.
# Train, validation, and test contain the same 64 identities with independent
# numeric inputs and noise. This tests new observations of known identities;
# it does not test previously unseen identities.
#
# **Grouping:** the adjusted Rand index (ARI) compares which identities share
# a group, regardless of the names or numbers assigned to clusters. An ARI of
# 1 means the partitions match, 0 means chance-level agreement, and negative
# values mean less agreement than expected by chance. The stable arm must reach
# at least 0.80; the no-group control must remain at or below 0.15.
# If two devices truly share a rule, putting them in groups called 2 and 5 is
# a disagreement; renaming a correctly recovered group from 2 to 5 is harmless.
# ARI scores that distinction, so merely producing four or eight clusters is
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
# {{< evidence P078 status >}}
#
# The selected run does not meet the stable arm's grouping or prediction gates.
# Identity information helps prediction, but producing several clusters does
# not show that the right groups were discovered. These are final measurements
# after 600 updates per arm. No ARI history was recorded, so they do not show
# how grouping agreement changes during training.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P078 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P078 script >}}
#
# ### Process, observability, and split unit
#
# Randomly assign 64 opaque identities to four balanced groups. Each identity
# follows `y = intercept[group] + slope[group] * x + Normal(0, 0.1)`, with
# `x ~ Uniform(-1, 1)`. Intercepts are `(-6, -2, 2, 6)` and slopes are
# `(-0.6, 0.6, -0.6, 0.6)`. The partition stays fixed across independent train,
# validation, and test observations. Neither the group nor its parameters are
# schema fields. Repeated IDs supply behavioral evidence through the hidden
# numeric target, and a reconstructing identity mask engages Cluster's head.
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
# Fit the same `xs` model from the same initialization for stable-regime and
# no-regime data, each for 600 AdamW updates, batch 128, learning rate 0.003.
# Use adaptive bounds 2 through 8; the true count four is not supplied. Each arm
# receives 4,096 train, 1,024 validation, and 2,048 test rows. Validation is
# independent and does not select checkpoints. The reconstruction mask hides
# 10% of entity inputs during training; all entity inputs are visible at test.
#
# The fixed gates require ARI at least 0.80 and source-unit normalized RMSE at
# most 0.35 for the stable arm. The separately trained no-regime control must
# have ARI at most 0.15 against the now-irrelevant generating partition and
# normalized RMSE at least 0.85. For a direct dependency check, shuffling stable
# test IDs must give normalized RMSE at least 0.85 and increase it by at least
# 0.40. Require finite metrics, full identity exposure, and frozen evaluation
# assignments. The population conditional-mean oracle for the no-regime arm
# has RMSE approximately 4.49; the informed regime oracle has noise RMSE 0.1.
# Count and usage have no success gate.
#
# ### Remaining work
#
# The selected run retains the original gates and 600-update budget. It fails
# both partition-recovery and prediction-quality gates; the thresholds are not
# redefined to make this outcome pass. Diagnose optimization
# and assignment behavior with training and validation data before proposing a
# distinct follow-up experiment or a ten-seed promotion panel.
#
# This task has strongly separated intercepts; crossing curves, unbalanced
# groups, unseen identities, and a benefit over a parameter-matched Category
# model remain separate questions. Occupying several clusters is not a success
# criterion: low ARI can reveal that these assignments have not recovered the
# generating groups. The no-regime arm checks that distinction directly.
#
# ### Complete experiment code
#

# %%
"""P078: require real partition recovery and held-out skill from Cluster."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
from relflow.presets.base import resolve
from reporting import experiment, report

BUDGET = 600
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
    """Adjusted Rand agreement, invariant to the arbitrary cluster labels."""
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


def build() -> rf.Model:
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
            entity=rf.Cluster(bounds=(2, 8), p_unavailable=0.0, mask=rf.Mask(rate=0.1, reconstruct=True)),
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
    assignments = rf.Cluster.assignments(model, rf.Address("entity"))
    groups = partition(partition_seed)
    # Partial-budget smoke runs may not yet encounter every ID; absence remains
    # an explicit failed coverage gate rather than silently dropping entities.
    assigned = np.asarray(
        [assignments.get(f"entity-{index:04d}", {}).get("cluster", -1) for index in range(ENTITY_COUNT)]
    )
    return {
        "rmse": rmse,
        "baseline_rmse": baseline,
        "normalized_rmse": rmse / baseline,
        "adjusted_rand": adjusted_rand(groups, assigned),
        "assigned_entities": int((assigned >= 0).sum()),
        "occupied_clusters": int(len(np.unique(assigned[assigned >= 0]))),
        "assignments": assigned.tolist(),
        "expected_partition": groups.tolist(),
        "status": rf.Cluster.status(model, rf.Address("entity")),
    }


@experiment("P078", seed=7801)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    metrics, checks = {}, {}
    partition_seed = seed * 100
    for arm, no_regime in (("stable", False), ("no_regime", True)):
        lit.seed_everything(seed, workers=True)
        model = build()
        options = {"partition_seed": partition_seed, "no_regime": no_regime}
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
        )
        trainer = model.trainer
        model.eval()
        state = rf.Cluster.assignments(model, rf.Address("entity"))
        training_mean = float(np.mean([row["y"] for row in records(rows=TRAIN_ROWS, seed=seed * 100 + 1, **options)]))
        test = list(records(rows=TEST_ROWS, seed=seed * 100 + 3, **options))
        measured = measure(model, test, training_mean, partition_seed)
        measured.update(steps=trainer.global_step, training_mean=training_mean, noise_rmse=0.1)
        if arm == "stable":
            shuffled = list(records(rows=TEST_ROWS, seed=seed * 100 + 3, partition_seed=partition_seed, no_regime=True))
            measured["shuffled"] = measure(model, shuffled, training_mean, partition_seed)
        metrics[arm] = measured
        checks[f"{arm}: finite prediction and partition scores"] = bool(
            np.isfinite([measured["rmse"], measured["normalized_rmse"], measured["adjusted_rand"]]).all()
        )
        checks[f"{arm}: all 64 known identities have assignments"] = measured["assigned_entities"] == ENTITY_COUNT
        checks[f"{arm}: evaluation preserves assignment state"] = state == rf.Cluster.assignments(
            model, rf.Address("entity")
        )
    stable, control = metrics["stable"], metrics["no_regime"]
    checks.update(
        {
            "Stable regimes recover ARI at least 0.80": stable["adjusted_rand"] >= 0.80,
            "Stable regimes predict with normalized RMSE at most 0.35": stable["normalized_rmse"] <= 0.35,
            "No-regime control has ARI at most 0.15": control["adjusted_rand"] <= 0.15,
            "No-regime control has normalized RMSE at least 0.85": control["normalized_rmse"] >= 0.85,
            "Shuffling stable test IDs gives normalized RMSE at least 0.85": stable["shuffled"]["normalized_rmse"]
            >= 0.85,
            "Shuffling stable test IDs increases normalized RMSE by at least 0.40": (
                stable["shuffled"]["normalized_rmse"] - stable["normalized_rmse"] >= 0.40
            ),
        }
    )
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
