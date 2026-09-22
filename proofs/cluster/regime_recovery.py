# %% [markdown]
# ---
# title: Recovering Hidden Behavioral Regimes
# categories: [Clustering]
# proof-id: P078
# description: Measure assignment agreement and independent predictions rather than treating cluster count as recovery.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P078 status >}}
#
# ## Insights
#
# **The three CPU seeds failed to recover the hidden regimes at this budget.**
# Assignment ARI ranged from 0.074 to 0.227, below the required 0.80. Held-out
# source-unit RMSE was 2.65–3.77, or 59.2–83.9% of the constant-baseline error,
# above the 35% gate. All runs occupied seven clusters; that count does not
# establish a correct partition of the four generating groups.
#
# Shuffling test identities increased error to 106.7–114.9% of baseline, so
# identity information was useful, but two seeds missed the required 0.40
# normalized-error gap. The separately trained no-regime control stayed near
# baseline (100.0–100.2%) with ARI between -0.005 and 0.040. Its matched numeric
# observations rule out target marginal differences as the source of the gain.
# These results show some identity-dependent prediction, while the proposed
# regime-recovery capability remains unestablished under the original gates.

# %%
"""P078: require real partition recovery and held-out skill from Cluster."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
from reporting import report

import relflow as rf

PROOF_ID = "P078"
BUDGET = 600
BATCH_SIZE = 128
TRAIN_ROWS = 4096
TEST_ROWS = 2048
ENTITY_COUNT = 64
REGIME_COUNT = 4
INTERCEPTS = np.asarray((-6.0, -2.0, 2.0, 6.0))
SLOPES = np.asarray((-0.6, 0.6, -0.6, 0.6))

# %% [markdown]
# ## Process, observability, and split unit
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
# ```yaml
# entity: entity-0017
# x: 0.5
# y: -6.27
# ```
#
# An illustrative first-regime observation has noiseless target -6.3; the
# displayed deviation is one possible noise realization. IDs do not encode
# regime membership; these examples use an illustrative partition.
#
# ```yaml
# entity: entity-0042
# x: 0.5
# y: -6.32
# ```
#
# A different identity in that same illustrative regime should be grouped with
# the first one, even though no shared label appears in their record fields.
#
# ```yaml
# entity: entity-0017
# x: 0.5
# y: 6.24
# ```
#
# In the no-regime control, an independently permuted identity can accompany
# any regime on each observation. Every numeric pair and the complete identity
# frequency distribution are retained, but an ID no longer identifies a curve.
# The conditional mean of the population remains zero for every x because the
# four intercepts and slopes each average to zero.


# %%
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
    model = rf.Model.xs(
        batch_size=BATCH_SIZE,
        entity=rf.Cluster(bounds=(2, 8), p_unavailable=0.0, mask=rf.Mask(rate=0.1, reconstruct=True)),
        x=rf.Number,
        y=rf.Number(mask=True, objective="mse"),
    )
    model.optimizer = rf.adamw(learning_rate=0.003)
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{"entity": row["entity"], "x": row["x"]} for row in rows]
    output = model.predict(pa.Table.from_pylist(visible))["predictions"].to_pylist()
    return np.asarray([row["/y"]["content"] for row in output])


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


# %% [markdown]
# ## Model and predeclared gates
#
# ```{typst}
# //| label: fig-proof-cluster-regime-recovery
# //| fig-cap: "Identity assignments must capture stable curves while predicting new numeric observations."
# //| fig-alt: "An event has a Cluster entity with a reconstructing 10 percent mask, a visible Number x, and hidden Number y. Hidden regimes are absent from the schema."
# #tree(node("event", kind: "root", children: (
#   node("entity", type: "Cluster", detail: "Repeated identity; 10% training mask"),
#   node("x", type: "Number"),
#   node("y", kind: "target", type: "Number"),
# )))
# ```
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


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    metrics, checks = {}, {}
    partition_seed = seed * 100
    for arm, no_regime in (("stable", False), ("no_regime", True)):
        lit.seed_everything(seed, workers=True)
        model = build()
        options = {"partition_seed": partition_seed, "no_regime": no_regime}
        data = rf.SyntheticDataModule(
            model=model,
            train=partial(records, rows=TRAIN_ROWS, seed=seed * 100 + 1, **options),
            validate=partial(records, rows=1024, seed=seed * 100 + 2, **options),
            seed=seed,
        )
        trainer = lit.Trainer(
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
        trainer.fit(model, datamodule=data)
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


# %% [markdown]
# ## Evidence
#
# {{< proof P078 evidence >}}
#
# ## Remaining work
#
# The three-seed panel retained the original gates and 600-update budget. All
# three runs failed both partition-recovery and prediction-quality gates; the
# threshold is not redefined to make these outcomes pass. Diagnose optimization
# and assignment behavior with training and validation data before proposing a
# distinct follow-up experiment or a ten-seed promotion panel.
#
# This task has strongly separated intercepts; crossing curves, unbalanced
# groups, unseen identities, and a benefit over a parameter-matched Category
# model remain separate questions. No-regime assignments occupied seven or
# eight clusters. Their low ARI measures a lack of recovered meaning; Cluster
# did not automatically reject the spurious groups.
#
# ## Reproduce
#
# {{< proof P078 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=7801)
