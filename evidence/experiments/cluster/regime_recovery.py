# %% [markdown]
# ---
# title: Can it discover hidden groups with different behaviors?
# categories:
# - Cluster
# evidence-id: P078
# description: Repeated identities follow four hidden numerical rules. Ask the Cluster field to recover the groups
#   while predicting new observations.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Stable hidden groups
#   metric:
#   - stable
#   - normalized_rmse
#   format: error
# - label: No persistent groups
#   metric:
#   - no_regime
#   - normalized_rmse
#   format: error
# evidence-group: Representation
# evidence-result:
#   accelerator: cpu
#   checks:
#     No-regime control has ARI at most 0.15: true
#     No-regime control has normalized RMSE at least 0.85: true
#     Shuffling stable test IDs gives normalized RMSE at least 0.85: true
#     Shuffling stable test IDs increases normalized RMSE by at least 0.40: true
#     Stable regimes predict with normalized RMSE at most 0.35: false
#     Stable regimes recover ARI at least 0.80: false
#     'no_regime: all 64 known identities have assignments': true
#     'no_regime: evaluation preserves assignment state': true
#     'no_regime: finite prediction and partition scores': true
#     'stable: all 64 known identities have assignments': true
#     'stable: evaluation preserves assignment state': true
#     'stable: finite prediction and partition scores': true
#   duration_seconds: 50.818
#   metrics:
#     no_regime:
#       adjusted_rand: -0.031176929072486367
#       assigned_entities: 64
#       assignments:
#       - 0
#       - 6
#       - 3
#       - 5
#       - 4
#       - 6
#       - 4
#       - 6
#       - 3
#       - 7
#       - 5
#       - 3
#       - 6
#       - 6
#       - 0
#       - 5
#       - 5
#       - 0
#       - 5
#       - 0
#       - 0
#       - 6
#       - 1
#       - 4
#       - 0
#       - 3
#       - 5
#       - 7
#       - 3
#       - 2
#       - 2
#       - 5
#       - 2
#       - 7
#       - 4
#       - 6
#       - 7
#       - 2
#       - 4
#       - 7
#       - 2
#       - 2
#       - 5
#       - 4
#       - 4
#       - 3
#       - 3
#       - 2
#       - 4
#       - 1
#       - 5
#       - 0
#       - 2
#       - 2
#       - 6
#       - 7
#       - 7
#       - 4
#       - 7
#       - 6
#       - 7
#       - 2
#       - 6
#       - 4
#       baseline_rmse: 4.4923299919737
#       expected_partition:
#       - 2
#       - 0
#       - 1
#       - 3
#       - 0
#       - 0
#       - 1
#       - 2
#       - 3
#       - 0
#       - 3
#       - 2
#       - 2
#       - 1
#       - 0
#       - 0
#       - 3
#       - 3
#       - 2
#       - 0
#       - 2
#       - 0
#       - 1
#       - 3
#       - 1
#       - 1
#       - 1
#       - 3
#       - 3
#       - 3
#       - 1
#       - 2
#       - 1
#       - 2
#       - 2
#       - 3
#       - 2
#       - 1
#       - 1
#       - 0
#       - 1
#       - 0
#       - 2
#       - 2
#       - 0
#       - 0
#       - 3
#       - 3
#       - 2
#       - 2
#       - 0
#       - 3
#       - 1
#       - 2
#       - 0
#       - 1
#       - 3
#       - 3
#       - 1
#       - 1
#       - 0
#       - 0
#       - 3
#       - 2
#       noise_rmse: 0.1
#       normalized_rmse: 1.001393979452723
#       occupied_clusters: 8
#       rmse: 4.498592207677363
#       status:
#         committed:
#         - 0
#         - 1
#         usage:
#         - 0.3892781436443329
#         - 0.6100784540176392
#         - 3.577067400328815e-05
#         - 5.786109613836743e-05
#         - 0.0001898727350635454
#         - 0.0002021004183916375
#         - 8.135511598084122e-05
#         - 7.71515115047805e-05
#       steps: 600
#       training_mean: 0.0089152195024117
#     stable:
#       adjusted_rand: 0.274748322147651
#       assigned_entities: 64
#       assignments:
#       - 4
#       - 2
#       - 6
#       - 5
#       - 2
#       - 6
#       - 2
#       - 5
#       - 5
#       - 6
#       - 5
#       - 5
#       - 4
#       - 6
#       - 6
#       - 1
#       - 4
#       - 4
#       - 5
#       - 0
#       - 4
#       - 2
#       - 2
#       - 5
#       - 2
#       - 6
#       - 6
#       - 5
#       - 4
#       - 3
#       - 2
#       - 5
#       - 6
#       - 5
#       - 4
#       - 5
#       - 4
#       - 2
#       - 2
#       - 2
#       - 2
#       - 6
#       - 4
#       - 7
#       - 6
#       - 6
#       - 7
#       - 5
#       - 7
#       - 5
#       - 2
#       - 5
#       - 6
#       - 7
#       - 6
#       - 3
#       - 4
#       - 5
#       - 6
#       - 6
#       - 6
#       - 6
#       - 5
#       - 4
#       baseline_rmse: 4.4923299919737
#       expected_partition:
#       - 2
#       - 0
#       - 1
#       - 3
#       - 0
#       - 0
#       - 1
#       - 2
#       - 3
#       - 0
#       - 3
#       - 2
#       - 2
#       - 1
#       - 0
#       - 0
#       - 3
#       - 3
#       - 2
#       - 0
#       - 2
#       - 0
#       - 1
#       - 3
#       - 1
#       - 1
#       - 1
#       - 3
#       - 3
#       - 3
#       - 1
#       - 2
#       - 1
#       - 2
#       - 2
#       - 3
#       - 2
#       - 1
#       - 1
#       - 0
#       - 1
#       - 0
#       - 2
#       - 2
#       - 0
#       - 0
#       - 3
#       - 3
#       - 2
#       - 2
#       - 0
#       - 3
#       - 1
#       - 2
#       - 0
#       - 1
#       - 3
#       - 3
#       - 1
#       - 1
#       - 0
#       - 0
#       - 3
#       - 2
#       noise_rmse: 0.1
#       normalized_rmse: 0.5524729339632773
#       occupied_clusters: 8
#       rmse: 2.4818907309969362
#       shuffled:
#         adjusted_rand: 0.274748322147651
#         assigned_entities: 64
#         assignments:
#         - 4
#         - 2
#         - 6
#         - 5
#         - 2
#         - 6
#         - 2
#         - 5
#         - 5
#         - 6
#         - 5
#         - 5
#         - 4
#         - 6
#         - 6
#         - 1
#         - 4
#         - 4
#         - 5
#         - 0
#         - 4
#         - 2
#         - 2
#         - 5
#         - 2
#         - 6
#         - 6
#         - 5
#         - 4
#         - 3
#         - 2
#         - 5
#         - 6
#         - 5
#         - 4
#         - 5
#         - 4
#         - 2
#         - 2
#         - 2
#         - 2
#         - 6
#         - 4
#         - 7
#         - 6
#         - 6
#         - 7
#         - 5
#         - 7
#         - 5
#         - 2
#         - 5
#         - 6
#         - 7
#         - 6
#         - 3
#         - 4
#         - 5
#         - 6
#         - 6
#         - 6
#         - 6
#         - 5
#         - 4
#         baseline_rmse: 4.4923299919737
#         expected_partition:
#         - 2
#         - 0
#         - 1
#         - 3
#         - 0
#         - 0
#         - 1
#         - 2
#         - 3
#         - 0
#         - 3
#         - 2
#         - 2
#         - 1
#         - 0
#         - 0
#         - 3
#         - 3
#         - 2
#         - 0
#         - 2
#         - 0
#         - 1
#         - 3
#         - 1
#         - 1
#         - 1
#         - 3
#         - 3
#         - 3
#         - 1
#         - 2
#         - 1
#         - 2
#         - 2
#         - 3
#         - 2
#         - 1
#         - 1
#         - 0
#         - 1
#         - 0
#         - 2
#         - 2
#         - 0
#         - 0
#         - 3
#         - 3
#         - 2
#         - 2
#         - 0
#         - 3
#         - 1
#         - 2
#         - 0
#         - 1
#         - 3
#         - 3
#         - 1
#         - 1
#         - 0
#         - 0
#         - 3
#         - 2
#         normalized_rmse: 1.2307000358925804
#         occupied_clusters: 8
#         rmse: 5.528710682363348
#         status:
#           committed:
#           - 0
#           - 1
#           usage:
#           - 0.5702924132347107
#           - 0.42859697341918945
#           - 9.824825974646956e-05
#           - 0.00026156831881962717
#           - 0.0001388096425216645
#           - 0.0001363397459499538
#           - 0.00021528090292122215
#           - 0.0002612356038298458
#       status:
#         committed:
#         - 0
#         - 1
#         usage:
#         - 0.5702924132347107
#         - 0.42859697341918945
#         - 9.824825974646956e-05
#         - 0.00026156831881962717
#         - 0.0001388096425216645
#         - 0.0001363397459499538
#         - 0.00021528090292122215
#         - 0.0002612356038298458
#       steps: 600
#       training_mean: 0.0089152195024117
#   mode: full
#   outcome: not_met
#   provenance:
#     code_sha256: c1471e20b511911620c8eec7a929c81280f23623c9431b756cfd5fe282ab943e
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 62efb5f9e501
#     packages:
#       lightning: 2.6.4
#       numpy: 2.4.6
#       polars: 2.0.0
#       pyarrow: 24.0.0
#       pydantic: 2.13.4
#       relflow: 0.4.1
#       tensordict: 0.12.4
#       torch: 2.12.0+cpu
#       torchmetrics: 1.9.0
#     platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.41
#     python: 3.12.16
#     relflow_path: /tmp/runtime/lib/python3.12/site-packages/relflow
#     relflow_revision: 41d6f5fb74c0ba9f0be104c896913a55adb125f5
#     relflow_source_sha256: 00fdf397aeefb3a4e705d431850e0a6294eef142989513fddcb903bb1aefa873
#     script_sha256: f4d20bacfa8dbbbbb7743b75129af1dda933f6016793d0492ff570524fca7075
#     source_sha256: 2919e626ef023270a89aa514cd8ecf5746eee12f684357debb1042c57558b17f
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 7801
#   started_at: '2026-11-01T01:50:52.034354+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41261482839-1
#   candidate:
#     revision: 41d6f5fb74c0ba9f0be104c896913a55adb125f5
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   baseline:
#     revision: 1add6d8919e686dc63b7fa6c4cdbdc437be2b570
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#   manifest_sha256: f4b3bfe706e43e38785ab5bed94c3f51eb44fe7314d0a1b2f0dd3c6f3f7bef92
#   passed: true
#   github_run:
#     event: pull_request
#     head_branch: dev/evidence-in-docs
#     head_sha: 8bb7764d8186c743eb554cb4a91dedb205ad4cb9
#     html_url: https://github.com/relflow/relflow/actions/runs/41261482839
#     pull_requests:
#     - base_sha: 1add6d8919e686dc63b7fa6c4cdbdc437be2b570
#       head_sha: 8bb7764d8186c743eb554cb4a91dedb205ad4cb9
#       number: 64
#     run_attempt: 1
#     run_id: 41261482839
#   release: https://github.com/relflow/docs/releases/tag/ci-41261482839-attempt-1
#   reports_sha256: bc67a1cb3725cbf9c5a8deb65844297d70f6d2122cd3b7de5c1d15f56603864f
# ---
#
# ## Example
#
# ```yaml
# entity: entity-0017
# x: 0.5
# y: -6.27
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
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
# Measure actual grouping agreement and held-out prediction, then shuffle identities and train a control with no persistent groups.
#
# ## Result
#
# {{< evidence P078 status >}}
#
# All three seeds fail the recovery checks at this budget. Some identity information helps prediction, but producing several clusters does not show the right groups were discovered.
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
# ### Complete experiment code
#

# %%
"""P078: require real partition recovery and held-out skill from Cluster."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P078"
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
    model = rf.Model.xs(
        entity=rf.Cluster(bounds=(2, 8), p_unavailable=0.0, mask=rf.Mask(rate=0.1, reconstruct=True)),
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
    report(PROOF_ID, run, seed=7801)

# %% [markdown]
# </details>
