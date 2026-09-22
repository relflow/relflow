# %% [markdown]
# ---
# title: Combining noisy and incomplete evidence
# categories: [Representation]
# proof-id: P074
# description: Compare learned sensor fusion with the exact Gaussian conditional mean under familiar and unseen missingness patterns.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P074 status >}}
#
# ## Insights
#
# Several imperfect measurements can reduce expected prediction error even
# when no individual reading determines the answer. This experiment compares
# the learned prediction with the known conditional mean under every sensor
# visibility pattern. One combination is absent during fitting. An irrelevant
# sensor, no-evidence condition, and deliberately corrupted reading separate
# evidence fusion from assumptions of universal robustness.
#
# One of three CPU seeds met every gate. Joint evidence reduced MSE by
# 31.2–39.1% relative to the best single sensor in all three runs, and shuffling
# nuisance readings changed predictions by RMS 0.042–0.077. The unseen pattern's
# squared distance to the conditional mean ranged from 0.015 to 0.226: seed
# 7403 missed its 0.15 gate and several familiar-pattern gates. Seeds 7401 and
# 7403 also retained too much variation when every informative sensor was
# hidden, with prediction standard deviations 0.117 and 0.118 against a 0.10
# gate. Evidence fusion is useful here; reliable prior fallback and missingness
# transfer are not established across seeds.
#
# Corrupting a visible sensor raised MSE to 3.70–4.82, compared with 0.255–0.367
# with intact joint evidence. Skipping that sensor exactly restored its paired
# two-sensor prediction. Every hidden-value invariance check passed.

# %%
"""P074: Gaussian evidence fusion, missingness transfer, and a known oracle."""

from collections.abc import Iterator
from functools import partial
from itertools import product

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
from reporting import report

import relflow as rf

PROOF_ID = "P074"
BUDGET = 600
TRAIN_ROWS = 4096
TEST_ROWS = 2048
SENSORS = ("first", "second", "third")
NOISE = np.asarray([0.8, 1.0, 1.2])
UNSEEN = (False, True, False)
PATTERNS = tuple(product((False, True), repeat=3))

# %% [markdown]
# ## Process and information boundary
#
# Independently draw `z ~ Normal(0,1)` and three sensors `x_i = z + e_i`, with
# independent Gaussian errors of standard deviations 0.8, 1.0, and 1.2. A
# fourth, nuisance reading is independent `Normal(0,1)`. The target is z and is
# always hidden. For any visible subset V, the exact conditional mean is
# `sum(x_i / sigma_i**2) / (1 + sum(1 / sigma_i**2))`; its conditional variance
# is `1 / (1 + sum(1 / sigma_i**2))`. With no sensors visible, the mean is zero.
#
# ```yaml
# first: 0.5
# second: 1.1
# third: -0.2
# hide_first: false
# hide_second: false
# hide_third: false
# nuisance: 1.4
# target: 0.4
# ```
#
# Hiding the middle sensor gives the combination withheld from training:
#
# ```yaml
# first: 0.5
# second: 1.1
# third: -0.2
# hide_first: false
# hide_second: true
# hide_third: false
# nuisance: 1.4
# target: 0.4
# ```
#
# When all informative sources disappear, only the prior remains:
#
# ```yaml
# first: 0.5
# second: 1.1
# third: -0.2
# hide_first: true
# hide_second: true
# hide_third: true
# nuisance: 1.4
# target: 0.4
# ```
#
# ```{typst}
# //| label: fig-proof-incomplete-evidence
# //| fig-cap: "Three independently noisy sensors can be skipped; an unrelated field remains visible."
# //| fig-alt: "Record contains three Number sensors, each skipped by its own hide flag, a Number nuisance sensor, and an always-hidden Number target."
# #tree(node("record", kind: "root", children: (
#   node("first", type: "Number", body: [Skip when `hide_first`]),
#   node("second", type: "Number", body: [Skip when `hide_second`]),
#   node("third", type: "Number", body: [Skip when `hide_third`]),
#   node("nuisance", type: "Number", body: [Independent noise]),
#   node("target", kind: "target", type: "Number"),
# )))
# ```


# %%
def records(*, rows: int, seed: int, training: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    target = rng.normal(size=rows)
    sensors = target[:, None] + rng.normal(size=(rows, 3)) * NOISE
    nuisance = rng.normal(size=rows)
    hidden = rng.random((rows, 3)) < 0.5
    if training:
        # Condition only on visibility, never on values or targets.
        excluded = np.all(hidden == UNSEEN, axis=1)
        while excluded.any():
            hidden[excluded] = rng.random((int(excluded.sum()), 3)) < 0.5
            excluded = np.all(hidden == UNSEEN, axis=1)
    for values, mask, noise, answer in zip(sensors, hidden, nuisance, target, strict=True):
        yield {
            **{name: float(value) for name, value in zip(SENSORS, values, strict=True)},
            **{f"hide_{name}": bool(value) for name, value in zip(SENSORS, mask, strict=True)},
            "nuisance": float(noise),
            "target": float(answer),
        }


def build() -> rf.Model:
    return rf.Model.xs(
        batch_size=64,
        fields={
            **{name: rf.Number(mask=rf.Mask(query=f"hide_{name}", skip=True, dropout=False)) for name in SENSORS},
            "nuisance": rf.Number,
            "target": rf.Number(mask=True, objective="mse"),
        },
    )


def select(rows: list[dict], pattern: tuple[bool, ...]) -> list[dict]:
    return [{**row, **{f"hide_{name}": hidden for name, hidden in zip(SENSORS, pattern, strict=True)}} for row in rows]


def oracle(rows: list[dict], pattern: tuple[bool, ...]) -> tuple[np.ndarray, float]:
    precision = np.asarray([not hidden for hidden in pattern]) / NOISE**2
    readings = np.asarray([[row[name] for name in SENSORS] for row in rows])
    variance = float(1.0 / (1.0 + precision.sum()))
    return readings @ precision * variance, variance


def prediction(model: rf.Model, rows: list[dict]) -> np.ndarray:
    output = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    return np.asarray([row["/target"]["content"] for row in output], dtype=np.float64)


# %% [markdown]
# ## Training and predeclared measurements
#
# Train an xs model for 600 AdamW updates at learning rate 0.002, batch size
# 64, with independent 4,096/512 training/validation rows. Missingness begins
# with three independent Bernoulli(0.5) flags; fitting rejects only the pattern
# with first and third visible and second hidden. This conditioning makes the
# retained flags dependent on one another, but keeps them independent of every
# numerical value. No selector is an embedded field.
#
# Apply all eight fixed patterns to the same 2,048 independent test records.
# For each, record prediction MSE, empirical oracle MSE, and squared distance
# to the conditional mean. Predeclared gates require mean-squared distance to
# the oracle below 0.10 for seen patterns and below 0.15 for the unseen one.
# Joint evidence must improve MSE by at least 15% over the best single sensor.
# The no-evidence prediction must have absolute mean below 0.20 and standard
# deviation below 0.10. Permuting nuisance readings must change predictions by
# RMS less than 0.12. Hidden-value poisoning must have maximum effect <1e-5.
#
# Add six to the first visible sensor in a separate stress control. Its effect
# is diagnostic: this model was not trained to identify corrupted sensors.
# Hiding that same corrupted sensor must exactly recover its corresponding
# two-sensor prediction. All comparisons use the final fixed-budget model.


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    model.optimizer = rf.adamw(learning_rate=0.002)
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, rows=TRAIN_ROWS, seed=seed * 100 + 1, training=True),
        validate=partial(records, rows=512, seed=seed * 100 + 2, training=True),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=BUDGET if steps is None else min(steps, BUDGET),
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=data)
    model.eval()
    test = list(records(rows=TEST_ROWS, seed=seed * 100 + 3))
    actual = np.asarray([row["target"] for row in test])
    train_mean = float(np.mean([row["target"] for row in records(rows=TRAIN_ROWS, seed=seed * 100 + 1, training=True)]))
    metrics = {
        "steps": trainer.global_step,
        "train_rows": TRAIN_ROWS,
        "validation_rows": 512,
        "test_rows": TEST_ROWS,
        "constant_baseline_mse": float(np.mean((actual - train_mean) ** 2)),
        "patterns": {},
    }
    checks = {}
    predicted_by_pattern = {}
    for pattern in PATTERNS:
        key = "".join("0" if hidden else "1" for hidden in pattern)
        selected = select(test, pattern)
        predicted = prediction(model, selected)
        predicted_by_pattern[pattern] = predicted
        conditional_mean, variance = oracle(selected, pattern)
        poisoned = [
            {
                **row,
                **{name: 1000.0 for name, hidden in zip(SENSORS, pattern, strict=True) if hidden},
                "target": -1000.0,
            }
            for row in selected
        ]
        measured = {
            "unseen_pattern": pattern == UNSEEN,
            "mse": float(np.mean((predicted - actual) ** 2)),
            "oracle_mse": float(np.mean((conditional_mean - actual) ** 2)),
            "oracle_expected_mse": variance,
            "distance_to_oracle": float(np.mean((predicted - conditional_mean) ** 2)),
            "hidden_value_max_drift": float(np.max(np.abs(predicted - prediction(model, poisoned)))),
        }
        metrics["patterns"][key] = measured
        checks[f"{key}: finite predictions"] = bool(np.isfinite(predicted).all())
        checks[f"{key}: tracks conditional mean"] = measured["distance_to_oracle"] < (
            0.15 if pattern == UNSEEN else 0.10
        )
        checks[f"{key}: hidden sensor and target values cannot affect predictions"] = (
            measured["hidden_value_max_drift"] < 1e-5
        )
    single_mse = [metrics["patterns"][key]["mse"] for key in ("100", "010", "001")]
    joint_mse = metrics["patterns"]["111"]["mse"]
    metrics["joint_to_best_single_mse"] = joint_mse / min(single_mse)
    checks["Joint evidence improves MSE by at least 15 percent over every single sensor"] = (
        metrics["joint_to_best_single_mse"] < 0.85
    )
    no_evidence = predicted_by_pattern[(True, True, True)]
    metrics["no_evidence_prediction_mean"] = float(no_evidence.mean())
    metrics["no_evidence_prediction_std"] = float(no_evidence.std())
    checks["Absent evidence returns the zero prior mean"] = (
        abs(float(no_evidence.mean())) < 0.20 and float(no_evidence.std()) < 0.10
    )
    complete = select(test, (False, False, False))
    order = np.random.default_rng(seed * 100 + 4).permutation(TEST_ROWS)
    nuisance = [{**row, "nuisance": test[index]["nuisance"]} for row, index in zip(complete, order, strict=True)]
    metrics["nuisance_permutation_rms_drift"] = float(
        np.sqrt(np.mean((prediction(model, nuisance) - predicted_by_pattern[(False, False, False)]) ** 2))
    )
    checks["Unrelated sensor has small prediction influence"] = metrics["nuisance_permutation_rms_drift"] < 0.12
    corrupted = [{**row, "first": row["first"] + 6.0} for row in complete]
    metrics["corrupted_first_sensor_mse"] = float(np.mean((prediction(model, corrupted) - actual) ** 2))
    hidden_corrupted = select(corrupted, (True, False, False))
    metrics["hidden_corruption_max_drift"] = float(
        np.max(np.abs(prediction(model, hidden_corrupted) - predicted_by_pattern[(True, False, False)]))
    )
    checks["Skipping the corrupted sensor restores the paired two-sensor prediction"] = (
        metrics["hidden_corruption_max_drift"] < 1e-5
    )
    return metrics, checks


# %% [markdown]
# ## Evidence and limits
#
# {{< proof P074 evidence >}}
#
# The known Gaussian prior and stationary independent errors make this a
# controlled denoising task. Missingness depends on neither the target nor
# sensor values. Success does not establish handling of informative missingness,
# reliability shifts, correlated errors, uncertainty calibration, or automatic
# corruption detection. Failure on the unseen pattern is retained as evidence
# of the learned missingness boundary. Gates precede the first full runs.
# The three-seed panel retains its failures without adjusting thresholds or
# budgets. Gates remain provisional pending ten seeds.
#
# ## Reproduce
#
# {{< proof P074 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=7401)
