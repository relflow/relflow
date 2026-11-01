# %% [markdown]
# ---
# title: Can it combine noisy measurements with missing readings?
# categories:
# - Representation
# evidence-id: P074
# description: Predict a signal from three noisy sensors, sometimes hiding one or more readings. An extra sensor carries
#   no useful information.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Error with all three sensors
#   metric:
#   - patterns
#   - '111'
#   - mse
#   format: number
# - label: Error with a corrupted sensor
#   metric:
#   - corrupted_first_sensor_mse
#   format: number
# evidence-group: Representation
# evidence-result:
#   accelerator: cpu
#   checks:
#     '000: finite predictions': true
#     '000: hidden sensor and target values cannot affect predictions': true
#     '000: tracks conditional mean': true
#     '001: finite predictions': true
#     '001: hidden sensor and target values cannot affect predictions': true
#     '001: tracks conditional mean': true
#     '010: finite predictions': true
#     '010: hidden sensor and target values cannot affect predictions': true
#     '010: tracks conditional mean': true
#     '011: finite predictions': true
#     '011: hidden sensor and target values cannot affect predictions': true
#     '011: tracks conditional mean': true
#     '100: finite predictions': true
#     '100: hidden sensor and target values cannot affect predictions': true
#     '100: tracks conditional mean': true
#     '101: finite predictions': true
#     '101: hidden sensor and target values cannot affect predictions': true
#     '101: tracks conditional mean': true
#     '110: finite predictions': true
#     '110: hidden sensor and target values cannot affect predictions': true
#     '110: tracks conditional mean': true
#     '111: finite predictions': true
#     '111: hidden sensor and target values cannot affect predictions': true
#     '111: tracks conditional mean': true
#     Absent evidence returns the zero prior mean: false
#     Joint evidence improves MSE by at least 15 percent over every single sensor: true
#     Skipping the corrupted sensor restores the paired two-sensor prediction: true
#     Unrelated sensor has small prediction influence: true
#   duration_seconds: 21.978
#   metrics:
#     constant_baseline_mse: 0.988053225656172
#     corrupted_first_sensor_mse: 2.89892233184262
#     hidden_corruption_max_drift: 0.0
#     joint_to_best_single_mse: 0.6171874901198882
#     no_evidence_prediction_mean: 0.15036732231480127
#     no_evidence_prediction_std: 0.16485731103403406
#     nuisance_permutation_rms_drift: 0.08661957403981585
#     patterns:
#       '000':
#         distance_to_oracle: 0.049788264621495576
#         hidden_value_max_drift: 0.0
#         mse: 1.0338120146130956
#         oracle_expected_mse: 1.0
#         oracle_mse: 0.988031778745079
#         unseen_pattern: false
#       '001':
#         distance_to_oracle: 0.0691476408013513
#         hidden_value_max_drift: 0.0
#         mse: 0.6577047043944804
#         oracle_expected_mse: 0.5901639344262295
#         oracle_mse: 0.590150262266959
#         unseen_pattern: false
#       '010':
#         distance_to_oracle: 0.041033186211170705
#         hidden_value_max_drift: 0.0
#         mse: 0.536985135256502
#         oracle_expected_mse: 0.5
#         oracle_mse: 0.48948520488805025
#         unseen_pattern: false
#       '011':
#         distance_to_oracle: 0.03549847549272624
#         hidden_value_max_drift: 0.0
#         mse: 0.4064668611496632
#         oracle_expected_mse: 0.37113402061855666
#         oracle_mse: 0.3678902899425125
#         unseen_pattern: false
#       '100':
#         distance_to_oracle: 0.017858166720857247
#         hidden_value_max_drift: 0.0
#         mse: 0.4246168434800212
#         oracle_expected_mse: 0.3902439024390244
#         oracle_mse: 0.4061850665424861
#         unseen_pattern: false
#       '101':
#         distance_to_oracle: 0.02178038823645926
#         hidden_value_max_drift: 0.0
#         mse: 0.3389138695391476
#         oracle_expected_mse: 0.3070362473347548
#         oracle_mse: 0.31400437689019883
#         unseen_pattern: true
#       '110':
#         distance_to_oracle: 0.014695487780777883
#         hidden_value_max_drift: 0.0
#         mse: 0.3124629919293869
#         oracle_expected_mse: 0.2807017543859649
#         oracle_mse: 0.29230042731279604
#         unseen_pattern: false
#       '111':
#         distance_to_oracle: 0.016865970937507333
#         hidden_value_max_drift: 0.0
#         mse: 0.2620682038900637
#         oracle_expected_mse: 0.23491027732463293
#         oracle_mse: 0.2416990730468406
#         unseen_pattern: false
#     steps: 600
#     test_rows: 2048
#     train_rows: 4096
#     validation_rows: 512
#   mode: full
#   outcome: not_met
#   provenance:
#     code_sha256: a801f5c0d5c3aaa2911cb96af024f05cbe6640c9609eec891991bb4b3eaf3acd
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 9d96eac92146
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
#     script_sha256: 1bd3654bb149602966e2f5eae0a05bb31c373d50b1d020800865fa41cfd98198
#     source_sha256: ae1e331a4494109e02d05760c8fd2a1564089498062ad3c6fa76a5fece8f3be8
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 7401
#   started_at: '2026-11-01T01:49:23.052447+00:00'
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
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-incomplete-evidence
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with record, first, second, third, nuisance, target. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("record", kind: "root", children: (
#   node("first", type: "Number", body: [Skip when `hide_first`]),
#   node("second", type: "Number", body: [Skip when `hide_second`]),
#   node("third", type: "Number", body: [Skip when `hide_third`]),
#   node("nuisance", type: "Number", body: [Independent noise]),
#   node("target", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Compare combined and single-sensor evidence, no evidence, an unseen visibility pattern, and a deliberately corrupted reading.
#
# ## Result
#
# {{< evidence P074 status >}}
#
# Combining readings helps in all three seeds, but only one passes every check. Missing-evidence fallback remains inconsistent, and a corrupted visible sensor can sharply increase error.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P074 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P074 script >}}
#
# ### Process and information boundary
#
# Independently draw `z ~ Normal(0,1)` and three sensors `x_i = z + e_i`, with
# independent Gaussian errors of standard deviations 0.8, 1.0, and 1.2. A
# fourth, nuisance reading is independent `Normal(0,1)`. The target is z and is
# always hidden. For any visible subset V, the exact conditional mean is
# `sum(x_i / sigma_i**2) / (1 + sum(1 / sigma_i**2))`; its conditional variance
# is `1 / (1 + sum(1 / sigma_i**2))`. With no sensors visible, the mean is zero.
#
# ### Training and predeclared measurements
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
#
# ### Complete experiment code
#

# %%
"""P074: Gaussian evidence fusion, missingness transfer, and a known oracle."""

from collections.abc import Iterator
from functools import partial
from itertools import product

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P074"
BUDGET = 600
TRAIN_ROWS = 4096
TEST_ROWS = 2048
SENSORS = ("first", "second", "third")
NOISE = np.asarray([0.8, 1.0, 1.2])
UNSEEN = (False, True, False)
PATTERNS = tuple(product((False, True), repeat=3))


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
    return np.asarray([row["target"]["content"] for row in output], dtype=np.float64)


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    data = rf.SyntheticDataModule(
        train=partial(records, rows=TRAIN_ROWS, seed=seed * 100 + 1, training=True),
        validate=partial(records, rows=512, seed=seed * 100 + 2, training=True),
        seed=seed,
    )
    model.fit(
        data,
        batch_size=64,
        learning_rate=0.002,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
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
    trainer = model.trainer
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


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7401)

# %% [markdown]
# </details>
