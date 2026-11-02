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
#   duration_seconds: 22.781
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
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: 14c5f95ca09f
#     packages:
#       lightning: 2.6.4
#       numpy: 2.4.6
#       polars: 2.0.0
#       pyarrow: 24.0.0
#       pydantic: 2.13.4
#       relflow: 0.4.2
#       tensordict: 0.12.4
#       torch: 2.12.0+cpu
#       torchmetrics: 1.9.0
#     platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.41
#     python: 3.12.16
#     relflow_path: /tmp/runtime/lib/python3.12/site-packages/relflow
#     relflow_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     relflow_source_sha256: 77953506049ed76c92adc5f32a9bcc32432a788068a4e366b2f78bcf709f29a2
#     script_sha256: 159ff6edc159167fd12b263c648a9ca5a21257cc52fe7275e61985a080c65816
#     source_sha256: de84a71d945bb6293a4e6d2a550654fbbfa4b5861322dfa23b60cbd42f86ee45
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 7401
#   started_at: '2026-11-02T17:54:59.861544+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41494093900-1
#   candidate:
#     payload_sha256: 2169c968a7733085889a6eed6e895bcb2590059bcc46243d49a61ffacc42f67f
#     revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   baseline:
#     payload_sha256: 2a6af4bd3869faac23f6b7327f2c59e8f5b367580b2f2fa609710958775c08ed
#     revision: 7dec5e860dc389c37c3642d8904867249e831773
#     wheel_sha256: d3da50b65e0c0c7b98059f095b82bf6bf5d0ea0c5ccd0f302015118075f1dd85
#   evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#   executor_revision: 92b3c887eabc598ee2fc6c9dba4e05b9166cb8ba
#   manifest_sha256: 4f0926f3fc366fd8d14bf728b2d06f4a475d61852679176704816c110ae51cfd
#   passed: true
#   github_run:
#     event: push
#     head_branch: main
#     head_sha: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     html_url: https://github.com/relflow/relflow/actions/runs/41494093900
#     path: .github/workflows/ci.yml
#     run_attempt: 1
#     run_id: 41494093900
#     workflow_id: 403231172
#   release: https://github.com/relflow/relflow/releases/tag/evidence-ci-41494093900-attempt-1
#   reports_sha256: b1bfee5622ffbd389b1777443dc05e6d8f08dccff9e2540a601f0c2608948594
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
