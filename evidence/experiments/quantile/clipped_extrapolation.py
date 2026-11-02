# %% [markdown]
# ---
# title: What happens beyond the learned percentile range?
# categories:
# - Quantile
# evidence-id: P068
# description: Train a Quantile relationship within a bounded amount range, then supply much smaller and larger amounts.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: In-range error, original units
#   metric:
#   - in_range_rmse
#   format: number
# - label: Below-range error, original units
#   metric:
#   - below
#   - rmse
#   format: number
# - label: Above-range error, original units
#   metric:
#   - above
#   - rmse
#   format: number
# evidence-group: Representation
# evidence-result:
#   accelerator: cpu
#   checks:
#     Both digests count exactly the consumed training rows: true
#     Evaluation preserves both training distributions: true
#     In-range predictions are finite: true
#     In-range source-unit normalized RMSE below 0.20: true
#     Shuffled input normalized RMSE above 0.90: true
#     'above: input percentiles collapse to the same tail': true
#     'above: mean prediction stays within 15 percent of the corresponding edge': true
#     'above: observed error meets the unavoidable clipping lower bound': true
#     'above: out-of-range finite inputs produce finite predictions': true
#     'above: predictions saturate within 0.00001 source units': true
#     'above: predictions stay inside learned target support': true
#     'below: input percentiles collapse to the same tail': true
#     'below: mean prediction stays within 15 percent of the corresponding edge': true
#     'below: observed error meets the unavoidable clipping lower bound': true
#     'below: out-of-range finite inputs produce finite predictions': true
#     'below: predictions saturate within 0.00001 source units': true
#     'below: predictions stay inside learned target support': true
#   duration_seconds: 11.203
#   metrics:
#     above:
#       baseline_rmse: 121.32069478122891
#       endpoint_distance_fraction: 0.019401782665147953
#       normalized_rmse: 0.6884577848599096
#       prediction_mean: 218.97874852032194
#       prediction_spread: 0.0
#       rmse: 83.52417678675005
#       rmse_above_clipping_bound: 1.5160950857091677
#       shift: 30.0
#       support_clipping_rmse_lower_bound: 82.00808170104088
#       target_maximum: 340.608566329766
#       target_minimum: 259.8088142780439
#     below:
#       baseline_rmse: 123.07858363150321
#       endpoint_distance_fraction: 0.021408523991806056
#       normalized_rmse: 0.6896167324316166
#       prediction_mean: 140.92914752461934
#       prediction_spread: 0.0
#       rmse: 84.8770506762687
#       rmse_above_clipping_bound: 1.675025041465645
#       shift: -30.0
#       support_clipping_rmse_lower_bound: 83.20202563480305
#       target_maximum: 100.60856632976598
#       target_minimum: 19.808814278043897
#     in_range_baseline_rmse: 23.09814636434094
#     in_range_normalized_rmse: 0.05301209004503605
#     in_range_rmse: 1.224481014939864
#     noise_rmse: 0.5773502691896258
#     normalization:
#       /amount:
#         compression: 100
#         count: 51200
#         maximum: 29.99448916977137
#         median: 20.10375657728904
#         minimum: 10.017885500429006
#       /cost:
#         compression: 100
#         count: 51200
#         maximum: 220.55747836503323
#         median: 180.31984116606625
#         minimum: 139.1871284359407
#     shuffled_normalized_rmse: 1.4246887064731133
#     shuffled_rmse: 32.90766826573954
#     steps: 400
#     test_rows_per_condition: 2048
#     train_rows: 4096
#     training_mean: 180.14562292071815
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 830c0b9f2e9c5223003f8dd63cc59fb36fd457bf7caeb1e845528874d909a55f
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 8ec5781a607d
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
#     script_sha256: aedeb6319d7e8e7c2d5a81804e8d126f1b75ad13d84cc4af0b9fef8eb20b4ae7
#     source_sha256: 7e91b3327b4d455959ef8d2d6772e17462a7bc1bb16ada2eda84c7fb2ac68c00
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 6801
#   started_at: '2026-11-01T01:48:41.795058+00:00'
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
# amount: 20.0
# cost: 180.5
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-quantile-clipped-extrapolation
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with order, amount, cost. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("order", kind: "root", children: (
#   node("amount", type: "Quantile", detail: "In-range or shifted input"),
#   node("cost", kind: "target", type: "Quantile", detail: "Bounded learned support"),
# )))
# ```
#
# ## Comparison
#
# Compare in-range error with each outer range. Check whether distinct amounts in a tail receive the same prediction.
#
# ## Result
#
# {{< evidence P068 status >}}
#
# Outside the learned support, percentile inputs and outputs hit their boundaries. Tail predictions become constant; accepting a number does not guarantee extrapolation.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P068 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P068 script >}}
#
# ### Synthetic process and paired observations
#
# Each independent row has `amount ~ Uniform(10, 30)` and
# `cost = 100 + 4 * amount + Uniform(-1, 1)`. Only amount is visible; the
# conditional mean is identifiable, with noise RMSE `1 / sqrt(3)`. There are
# no identities or shared row-level latent variables across splits.
#
# ### Training, metric, and boundary gates
#
# Use 4,096 training rows, 1,024 validation rows, and 2,048 independently
# generated test rows per evaluation condition. The in-range and shifted
# tests share latent draws only with each other. Train `xs` for 400 AdamW
# updates at learning rate 0.002 and batch size 128. Use the fixed final
# update, with percentile MSE and default digest compression.
#
# Predeclare in-range normalized RMSE below 0.20 against the training-mean
# baseline, and shuffled-input normalized RMSE above 0.90. For each shifted
# test, compute the best possible bounded output separately for every target:
# `clip(cost, learned_minimum, learned_maximum)`. Its RMSE is an exact lower
# bound for every predictor confined to that support, even one with an oracle
# target. The learned predictor must stay within support and incur at least
# that lower-bound error. Same-tail inputs must produce the same percentile
# and predictions within 0.00001 source units, with mean predictions within
# 15% of the support width of the corresponding endpoint. This gate records
# expected saturation as successful evidence of a limitation.
#
# ### Remaining work
#
# The ten-seed calibration retained the predeclared gates and fixed training
# budget. This evidence establishes the boundary of this Quantile schema, not the
# extrapolation ability of Number or models with additional informative fields.
# Accepting an out-of-support value gives no OOD warning or calibrated uncertainty.
#
# ### Complete experiment code
#

# %%
"""P068: distinguish learned in-range prediction from bounded Quantile extrapolation."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P068"
BUDGET = 400
BATCH_SIZE = 128
TRAIN_ROWS = 4096
TEST_ROWS = 2048


def records(*, rows: int, seed: int, shift: float = 0.0) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    amount = rng.uniform(10.0, 30.0, rows) + shift
    cost = 100.0 + 4.0 * amount + rng.uniform(-1.0, 1.0, rows)
    for value, target in zip(amount, cost, strict=True):
        yield {"amount": float(value), "cost": float(target)}


def build() -> rf.Model:
    model = rf.Model.xs(
        amount=rf.Quantile,
        cost=rf.Quantile(objective="mse", mask=True),
    )
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = pa.Table.from_pylist([{"amount": row["amount"]} for row in rows])
    predictions = model.predict(visible)["predictions"].to_pylist()
    return np.asarray([row["cost"]["content"] for row in predictions], dtype=np.float64)


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    data = rf.SyntheticDataModule(
        train=partial(records, rows=TRAIN_ROWS, seed=seed + 1),
        validate=partial(records, rows=1024, seed=seed + 2),
        seed=seed,
    )
    model.fit(
        data,
        batch_size=BATCH_SIZE,
        learning_rate=0.002,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=BUDGET if steps is None else min(steps, BUDGET),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer
    model.eval()
    addresses = (rf.Address("amount"), rf.Address("cost"))
    normalization = {str(address): rf.Quantile.normalization(model, address) for address in addresses}
    lower = float(normalization["/cost"]["minimum"])
    upper = float(normalization["/cost"]["maximum"])
    width = upper - lower
    rows = list(records(rows=TEST_ROWS, seed=seed + 3))
    target = np.asarray([row["cost"] for row in rows])
    training_mean = float(np.mean([row["cost"] for row in records(rows=TRAIN_ROWS, seed=seed + 1)]))
    baseline_rmse = float(np.sqrt(np.mean((training_mean - target) ** 2)))
    output = predict(model, rows)
    rmse = float(np.sqrt(np.mean((output - target) ** 2)))
    permutation = np.random.default_rng(seed + 4).permutation(TEST_ROWS)
    shuffled = predict(model, [rows[index] for index in permutation])
    shuffled_rmse = float(np.sqrt(np.mean((shuffled - target) ** 2)))
    metrics = {
        "steps": trainer.global_step,
        "train_rows": TRAIN_ROWS,
        "test_rows_per_condition": TEST_ROWS,
        "training_mean": training_mean,
        "noise_rmse": float(1 / np.sqrt(3)),
        "in_range_rmse": rmse,
        "in_range_baseline_rmse": baseline_rmse,
        "in_range_normalized_rmse": rmse / baseline_rmse,
        "shuffled_rmse": shuffled_rmse,
        "shuffled_normalized_rmse": shuffled_rmse / baseline_rmse,
        "normalization": normalization,
    }
    checks = {
        "In-range source-unit normalized RMSE below 0.20": rmse / baseline_rmse < 0.20,
        "Shuffled input normalized RMSE above 0.90": shuffled_rmse / baseline_rmse > 0.90,
        "In-range predictions are finite": bool(np.isfinite(output).all()),
        "Both digests count exactly the consumed training rows": all(
            state["count"] == trainer.global_step * BATCH_SIZE for state in normalization.values()
        ),
    }
    for name, shift, percentile, endpoint in (("below", -30.0, 0.0, lower), ("above", 30.0, 1.0, upper)):
        shifted = list(records(rows=TEST_ROWS, seed=seed + 3, shift=shift))
        targets = np.asarray([row["cost"] for row in shifted])
        predictions = predict(model, shifted)
        baseline = float(np.sqrt(np.mean((training_mean - targets) ** 2)))
        observed_rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
        bounded_oracle = np.clip(targets, lower, upper)
        lower_bound = float(np.sqrt(np.mean((bounded_oracle - targets) ** 2)))
        spread = float(np.ptp(predictions))
        distance = float(abs(predictions.mean() - endpoint) / width)
        fields = model.encode(pa.Table.from_pylist(shifted), strata="test")
        metrics[name] = {
            "shift": shift,
            "target_minimum": float(targets.min()),
            "target_maximum": float(targets.max()),
            "prediction_mean": float(predictions.mean()),
            "prediction_spread": spread,
            "rmse": observed_rmse,
            "baseline_rmse": baseline,
            "normalized_rmse": observed_rmse / baseline,
            "support_clipping_rmse_lower_bound": lower_bound,
            "rmse_above_clipping_bound": observed_rmse - lower_bound,
            "endpoint_distance_fraction": distance,
        }
        checks[f"{name}: out-of-range finite inputs produce finite predictions"] = bool(np.isfinite(predictions).all())
        checks[f"{name}: predictions stay inside learned target support"] = bool(
            np.all(predictions >= lower) and np.all(predictions <= upper)
        )
        checks[f"{name}: observed error meets the unavoidable clipping lower bound"] = (
            lower_bound > 50.0 and observed_rmse >= lower_bound - 1e-8
        )
        checks[f"{name}: input percentiles collapse to the same tail"] = bool(
            fields["/amount"].content.eq(percentile).all()
        )
        checks[f"{name}: predictions saturate within 0.00001 source units"] = spread < 1e-5
        checks[f"{name}: mean prediction stays within 15 percent of the corresponding edge"] = distance < 0.15
    checks["Evaluation preserves both training distributions"] = normalization == {
        str(address): rf.Quantile.normalization(model, address) for address in addresses
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=6801)

# %% [markdown]
# </details>
