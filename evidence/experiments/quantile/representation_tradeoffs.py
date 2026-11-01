# %% [markdown]
# ---
# title: Should a number be represented by size or percentile?
# categories:
# - Quantile
# evidence-id: P077
# description: Compare Number and Quantile on rank and amount tasks, with clean training and rare corrupt measurements.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Corrupt training, Number
#   metric:
#   - input_tasks
#   - amount_number_contaminated
#   - normalized_rmse
#   format: error
# - label: Corrupt training, Quantile
#   metric:
#   - input_tasks
#   - amount_quantile_contaminated
#   - normalized_rmse
#   format: error
# evidence-group: Representation
# evidence-result:
#   accelerator: cpu
#   checks:
#     Learned MSE predictions differ by at least 10 percent of source means: true
#     Percentile and source-unit MSE population optima differ materially: true
#     'amount_number_clean: finite measurements': true
#     'amount_number_clean: normalized RMSE at most 0.35': true
#     'amount_number_clean: shuffled normalized RMSE at least 0.90': true
#     'amount_number_contaminated: finite measurements': true
#     'amount_quantile_clean: finite measurements': true
#     'amount_quantile_clean: normalized RMSE at most 0.35': true
#     'amount_quantile_clean: shuffled normalized RMSE at least 0.90': true
#     'amount_quantile_contaminated: finite measurements': true
#     'number_mae: predictions are finite': true
#     'number_mae: relative optimum error at most 0.25': true
#     'number_mse: predictions are finite': true
#     'number_mse: relative optimum error at most 0.25': true
#     'quantile_mae: predictions are finite': true
#     'quantile_mae: relative optimum error at most 0.25': true
#     'quantile_mse: predictions are finite': true
#     'quantile_mse: relative optimum error at most 0.25': true
#     'rank_number_clean: accuracy reaches 0.90': true
#     'rank_number_clean: finite measurements': true
#     'rank_number_clean: shuffled accuracy at most 0.60': true
#     'rank_number_contaminated: finite measurements': true
#     'rank_quantile_clean: accuracy reaches 0.90': true
#     'rank_quantile_clean: finite measurements': true
#     'rank_quantile_clean: shuffled accuracy at most 0.60': true
#     'rank_quantile_contaminated: finite measurements': true
#   duration_seconds: 127.976
#   metrics:
#     central_rows: 3901
#     input_tasks:
#       amount_number_clean:
#         baseline_rmse: 2.7796081299264612
#         central_rmse: 0.1721292415421181
#         normalized_rmse: 0.06602526107350982
#         rmse: 0.18352435246044502
#         shuffled_normalized_rmse: 1.382963800961026
#         steps: 400
#         tail_rmse: 0.3387568658027227
#       amount_number_contaminated:
#         baseline_rmse: 2.7796081299264612
#         central_rmse: 0.136498142576108
#         normalized_rmse: 0.050559243770540496
#         rmse: 0.14053488502752814
#         shuffled_normalized_rmse: 1.4042224179493201
#         steps: 400
#         tail_rmse: 0.20523646970574982
#       amount_quantile_clean:
#         baseline_rmse: 2.7796081299264612
#         central_rmse: 0.27930527422777673
#         normalized_rmse: 0.09880637787914434
#         rmse: 0.2746430112414557
#         shuffled_normalized_rmse: 1.4119279036557555
#         steps: 400
#         tail_rmse: 0.15415085954332416
#       amount_quantile_contaminated:
#         baseline_rmse: 2.7796081299264612
#         central_rmse: 0.1618314563943457
#         normalized_rmse: 0.05894220751730564
#         rmse: 0.16383623921091534
#         shuffled_normalized_rmse: 1.3787370839096826
#         steps: 400
#         tail_rmse: 0.19975865800818857
#       rank_number_clean:
#         accuracy: 0.9990234375
#         central_accuracy: 0.9989746218918226
#         majority_accuracy: 0.5
#         shuffled_accuracy: 0.52001953125
#         steps: 400
#         tail_accuracy: 1.0
#       rank_number_contaminated:
#         accuracy: 0.996826171875
#         central_accuracy: 0.9966675211484235
#         majority_accuracy: 0.5
#         shuffled_accuracy: 0.520751953125
#         steps: 400
#         tail_accuracy: 1.0
#       rank_quantile_clean:
#         accuracy: 0.995361328125
#         central_accuracy: 0.9951294539861574
#         majority_accuracy: 0.5
#         shuffled_accuracy: 0.520263671875
#         steps: 400
#         tail_accuracy: 1.0
#       rank_quantile_contaminated:
#         accuracy: 0.9990234375
#         central_accuracy: 0.9989746218918226
#         majority_accuracy: 0.5
#         shuffled_accuracy: 0.51953125
#         steps: 400
#         tail_accuracy: 1.0
#     learned_mse_separation_fraction: 0.2669468619846949
#     noisy_targets:
#       number_mae:
#         group_predictions:
#         - 0.5570969384426183
#         - 0.9694045143588924
#         - 1.6236446103080078
#         mean_relative_optimum_error: 0.04243589225328844
#         optimum_source_mae: 0.8753664718008379
#         optimum_source_rmse: 1.6526635866958435
#         percentile_mae: 0.21478285740935898
#         percentile_mse: 0.06870025711300654
#         source_mae: 0.8770206507251517
#         source_rmse: 1.6614524752627977
#         steps: 400
#       number_mse:
#         group_predictions:
#         - 0.8366915335441786
#         - 1.3927435718132786
#         - 2.4202991944819447
#         mean_relative_optimum_error: 0.026339137908036755
#         optimum_source_mae: 0.9374528597987286
#         optimum_source_rmse: 1.5815915450641407
#         percentile_mae: 0.23469993561657493
#         percentile_mse: 0.08302519937491024
#         source_mae: 0.9550441021562887
#         source_rmse: 1.5814917957075545
#         steps: 400
#       quantile_mae:
#         group_predictions:
#         - 0.6084054160837571
#         - 0.984106792663203
#         - 1.5461976758268106
#         mean_relative_optimum_error: 0.027055953892624423
#         optimum_source_mae: 0.8753664718008379
#         optimum_source_rmse: 1.6526635866958435
#         percentile_mae: 0.21453783190908954
#         percentile_mse: 0.06680493744855936
#         source_mae: 0.8778764584692968
#         source_rmse: 1.6692542452317325
#         steps: 400
#       quantile_mse:
#         group_predictions:
#         - 0.7295130318312212
#         - 1.0331054901216066
#         - 1.486275102830487
#         mean_relative_optimum_error: 0.03869281960393878
#         optimum_source_mae: 0.8834192118203341
#         optimum_source_rmse: 1.6798647788753727
#         percentile_mae: 0.21717363826390337
#         percentile_mse: 0.06582571929664477
#         source_mae: 0.8838590016601799
#         source_rmse: 1.6709552623065143
#         steps: 400
#     population_optima:
#       number_mae:
#       - 0.6065306597126334
#       - 1.0
#       - 1.6487212707001282
#       number_mse:
#       - 0.8352702114112721
#       - 1.3771277643359572
#       - 2.2704998375324057
#       quantile_mae:
#       - 0.6065306597126334
#       - 1.0
#       - 1.6487212707001282
#       quantile_mse:
#       - 0.6866702972090775
#       - 0.9999999999999999
#       - 1.4563029798499054
#     tail_rows: 195
#     test_rows: 4096
#     train_rows: 8192
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: a0c88a124d8bf6e11c0e4d10311b88141a3fbe971299b6284dcd38babb5cea8f
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: d900a8c6d73d
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
#     script_sha256: aa134c4ef643be282f22621503699a3cdf194e8c698ca3559c2a8944b74acaa8
#     source_sha256: 5091065d78a4884f3a9c81dbda8cc24e4e29aa7926d3d2833430265df92759fe
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 7701
#   started_at: '2026-11-01T01:50:15.706109+00:00'
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
# x: 1.0
# rank: lower
# amount: 0.24
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-numeric-tradeoffs
# //| fig-cap: "Amber cards are hidden prediction targets. Separate models predict rank or amount."
# //| fig-alt: "Model tree with measurement, x, target. Amber cards are hidden prediction targets. Separate models predict rank or amount."
# #tree(node("measurement", kind: "root", children: (
#   node("x", type: "Number / Quantile", detail: "Matched input choice"),
#   node("target", kind: "target", type: "Enum / Number", detail: "Separate task models"),
# )))
# ```
#
# ## Comparison
#
# Measure overall and tail errors separately. Also compare noisy output targets learned in original units versus percentile units.
#
# ## Result
#
# {{< evidence P077 status >}}
#
# Neither representation wins every comparison. Percentiles can reduce some contamination effects while worsening tail error; changing output units can also change which answer the loss prefers.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P077 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P077 script >}}
#
# ### Input representation experiment
#
# Draw `u ~ Uniform(0, 1)` and `x = exp(4u) - 1`. The hidden rank label is
# `upper` when u exceeds 0.5. The numeric target is `0.2*x + Normal(0, 0.1)`.
# Only x is visible, and each task gets its own model. In the contaminated arm,
# multiply 2% of training x values by 100 while retaining their original
# targets. This models input measurement errors, not a change in the true rule.
# Validation and testing stay clean, with paired observations across all arms.
#
# Test rows with u > 0.95 receive separate tail metrics.
#
# ### Noisy-target experiment and analytical baselines
#
# Three equally likely Enum groups generate `target = exp(mu[group] + 0.8*z)`,
# with z standard normal and mu equal to -0.5, 0, or 0.5. Neither z nor mu is a
# model input. The source-unit mean is `exp(mu + 0.8**2/2)` and the median is
# `exp(mu)`. The population CDF F is the equal mixture of these three lognormal
# distributions. For percentile MSE, the optimum is `F^-1(E[F(target)|group])`.
# The expectation has an analytical normal-CDF expression; bisection computes
# its inverse. These evaluator-only oracles never fit a test distribution.
#
# Four matched models compare Number/Quantile targets under MSE/MAE. Report
# source-unit MAE/RMSE, percentile MAE/MSE, and distance to the corresponding
# population optimum. Quantile learns an approximate empirical training CDF,
# so the population optimum is a reference rather than an exact sketch optimum.
#
# Every arm has 8,192 training rows, 1,024 validation rows, 400 AdamW updates
# at learning rate 0.002, and batch size 128. Evaluation uses 4,096 fresh rows.
# Architecture width and optimizer budget match; datatype parameter counts may
# differ. Training, validation, and testing use seed*100 plus distinct offsets.
#
# Gates are fixed before execution: clean rank accuracy >=0.90; clean amount
# nRMSE <=0.35; input shuffling leaves rank accuracy <=0.60 and amount nRMSE
# >=0.90. Each noisy-target model must approach its own optimum with mean
# relative error <=0.25. Contamination comparisons and tail differences are
# diagnostics, not gates requiring Quantile superiority.
# The learned Number-MSE predictions must exceed Quantile-MSE by an average
# of at least 10% of the conditional source means, establishing a behavioral
# distinction in addition to the analytical one.
#
# ### Complete experiment code
#

# %%
"""P077: compare numerical representations and the quantities their objectives estimate."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P077"
BUDGET = 400
TRAIN_ROWS = 8192
TEST_ROWS = 4096
GROUPS = ("amber", "birch", "cove")
LOCATIONS = np.array([-0.5, 0.0, 0.5])
SIGMA = 0.8


def records(*, rows: int, seed: int, contaminated: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, rows)
    x = np.expm1(4 * u)
    amount = 0.2 * x + rng.normal(0, 0.1, rows)
    visible = x.copy()
    if contaminated:
        visible[rng.random(rows) < 0.02] *= 100
    for index in range(rows):
        yield {
            "x": float(visible[index]),
            "rank": "upper" if u[index] > 0.5 else "lower",
            "amount": float(amount[index]),
        }


def build(representation: str, task: str, objective: str = "mse") -> rf.Model:
    scalar = rf.Quantile if representation == "quantile" else rf.Number
    if task == "noisy":
        model = rf.Model.xs(
            group=rf.Enum(values=GROUPS, p_unavailable=0.0),
            target=scalar(objective=objective, mask=True),
        )
    else:
        target = (
            rf.Enum(values=("lower", "upper"), p_unavailable=0.0, mask=True)
            if task == "rank"
            else rf.Number(objective="mse", mask=True)
        )
        model = rf.Model.xs(x=scalar, fields={task: target})
    return model


def fit(model: rf.Model, train, validate, seed: int, steps: int | None, accelerator: str) -> int:
    data = rf.SyntheticDataModule(train=train, validate=validate, seed=seed)
    model.fit(
        data,
        batch_size=128,
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
    return trainer.global_step


def predict(model: rf.Model, rows: list[dict], task: str) -> np.ndarray:
    visible = [{"group": row["group"]} if task == "noisy" else {"x": row["x"]} for row in rows]
    predictions = model.predict(pa.Table.from_pylist(visible))["predictions"].to_pylist()
    name = "target" if task == "noisy" else task
    content = [row[name]["content"] for row in predictions]
    if task == "rank":
        return np.asarray([row["value"] == "upper" for row in content])
    return np.asarray(content, dtype=float)


def noisy_records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    group = np.resize(np.arange(len(GROUPS)), rows)
    rng.shuffle(group)
    target = np.exp(LOCATIONS[group] + SIGMA * rng.standard_normal(rows))
    for index in range(rows):
        yield {"group": GROUPS[group[index]], "target": float(target[index])}


def cdf(values: np.ndarray) -> np.ndarray:
    """Analytical CDF of the equally weighted three-lognormal population mixture."""
    scaled = (np.log(np.maximum(values, 1e-300))[..., None] - LOCATIONS) / SIGMA
    return torch.special.ndtr(torch.from_numpy(scaled)).mean(dim=-1).numpy()


def inverse(percentiles: np.ndarray) -> np.ndarray:
    lower = np.zeros_like(percentiles)
    upper = np.full_like(percentiles, np.exp(LOCATIONS.max() + 10 * SIGMA))
    for _ in range(64):
        midpoint = (lower + upper) / 2
        below = cdf(midpoint) < percentiles
        lower = np.where(below, midpoint, lower)
        upper = np.where(below, upper, midpoint)
    return (lower + upper) / 2


def optima() -> dict[str, np.ndarray]:
    differences = (LOCATIONS[:, None] - LOCATIONS[None, :]) / (np.sqrt(2) * SIGMA)
    mean_percentiles = torch.special.ndtr(torch.from_numpy(differences)).mean(dim=1).numpy()
    return {
        "number_mse": np.exp(LOCATIONS + SIGMA**2 / 2),
        "number_mae": np.exp(LOCATIONS),
        "quantile_mse": inverse(mean_percentiles),
        "quantile_mae": np.exp(LOCATIONS),
    }


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    stream = seed * 100
    test = list(records(rows=TEST_ROWS, seed=stream + 3))
    tail = np.asarray([row["x"] > np.expm1(4 * 0.95) for row in test])
    permutation = np.random.default_rng(stream + 4).permutation(TEST_ROWS)
    shuffled = [{**row, "x": test[index]["x"]} for row, index in zip(test, permutation, strict=True)]
    training = list(records(rows=TRAIN_ROWS, seed=stream + 1))
    mean = float(np.mean([row["amount"] for row in training]))
    target = np.asarray([row["amount"] for row in test])
    baseline = float(np.sqrt(np.mean((target - mean) ** 2)))
    labels = np.asarray([row["rank"] == "upper" for row in test])
    metrics = {
        "train_rows": TRAIN_ROWS,
        "test_rows": TEST_ROWS,
        "tail_rows": int(tail.sum()),
        "central_rows": int((~tail).sum()),
        "input_tasks": {},
        "noisy_targets": {},
    }
    checks = {}
    for task in ("rank", "amount"):
        for representation in ("number", "quantile"):
            for contaminated in (False, True):
                lit.seed_everything(seed, workers=True)
                model = build(representation, task)
                updates = fit(
                    model,
                    partial(records, rows=TRAIN_ROWS, seed=stream + 1, contaminated=contaminated),
                    partial(records, rows=1024, seed=stream + 2),
                    seed,
                    steps,
                    accelerator,
                )
                output = predict(model, test, task)
                control = predict(model, shuffled, task)
                key = f"{task}_{representation}_{'contaminated' if contaminated else 'clean'}"
                if task == "rank":
                    result = {
                        "steps": updates,
                        "accuracy": float(np.mean(output == labels)),
                        "central_accuracy": float(np.mean(output[~tail] == labels[~tail])),
                        "tail_accuracy": float(np.mean(output[tail] == labels[tail])),
                        "shuffled_accuracy": float(np.mean(control == labels)),
                        "majority_accuracy": float(max(labels.mean(), 1 - labels.mean())),
                    }
                    if not contaminated:
                        checks[f"{key}: accuracy reaches 0.90"] = result["accuracy"] >= 0.90
                        checks[f"{key}: shuffled accuracy at most 0.60"] = result["shuffled_accuracy"] <= 0.60
                else:
                    result = {
                        "steps": updates,
                        "rmse": float(np.sqrt(np.mean((output - target) ** 2))),
                        "baseline_rmse": baseline,
                        "normalized_rmse": float(np.sqrt(np.mean((output - target) ** 2)) / baseline),
                        "tail_rmse": float(np.sqrt(np.mean((output[tail] - target[tail]) ** 2))),
                        "central_rmse": float(np.sqrt(np.mean((output[~tail] - target[~tail]) ** 2))),
                        "shuffled_normalized_rmse": float(np.sqrt(np.mean((control - target) ** 2)) / baseline),
                    }
                    if not contaminated:
                        checks[f"{key}: normalized RMSE at most 0.35"] = result["normalized_rmse"] <= 0.35
                        checks[f"{key}: shuffled normalized RMSE at least 0.90"] = (
                            result["shuffled_normalized_rmse"] >= 0.90
                        )
                metrics["input_tasks"][key] = result
                checks[f"{key}: finite measurements"] = bool(np.isfinite(list(result.values())).all())
    noisy = list(noisy_records(rows=TEST_ROWS, seed=stream + 13))
    groups = np.asarray([GROUPS.index(row["group"]) for row in noisy])
    observed = np.asarray([row["target"] for row in noisy])
    targets = optima()
    metrics["population_optima"] = {key: values.tolist() for key, values in targets.items()}
    for representation in ("number", "quantile"):
        for objective in ("mse", "mae"):
            lit.seed_everything(seed, workers=True)
            model = build(representation, "noisy", objective)
            updates = fit(
                model,
                partial(noisy_records, rows=TRAIN_ROWS, seed=stream + 11),
                partial(noisy_records, rows=1024, seed=stream + 12),
                seed,
                steps,
                accelerator,
            )
            output = predict(model, noisy, "noisy")
            key = f"{representation}_{objective}"
            optimum = targets[key][groups]
            group_predictions = np.asarray([output[groups == group].mean() for group in range(len(GROUPS))])
            relative = float(np.mean(np.abs(group_predictions - targets[key]) / targets[key]))
            rank_error = cdf(output) - cdf(observed)
            result = {
                "steps": updates,
                "group_predictions": group_predictions.tolist(),
                "mean_relative_optimum_error": relative,
                "source_rmse": float(np.sqrt(np.mean((output - observed) ** 2))),
                "source_mae": float(np.mean(np.abs(output - observed))),
                "percentile_mse": float(np.mean(rank_error**2)),
                "percentile_mae": float(np.mean(np.abs(rank_error))),
                "optimum_source_rmse": float(np.sqrt(np.mean((optimum - observed) ** 2))),
                "optimum_source_mae": float(np.mean(np.abs(optimum - observed))),
            }
            metrics["noisy_targets"][key] = result
            checks[f"{key}: predictions are finite"] = bool(np.isfinite(output).all())
            checks[f"{key}: relative optimum error at most 0.25"] = relative <= 0.25
    checks["Percentile and source-unit MSE population optima differ materially"] = bool(
        np.mean(np.abs(targets["number_mse"] - targets["quantile_mse"]) / targets["number_mse"]) > 0.10
    )
    number_mse = np.asarray(metrics["noisy_targets"]["number_mse"]["group_predictions"])
    quantile_mse = np.asarray(metrics["noisy_targets"]["quantile_mse"]["group_predictions"])
    separation = float(np.mean((number_mse - quantile_mse) / targets["number_mse"]))
    metrics["learned_mse_separation_fraction"] = separation
    checks["Learned MSE predictions differ by at least 10 percent of source means"] = separation >= 0.10
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7701)

# %% [markdown]
# </details>
