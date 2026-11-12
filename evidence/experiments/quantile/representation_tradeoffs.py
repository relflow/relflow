# %% [markdown]
# ---
# title: Should a number be represented by size or percentile?
# categories:
# - Quantile
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
# ---
#
# A number tells us both an amount and a position in a distribution. For example,
# 100 is ten times 10 in original units, but their percentile positions depend on
# what values were seen in training. This experiment separates questions about
# ordering from questions about amount, then asks how rare bad measurements affect
# each representation.
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
# The illustrated x = 1 is in the lower half of this generator's population, so
# its rank answer is lower. Its amount answer is roughly 0.2 times x, plus small
# noise; 0.24 is an illustrative noisy observation. Separate models predict the
# rank label and the numeric amount. Neither answer is visible as an input.
#
# For a concrete corruption, imagine recording that same x as 100 while keeping
# its original answer. This creates a misleading input measurement. The test does
# this to 2% of training inputs; its validation and test observations stay clean.
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
# The input comparison gives Number and Quantile the same underlying observations.
# It measures both ordinary rows and the largest 5% separately, since a good
# average score can conceal large mistakes at the extremes. Shuffling the test
# inputs while retaining their answers removes the useful input-answer relationship.
#
# A second comparison changes the output representation for a noisy positive
# quantity. Squared error in original units puts a large penalty on large amount
# misses; squared error in percentile units measures mistakes in distribution
# position. Those objectives can prefer different answers even with unlimited
# training. Each model is therefore compared with the optimum for its own
# objective, as well as scored in common original units.
#
#
# Measure overall and tail errors separately. Also compare noisy output targets learned in original units versus percentile units.
#
# ## Result
#
# {{< evidence P077 status >}}
#
# Read rank accuracy and amount error as different questions. Contamination and
# tail comparisons are diagnostics; the checks do not require Quantile to win.
# The noisy-output measurements also separate learning quality from the quantity
# the objective asks the model to estimate. These generated distributions support
# a controlled comparison, rather than a universal datatype recommendation.
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
from relflow.presets.base import resolve
from reporting import experiment, report

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
                group=rf.Enum(values=GROUPS, p_unavailable=0.0),
                target=scalar(objective=objective, mask=True),
            )
        )
        model.preset = control_preset
    else:
        target = (
            rf.Enum(values=("lower", "upper"), p_unavailable=0.0, mask=True)
            if task == "rank"
            else rf.Number(objective="mse", mask=True)
        )
        # Freeze the experimental architecture independently of SDK preset changes.
        control_preset = rf.presets.Preset(
            name="xs",
            d_model=64,
            root=rf.presets.BranchDefaults(n_layers=1, n_heads=2, reduction=rf.Attention()),
            branch=rf.presets.BranchDefaults(n_layers=1, n_heads=2, reduction=rf.Attention()),
            leaf=rf.presets.LeafDefaults(n_heads=2),
        )
        model = rf.Model(schema=resolve(control_preset, x=scalar, fields={task: target}))
        model.preset = control_preset
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


@experiment("P077", seed=7701)
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
    report(run)

# %% [markdown]
# </details>
