# %% [markdown]
# ---
# title: Percentiles, Magnitudes, and Noisy Targets
# categories: [Quantile]
# proof-id: P077
# description: Compare Number and Quantile under rank and magnitude tasks, contaminated inputs, and skewed conditional target distributions.
# execute: {enabled: false, eval: false}
# code-fold: true
# ---
#
# {{< proof P077 status >}}
#
# ## Insights
#
# Bounded percentile coordinates may help when rare corrupt measurements distort
# numerical scale, but the benefit must be measured separately from tail errors.
# This matched experiment compares input representations on a threshold task
# and a source-unit regression task, under clean and contaminated training.
# It does not require either representation to win every comparison.
#
# A second experiment separates output semantics: percentile MSE targets the
# conditional mean percentile, whose inverse generally differs from the
# conditional source-unit mean. On continuous distributions, percentile MAE
# and source-unit MAE instead share the conditional median as their optimum.
#
# All three initial CPU seeds met the gates. Clean amount regression favored
# Number (nRMSE 0.038–0.048) over Quantile (0.066–0.089). Under contamination,
# Quantile had lower overall error in two seeds but higher tail error in two;
# neither representation was uniformly better. Both learned the rank threshold.
# The noisy-target MSE predictions separated by 23.0–40.2% of the source means
# on average, confirming a learned distinction between the two objectives.

# %%
"""P077: compare numerical representations and the quantities their objectives estimate."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P077"
BUDGET = 400
TRAIN_ROWS = 8192
TEST_ROWS = 4096
GROUPS = ("amber", "birch", "cove")
LOCATIONS = np.array([-0.5, 0.0, 0.5])
SIGMA = 0.8

# %% [markdown]
# ## Input representation experiment
#
# Draw `u ~ Uniform(0, 1)` and `x = exp(4u) - 1`. The hidden rank label is
# `upper` when u exceeds 0.5. The numeric target is `0.2*x + Normal(0, 0.1)`.
# Only x is visible, and each task gets its own model. In the contaminated arm,
# multiply 2% of training x values by 100 while retaining their original
# targets. This models input measurement errors, not a change in the true rule.
# Validation and testing stay clean, with paired observations across all arms.
#
# ```yaml
# x: 1.0
# rank: lower
# amount: 0.24
# ```
#
# ```yaml
# x: 100.0
# rank: lower
# amount: 0.24
# ```
#
# The second record illustrates corrupted training input. The label and amount
# still describe the original x=1 observation.
#
# ```yaml
# x: 50.0
# rank: upper
# amount: 10.03
# ```
#
# This is a valid high-value observation. Test rows with u>0.95 receive separate
# tail metrics, so an average improvement cannot conceal harm in that region.
#
# ```{typst}
# //| label: fig-proof-numeric-tradeoffs
# //| fig-cap: "Each matched model observes the same x through Number or Quantile and predicts one hidden task."
# //| fig-alt: "A measurement contains visible numerical x and one hidden rank Enum or amount Number target."
# #tree(node("measurement", kind: "root", children: (
#   node("x", type: "Number / Quantile", detail: "Matched input choice"),
#   node("target", kind: "target", type: "Enum / Number", detail: "Separate task models"),
# )))
# ```


# %%
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
            batch_size=128,
            group=rf.Enum(values=GROUPS, p_unavailable=0.0),
            target=scalar(objective=objective, mask=True),
        )
    else:
        target = (
            rf.Enum(values=("lower", "upper"), p_unavailable=0.0, mask=True)
            if task == "rank"
            else rf.Number(objective="mse", mask=True)
        )
        model = rf.Model.xs(batch_size=128, x=scalar, fields={task: target})
    model.optimizer = rf.adamw(learning_rate=0.002)
    return model


def fit(model: rf.Model, train, validate, seed: int, steps: int | None, accelerator: str) -> int:
    data = rf.SyntheticDataModule(model=model, train=train, validate=validate, seed=seed)
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
    return trainer.global_step


def predict(model: rf.Model, rows: list[dict], task: str) -> np.ndarray:
    visible = [{"group": row["group"]} if task == "noisy" else {"x": row["x"]} for row in rows]
    predictions = model.predict(pa.Table.from_pylist(visible))["predictions"].to_pylist()
    address = "/target" if task == "noisy" else f"/{task}"
    content = [row[address]["content"] for row in predictions]
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


# %% [markdown]
# ## Noisy-target experiment and analytical baselines
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


# %%
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


# %% [markdown]
# ## Evidence and limits
#
# {{< proof P077 evidence >}}
#
# Three CPU seeds are recorded with the original budgets and gates. Thresholds
# remain provisional pending ten seeds. This measures one skewed input
# distribution, one measurement-error mechanism, and one conditional target
# family. It does not establish invariance of frozen models to unit changes,
# universal outlier robustness, distribution-shift adaptation, or uncertainty
# calibration. Compression and tail sample size also limit percentile accuracy.
#
# ## Reproduce
#
# {{< proof P077 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=7701)
