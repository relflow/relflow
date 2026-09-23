# %% [markdown]
# ---
# title: Quantile Predictions Saturate at Learned Support
# categories: [Quantile]
# proof-id: P068
# description: A Quantile model can learn an in-range relationship while accepting out-of-range numbers whose predictions remain bounded by the training distribution.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P068 status >}}
#
# ## Insights
#
# Accepting a finite numeric input does not establish numerical extrapolation.
# Quantile represents values below or above its training support by the same
# tail percentiles. A Quantile target also inverts only into its learned
# support. This proof pairs a learned in-range relationship with out-of-range
# observations and measures the unavoidable error of that output boundary.
#
# Across ten CPU seeds, in-range RMSE was 2.78–5.94% of the constant-baseline
# RMSE. Out-of-range RMSE rose to 83.59–85.78 source units, and predictions
# were constant within each tail. Every limitation gate passed on those seeds
# and in a separate RTX 3090 run; the proof remains marked Limited.

# %%
"""P068: distinguish learned in-range prediction from bounded Quantile extrapolation."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
from reporting import report

import relflow as rf

PROOF_ID = "P068"
BUDGET = 400
BATCH_SIZE = 128
TRAIN_ROWS = 4096
TEST_ROWS = 2048

# %% [markdown]
# ## Synthetic process and paired observations
#
# Each independent row has `amount ~ Uniform(10, 30)` and
# `cost = 100 + 4 * amount + Uniform(-1, 1)`. Only amount is visible; the
# conditional mean is identifiable, with noise RMSE `1 / sqrt(3)`. There are
# no identities or shared row-level latent variables across splits.
#
# ```yaml
# amount: 20.0
# cost: 180.5
# ```
#
# The cost is 180 plus one possible noise realization of 0.5. Evaluation also
# makes paired rows at `amount - 30` and `amount + 30`, retaining that noise:
#
# ```yaml
# amount: 50.0
# cost: 300.5
# ```
#
# Its correct target is outside the training target support, roughly 139 to
# 221. It remains a valid numeric input. All amounts above the observed
# training maximum nevertheless map to percentile 1; the model cannot
# distinguish how far above the maximum they lie.
#
# ```yaml
# amount: -10.0
# cost: 60.5
# ```
#
# The corresponding lower shift preserves the same noise but places the
# correct cost below the learned support. Both tails are evaluated separately.
#
# ```{typst}
# //| label: fig-proof-quantile-clipped-extrapolation
# //| fig-cap: "The same numeric schema accepts shifted inputs while its learned percentile support stays fixed."
# //| fig-alt: "An order contains visible Quantile amount and hidden Quantile cost, with evaluation values outside the training ranges."
# #tree(node("order", kind: "root", children: (
#   node("amount", type: "Quantile", detail: "In-range or shifted input"),
#   node("cost", kind: "target", type: "Quantile", detail: "Bounded learned support"),
# )))
# ```


# %%
def records(*, rows: int, seed: int, shift: float = 0.0) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    amount = rng.uniform(10.0, 30.0, rows) + shift
    cost = 100.0 + 4.0 * amount + rng.uniform(-1.0, 1.0, rows)
    for value, target in zip(amount, cost, strict=True):
        yield {"amount": float(value), "cost": float(target)}


def build() -> rf.Model:
    model = rf.Model.xs(
        batch_size=BATCH_SIZE,
        amount=rf.Quantile,
        cost=rf.Quantile(objective="mse", mask=True),
    )
    model.optimizer = rf.adamw(learning_rate=0.002)
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = pa.Table.from_pylist([{"amount": row["amount"]} for row in rows])
    predictions = model.predict(visible)["predictions"].to_pylist()
    return np.asarray([row["/cost"]["content"] for row in predictions], dtype=np.float64)


# %% [markdown]
# ## Training, metric, and boundary gates
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


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, rows=TRAIN_ROWS, seed=seed + 1),
        validate=partial(records, rows=1024, seed=seed + 2),
        seed=seed,
    )
    trainer = lit.Trainer(
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
    trainer.fit(model, datamodule=data)
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


# %% [markdown]
# ## Evidence
#
# {{< proof P068 evidence >}}
#
# ## Remaining work
#
# The ten-seed calibration retained the predeclared gates and fixed training
# budget. This proof establishes the boundary of this Quantile schema, not the
# extrapolation ability of Number or models with additional informative fields.
# Accepting an out-of-support value gives no OOD warning or calibrated uncertainty.
#
# ## Reproduce
#
# {{< proof P068 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=6801)
