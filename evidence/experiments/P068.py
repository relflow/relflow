# %% [markdown]
# ---
# title: What happens beyond the learned percentile range?
# description: Train a Quantile relationship within a bounded amount range, then supply much smaller and larger amounts.
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
# evidence-review:
#   question: What happens beyond the learned percentile range?
#   modeling: Train Quantile inputs and targets on amounts between 10 and 30, with cost = 100 + 4 × amount plus small
#     noise. Test within that interval and in matched intervals below and above it. Compare RMSE in the original cost
#     units.
#   comparison:
#     kind: Recorded conditions
#     metric: RMSE
#     conditions:
#     - label: In-range error, original units
#       path:
#       - in_range_rmse
#       format: number
#     - label: Below-range error, original units
#       path:
#       - below
#       - rmse
#       format: number
#     - label: Above-range error, original units
#       path:
#       - above
#       - rmse
#       format: number
#     definition: Root mean squared prediction error, in the target’s original units. Zero is an exact prediction.
#     direction: Lower error is better
#   note: The checks deliberately measure tail saturation and bounded target support. Checks met does not mean the
#     model extrapolated the relationship.
# evidence-plot:
#   target: cost
#   focus: What happens beyond the output representation's observed range?
#   comparison: Separate in-range, below-range, and above-range targets on common source-unit axes.
#   limit: Clipped output endpoints impose a representational limit that extra input evidence cannot remove.
#   labels:
#     rows: In-range held-out records
#     shifted: Shifted outer range
#   renderer: plot
# evidence-cost:
#   baseline: 14
#   candidate: 14
# ---
#
# This experiment asks what a percentile representation cannot distinguish. Training amounts lie
# between 10 and 30, with cost increasing linearly. Evaluation then moves populations below or above
# that range. The question is whether valid numeric input can still lose the information needed for
# extrapolation after percentile conversion.
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
#
# The illustrated amount of 20 has average cost `100 + 4 × 20 = 180`; 180.5 is a possible noisy
# observation. An illustrative amount of 50 instead has average cost 300. Training costs occupy
# approximately 140 to 220. A decoder restricted to its learned target support cannot reach 300 merely
# because the original relationship is linear.
#
# ## Comparison
#
# Compare in-range error with each outer range. Check whether distinct amounts in a tail receive the
# same prediction.
#
# The shifted tests reuse the same latent draws as the in-range evaluation and add a fixed amount
# shift, so the underlying rule remains unchanged. Within each outer range, inputs collapse to the
# same endpoint percentile. The experiment measures prediction spread to check this saturation.
# Shuffling the in-range inputs separately confirms that ordinary prediction depended on the amount.
#
# ## Result
#
# Outside the learned support, percentile inputs and outputs hit their boundaries. Tail predictions
# become constant; accepting a number does not guarantee extrapolation.
#
# The clipping lower bound is calculated by an evaluator allowed to see each true target, then
# restricted to the learned output interval. It describes an unavoidable error floor, not a trained
# prediction. Here a met limitation check can accompany large tail error: it confirms saturation and
# finite bounded outputs, rather than successful extrapolation or an automatic warning about
# unfamiliar input.
#
# ## Method
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
# tests share latent draws only with each other. Train the frozen control model for 400 AdamW
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

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.10",
#     "lightning",
#     "numpy",
#     "polars",
#     "pyarrow",
#     "pyyaml>=6",
#     "relflow>=0.5.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "representation"
# types = ["Quantile"]
# conditions = ["extrapolation", "changed-associations", "numeric-extremes"]
# ///

"""P068: distinguish learned in-range prediction from bounded Quantile extrapolation."""

import sys
from collections.abc import Iterator
from functools import partial
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
from policies import CONTROL
from reporting import experiment, report
from visibility import TARGET

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


class Model(rf.Model):
    amount = rf.Quantile
    cost = rf.Quantile(objective="mse", **TARGET)


def build() -> rf.Model:

    model = Model(preset=CONTROL.name)
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = pa.Table.from_pylist([{"amount": row["amount"]} for row in rows])
    predictions = pl.DataFrame(model.predict(visible))["predictions"].to_list()
    return np.asarray([row["cost"]["content"] for row in predictions], dtype=np.float64)


@experiment("P068", seed=6801)
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


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    import textwrap

    import numpy as np

    from evidence.diagnostics.plots import COLORS, axes_pair

    pairs = [
        pair
        for pair in bundle.pairs(spec["target"])
        if pair["population"] == "test" and "permutation" not in pair["call"]
    ]
    if not pairs:
        raise ValueError("Extrapolation requires retained source-unit predictions")
    fig, axes = axes_pair(width=10, height=4.8)
    selected = pairs[:3]
    for index, pair in enumerate(selected):
        axes[0].scatter(
            pair["actual"], pair["predicted"], s=5, alpha=0.2, color=COLORS[index], label=pair["label"], rasterized=True
        )
    values = np.concatenate([pair["actual"] for pair in selected])
    lower, upper = (float(values.min()), float(values.max()))
    axes[0].plot([lower, upper], [lower, upper], ls="--", color="gray")
    axes[0].set(
        xlabel="Actual source-unit answer",
        ylabel="Model source-unit prediction",
        title="Does the prediction reach the requested range?",
    )
    axes[0].legend(fontsize=8)
    axes[1].boxplot(
        [np.abs(pair["actual"] - pair["predicted"]) for pair in selected],
        orientation="horizontal",
        tick_labels=[textwrap.fill(pair["label"], 22) for pair in selected],
        showfliers=False,
        medianprops={"color": COLORS[1]},
    )
    axes[1].set(xlabel="Absolute source-unit error", title="Error remains separate for each range")
    return (
        fig,
        "Every scatter point is a captured prediction. The diagonal denotes exact answers; clipping appears as a plateau. Error boxes show medians, quartiles, and 1.5-IQR whiskers with outliers omitted from that display.",
        {},
    )


if __name__ == "__main__":
    report(run)
