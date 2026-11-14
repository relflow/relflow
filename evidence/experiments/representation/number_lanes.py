# %% [markdown]
# ---
# title: Can one number representation serve additive and multiplicative structure?
# categories:
# - Representation
# description: Compare Number, which embeds raw and log lanes and reads out standardized log-scale deviations, with
#   Quantile ranks on additive, moderate-tail, heavy-tail, and far-from-zero targets.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Heavy tails, Number
#   metric:
#   - heavy
#   - number_number
#   - normalized_mae
#   format: error
# - label: Far from zero, Number
#   metric:
#   - narrow
#   - number_number
#   - normalized_mae
#   format: error
# - label: Additive, Number
#   metric:
#   - additive
#   - number_number
#   - normalized_mae
#   format: error
# evidence-group: Representation
# ---
#
# Can the same numerical datatype handle adding ordinary values, multiplying
# widely varying positive values, and predicting a small variation around a large
# constant? These tasks stress different aspects of numerical representation.
# The comparison keeps the learning setup matched while changing how the inputs
# or target are represented.
#
# ## Example
#
# ```yaml
# price: 18.4
# rate: 0.62
# ltv: 15.1
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# For the illustrated price 18.4 and rate 0.62, the generator's noise-free median
# answer is 18.4 times the square root of 0.62, about 14.49. Multiplicative noise
# can produce a nearby observation such as ltv = 15.1. The name ltv labels this
# synthetic target; no real customer-value process is being estimated.
#
# The other tasks use left and right inputs. An illustrative left = 10 and
# right = -4 gives a median total of 6 in the additive task and 1006 when the
# fixed 1000 offset is added. The target formula stays simple even though its
# scale changes considerably.
#
# ```{typst}
# //| label: fig-evidence-number-lanes
# //| fig-cap: "Amber cards are hidden prediction targets. Each arm changes only how numbers are represented."
# //| fig-alt: "Model tree with customer, price, rate, ltv. Amber cards are hidden prediction targets. Each arm changes only how numbers are represented."
# #tree(node("customer", kind: "root", children: (
#   node("price", type: "Number / Quantile", detail: "raw and log lanes, or ranks"),
#   node("rate", type: "Number / Quantile", detail: "raw and log lanes, or ranks"),
#   node("ltv", kind: "target", type: "Number / Quantile", detail: "log-scale readout, or ranks"),
# )))
# ```
#
# ## Comparison
#
# Number exposes original-scale and signed-log information; Quantile exposes
# training-percentile information. A log scale makes multiplicative changes easier
# to express, while a percentile says where a value sits rather than how many
# units separate it from another value. The experiment checks which representation
# can learn each generated relationship under its declared budget.
#
# The evaluator knows the conditional median: the best population answer for the
# absolute-error task. It compares model error with that median's error and with
# always guessing the training median. A shuffled-input control retains each
# answer but pairs it with another observation's inputs. Separate bulk, tail, and
# outside-training-range measurements reveal where mistakes occur.
#
#
# Hold the data, model width, optimizer, and update budget fixed. Change only the
# numeric datatype: Number, which embeds a raw lane and a signed-log lane and
# reads targets out as standardized deviations on the signed-log scale, or
# Quantile, which represents values by their training percentile. Score every
# arm in source units against the known conditional median, on the bulk of the
# distribution and on its tails, for an additive task, for multiplicative tasks
# with moderate and very heavy tails, and for a target that sits near 1,000.
#
# ## Result
#
# {{< evidence P079 status >}}
#
# Read the task-specific errors together. The checks ask whether Number approaches
# the known median answer and remains close to the Quantile-input comparison;
# they do not establish that one representation wins on every distribution.
# Most arms receive 500 updates, and a separate 1500-update arm probes budget
# sensitivity. Extreme-tail errors and the Quantile-target comparison are
# additional diagnostics, so a single overall percentage does not summarize all
# of the tradeoffs. These are controlled synthetic tasks with known formulas.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P079 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P079 script >}}
#
# ### Synthetic processes and oracles
#
# The additive task draws `left` and `right` uniformly on [-50, 50] and sets
# `total = left + right + 2 z`; the far-from-zero task adds 1,000 to that
# total, so its target spans roughly 900 to 1,100. The multiplicative tasks
# draw `price = exp(3 + 1.2 z1)` and
# `rate = exp(s z2)` from independent standard normals, then
# `ltv = price * sqrt(rate) * exp(0.3 z3)`. Prices have a median near 20 and a
# long tail; rates have a median of 1. The heavy task uses `s = 2.5`, where the
# largest training rates can be much greater than the median. The moderate
# task uses `s = 1.5`, reducing that spread. The actual extremes depend on the
# generated sample. The noise is symmetric in log units, so
# the conditional median is exactly `price * sqrt(rate)`, and the evaluator
# computes it from the visible inputs without any latent variable.
#
# Oracle MAE is the test-set error of the conditional median. Baseline MAE is
# the error of the training-median constant. The multiplicative bulk is the set
# of test rows whose price and rate both fall between the 10th and 90th training
# percentiles; the tail is any row above the 95th percentile of either input.
# Every arm also records how much of its absolute error comes from predictions
# outside the training target range.
#
# ### Arms and budget
#
# Inputs embed as `Number` or as `Quantile` ranks; targets train as a `Number`
# or as a `Quantile` with the MAE objective. Every task compares the two input
# choices with a Number target; the heavy task also compares the two target
# choices with Number inputs, and gives the Number arm three times the budget
# once.
#
# Every arm uses the `xs` preset, 8,192 training rows, 1,024 validation rows,
# batch size 128, learning rate 0.002, and 500 AdamW updates (1,500 for the
# long arms), scored once on 4,096 fresh rows. A shuffled-input control pairs
# each test target with another row's inputs. Lane weight norms of the first
# projection are recorded for the combined arms as a diagnostic.
#
# ### Gates
#
# This experiment compares Number with Quantile across both tail regimes.
#
# Gates remain provisional until ten-seed calibration. Number must reach at most
# 1.5 times the oracle MAE on every task and stay within 10 percent of Quantile
# inputs on every task. Every arm must produce finite predictions and lose the
# relationship under shuffled inputs. The Quantile target, the long-budget arm,
# and the beyond-support error shares are diagnostics without gates.
#
# ### Remaining work
#
# The tasks are two-input and noise-free apart from symmetric log or additive
# noise. They do not cover bounded or bimodal bulks, where a rank lane may add
# information beyond the log lane, or mixed-sign heavy tails. Mixed precision
# needs its own explicit comparison; these fits declare full float32 precision.
#
# ### Complete experiment code
#

# %%
"""P079: compare Number's lanes and log-scale readout with Quantile ranks across numeric regimes."""

from collections.abc import Callable, Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
from relflow.presets.base import resolve
from reporting import experiment, report

BUDGET = 500
LONG_BUDGET = 1500
BATCH_SIZE = 128
TRAIN_ROWS = 8192
TEST_ROWS = 4096
PRICE_SCALE = 1.2
HEAVY_SCALE = 2.5
MODERATE_SCALE = 1.5
NOISE = 0.3
SPAN = 50.0
ADDITIVE_NOISE = 2.0
OFFSET = 1000.0
INPUTS = {
    "heavy": ("price", "rate"),
    "moderate": ("price", "rate"),
    "additive": ("left", "right"),
    "narrow": ("left", "right"),
}
TARGETS = {"heavy": "ltv", "moderate": "ltv", "additive": "total", "narrow": "total"}
ARMS = (
    ("heavy", "number", "number", BUDGET),
    ("heavy", "rank", "number", BUDGET),
    ("heavy", "number", "rank", BUDGET),
    ("heavy", "number", "number", LONG_BUDGET),
    ("moderate", "number", "number", BUDGET),
    ("moderate", "rank", "number", BUDGET),
    ("additive", "number", "number", BUDGET),
    ("additive", "rank", "number", BUDGET),
    ("narrow", "number", "number", BUDGET),
    ("narrow", "rank", "number", BUDGET),
)


def multiplicative(*, rows: int, seed: int, scale: float = HEAVY_SCALE) -> Iterator[dict]:
    """Lognormal prices and rates multiply under symmetric log noise; ``scale`` sets the rate tail."""
    rng = np.random.default_rng(seed)
    price = np.exp(3.0 + PRICE_SCALE * rng.standard_normal(rows))
    rate = np.exp(scale * rng.standard_normal(rows))
    ltv = price * np.sqrt(rate) * np.exp(NOISE * rng.standard_normal(rows))
    for index in range(rows):
        yield {"price": float(price[index]), "rate": float(rate[index]), "ltv": float(ltv[index])}


def additive(*, rows: int, seed: int, offset: float = 0.0) -> Iterator[dict]:
    """Two uniform terms add under symmetric noise; ``offset`` moves the total away from zero."""
    rng = np.random.default_rng(seed)
    left = rng.uniform(-SPAN, SPAN, rows)
    right = rng.uniform(-SPAN, SPAN, rows)
    total = offset + left + right + ADDITIVE_NOISE * rng.standard_normal(rows)
    for index in range(rows):
        yield {"left": float(left[index]), "right": float(right[index]), "total": float(total[index])}


RECORDS: dict[str, Callable[..., Iterator[dict]]] = {
    "heavy": multiplicative,
    "moderate": partial(multiplicative, scale=MODERATE_SCALE),
    "additive": additive,
    "narrow": partial(additive, offset=OFFSET),
}


def median(task: str, rows: list[dict]) -> np.ndarray:
    """Conditional median of the target given the visible inputs."""
    if task == "additive":
        return np.asarray([row["left"] + row["right"] for row in rows])
    if task == "narrow":
        return np.asarray([OFFSET + row["left"] + row["right"] for row in rows])
    return np.asarray([row["price"] * np.sqrt(row["rate"]) for row in rows])


def build(task: str, inputs: str, target: str) -> rf.Model:
    scalar = rf.Quantile if inputs == "rank" else rf.Number
    fields: dict[str, object] = {name: scalar for name in INPUTS[task]}
    if target == "rank":
        fields[TARGETS[task]] = rf.Quantile(objective="mae", mask=True)
    else:
        fields[TARGETS[task]] = rf.Number(mask=True)
    # Freeze the experimental architecture independently of SDK preset changes.
    control_preset = rf.presets.Preset(
        name="xs",
        d_model=64,
        root=rf.presets.BranchDefaults(n_layers=1, n_heads=2, reduction=rf.Attention()),
        branch=rf.presets.BranchDefaults(n_layers=1, n_heads=2, reduction=rf.Attention()),
        leaf=rf.presets.LeafDefaults(n_heads=2),
    )
    model = rf.Model(schema=resolve(control_preset, fields=fields))
    model.preset = control_preset
    return model


def fit(model: rf.Model, train, validate, seed: int, steps: int | None, accelerator: str, budget: int) -> int:
    data = rf.SyntheticDataModule(train=train, validate=validate, seed=seed)
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
        max_steps=budget if steps is None else min(steps, budget),
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    model.eval()
    return model.trainer.global_step


def predict(model: rf.Model, task: str, rows: list[dict]) -> np.ndarray:
    visible = [{name: row[name] for name in INPUTS[task]} for row in rows]
    predictions = pl.DataFrame(model.predict(pa.Table.from_pylist(visible)))["predictions"].to_list()
    return np.asarray([row[TARGETS[task]]["content"] for row in predictions], dtype=np.float64)


def evaluate(output: np.ndarray, control: np.ndarray, reference: dict, bulk: np.ndarray | None) -> dict:
    target, oracle = reference["target"], reference["oracle"]
    error = np.abs(output - target)
    mae = float(error.mean())
    beyond = (output < reference["support"][0]) | (output > reference["support"][1])
    result = {
        "mae": mae,
        "oracle_ratio": mae / reference["oracle_mae"],
        "normalized_mae": mae / reference["baseline_mae"],
        "median_distance": float(np.mean(np.abs(output - oracle))),
        "shuffled_normalized_mae": float(np.mean(np.abs(control - target))) / reference["baseline_mae"],
        "beyond_support_rows": int(beyond.sum()),
        "beyond_support_error_share": float(error[beyond].sum() / error.sum()) if error.sum() else 0.0,
        "finite": bool(np.isfinite(output).all() and np.isfinite(control).all()),
    }
    if bulk is not None:
        tail = reference["tail"]
        irreducible = np.abs(target - oracle)
        result["bulk_oracle_ratio"] = float(error[bulk].mean() / irreducible[bulk].mean())
        result["tail_oracle_ratio"] = float(error[tail].mean() / irreducible[tail].mean())
    return result


@experiment("P079", seed=7901)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    stream = seed * 100
    metrics: dict = {"train_rows": TRAIN_ROWS, "test_rows": TEST_ROWS, "budget": BUDGET, "long_budget": LONG_BUDGET}
    for offset, task in ((0, "heavy"), (20, "moderate"), (10, "additive"), (30, "narrow")):
        records, name = RECORDS[task], TARGETS[task]
        training = list(records(rows=TRAIN_ROWS, seed=stream + offset + 1))
        test = list(records(rows=TEST_ROWS, seed=stream + offset + 3))
        target = np.asarray([row[name] for row in test])
        oracle = median(task, test)
        observed = np.asarray([row[name] for row in training])
        constant = float(np.median(observed))
        reference = {
            "target": target,
            "oracle": oracle,
            "oracle_mae": float(np.mean(np.abs(target - oracle))),
            "baseline_mae": float(np.mean(np.abs(target - constant))),
            "support": (float(observed.min()), float(observed.max())),
        }
        bulk = None
        if task in ("heavy", "moderate"):
            columns = {key: np.asarray([row[key] for row in training]) for key in INPUTS[task]}
            low = {key: np.percentile(values, 10) for key, values in columns.items()}
            high = {key: np.percentile(values, 90) for key, values in columns.items()}
            extreme = {key: np.percentile(values, 95) for key, values in columns.items()}
            bulk = np.all([[low[key] <= row[key] <= high[key] for key in INPUTS[task]] for row in test], axis=1)
            reference["tail"] = np.any([[row[key] > extreme[key] for key in INPUTS[task]] for row in test], axis=1)
        permutation = np.random.default_rng(stream + offset + 4).permutation(TEST_ROWS)
        shuffled = [{**test[index], name: row[name]} for row, index in zip(test, permutation, strict=True)]
        metrics[task] = {
            "oracle_mae": reference["oracle_mae"],
            "baseline_mae": reference["baseline_mae"],
            "training_median": constant,
            "training_support": list(reference["support"]),
            "bulk_rows": None if bulk is None else int(bulk.sum()),
            "tail_rows": None if bulk is None else int(reference["tail"].sum()),
        }
        for arm_task, inputs, space, budget in ARMS:
            if arm_task != task:
                continue
            lit.seed_everything(seed, workers=True)
            model = build(task, inputs, space)
            updates = fit(
                model,
                partial(records, rows=TRAIN_ROWS, seed=stream + offset + 1),
                partial(records, rows=1024, seed=stream + offset + 2),
                seed,
                steps,
                accelerator,
                budget,
            )
            result = evaluate(predict(model, task, test), predict(model, task, shuffled), reference, bulk)
            result["steps"] = updates
            metrics[task][f"{inputs}_{space}" + ("_long" if budget != BUDGET else "")] = result
    arms = [value for task in TARGETS for value in metrics[task].values() if isinstance(value, dict)]
    checks = {}
    for task in TARGETS:
        label = {
            "heavy": "Heavy tails",
            "moderate": "Moderate tails",
            "additive": "Additive",
            "narrow": "Far from zero",
        }[task]
        checks[f"{label}: Number reaches 1.5 times the oracle MAE"] = (
            metrics[task]["number_number"]["oracle_ratio"] <= 1.5
        )
        checks[f"{label}: Number stays within 10 percent of Quantile inputs"] = (
            metrics[task]["number_number"]["mae"] <= 1.10 * metrics[task]["rank_number"]["mae"]
        )
    checks["Shuffled inputs lose the relationship in every arm"] = all(
        arm["shuffled_normalized_mae"] >= 0.90 for arm in arms
    )
    checks["Every arm produces finite predictions"] = all(arm["finite"] for arm in arms)
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
