# %% [markdown]
# ---
# title: How do open ordinal targets behave beyond seen ranks?
# description: Compare default and MSE open Ordinal targets with matching Number heads on seen, withheld, and out-of-range
#   ranks.
# evidence-readout:
# - label: Ordinal MAE unseen rank error
#   metric:
#   - arms
#   - ordinal_mae
#   - unseen
#   - rank_mae
#   format: number
# - label: Ordinal MAE extrapolation rank error
#   metric:
#   - arms
#   - ordinal_mae
#   - extrapolation
#   - rank_mae
#   format: number
# - label: Ordinal MSE extrapolation rank error
#   metric:
#   - arms
#   - ordinal_mse
#   - extrapolation
#   - rank_mae
#   format: number
# evidence-group: Representation
# evidence-review:
#   question: How do open ordinal targets behave beyond seen ranks?
#   modeling: Give a noisy numeric clue to an open integer target. Train on even ranks, test seen and intermediate
#     ranks, and separately evaluate ranks outside the training range. Compare default and MSE Ordinal with matching
#     Number outputs.
#   comparison:
#     metric: Rank MAE
#     kind: Controlled comparison
#     conditions:
#     - label: Default Ordinal · unseen
#       path:
#       - arms
#       - ordinal_mae
#       - unseen
#       - rank_mae
#       format: number
#     - label: Default Ordinal · extrapolation
#       path:
#       - arms
#       - ordinal_mae
#       - extrapolation
#       - rank_mae
#       format: number
#     - label: MSE Ordinal · extrapolation
#       path:
#       - arms
#       - ordinal_mse
#       - extrapolation
#       - rank_mae
#       format: number
#     definition: Mean absolute error between predicted and observed integer ranks after rounding.
#     direction: Lower error is better
#   note: Extrapolation is reported without a predeclared quality guarantee. The clue is a noisy rank.
# evidence-plot:
#   renderer: plot
#   target: rank
#   focus: How do open ordinal outputs behave on ranks outside training?
#   comparison: Compare default and MSE Ordinal with Number and a shuffled-clue control.
#   limit: This synthetic clue is directly derived from the rank; extrapolation is not guaranteed.
# ---
#
# Each record has a noisy numeric signal and a hidden integer rank. Training sees only
# even ranks in [-8, 8]; tests separately cover seen ranks, intermediate odd ranks,
# and held-out ranks outside that range.
#
# ## Example
#
# ```yaml
# signal: 9.9
# rank: 10
# ```
#
# Rank 10 lies outside the training range. The signal is visible; the rank is hidden.
#
# ## Comparison
#
# Compare default and MSE Ordinal, matching Number targets, and a shuffled-clue control.
# All arms receive matching rows, seeds, model capacity and update budgets.
#
# ## Result
#
# Report post-rounding exact accuracy and rank error for each condition and rank.
# Extrapolation is diagnostic rather than a promised capability.
#
# ## Method
#
# Generate 4,096 train and 512 validation records from even ranks -8 through 8, and
# 1,024 independent test records per condition. Intermediate ranks are odd within
# that range; extrapolation ranks are -12, -11, -10, 10, 11 and 12. The clue equals
# the rank plus Gaussian noise of standard deviation 0.15. Train each arm for 500
# updates and preserve aligned predictions, fixed split seeds, and per-rank errors.
# Content and predicted value states are recorded separately; all test targets are valued.
#
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["lightning", "matplotlib>=3.10", "numpy", "polars", "pyarrow", "pyyaml>=6", "relflow>=0.5.0", "torch"]
#
# [tool.relflow.evidence]
# capability = "representation"
# types = ["Ordinal", "Number"]
# conditions = ["unseen-values", "extrapolation", "objectives", "representations"]
# ///

"""P121: compare open Ordinal and Number targets on seen and held-out ranks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
from policies import CONTROL
from reporting import experiment, report
from visibility import TARGET

BUDGET = 500
TRAIN_ROWS = 4096
VALIDATION_ROWS = 512
TEST_ROWS = 1024
RANKS = {"seen": tuple(range(-8, 9, 2)), "unseen": tuple(range(-7, 8, 2)), "extrapolation": (-12, -11, -10, 10, 11, 12)}
ARMS = ("ordinal_mae", "ordinal_mse", "number_mae", "number_mse", "shuffled")


class OrdinalMae(rf.Model):
    signal = rf.Number
    rank = rf.Ordinal(**TARGET)


class OrdinalMse(rf.Model):
    signal = rf.Number
    rank = rf.Ordinal(**TARGET, objective="mse")


class NumberMae(rf.Model):
    signal = rf.Number
    rank = rf.Number(**TARGET)


class NumberMse(rf.Model):
    signal = rf.Number
    rank = rf.Number(**TARGET, objective="mse")


def records(*, rows: int, seed: int, condition: str):
    rng = np.random.default_rng(seed)
    for index in range(rows):
        rank = int(rng.choice(RANKS[condition]))
        yield {"case": f"{seed}:{index}", "signal": float(rank + rng.normal(0, 0.15)), "rank": rank}


def frame(rows: list[dict], *, targets: bool = True, shuffle: bool = False, seed: int = 0) -> pl.DataFrame:
    indices = np.arange(len(rows))
    if shuffle:
        indices = np.random.default_rng(seed).permutation(indices)
    data = {"signal": [rows[int(index)]["signal"] for index in indices]}
    if targets:
        data["rank"] = [row["rank"] for row in rows]
    return pl.DataFrame(data)


def fit(
    model: rf.Model, train: pl.DataFrame, validate: pl.DataFrame, seed: int, steps: int | None, accelerator: str
) -> int:
    data = rf.PolarsDataModule(
        train=train, validate=validate, seed=seed, num_workers=0, persistent_workers=False, pin_memory=False
    )
    model.fit(
        data,
        batch_size=64,
        learning_rate=0.003,
        weight_decay=0.01,
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
    model.eval()
    return int(model.trainer.global_step)


def evaluate(model: rf.Model, rows: list[dict]) -> dict:
    expected = np.asarray([row["rank"] for row in rows], dtype=np.int64)
    output = model.predict(frame(rows, targets=False), batch_size=256)["predictions"].to_list()
    raw = np.asarray([row["rank"]["content"] for row in output], dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("open rank model produced nonfinite predictions")
    predicted = np.rint(raw).astype(np.int64)
    errors = predicted - expected
    states = [max(row["rank"]["state"], key=row["rank"]["state"].get) for row in output]
    return {
        "rows": len(expected),
        "accuracy": float(np.mean(errors == 0)),
        "rank_mae": float(np.mean(np.abs(errors))),
        "rank_rmse": float(np.sqrt(np.mean(errors**2))),
        "inferred": all(row["rank"]["inferred"] for row in output),
        "state_accuracy": float(np.mean(np.asarray(states) == "valued")),
        "states": states,
        "ranks": {
            str(rank): {
                "rows": int((expected == rank).sum()),
                "rank_mae": float(np.mean(np.abs(errors[expected == rank]))),
                "accuracy": float(np.mean(errors[expected == rank] == 0)),
            }
            for rank in np.unique(expected)
        },
        "predictions": predicted,
        "raw_predictions": raw,
    }


@experiment("P121", seed=12101)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train = list(records(rows=TRAIN_ROWS, seed=seed + 1, condition="seen"))
    validate = list(records(rows=VALIDATION_ROWS, seed=seed + 2, condition="seen"))
    tests = {
        condition: list(records(rows=TEST_ROWS, seed=seed + 3 + index, condition=condition))
        for index, condition in enumerate(RANKS)
    }
    metrics = {
        "budget": BUDGET,
        "rows": {"train": len(train), "validation": len(validate), "test_per_condition": TEST_ROWS},
        "test": {
            condition: {
                "case": [row["case"] for row in rows],
                "rank": [row["rank"] for row in rows],
                "signal": [row["signal"] for row in rows],
            }
            for condition, rows in tests.items()
        },
        "arms": {},
    }
    checks = {}
    models = {
        "ordinal_mae": OrdinalMae,
        "ordinal_mse": OrdinalMse,
        "number_mae": NumberMae,
        "number_mse": NumberMse,
        "shuffled": OrdinalMae,
    }
    for arm in ARMS:
        lit.seed_everything(seed, workers=True)
        model = models[arm](preset=CONTROL.name)
        updates = fit(
            model, frame(train, shuffle=arm == "shuffled", seed=seed + 6), frame(validate), seed, steps, accelerator
        )
        measured = {condition: evaluate(model, rows) for condition, rows in tests.items()}
        measured["steps"] = updates
        metrics["arms"][arm] = measured
        checks[f"{arm}: full budget retained"] = updates == BUDGET
        checks[f"{arm}: inferred outputs"] = all(measured[condition]["inferred"] for condition in tests)
        checks[f"{arm}: valued state accuracy at least 0.95"] = all(
            measured[condition]["state_accuracy"] >= 0.95 for condition in tests
        )
    checks["Default Ordinal seen rank MAE below 1"] = metrics["arms"]["ordinal_mae"]["seen"]["rank_mae"] < 1.0
    checks["Shuffled signal seen rank MAE above 2"] = metrics["arms"]["shuffled"]["seen"]["rank_mae"] > 2.0
    return metrics, checks


def plot(bundle, spec):
    """Separate interpolation and extrapolation without hiding rank-level errors."""
    import matplotlib.pyplot as plt

    from evidence.diagnostics.plots import dots

    arms = bundle.record["metrics"]["arms"]
    names = ("ordinal_mae", "ordinal_mse", "number_mae", "number_mse", "shuffled")
    labels = ("Ordinal MAE", "Ordinal MSE", "Number MAE", "Number MSE", "Shuffled clues")
    conditions = ("seen", "unseen", "extrapolation")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    dots(
        axes[0],
        ("Seen ranks", "Withheld inside range", "Outside training range"),
        {
            label: [arms[name][condition]["rank_mae"] for condition in conditions]
            for name, label in zip(names, labels, strict=True)
        },
        xlabel="Post-rounding rank MAE (lower is better)",
    )
    for name, label in zip(names[:2], labels[:2], strict=True):
        ranks = {
            int(rank): score["rank_mae"]
            for condition in conditions
            for rank, score in arms[name][condition]["ranks"].items()
        }
        axes[1].scatter(list(ranks), list(ranks.values()), label=label, alpha=0.75)
    for boundary in (-8, 8):
        axes[1].axvline(boundary, color="gray", linestyle=":", linewidth=1)
    axes[1].set(xlabel="Actual rank", ylabel="Post-rounding rank MAE")
    axes[1].legend(fontsize=8, loc="best")
    axes[1].grid(alpha=0.22)
    axes[0].set_title("Held-out populations")
    axes[1].set_title("Ordinal error by actual rank")
    paths = [["arms", name, condition, "rank_mae"] for name in names for condition in conditions]
    paths += [["arms", name, condition, "ranks"] for name in names[:2] for condition in conditions]
    return (
        fig,
        "Dotted lines mark the ends of the training range. Withheld odd ranks are inside that range; extrapolation ranks are outside it. Extrapolation is measured without a quality guarantee.",
        {"metric_paths": paths},
    )


if __name__ == "__main__":
    report(run)
