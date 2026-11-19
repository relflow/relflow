# %% [markdown]
# ---
# title: Do ordered inputs generalize to unseen ranks?
# description: Compare ordinal labels and integer ranks with Number, Category, and shuffled-signal controls.
# evidence-readout:
# - label: Ordered labels on withheld inputs
#   metric:
#   - arms
#   - ordinal_labels
#   - unseen
#   - nrmse
#   format: number
# - label: Number on withheld inputs
#   metric:
#   - arms
#   - number
#   - unseen
#   - nrmse
#   format: number
# - label: Category on withheld inputs
#   metric:
#   - arms
#   - category
#   - unseen
#   - nrmse
#   format: number
# evidence-group: Representation
# evidence-review:
#   question: Do ordered inputs generalize to unseen ranks?
#   modeling: Predict a continuous response from an ordered input. Train on even ranks and test on withheld odd ranks,
#     comparing declared labels, integer Ordinal, Number, Category, and a shuffled numeric control.
#   comparison:
#     metric: Normalized RMSE
#     kind: Controlled comparison
#     conditions:
#     - label: Shuffled numeric input
#       path:
#       - arms
#       - shuffled
#       - unseen
#       - nrmse
#       format: number
#     - label: Category input
#       path:
#       - arms
#       - category
#       - unseen
#       - nrmse
#       format: number
#     - label: Prepared Number
#       path:
#       - arms
#       - number
#       - unseen
#       - nrmse
#       format: number
#     - label: Integer Ordinal
#       path:
#       - arms
#       - ordinal_ranks
#       - unseen
#       - nrmse
#       format: number
#     - label: Declared-label Ordinal
#       path:
#       - arms
#       - ordinal_labels
#       - unseen
#       - nrmse
#       format: number
#     definition: Root mean squared error divided by the error of the training-mean predictor on the same split.
#     direction: Lower error is better
#   note: Withheld labels are declared but absent from training. Numeric extrapolation is reported separately.
# evidence-plot:
#   mode: summary
#   target: target
#   focus: How do ordered inputs generalize to withheld ranks?
#   comparison: Compare Ordinal, Number, Category, and shuffled controls on familiar, withheld, and outside-range
#     ranks.
#   limit: Label arms cannot represent outside-range ranks; interpolation and extrapolation are separate.
# ---
#
# A model receives an ordered level and predicts a hidden continuous response. Training
# observes only even positions in a nine-level order; test separately evaluates familiar
# levels and withheld odd positions between them. Labels are nonnumeric and their declared
# order, rather than alphabetical order, gives the model the numeric position.
#
# ## Example
#
# ```yaml
# level: "silver"
# target: -4.5
# ```
#
# `silver` occupies position one in the declared order. The generating mean is
# `1.5 * (1 - 4) = -4.5`; generated observations add independent noise. The target is
# omitted from prediction inputs. Training sees positions zero, two, four, six, and eight,
# so this level is withheld while remaining inside the trained range.
#
#
# ## Comparison
#
# Compare declared-label Ordinal, implicit-integer Ordinal, prepared Number, and Category
# inputs using identical underlying ranks, independent observation splits, model seeds,
# and fixed training budgets. A fifth model receives a shuffled Number rank during
# training; validation and test records remain intact. The Category control sees the
# same names but has no declared order. Familiar and withheld ranks are scored separately.
# Integer Ordinal and Number also report out-of-range ranks, without claiming extrapolation
# must succeed. The declared-label arms cannot accept undeclared out-of-range labels.
#
# ## Result
#
#
# The predeclared checks compare interpolation with the controls and the prepared numeric
# input. This synthetic linear relationship does not establish that a declared order is
# appropriate for arbitrary business labels, nor that a model will extrapolate.
#
#
#
#
#
# ## Method
#
# ### Population, controls, and scoring
#
# Each arm receives 4,096 training and 512 validation rows; seen, withheld, and, where
# applicable, extrapolation tests each contain 1,024 fresh rows. Rank counts are balanced
# within each split. Target noise has standard deviation 0.1. All arms train for 300 updates
# with the same frozen architecture and optimizer. Report RMSE, MAE, normalized RMSE,
# per-rank errors, and aligned held-out predictions. Category cannot distinguish unfamiliar
# input labels without information beyond the trained vocabulary.
#
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = [
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
# types = ["Ordinal", "Number", "Category"]
# conditions = ["unseen-values", "extrapolation", "changed-associations", "representations"]
# ///

"""P114: measure ordered inputs on withheld ranks against matched representations."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
from policies import CONTROL
from reporting import experiment, report

BUDGET = 300
TRAIN_ROWS = 4096
VALIDATION_ROWS = 512
TEST_ROWS = 1024
LABELS = ("bronze", "silver", "gold", "platinum", "diamond", "opal", "sapphire", "ruby", "emerald")
RANKS = {"seen": (0, 2, 4, 6, 8), "unseen": (1, 3, 5, 7), "extrapolation": (-2, -1, 9, 10)}
ARMS = ("ordinal_labels", "ordinal_ranks", "number", "category", "shuffled")


class LabelModel(rf.Model):
    level = rf.Ordinal(values=LABELS)
    target = rf.Number(mask=True, objective="mse")


class RankModel(rf.Model):
    level = rf.Ordinal
    target = rf.Number(mask=True, objective="mse")


class NumericModel(rf.Model):
    level = rf.Number
    target = rf.Number(mask=True, objective="mse")


class CategoryModel(rf.Model):
    level = rf.Category
    target = rf.Number(mask=True, objective="mse")


def records(*, rows: int, seed: int, condition: str) -> Iterator[dict]:
    """Generate balanced ranks and independent noisy answers for one split."""
    rng = np.random.default_rng(seed)
    ranks = np.resize(np.asarray(RANKS[condition], dtype=np.int64), rows)
    rng.shuffle(ranks)
    noise = rng.normal(0, 0.1, rows)
    for index, (rank, error) in enumerate(zip(ranks, noise, strict=True)):
        yield {"case": f"{seed}:{index}", "rank": int(rank), "target": 1.5 * (int(rank) - 4) + float(error)}


def frame(rows: list[dict], *, arm: str, targets: bool = True, shuffle: bool = False, seed: int = 0) -> pl.DataFrame:
    """Select the same underlying ranks in each arm and omit answers at prediction."""
    indices = np.arange(len(rows))
    if shuffle:
        indices = np.random.default_rng(seed).permutation(indices)
    ranks = [rows[int(index)]["rank"] for index in indices]
    if arm in ("ordinal_labels", "category") and any(rank < 0 or rank >= len(LABELS) for rank in ranks):
        raise ValueError(f"{arm} accepts only declared label positions 0 through {len(LABELS) - 1}")
    values = [LABELS[rank] for rank in ranks] if arm in ("ordinal_labels", "category") else ranks
    columns = {"level": values}
    if targets:
        columns["target"] = [row["target"] for row in rows]
    return pl.DataFrame(columns)


def build(arm: str) -> rf.Model:
    if arm == "ordinal_labels":
        return LabelModel(preset=CONTROL.name)
    if arm == "ordinal_ranks":
        return RankModel(preset=CONTROL.name)
    if arm == "category":
        return CategoryModel(preset=CONTROL.name)
    if arm in ("number", "shuffled"):
        return NumericModel(preset=CONTROL.name)
    raise ValueError(f"unknown Ordinal experiment arm {arm!r}")


def prediction(model: rf.Model, rows: list[dict], arm: str) -> np.ndarray:
    return (
        model.predict(frame(rows, arm=arm, targets=False), batch_size=256)["predictions"]
        .struct.field("target")
        .struct.field("content")
        .to_numpy()
    )


def metrics(actual: np.ndarray, predicted: np.ndarray, mean: float) -> dict:
    error = predicted - actual
    baseline = float(np.sqrt(np.mean((actual - mean) ** 2)))
    rmse = float(np.sqrt(np.mean(error**2)))
    return {"rows": int(len(actual)), "rmse": rmse, "mae": float(np.mean(np.abs(error))), "nrmse": rmse / baseline}


def evaluate(model: rf.Model, rows: list[dict], arm: str, mean: float) -> dict:
    actual = np.asarray([row["target"] for row in rows])
    ranks = np.asarray([row["rank"] for row in rows])
    predicted = prediction(model, rows, arm)
    if not np.isfinite(predicted).all():
        raise ValueError(f"{arm} produced nonfinite held-out predictions")
    result = metrics(actual, predicted, mean)
    result["finite"] = bool(np.isfinite(predicted).all())
    result["ranks"] = {
        str(rank): metrics(actual[ranks == rank], predicted[ranks == rank], mean) for rank in np.unique(ranks)
    }
    result["predictions"] = predicted
    return result


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


@experiment("P114", seed=11401)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train = list(records(rows=TRAIN_ROWS, seed=seed + 1, condition="seen"))
    validate = list(records(rows=VALIDATION_ROWS, seed=seed + 2, condition="seen"))
    tests = {
        condition: list(records(rows=TEST_ROWS, seed=seed + 3 + index, condition=condition))
        for index, condition in enumerate(RANKS)
    }
    mean = float(np.mean([row["target"] for row in train]))
    output = {
        "budget": BUDGET,
        "rows": {"train": len(train), "validation": len(validate), "test_per_condition": TEST_ROWS},
        "train_mean": mean,
        "arms": {},
        "test": {
            condition: {
                "case": [row["case"] for row in rows],
                "rank": [row["rank"] for row in rows],
                "target": [row["target"] for row in rows],
            }
            for condition, rows in tests.items()
        },
    }
    checks = {}
    for arm in ARMS:
        lit.seed_everything(seed, workers=True)
        model = build(arm)
        updates = fit(
            model,
            frame(train, arm=arm, shuffle=arm == "shuffled", seed=seed + 6),
            frame(validate, arm=arm),
            seed,
            steps,
            accelerator,
        )
        conditions = (
            ("seen", "unseen", "extrapolation")
            if arm in ("ordinal_ranks", "number", "shuffled")
            else ("seen", "unseen")
        )
        scores = {condition: evaluate(model, tests[condition], arm, mean) for condition in conditions}
        scores["steps"] = updates
        output["arms"][arm] = scores
        checks[f"{arm}: full budget retained"] = updates == BUDGET
        checks[f"{arm}: finite predictions"] = all(scores[condition]["finite"] for condition in conditions)
        if arm in ("ordinal_labels", "ordinal_ranks", "number"):
            for condition in ("seen", "unseen"):
                checks[f"{arm}: {condition} nRMSE below 0.25"] = scores[condition]["nrmse"] < 0.25
        if arm == "category":
            checks["Category: seen nRMSE below 0.35"] = scores["seen"]["nrmse"] < 0.35
            checks["Category: unseen nRMSE above 0.80"] = scores["unseen"]["nrmse"] > 0.80
        if arm == "shuffled":
            checks["Shuffled numeric input: unseen nRMSE above 0.80"] = scores["unseen"]["nrmse"] > 0.80
    number = output["arms"]["number"]["unseen"]["nrmse"]
    for arm in ("ordinal_labels", "ordinal_ranks"):
        checks[f"{arm}: within 0.05 nRMSE of Number on unseen ranks"] = (
            abs(output["arms"][arm]["unseen"]["nrmse"] - number) < 0.05
        )
    return output, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
