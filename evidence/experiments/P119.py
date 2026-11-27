# %% [markdown]
# ---
# title: How do ordinal output objectives behave with uncertain labels?
# description: Compare default and MSE ordinal targets with a fixed-label Enum on noisy ordered outcomes.
# evidence-readout:
# - label: Enum
#   metric:
#   - arms
#   - enum
#   - overall
#   - rank_mae
#   format: number
# - label: Ordinal MSE
#   metric:
#   - arms
#   - mse
#   - overall
#   - rank_mae
#   format: number
# - label: Ordinal default MAE
#   metric:
#   - arms
#   - mae
#   - overall
#   - rank_mae
#   format: number
# evidence-group: Representation
# evidence-review:
#   question: How do ordinal output objectives behave with uncertain labels?
#   modeling: Predict one of five declared ordered labels from a numeric clue with overlapping noisy categories. Compare
#     the two numeric objectives with Enum on independent held-out observations, and retain class probabilities.
#   comparison:
#     metric: Rank MAE
#     kind: Controlled comparison
#     conditions:
#     - label: Enum
#       path:
#       - arms
#       - enum
#       - overall
#       - rank_mae
#       format: number
#     - label: Ordinal MSE
#       path:
#       - arms
#       - mse
#       - overall
#       - rank_mae
#       format: number
#     - label: Ordinal default MAE
#       path:
#       - arms
#       - mae
#       - overall
#       - rank_mae
#       format: number
#     definition: Mean absolute difference between the predicted and observed declared rank after writing.
#     direction: Lower error is better
#   note: Unequal latent boundaries and measurement noise are synthetic; no universal objective ranking is assumed.
# evidence-plot:
#   renderer: plot
#   target: target
#   focus: How do ordinal output objectives behave with uncertain labels?
#   comparison: Compare default and MSE ordinal targets with a fixed-label Enum on noisy ordered outcomes.
#   limit: Unequal latent boundaries and measurement noise are synthetic; no universal objective ranking is assumed.
# ---
#
# The input signal is noisy evidence of an ordered label. All labels are declared for every model; held-out records use independent draws and the level is hidden at inference.
#
# ## Example
#
# ```yaml
# signal: 1.1
# level: high
# ```
#
# ## Comparison
#
# Predict one of five declared ordered labels from a numeric clue with overlapping noisy categories. Score post-write labels and retain class probabilities for Enum.
# Independent splits, matched seeds, equal update budgets, and a negative control are retained.
#
# ## Result
#
# Measurements and checks are recorded independently; one synthetic task does not establish general performance.
#
# ## Method
#
# Each arm uses the same declared budget, optimizer, model/data seeds, and independent held-out observations.
# Per-arm metrics retain subgroup results and aligned predictions.
# Content and predicted value states are recorded separately; all test targets are valued.
#
# Generate 4,096 training, 512 validation and 2,048 held-out signals uniformly in
# [-4, 4]; add independent Gaussian noise with standard deviation 0.75 and partition
# at -2.8, -1.0, 0.25 and 2.0 into five ordered labels. All labels are declared
# and present in training. Run 500 updates per arm. Score post-write rank accuracy
# and error overall, per class and near boundaries. Enum additionally records
# negative log likelihood, ranked probability score and threshold calibration bins.
#
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["lightning", "matplotlib>=3.10", "numpy", "polars", "pyarrow", "pyyaml>=6", "relflow>=0.6.0", "torch"]
#
# [tool.relflow.evidence]
# capability = "representation"
# types = ["Ordinal", "Enum", "Number"]
# conditions = ["objectives", "noise", "representations", "unseen-inputs"]
# ///

"""P119: compare Ordinal objectives and fixed-class Enum on noisy ordered targets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
from policies import CONTROL
from reporting import experiment, report

BUDGET = 500
TRAIN_ROWS = 4096
VALIDATION_ROWS = 512
TEST_ROWS = 2048
LABELS = ("very_low", "low", "medium", "high", "very_high")
THRESHOLDS = np.asarray([-2.8, -1.0, 0.25, 2.0])
ARMS = ("mae", "mse", "enum", "shuffled")


class OrdinalMae(rf.Model):
    signal = rf.Number
    level = rf.Ordinal(values=LABELS, predict=True)


class OrdinalMse(rf.Model):
    signal = rf.Number
    level = rf.Ordinal(values=LABELS, predict=True, objective="mse")


class EnumModel(rf.Model):
    signal = rf.Number
    level = rf.Enum(*LABELS, predict=True, topk=[len(LABELS)])


def records(*, rows: int, seed: int):
    rng = np.random.default_rng(seed)
    for index in range(rows):
        signal = float(rng.uniform(-4.0, 4.0))
        latent = signal + float(rng.normal(0, 0.75))
        rank = int(np.searchsorted(THRESHOLDS, latent))
        yield {"case": f"{seed}:{index}", "signal": signal, "level": LABELS[rank], "rank": rank}


def frame(rows: list[dict], *, targets: bool = True, shuffle: bool = False, seed: int = 0) -> pl.DataFrame:
    indices = np.arange(len(rows))
    if shuffle:
        indices = np.random.default_rng(seed).permutation(indices)
    data = {"signal": [rows[int(index)]["signal"] for index in indices]}
    if targets:
        data["level"] = [row["level"] for row in rows]
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


def score(actual: np.ndarray, predicted: np.ndarray, selected: np.ndarray) -> dict:
    expected, result = actual[selected], predicted[selected]
    return {
        "rows": int(len(expected)),
        "accuracy": float(np.mean(expected == result)),
        "rank_mae": float(np.mean(np.abs(expected - result))),
        "rank_rmse": float(np.sqrt(np.mean((expected - result) ** 2))),
    }


@experiment("P119", seed=11901)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train = list(records(rows=TRAIN_ROWS, seed=seed + 1))
    validate = list(records(rows=VALIDATION_ROWS, seed=seed + 2))
    test = list(records(rows=TEST_ROWS, seed=seed + 3))
    actual = np.asarray([row["rank"] for row in test])
    signals = np.asarray([row["signal"] for row in test])
    close = np.min(np.abs(signals[:, None] - THRESHOLDS), axis=1) < 0.4
    groups = {"overall": np.ones(len(test), dtype=bool), "near_boundary": close, "away_from_boundary": ~close}
    groups.update({label: actual == index for index, label in enumerate(LABELS)})
    training = np.asarray([row["rank"] for row in train])
    median = int(np.median(training))
    mode = int(np.bincount(training, minlength=len(LABELS)).argmax())
    metrics = {
        "budget": BUDGET,
        "rows": {"train": len(train), "validation": len(validate), "test": len(test)},
        "baseline": {
            "median": score(actual, np.full_like(actual, median), groups["overall"]),
            "mode": score(actual, np.full_like(actual, mode), groups["overall"]),
        },
        "test": {"case": [row["case"] for row in test], "signal": signals, "rank": actual},
        "arms": {},
    }
    checks = {}
    for arm in ARMS:
        lit.seed_everything(seed, workers=True)
        model = {"mae": OrdinalMae, "mse": OrdinalMse, "enum": EnumModel, "shuffled": OrdinalMae}[arm](
            preset=CONTROL.name
        )
        updates = fit(
            model, frame(train, shuffle=arm == "shuffled", seed=seed + 4), frame(validate), seed, steps, accelerator
        )
        output = model.predict(frame(test, targets=False), batch_size=256)["predictions"].to_list()
        labels = [row["level"]["content"]["value"] if arm == "enum" else row["level"]["content"] for row in output]
        predicted = np.asarray([LABELS.index(value) for value in labels], dtype=np.int64)
        inferred = all(row["level"]["inferred"] for row in output)
        states = [max(row["level"]["state"], key=row["level"]["state"].get) for row in output]
        measured = {name: score(actual, predicted, selected) for name, selected in groups.items()}
        measured.update(
            steps=updates,
            inferred=inferred,
            state_accuracy=float(np.mean(np.asarray(states) == "valued")),
            states=states,
            parameters=sum(p.numel() for p in model.parameters()),
            predictions=predicted,
        )
        if arm == "enum":
            probabilities = np.zeros((len(output), len(LABELS)), dtype=np.float64)
            for row, item in enumerate(output):
                for candidate in item["level"]["content"]["topk"]:
                    probabilities[row, LABELS.index(candidate["value"])] = candidate["probability"]
            observed = probabilities[np.arange(len(actual)), actual]
            cdf = np.cumsum(probabilities, axis=1)[:, :-1]
            truth = actual[:, None] <= np.arange(len(LABELS) - 1)
            calibration = []
            for threshold in range(len(LABELS) - 1):
                for index in range(10):
                    selected = (cdf[:, threshold] >= index / 10) & (cdf[:, threshold] < (index + 1) / 10)
                    if index == 9:
                        selected |= cdf[:, threshold] == 1
                    if selected.any():
                        calibration.append(
                            {
                                "threshold": threshold,
                                "bin": index,
                                "rows": int(selected.sum()),
                                "predicted": float(cdf[selected, threshold].mean()),
                                "observed": float(truth[selected, threshold].mean()),
                            }
                        )
            measured["probability"] = {
                "nll": float(-np.log(np.maximum(observed, 1e-12)).mean()),
                "ranked_probability_score": float(np.mean(np.sum((cdf - truth) ** 2, axis=1))),
                "threshold_calibration": calibration,
                "probabilities": probabilities,
            }
            checks["Enum probabilities sum to one"] = bool(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6))
        metrics["arms"][arm] = measured
        checks[f"{arm}: full budget retained"] = updates == BUDGET
        checks[f"{arm}: inferred output"] = inferred
        checks[f"{arm}: valued state accuracy at least 0.95"] = measured["state_accuracy"] >= 0.95
    checks["Default Ordinal held-out rank MAE below 1"] = metrics["arms"]["mae"]["overall"]["rank_mae"] < 1.0
    checks["Default Ordinal improves on shuffled signal"] = (
        metrics["arms"]["mae"]["overall"]["rank_mae"] + 0.15 < metrics["arms"]["shuffled"]["overall"]["rank_mae"]
    )
    return metrics, checks


def plot(bundle, spec):
    """Show the declared rank comparison and Enum threshold calibration."""
    import matplotlib.pyplot as plt

    from evidence.diagnostics.plots import dots

    arms = bundle.record["metrics"]["arms"]
    names = ("mae", "mse", "enum", "shuffled")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    dots(
        axes[0],
        ("Ordinal MAE", "Ordinal MSE", "Enum", "Shuffled clues"),
        {"Held-out rank MAE": [arms[name]["overall"]["rank_mae"] for name in names]},
        xlabel="Post-write rank MAE (lower is better)",
    )
    calibration = arms["enum"]["probability"]["threshold_calibration"]
    axes[1].plot([0, 1], [0, 1], linestyle=":", color="gray", label="Ideal")
    for threshold in range(4):
        bins = [item for item in calibration if item["threshold"] == threshold]
        axes[1].scatter(
            [item["predicted"] for item in bins],
            [item["observed"] for item in bins],
            s=[max(12, item["rows"] / 5) for item in bins],
            label=f"Rank at most {threshold}",
            alpha=0.75,
        )
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed frequency")
    axes[1].legend(fontsize=8, loc="best")
    axes[1].grid(alpha=0.22)
    axes[0].set_title("Held-out comparison")
    axes[1].set_title("Enum threshold calibration")
    paths = [["arms", name, "overall", "rank_mae"] for name in names]
    paths.append(["arms", "enum", "probability", "threshold_calibration"])
    return (
        fig,
        "Each calibration point is a nonempty probability bin; marker size reflects its row count. Rank errors are evaluated after writing. This synthetic task does not establish a universal objective ranking.",
        {"metric_paths": paths},
    )


if __name__ == "__main__":
    report(run)
