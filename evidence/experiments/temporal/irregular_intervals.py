# %% [markdown]
# ---
# title: Can a forecast use irregular elapsed time?
# categories:
# - Temporal
# description: An observation sequence supplies elapsed times and numeric values, and a hidden future value is
#   requested.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Elapsed-time model
#   metric:
#   - elapsed
#   - familiar
#   - normalized_rmse
#   format: number
# - label: Position-only control
#   metric:
#   - position_only
#   - familiar
#   - normalized_rmse
#   format: number
# evidence-group: Temporal
# ---
#
# Four noisy observations follow one unknown linear rate. Elapsed times are measured relative to the last
# observation, and forecast_at requests a future time in those same units. The displayed trajectory rises
# by one across four time units, so one additional unit gives 1.25. Equal sequence positions alone would
# not reveal that rate. The true answer is hidden during inference.
#
# ## Example
#
# ```yaml
# observations:
#   - {elapsed: -4.0, value: 0.0}
#   - {elapsed: -3.0, value: 0.25}
#   - {elapsed: -1.0, value: 0.75}
#   - {elapsed: 0.0, value: 1.0}
# forecast_at: 1.0
# answer: 1.25
# ```
#
# ```{typst}
# //| label: fig-evidence-p093
# //| fig-cap: "Visible inputs and the hidden prediction target for this controlled task."
# //| fig-alt: "An observation sequence supplies elapsed times and numeric values, and a hidden future value is requested."
# #tree(node("trajectory", kind: "root", children: (node("observations", kind: "branch", children: (node("elapsed", type: "Number"), node("value", type: "Number"))), node("forecast_at", type: "Number"), node("answer", kind: "target", type: "Number"),)))
# ```
#
# ## Comparison
#
# Compare the actual elapsed-time records with a separately trained model that sees ordinal positions
# -3,-2,-1,0 and a fixed next-position query. Both receive the same observed values and answers. A
# least-squares reference fits the visible elapsed-time/value pairs, establishing that the observations
# contain enough information. Evaluate familiar gaps separately from larger intervals and forecast
# horizons.
#
# ## Result
#
# {{< evidence P093 status >}}
#
# These checks define a new experiment. A full recorded run is required before claiming that the model
# meets them. Lower normalized error is better; a value of one matches the stated constant reference. The
# numerical gates are provisional and must be calibrated on separate validation seeds rather than adjusted
# after seeing acceptance-test outcomes.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P093 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P093 script >}}
#
# ### Process, controls, and held-out evaluation
#
# Each row samples a slope Uniform(-1,1), intercept Uniform(-0.5,0.5), and independent Gaussian
# observation/target noise with standard deviation 0.01. Training has three independent gaps and a forecast
# horizon between 0.25 and 1.0; shifted tests use 1.25-2.0. Four observed values are available at every
# split. Independent streams supply 4096/512 training/validation rows and 1024 rows per test condition.
# Each xs arm receives 600 AdamW updates, batch 64, learning rate 0.002, weight decay 0.01, FP32. Both
# models have identical architecture and initialization; the control explicitly erases timing information.
# Every normalized error uses the same training-mean answer reference on the corresponding test
# distribution. Larger-gap quality is a distinct provisional gate, not implied by familiar-gap accuracy.
# Exact elapsed time belongs in preprocessing and Number fields; calendar Period components are not treated
# as durations.
#
# ### Complete experiment code

# %%
"""P093: forecast an unknown rate using irregular elapsed times."""

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
import torch
from relflow.presets.base import resolve
from reporting import experiment, report

BUDGET = 600
TRAIN_ROWS = 4096
TEST_ROWS = 1024


def records(*, rows: int, seed: int, shifted: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        gaps = rng.uniform(1.25, 2.0, 3) if shifted else rng.uniform(0.25, 1.0, 3)
        elapsed = np.r_[-np.cumsum(gaps)[::-1], 0.0]
        slope, intercept = rng.uniform(-1, 1), rng.uniform(-0.5, 0.5)
        horizon = float(rng.uniform(1.25, 2.0) if shifted else rng.uniform(0.25, 1.0))
        values = intercept + slope * elapsed + rng.normal(0, 0.01, 4)
        yield {
            "observations": [{"elapsed": float(t), "value": float(v)} for t, v in zip(elapsed, values, strict=True)],
            "forecast_at": horizon,
            "answer": float(intercept + slope * horizon + rng.normal(0, 0.01)),
        }


def position_only(rows: list[dict]) -> list[dict]:
    """Erase elapsed-time information while preserving all observations and answers."""
    return [
        {
            "observations": [
                {"elapsed": float(index - 3), "value": item["value"]} for index, item in enumerate(row["observations"])
            ],
            "forecast_at": 1.0,
            "answer": row["answer"],
        }
        for row in rows
    ]


def build() -> rf.Model:
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
            observations=rf.Branch(length=4, reduction=None, elapsed=rf.Number, value=rf.Number),
            forecast_at=rf.Number,
            answer=rf.Number(mask=True, objective="mse"),
        )
    )
    model.preset = control_preset
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    output = model.predict(
        pa.Table.from_pylist([{key: value for key, value in row.items() if key != "answer"} for row in rows]),
        batch_size=128,
    )["predictions"].to_pylist()
    return np.asarray([row["answer"]["content"] for row in output])


def least_squares(rows: list[dict]) -> np.ndarray:
    result = []
    for row in rows:
        x = np.asarray([item["elapsed"] for item in row["observations"]])
        y = np.asarray([item["value"] for item in row["observations"]])
        centered = x - x.mean()
        slope = float(np.dot(centered, y) / np.dot(centered, centered))
        result.append(float(y.mean() + slope * (row["forecast_at"] - x.mean())))
    return np.asarray(result)


def score(output: np.ndarray, rows: list[dict], mean: float) -> dict:
    target = np.asarray([row["answer"] for row in rows])
    rmse = float(np.sqrt(np.mean((output - target) ** 2)))
    return {"rmse": rmse, "normalized_rmse": rmse / float(np.sqrt(np.mean((target - mean) ** 2)))}


def fit(model: rf.Model, train: list[dict], validate: list[dict], *, seed: int, budget: int, accelerator: str) -> int:
    """Own the fixed optimizer budget and independent synthetic splits."""
    data = rf.ArrowDataModule(
        train=pa.Table.from_pylist(train),
        validate=pa.Table.from_pylist(validate),
        seed=seed,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    model.fit(
        data,
        batch_size=64,
        learning_rate=0.002,
        weight_decay=0.01,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=budget,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    model.eval()
    return int(model.trainer.global_step)


@experiment("P093", seed=9301)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train = list(records(rows=TRAIN_ROWS, seed=seed * 100 + 1))
    validate = list(records(rows=512, seed=seed * 100 + 2))
    tests = {
        "familiar": list(records(rows=TEST_ROWS, seed=seed * 100 + 3)),
        "long_intervals": list(records(rows=TEST_ROWS, seed=seed * 100 + 4, shifted=True)),
    }
    mean = float(np.mean([row["answer"] for row in train]))
    metrics, checks = {}, {}
    for actual_time in (True, False):
        lit.seed_everything(seed, workers=True)
        model = build()
        transformed_train = train if actual_time else position_only(train)
        transformed_validate = validate if actual_time else position_only(validate)
        completed = fit(
            model,
            transformed_train,
            transformed_validate,
            seed=seed,
            budget=BUDGET if steps is None else min(steps, BUDGET),
            accelerator=accelerator,
        )
        before = deepcopy(model.state_dict())
        name = "elapsed" if actual_time else "position_only"
        arm = {"steps": completed}
        checks[f"{name}: full budget completed"] = completed == BUDGET
        for condition, rows in tests.items():
            visible = rows if actual_time else position_only(rows)
            output = predict(model, visible)
            arm[condition] = score(output, rows, mean)
            checks[f"{name}/{condition}: finite predictions"] = bool(np.isfinite(output).all())
            if actual_time:
                oracle = score(least_squares(rows), rows, mean)
                arm[condition]["least_squares"] = oracle
                checks[f"{condition}: visible-time oracle RMSE below 0.06"] = oracle["rmse"] < 0.06
        checks[f"{name}: evaluation preserves full state"] = all(
            (
                torch.equal(value, model.state_dict()[key])
                if isinstance(value, torch.Tensor)
                else value == model.state_dict()[key]
            )
            for key, value in before.items()
        )
        metrics[name] = arm
    for condition, maximum in (("familiar", 0.35), ("long_intervals", 0.60)):
        actual = metrics["elapsed"][condition]["normalized_rmse"]
        control = metrics["position_only"][condition]["normalized_rmse"]
        checks[f"{condition}: elapsed-time normalized RMSE below {maximum}"] = actual < maximum
        checks[f"{condition}: elapsed times improve RMSE by at least 15 percent"] = actual < 0.85 * control
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
