# %% [markdown]
# ---
# title: Can it total contributions already calculated?
# description: Supply each item’s contribution to a total, instead of its separate value and weight.
# evidence-readout:
# - label: Total prediction
#   metric:
#   - measured
#   - nrmse
#   format: error
# evidence-group: Aggregation
# evidence-review:
#   question: Can relflow total contributions already calculated?
#   modeling: Compute value × weight for each of six items before loading the record. Give only these contributions
#     to the model and hide their sum. Compare held-out total error with a constant predictor.
#   comparison:
#     kind: Baseline reference
#     metric: RMSE
#     conditions:
#     - label: Constant-prediction baseline
#       path:
#       - measured
#       - baseline_rmse
#       format: number
#     - label: Trained model
#       path:
#       - measured
#       - rmse
#       format: number
#     definition: Root mean squared prediction error, in the target’s original units. Zero is an exact prediction.
#     direction: Lower error is better
#   note: This diagnostic supplies the multiplication step. P014 tests whether the model can learn it from separate
#     values and weights.
# evidence-plot:
#   mode: regression
#   target: weighted_sum
#   focus: Can the model sum already prepared weighted contributions?
#   comparison: Compare held-out weighted sums with the actual prepared-contribution totals.
#   limit: This control supplies multiplication in the input and isolates the pooling problem.
# evidence-cost:
#   baseline: 19
#   candidate: 19
# ---
#
# Every input number is already an item's contribution to a weighted total. The separate value and
# weight have been multiplied before the record reaches the model. The hidden task is therefore to
# add six signed contributions, rather than discover their original pairings or learn
# multiplication.
#
# ## Example
#
# ```yaml
# items:
#   - {contribution: 0.4}
#   - {contribution: -0.4}
#   - {contribution: 0.3}
#   - {contribution: 0.1}
#   - {contribution: -0.2}
#   - {contribution: 0.3}
# weighted_sum: 0.5
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# In the illustration, 0.4 and −0.4 cancel. The remaining 0.3, 0.1, −0.2, and 0.3 sum to 0.5. A
# contribution could have come from many different value–weight pairs; those original pairs are not
# needed once their product is known.
#
# ## Comparison
#
# Compare predictions with the correct totals on new lists of the same length.
#
# Training and test use independently generated lists, always with six items. Compare this
# diagnostic with the raw value-and-weight experiment: both predict a weighted total, but this
# version supplies the difficult item-level interaction directly. There is no pairing corruption
# here because the original value and weight fields are absent. It tests whether the remaining
# numerical information can be combined successfully.
#
# ## Result
#
# The simpler summation task works. Fixed length means averaging and a fixed multiplier could also
# solve it; varying lengths are not tested.
#
# The error percentage compares held-out totals with a constant training-mean prediction. A low
# score establishes useful fixed-length prediction, not a particular internal algorithm: multiplying
# an accurately decoded average by six would also solve this task. Variable-length bags and
# complete-list duplication would be needed to distinguish that shortcut from a rule that scales
# across counts.
#
# ## Method
#
# ### How it works
#
# The generator computes each `contribution = value * weight` before the
# record reaches the model. Learned `rf.Attention` reductions on the item branch
# and root combine six contributions into the masked sum target.
#
# This is a diagnostic control for the
# [raw-pair evidence](raw-value-weight-sum.html). Success establishes fixed-length
# summation of supplied products. It does not establish that the model learned
# multiplication or same-item value–weight binding.
#
# ### Remaining work
#
# Repeat the control across seeds and test variable lengths and complete-item
# duplication. Keep the raw-pair evidence alongside it so that supplying products does
# not conceal a missing learned interaction.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "lightning",
#     "numpy",
#     "polars",
#     "pyyaml>=6",
#     "relflow>=0.5.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "computation"
# types = ["Number"]
# conditions = ["fresh-records"]
# ///

"""A supplied item-local product isolates fixed-width sum reduction.

Attention learns a fixed-width sum after products are supplied.

Run: uv run evidence/experiments/run.py P015 --wheel /path/to/relflow-0.5.0-py3-none-any.whl"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
import torch
from reporting import experiment, report
from visibility import TARGET

ITEMS = 6


def weighted_records(*, rows: int, length: int, seed: int) -> Iterator[dict]:
    """Draw random value/weight pairs and expose raw fields or their products."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        values = rng.uniform(-1.0, 1.0, size=length)
        weights = rng.uniform(0.15, 1.5, size=length)
        contributions = values * weights
        items = [{"contribution": float(contribution)} for contribution in contributions]
        yield {
            "items": items,
            "weighted_sum": float(contributions.sum()),
            "weighted_mean": float(contributions.sum() / weights.sum()),
        }


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = pl.DataFrame(model.predict(inputs))["predictions"].to_list()
    return np.asarray([row["weighted_sum"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["weighted_sum"] for row in test], dtype=np.float64)
    baseline_rmse = rmse(actual, float(np.asarray([row["weighted_sum"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline_rmse, "nrmse": measured / baseline_rmse}


class Item(rf.Branch):
    contribution = rf.Number


class PredictionModel(rf.Model):
    items = Item(length=ITEMS, n_layers=2, reduction=rf.Attention(n_layers=2))
    weighted_sum = rf.Number(**TARGET, objective="mse")


@experiment("P015", seed=3300)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(weighted_records(rows=768, length=ITEMS, seed=seed + 1))
    test = list(weighted_records(rows=384, length=ITEMS, seed=seed + 3))
    model = PredictionModel(d_model=32, n_layers=2, n_heads=4, reduction=rf.Attention(n_layers=2))
    datamodule = rf.SyntheticDataModule(
        train=lambda: weighted_records(rows=768, length=ITEMS, seed=seed + 1),
        validate=lambda: weighted_records(rows=192, length=ITEMS, seed=seed + 2),
        seed=seed,
    )
    model.fit(
        datamodule,
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=500 if steps is None else min(steps, 500),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer

    # Evaluate held-out answers and retain their original labels in corruption controls.
    measured = score(train=train, test=test, predicted=prediction(model, test))
    metrics = {"measured": measured, "steps": trainer.global_step}
    checks = {"Contribution sum nRMSE below 0.25": bool(measured["nrmse"] < 0.25)}
    return metrics, checks


if __name__ == "__main__":
    report(run)
