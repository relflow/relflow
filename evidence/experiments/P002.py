# %% [markdown]
# ---
# title: Can it add longer lists than it saw in training?
# description: Train on short lists, then ask for totals of longer lists.
# evidence-readout:
# - label: Familiar lengths
#   metric:
#   - in_range_score
#   - nrmse
#   format: error
# - label: Longer lists
#   metric:
#   - unseen_score
#   - nrmse
#   format: error
# evidence-group: Aggregation
# evidence-review:
#   question: Can relflow add longer lists than relflow saw in training?
#   modeling: Train on lists containing one to six signed amounts, with their total hidden. Test on new lists containing
#     seven to ten amounts. The model receives the individual amounts without a separate count field. Compare its
#     error on familiar and longer lengths.
#   comparison:
#     kind: Recorded conditions
#     metric: RMSE
#     conditions:
#     - label: Familiar lengths
#       path:
#       - in_range_score
#       - rmse
#       format: number
#     - label: Longer lists
#       path:
#       - unseen_score
#       - rmse
#       format: number
#     definition: Root mean squared prediction error, in the target’s original units. Zero is an exact prediction.
#     direction: Lower error is better
#   note: The branch can hold these lists; this tests learned generalization to lengths seven through ten, not arbitrary-length
#     arithmetic.
# evidence-plot:
#   mode: regression
#   target: total
#   focus: Does a learned sum continue to work on longer bags?
#   comparison: Separate training-range bags, unseen lengths, duplicated bags, and equal-value probes.
#   limit: Only the recorded lengths are tested; neural sums are approximate.
#   labels:
#     in_range: 'Familiar lengths: 1–6 items'
#     unseen: 'Unseen lengths: 7–10 items'
#     short: Short bags before duplication
#     duplicate(short): The same short bags duplicated
#     probes: Equal-value count probes
# evidence-cost:
#   baseline: 58
#   candidate: 57
# ---
#
# The question is whether a learned summation rule continues to work when there are more terms to
# add. The inputs remain ordinary item amounts; no count is supplied as a separate field. Only the
# number of items changes beyond the training range.
#
# ## Example
#
# ```yaml
# items:
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
#   - {amount: 0.5}
# total: 4.0
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# The illustrated list has eight copies of 0.5, so its total is 4.0. The model trained on at most
# six items, making this an example of an unfamiliar length. Four copies would total 2.0;
# duplicating that complete four-item list must double the answer without changing the individual
# values.
#
# ## Comparison
#
# Compare familiar and unfamiliar lengths. Also repeat a whole list: its total should double.
#
# The familiar test uses lengths one through six; the longer-list test uses seven through ten.
# Separately, the experiment duplicates bags of at most five items and compares their predictions
# before and after duplication. Equal-value probes also hold the amount fixed while changing
# repetition count. Together these checks distinguish responding to extra items from merely
# producing a plausible total.
#
# ## Result
#
# The learned total extends to the tested longer lists. This does not establish exact addition or
# unlimited length.
#
# The error percentages compare each test set with a constant training-mean guess. The duplication
# check asks a different question: does doubling the input approximately double the prediction?
# Passing both supports the tested extension in length; all lists still fit a capacity of twelve,
# and no claim is made about unlimited lengths or exact arithmetic.
#
# ## Method
#
# ### How it works
#
# The schema matches the [in-range sum](attention-sum-in-range.html):
# learned `rf.Attention` reductions receive raw amounts without a supplied count.
# Training lengths are one through six, while the unseen test set uses seven
# through ten. Branch capacity is twelve, allowing these inputs without overflow.
#
# A second intervention duplicates complete bags of at most five items and checks
# that predictions approximately double. Equal-value probes at lengths one,
# three, and six isolate sensitivity to multiplicity. The capacity bound permits
# these shapes; the accuracy and intervention gates establish the learned behavior.
#
# ### Remaining work
#
# Repeat the gates across seeds, capacities, and nested branches. Heavy tails,
# high duplication, empty collections, missing values, and overflow semantics need
# separate coverage. The doubling gate allows approximation error; it is not an
# exact-sum contract.
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
# conditions = ["longer-collections", "extrapolation", "transformations", "varying-collections"]
# ///

"""Characterize sum extrapolation beyond every trained collection length.

Require unseen-length accuracy and exact complete-bag scaling together.

Run: uv run evidence/experiments/run.py P002 --wheel /path/to/relflow-0.5.0-py3-none-any.whl"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from copy import deepcopy
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

TRAIN_MAX = 6
CAPACITY = 12


def random_records(*, rows: int, seed: int, minimum: int = 1, maximum: int = TRAIN_MAX) -> Iterator[dict]:
    """Draw variable-length numerical bags and their mean and sum."""
    if not 1 <= minimum <= maximum <= CAPACITY:
        raise ValueError(f"length range must satisfy 1 <= minimum <= maximum <= {CAPACITY}")
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        length = int(rng.integers(minimum, maximum + 1))
        values = rng.uniform(-1.0, 1.0, size=length)
        row: dict[str, object] = {
            "items": [{"amount": float(value)} for value in values],
            "mean_amount": float(values.mean()),
            "total": float(values.sum()),
        }
        yield row


def duplicate(observations: list[dict]) -> list[dict]:
    """Duplicate every complete bag and update its algebraic targets."""
    rows = deepcopy(observations)
    if any(len(row["items"]) * 2 > CAPACITY for row in rows):
        raise ValueError(f"duplicated collection exceeds configured capacity {CAPACITY}")
    for row in rows:
        row["items"] = [*row["items"], *row["items"]]
        row["total"] *= 2.0
    return rows


def equal_value_probes(*, value: float, lengths: tuple[int, ...]) -> list[dict]:
    """Create matched bags that differ only in repetition count."""
    if not lengths or min(lengths) < 1 or max(lengths) > CAPACITY:
        raise ValueError(f"probe lengths must be within 1..{CAPACITY}, got {lengths!r}")
    rows: list[dict[str, object]] = []
    for length in lengths:
        row: dict[str, object] = {"items": [{"amount": value}] * length, "mean_amount": value, "total": length * value}
        rows.append(row)
    return rows


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = pl.DataFrame(model.predict(inputs))["predictions"].to_list()
    return np.asarray([row["total"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["total"] for row in test], dtype=np.float64)
    baseline = rmse(actual, float(np.asarray([row["total"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline, "nrmse": measured / baseline}


class Item(rf.Branch):
    amount = rf.Number


class PredictionModel(rf.Model):
    items = Item(length=CAPACITY, attention="mha", n_layers=2, reduction=rf.Attention(n_layers=2))
    total = rf.Number(**TARGET, objective="mse")


@experiment("P002", seed=3615)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(random_records(rows=1536, seed=seed + 1))
    in_range = list(random_records(rows=512, seed=seed + 3))
    unseen = list(random_records(rows=512, seed=seed + 4, minimum=7, maximum=10))
    short = list(random_records(rows=256, seed=seed + 5, maximum=5))
    model = PredictionModel(d_model=32, n_layers=2, n_heads=4, reduction=rf.Attention(n_layers=2))
    datamodule = rf.SyntheticDataModule(
        train=lambda: random_records(rows=1536, seed=seed + 1),
        validate=lambda: random_records(rows=384, seed=seed + 2),
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
        max_steps=1100 if steps is None else min(steps, 1100),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer

    # Evaluate held-out answers and retain their original labels in corruption controls.
    in_range_score = score(train=train, test=in_range, predicted=prediction(model, in_range))
    unseen_score = score(train=train, test=unseen, predicted=prediction(model, unseen))
    short_prediction = prediction(model, short)
    duplicated_prediction = prediction(model, duplicate(short))
    duplication_error = rmse(2.0 * short_prediction, duplicated_prediction) / float(
        np.std(np.asarray([row["total"] for row in unseen], dtype=np.float64))
    )
    probes = equal_value_probes(value=0.65, lengths=(1, 3, 6))
    probe_prediction = prediction(model, probes)
    probe_target = np.asarray([row["total"] for row in probes], dtype=np.float64)
    probe_error = rmse(probe_target, probe_prediction) / float(
        np.std(np.asarray([row["total"] for row in unseen], dtype=np.float64))
    )
    metrics = {
        "in_range_score": in_range_score,
        "unseen_score": unseen_score,
        "duplication_error": duplication_error,
        "probe_error": probe_error,
        "probe_target": probe_target.tolist(),
        "probe_prediction": probe_prediction.tolist(),
        "steps": trainer.global_step,
    }
    checks = {
        "All measurements finite": bool(
            np.isfinite([in_range_score["nrmse"], unseen_score["nrmse"], duplication_error, probe_error]).all()
        ),
        "Seen nRMSE < 0.35, unseen nRMSE < 0.30, duplication and cardinality errors < 0.20": bool(
            in_range_score["nrmse"] < 0.35
            and unseen_score["nrmse"] < 0.3
            and (duplication_error < 0.2)
            and (probe_error < 0.2)
        ),
    }
    return metrics, checks


if __name__ == "__main__":
    report(run)
