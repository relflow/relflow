# %% [markdown]
# ---
# title: Can an average lose the total?
# categories:
# - Hierarchical statistics
# description: Predict both average and total after an average-only branch with item attention disabled; the branch's
#   count lane keeps the total recoverable.
# evidence-readout:
# - label: Average prediction
#   metric:
#   - mean_nrmse
#   format: error
# - label: Total prediction
#   metric:
#   - total_nrmse
#   format: error
# - label: 'Predicted total, six copies (correct: 7.5)'
#   metric:
#   - paired_total_predictions
#   - 1
#   format: number
# evidence-group: Structure
# evidence-diagram:
#   caption: Amber cards are hidden prediction targets. Repeated collections keep their fields together.
#   alt: Model tree with record, transactions, amount, mean_amount, global_total. Amber cards are hidden prediction
#     targets. Repeated collections keep their fields together.
# evidence-review:
#   question: Can relflow distinguish lists with the same average but different totals?
#   modeling: Repeat the same amount one to six times and hide both the average and total. Disable item attention
#     and summarize the collection with Mean, which also carries a count signal. Compare the learned total with a
#     constant predictor on held-out records.
#   comparison:
#     metric: RMSE
#     kind: Baseline reference
#     conditions:
#     - label: Constant-prediction baseline
#       path:
#       - total_baseline_rmse
#       format: number
#     - label: Predicted total
#       path:
#       - total_rmse
#       format: number
#     definition: Root mean squared prediction error, in the target’s original units. Zero is an exact prediction.
#     direction: Lower error is better
#   note: A separate pair uses one versus six copies of the same amount. This implementation tests the count-augmented
#     Mean summary, not a plain average that discards count.
#   description: Learn separate averages and totals from repeated values summarized with a count signal.
# evidence-plot:
#   mode: regression
#   target: global_total
#   focus: Can the same representation support both a mean and a total?
#   comparison: Inspect the two readouts and the matched one-copy/six-copy probes.
#   limit: Mean and total require different responses to repetition.
#   additional_targets:
#   - mean_amount
# evidence-cost:
#   baseline: 12
#   candidate: 12
# ---
#
# ## Example
#
# The model predicts two hidden quantities from the same collection: its average amount and its total. A
# single `1.25` gives both answers as `1.25`, as illustrated. Six copies of that amount would keep the
# average at `1.25` but change the total to `7.5`. Generated records contain one to six copies of one common
# amount, so the number of copies is the essential extra information needed for the total.
#
# ```yaml
# transactions:
#   - amount: 1.25
# mean_amount: 1.25
# global_total: 1.25
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```typst
# #tree(node("record", kind: "root", children: (
#   node("transactions", kind: "branch", repeated: true, width: 120pt, children: (
#     node("amount", type: "Number"),
#   )),
#   node("mean_amount", kind: "target", type: "Number",),
#   node("global_total", kind: "target", type: "Number",),
# )))
# ```
#
# ## Comparison
#
# Item attention is disabled and the branch uses Mean reduction. The average content of identical values is
# unchanged by duplication, while the branch's count lane can still distinguish how many items were present.
# The explicit one-versus-six probe tests whether the downstream prediction uses that retained count. It
# does not give the model a separate user-authored count feature or require the average itself to change.
#
# Compare one amount with six copies of the same amount. Their averages match, but their totals differ.
#
# ## Result
#
# The two normalized errors compare average and total predictions with their respective constant training-
# target guesses. The additional readout shows the predicted total for the concrete six-copy probe, whose
# mathematical target is `7.5`; the paired error is available in the details. A successful total prediction
# therefore should not be read as evidence that averaging destroys all cardinality. The experiment is
# restricted to equal values within each record and counts from one through six.
#
# The summary keeps the count: one and six copies produce different totals that track the correct values, while the average stays shared.
#
# ## Method
#
# ### How it works
#
# The transaction branch uses `attention=None, reduction=rf.Mean()`. Its
# identical encoded amounts are averaged without preceding branch attention
# introducing position information, and Mean adds a learned direction scaled
# by the number of present items over the branch capacity. The parent sees the
# same averaged content for one and six repeats and reads their counts from
# that lane, so it can recover their different totals.
#
# The synthetic training process varies the number of repeated equal amounts
# from one to six and varies the shared amount across observations. After 300
# deterministic steps, evaluation checks the average and the total on 512
# held-out rows and compares predictions for the explicit one-versus-six pair.
# Training and validation contain 1,024 and 256 independently generated rows.
#
# ### Remaining work
#
# Broaden the value patterns and nested partitions. The family still needs a
# nested `global_total` task and a `largest_session_average` target, then three
# paired core seeds and at least ten calibration seeds for the learned gates.
#
# Use [preprocessing](/advanced/data-modules/) for exact business
# arithmetic. For learned hierarchy compression, the
# [nested Attention case](attention-preserves-nested-cardinality.html) tests a
# route that retains cardinality information through attention instead.
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
# conditions = ["varying-collections"]
# ///

"""P041: learn an average and a total from repeated equal values through Mean's count lane.

With branch attention disabled, Mean averages one and six identical inputs to
the same content and tells them apart only through its count lane. A
downstream sum must read that lane to recover their different totals.
"""

import sys
from collections.abc import Iterator
from functools import partial
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
import torch
from reporting import experiment, report


def records(*, rows: int, seed: int) -> Iterator[dict]:
    """Yield variable-length bags of equal values with mean and sum targets."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        count = int(rng.integers(1, 7))
        amount = float(rng.uniform(0.5, 1.5))
        yield {
            "transactions": [{"amount": amount} for _ in range(count)],
            "mean_amount": amount,
            "global_total": count * amount,
        }


def prediction(model: rf.Model, rows: list[dict], source: str, target: str) -> np.ndarray:
    """Predict one hidden scalar from its visible repeated context."""
    inputs = [{source: row[source]} for row in rows]
    output = pl.DataFrame(model.predict(inputs)).to_dicts()
    return np.asarray([row["predictions"][target]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


class Transaction(rf.Branch):
    amount = rf.Number


class PredictionModel(rf.Model):
    transactions = Transaction(length=6, attention=None, reduction=rf.Mean())
    mean_amount = rf.Number(mask=True, objective="mse")
    global_total = rf.Number(mask=True, objective="mse")


@experiment("P041", seed=2599)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    data_seed = (seed - 2599) % (2**32)
    train = partial(records, rows=1024, seed=data_seed + 1)
    validate = partial(records, rows=256, seed=data_seed + 2)
    test = list(records(rows=512, seed=data_seed + 3))
    model = PredictionModel(d_model=24, n_layers=1, n_heads=4)
    data = rf.SyntheticDataModule(train=train, validate=validate, seed=seed)
    model.fit(
        data,
        batch_size=64,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.003),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
        accelerator=accelerator,
        max_epochs=-1,
        max_steps=steps if steps is not None else 300,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer = model.trainer

    training = list(train())
    scores = {}
    for target in ("mean_amount", "global_total"):
        actual = np.asarray([row[target] for row in test])
        baseline = rmse(actual, float(np.mean([row[target] for row in training])))
        measured = rmse(actual, prediction(model, test, "transactions", target))
        scores[target] = {"baseline_rmse": baseline, "rmse": measured, "nrmse": measured / baseline}
    paired = [
        {"transactions": [{"amount": 1.25}]},
        {"transactions": [{"amount": 1.25} for _ in range(6)]},
    ]
    paired_total = prediction(model, paired, "transactions", "global_total")
    correct = np.asarray([1.25, 7.5])
    return {
        "steps": trainer.global_step,
        "mean_baseline_rmse": scores["mean_amount"]["baseline_rmse"],
        "mean_rmse": scores["mean_amount"]["rmse"],
        "mean_nrmse": scores["mean_amount"]["nrmse"],
        "total_baseline_rmse": scores["global_total"]["baseline_rmse"],
        "total_rmse": scores["global_total"]["rmse"],
        "total_nrmse": scores["global_total"]["nrmse"],
        "paired_total_predictions": paired_total.tolist(),
        "paired_total_error": (paired_total - correct).tolist(),
    }, {
        "Mean normalized RMSE is below 0.25": scores["mean_amount"]["nrmse"] < 0.25,
        "Total normalized RMSE is below 0.35": scores["global_total"]["nrmse"] < 0.35,
        "Six equal values predict at least four times the one-value total": bool(
            paired_total[1] > 4.0 * paired_total[0] > 0.0
        ),
        "Paired totals land within 0.5 of their correct values": bool(np.all(np.abs(paired_total - correct) < 0.5)),
    }


if __name__ == "__main__":
    report(run)
