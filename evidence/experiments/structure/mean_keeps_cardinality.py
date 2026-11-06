# %% [markdown]
# ---
# title: Can an average lose the total?
# categories:
# - Hierarchical statistics
# description: Predict both average and total after an average-only branch with item attention disabled; the branch's
#   count lane keeps the total recoverable.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
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
# ---
#
# ## Example
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
# ```{typst}
# //| label: fig-evidence-structure-mean-keeps-cardinality
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, transactions, amount, mean_amount, global_total. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
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
# Compare one amount with six copies of the same amount. Their averages match, but their totals differ.
#
# ## Result
#
# {{< evidence P041 status >}}
#
# The summary keeps the count: one and six copies produce different totals that track the correct values, while the average stays shared.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P041 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P041 script >}}
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
# Use [preprocessing](../../guides/preprocessing.qmd) for exact business
# arithmetic. For learned hierarchy compression, the
# [nested Attention case](attention-preserves-nested-cardinality.html) tests a
# route that retains cardinality information through attention instead.
#
# ### Complete experiment code
#

# %%
"""P041: learn an average and a total from repeated equal values through Mean's count lane.

With branch attention disabled, Mean averages one and six identical inputs to
the same content and tells them apart only through its count lane. A
downstream sum must read that lane to recover their different totals.
"""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
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
    output = model.predict(inputs).to_pylist()
    return np.asarray([row["predictions"][target]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


@experiment("P041", seed=2599)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    data_seed = (seed - 2599) % (2**32)
    train = partial(records, rows=1024, seed=data_seed + 1)
    validate = partial(records, rows=256, seed=data_seed + 2)
    test = list(records(rows=512, seed=data_seed + 3))
    model = rf.Model(
        d_model=24,
        n_layers=1,
        n_heads=4,
        transactions=rf.Branch(length=6, attention=None, reduction=rf.Mean(), amount=rf.Number),
        mean_amount=rf.Number(mask=True, objective="mse"),
        global_total=rf.Number(mask=True, objective="mse"),
    )
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

# %% [markdown]
# </details>
