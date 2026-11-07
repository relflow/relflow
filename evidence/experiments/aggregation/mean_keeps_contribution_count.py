# %% [markdown]
# ---
# title: Can an average reveal how many items there were?
# categories:
# - Weighted aggregation
# description: Use a Mean branch, with item attention disabled, to predict both an average and a total; the
#   branch's count lane carries the multiplicity a plain average erases.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Average prediction
#   metric:
#   - mean_score
#   - nrmse
#   format: error
# - label: Total prediction
#   metric:
#   - sum_score
#   - nrmse
#   format: error
# - label: 'Predicted total, six copies (correct: 4.5)'
#   metric:
#   - paired_sum
#   - 1
#   format: number
# evidence-group: Aggregation
# ---
#
# An ordinary average cannot reveal how many identical numbers were averaged. This experiment asks
# the current Mean branch for two hidden outputs at once: the average contribution and its total.
# There is no separate visible item_count field; repetition must remain available through the branch
# representation.
#
# ## Example
#
# ```yaml
# items:
#   - {contribution: 0.75}
#   - {contribution: 0.75}
#   - {contribution: 0.75}
#   - {contribution: 0.75}
#   - {contribution: 0.75}
#   - {contribution: 0.75}
# mean_contribution: 0.75
# weighted_sum: 4.5
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# Six copies of 0.75 have an average of 0.75 and a total of 4.5, as illustrated. A single copy has
# the same average but a total of 0.75. These matched cases separate knowing the shared contribution
# from knowing how many times it should be counted.
#
# ```{typst}
# //| label: fig-evidence-mean-keeps-contribution-count
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, items, contribution, mean_contribution, weighted_sum. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("contribution", type: "Number"),
#   )),
#   node("mean_contribution", width: 150pt, kind: "target", type: "Number"),
#   node("weighted_sum", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Repeat the same contribution different numbers of times. The average stays fixed while the correct
# total changes.
#
# Training and test draw counts from one through six and a fresh shared contribution for each
# record. Item attention is disabled, so repetitions do not create new interactions before
# averaging. Mean's count lane distinguishes bags whose averaged content is identical. The
# experiment measures both outputs and includes the one-copy versus six-copy probe; accurate
# averages alone would not show that multiplicity survived.
#
# ## Result
#
# {{< evidence P013 status >}}
#
# The model learns both the average and a total that scales with the count. Mean's count lane
# preserves the multiplicity its plain average would erase.
#
#
# The two error percentages use separate constant-guess baselines for the average and total. The
# paired-total checks also require correctly separated predictions near 0.75 and 4.5. Success
# demonstrates usable count information for repeated equal contributions within the trained range;
# it does not establish unseen-length extrapolation or summation of arbitrary mixed contributions.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P013 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P013 script >}}
#
# ### How it works
#
# The item branch uses `attention=None` and `rf.Mean()`. Every item in a
# record repeats the same randomly drawn contribution, so averaging its
# identical encoded tokens gives the same content for one repetition or six.
# Mean adds a learned direction scaled by the number of present items over the
# branch capacity, and that lane is the only difference the parent sees between
# the two. The total head learns to multiply the shared contribution by it.
#
# The model should learn the contribution average and the total. A separate
# probe compares total predictions for one and six copies of 0.75, whose
# correct totals are 0.75 and 4.5: the six-copy prediction must exceed the
# one-copy prediction, and each must land within 0.5 of its correct value.
#
# ### Remaining work
#
# Check the count lane across seeds and at lengths the training data never
# showed. Count-preserving routes without the lane are covered separately by
# [Attention sums](attention-sum-unseen-lengths.html) and
# [visible count with Mean](visible-count-sum.html).
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""Mean keeps the count a weighted sum needs when item cardinality varies.

Learn the raw average and the total while a no-attention Mean carries
multiplicity through its count lane.

Run: uv run python evidence/experiments/run.py P013"""

from __future__ import annotations

from collections.abc import Iterator

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import experiment, report

ITEMS = 6


def repeated_contribution_records(*, rows: int, seed: int) -> Iterator[dict]:
    """Draw variable counts of one repeated contribution with its average and total."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        count = int(rng.integers(1, ITEMS + 1))
        contribution = float(rng.uniform(-1.25, 1.25))
        yield {
            "items": [{"contribution": contribution}] * count,
            "mean_contribution": contribution,
            "weighted_sum": count * contribution,
        }


def prediction(model: rf.Model, observations: list[dict], target: str) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row[target]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray, target: str) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row[target] for row in test], dtype=np.float64)
    baseline_rmse = rmse(actual, float(np.asarray([row[target] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline_rmse, "nrmse": measured / baseline_rmse}


@experiment("P013", seed=3309)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(repeated_contribution_records(rows=1024, seed=seed + 1))
    test = list(repeated_contribution_records(rows=512, seed=seed + 3))
    model = rf.Model(
        d_model=24,
        n_layers=1,
        n_heads=4,
        items=rf.Branch(length=ITEMS, attention=None, reduction=rf.Mean(), contribution=rf.Number),
        mean_contribution=rf.Number(mask=True, objective="mse"),
        weighted_sum=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        train=lambda: repeated_contribution_records(rows=1024, seed=seed + 1),
        validate=lambda: repeated_contribution_records(rows=256, seed=seed + 2),
        seed=seed,
    )
    model.fit(
        datamodule,
        batch_size=64,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.003),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=300 if steps is None else min(steps, 300),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer

    # Evaluate held-out answers, then probe one versus six copies of the same contribution.
    mean_score = score(
        train=train, test=test, predicted=prediction(model, test, "mean_contribution"), target="mean_contribution"
    )
    sum_score = score(train=train, test=test, predicted=prediction(model, test, "weighted_sum"), target="weighted_sum")
    paired = [{"items": [{"contribution": 0.75}]}, {"items": [{"contribution": 0.75}] * ITEMS}]
    paired_sum = prediction(model, paired, "weighted_sum")
    correct = np.asarray([0.75, 0.75 * ITEMS])
    metrics = {
        "mean_score": mean_score,
        "sum_score": sum_score,
        "paired_sum": paired_sum.tolist(),
        "paired_sum_error": (paired_sum - correct).tolist(),
        "steps": trainer.global_step,
    }
    checks = {
        "Mean contribution nRMSE below 0.25": bool(mean_score["nrmse"] < 0.25),
        "Weighted sum nRMSE below 0.35": bool(sum_score["nrmse"] < 0.35),
        "Six copies predict a larger total than one copy": bool(paired_sum[1] > paired_sum[0] + 1.0),
        "Paired totals land within 0.5 of their correct values": bool(np.all(np.abs(paired_sum - correct) < 0.5)),
    }
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
