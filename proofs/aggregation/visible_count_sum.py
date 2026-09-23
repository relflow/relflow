# %% [markdown]
# ---
# title: Restoring sums with a visible count
# categories:
# - Cardinality generalization
# proof-id: P004
# description: Expose the count that a Mean branch discards and test whether the root
#   learns to combine it with item content.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P004 status >}}
#
# ## Insights
#
# **Providing an item count lets the model distinguish totals that an average alone cannot.** Every item in
# a record repeats the same amount, so the branch summary stays unchanged when only the number of copies
# changes. The ordinary root count field supplies the missing factor needed for a total.
#
# The matched probes require different predictions for the same repeated value at different counts. This
# isolates learning the amount-times-count relationship from recovering multiplicity through the collection
# itself. It is a useful diagnostic when a reduction appears to lose mass. The proof covers repeated-value
# bags and familiar counts; it does not establish sums over arbitrary mixed-value collections.
#
# ## Setup

# %%
"""A visible item count diagnoses cardinality lost by Mean.

Learn value times visible cardinality and distinguish matched probes.

Run: uv run python proofs/run.py P004"""

from __future__ import annotations

from collections.abc import Iterator

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P004"
TRAIN_MAX = 6
CAPACITY = 12

# %% [markdown]
# ## Examples
#
# Targets use `mask=True`; the labels below are supervision hidden from the encoder.
#
# ### Six repetitions
#
# ```yaml
# item_count: 6
# items:
#   - {amount: 0.7}
#   - {amount: 0.7}
#   - {amount: 0.7}
#   - {amount: 0.7}
#   - {amount: 0.7}
#   - {amount: 0.7}
# total: 4.2
# ```
#
# The visible count supplies the factor needed to turn 0.7 into a total of 4.2.
#
# ### One repetition with the same branch summary
#
# ```yaml
# item_count: 1
# items:
#   - {amount: 0.7}
# total: 0.7
# ```
#
# This matched probe has the same encoded Mean content as the six-copy record.
# Only the visible count distinguishes its total of 0.7.
#
# ### A different amount and count
#
# ```yaml
# item_count: 3
# items:
#   - {amount: -0.4}
#   - {amount: -0.4}
#   - {amount: -0.4}
# total: -1.2
# ```
#
# Three repetitions of −0.4 require −1.2. Both the repeated value and the
# visible count vary across the generated records.
#
# ## Synthetic data and controls


# %%
def random_records(*, rows: int, seed: int, minimum: int = 1, maximum: int = TRAIN_MAX) -> Iterator[dict]:
    """Draw variable-length numerical bags and their mean and sum."""
    if not 1 <= minimum <= maximum <= CAPACITY:
        raise ValueError(f"length range must satisfy 1 <= minimum <= maximum <= {CAPACITY}")
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        length = int(rng.integers(minimum, maximum + 1))
        values = np.repeat(rng.uniform(-1.0, 1.0), length)
        row: dict[str, object] = {
            "items": [{"amount": float(value)} for value in values],
            "mean_amount": float(values.mean()),
            "total": float(values.sum()),
        }
        row["item_count"] = length
        yield row


def equal_value_probes(*, value: float, lengths: tuple[int, ...]) -> list[dict]:
    """Create matched bags that differ only in repetition count."""
    if not lengths or min(lengths) < 1 or max(lengths) > CAPACITY:
        raise ValueError(f"probe lengths must be within 1..{CAPACITY}, got {lengths!r}")
    rows: list[dict[str, object]] = []
    for length in lengths:
        row: dict[str, object] = {"items": [{"amount": value}] * length, "mean_amount": value, "total": length * value}
        row["item_count"] = length
        rows.append(row)
    return rows


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"], "item_count": row["item_count"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["/total"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["total"] for row in test], dtype=np.float64)
    baseline = rmse(actual, float(np.asarray([row["total"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline, "nrmse": measured / baseline}


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-visible-count-sum
# //| fig-cap: "Mean with item attention off removes multiplicity; a visible root count supplies the missing factor."
# //| fig-alt: "Record contains repeated items with amount inputs, visible item count, and hidden total targets. Root reduction: Attention. Item reduction: Mean; capacity 12; branch attention Off."
# #tree(node("record", kind: "root", width: 150pt, body: [
#     - *Reduction:* Attention
#   ], children: (
#   node("item_count", type: "Number"),
#   node("items", kind: "branch", repeated: true, width: 155pt, body: [
#       - *Reduction:* Mean
#       - *Branch attention:* Off
#       - *Capacity:* 12 items
#     ], children: (
#     node("amount", type: "Number"),
#   )),
#   node("total", kind: "target", type: "Number"),
# )))
# ```
#
# ## How it works
#
# Every item in a record repeats one random amount. The branch uses
# `attention=None` and `rf.Mean()`, so its summary does not change with repetition
# count. A visible root `item_count` supplies the missing factor, and learned root
# attention combines both inputs to predict their product.
#
# Train and test lengths are one through six. Matched probes repeat 0.7 once or
# six times, requiring totals 0.7 and 4.2. Their predictions must separate by more
# than 2.5 and stay close to those targets. This isolates learning from a supplied
# count; [the count-free Attention proof](attention-sum-in-range.html) tests
# structural recovery of multiplicity.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(random_records(rows=1024, seed=seed + 1))
    test = list(random_records(rows=512, seed=seed + 3))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        items=rf.Branch(length=CAPACITY, attention=None, n_layers=2, reduction=rf.Mean(), amount=rf.Number),
        item_count=rf.Number,
        total=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        model=model,
        train=lambda: random_records(rows=1024, seed=seed + 1),
        validate=lambda: random_records(rows=256, seed=seed + 2),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=700 if steps is None else min(steps, 700),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=datamodule)

    # Evaluate held-out answers and retain their original labels in corruption controls.
    predicted = prediction(model, test)
    measured = score(train=train, test=test, predicted=predicted)
    probes = equal_value_probes(value=0.7, lengths=(1, TRAIN_MAX))
    probe_prediction = prediction(model, probes)
    probe_target = np.asarray([row["total"] for row in probes], dtype=np.float64)
    probe_error = rmse(probe_target, probe_prediction)
    metrics = {
        "measured": measured,
        "probe_error": probe_error,
        "probe_target": probe_target.tolist(),
        "probe_prediction": probe_prediction.tolist(),
        "steps": trainer.global_step,
    }
    checks = {
        "Visible-count nRMSE below 0.25": bool(measured["nrmse"] < 0.25),
        "Matched-count prediction gap above 2.5": bool(probe_prediction[1] - probe_prediction[0] > 2.5),
        "Matched-count RMSE below 0.35": bool(probe_error < 0.35),
    }
    return metrics, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P004 evidence >}}
#
# ## Remaining work
#
# Repeat across seeds and test nonidentical bags, missing items, and nested
# collections. [Unseen visible counts](visible-count-unseen-lengths.html) have a
# separate proof; this case checks only the trained count range.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ## Reproduce
#
# {{< proof P004 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=3605)
