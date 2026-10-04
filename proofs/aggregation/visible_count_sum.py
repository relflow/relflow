# %% [markdown]
# ---
# title: Can a visible count help recover a total?
# categories:
# - Cardinality generalization
# proof-id: P004
# description: Every item has the same amount. Give the model the item count as an extra input and ask for
#   the total.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: Total prediction
#   metric:
#   - measured
#   - nrmse
#   format: error
# ---
#
# ## Example
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
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-proof-visible-count-sum
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, item_count, items, amount, total. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("item_count", type: "Number"),
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("amount", type: "Number"),
#   )),
#   node("total", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Keep the amount fixed while changing the count. The answers must differ even though an average-only branch sees the same value.
#
# ## Result
#
# {{< proof P004 status >}}
#
# The extra count supplies information the average loses. This test uses repeated equal amounts, rather than arbitrary mixed lists.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P004 evidence >}}
#
# ### Run this experiment
#
# {{< proof P004 script >}}
#
# ### How it works
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
# ### Remaining work
#
# Repeat across seeds and test nonidentical bags, missing items, and nested
# collections. [Unseen visible counts](visible-count-unseen-lengths.html) have a
# separate proof; this case checks only the trained count range.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

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


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3605)

# %% [markdown]
# </details>
