# %% [markdown]
# ---
# title: Can it add longer lists than it saw in training?
# categories:
# - Cardinality generalization
# proof-id: P002
# description: Train on short lists, then ask for totals of longer lists.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
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
# ---
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
# ```{typst}
# //| label: fig-proof-attention-sum-unseen-lengths
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, items, amount, total. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("amount", type: "Number"),
#   )),
#   node("total", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Compare familiar and unfamiliar lengths. Also repeat a whole list: its total should double.
#
# ## Result
#
# {{< proof P002 status >}}
#
# The learned total extends to the tested longer lists. This does not establish exact addition or unlimited length.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P002 evidence >}}
#
# ### Run this experiment
#
# {{< proof P002 script >}}
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
# ### Complete experiment code
#

# %%
"""Characterize sum extrapolation beyond every trained collection length.

Require unseen-length accuracy and exact complete-bag scaling together.

Run: uv run python proofs/run.py P002"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P002"
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
    if any((len(row["items"]) * 2 > CAPACITY for row in rows)):
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
    train = list(random_records(rows=1536, seed=seed + 1))
    in_range = list(random_records(rows=512, seed=seed + 3))
    unseen = list(random_records(rows=512, seed=seed + 4, minimum=7, maximum=10))
    short = list(random_records(rows=256, seed=seed + 5, maximum=5))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        items=rf.Branch(
            length=CAPACITY, attention="mha", n_layers=2, reduction=rf.Attention(n_layers=2), amount=rf.Number
        ),
        total=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        model=model,
        train=lambda: random_records(rows=1536, seed=seed + 1),
        validate=lambda: random_records(rows=384, seed=seed + 2),
        seed=seed,
    )
    trainer = lit.Trainer(
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
    trainer.fit(model, datamodule=datamodule)

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
    report(PROOF_ID, run, seed=3615)

# %% [markdown]
# </details>
