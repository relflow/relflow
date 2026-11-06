# %% [markdown]
# ---
# title: Can irregular hierarchies preserve which values belong together?
# categories:
# - Structure
# description: Account records contain days, sessions, transactions, and a hidden largest-session total.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Familiar structures
#   metric:
#   - familiar
#   - normalized_rmse
#   format: number
# - label: Unseen lengths
#   metric:
#   - unseen_lengths
#   - normalized_rmse
#   format: number
# evidence-group: Structure
# ---
#
# An account contains days, each day contains sessions, and each session contains transactions. The hidden
# answer is the largest session total anywhere in that account. The displayed answer is explanatory and is
# excluded from prediction inputs. In the example, the session totals are five and twelve. Moving the same
# amounts into sessions [2, 7] and [3, 5] changes the answer to nine without changing the amount multiset
# or session sizes.
#
# ## Example
#
# ```yaml
# days:
#   - sessions:
#       - transactions: [{amount: 2.0}, {amount: 3.0}]
#       - transactions: [{amount: 5.0}, {amount: 7.0}]
# answer: 12.0
# ```
#
# ```{typst}
# //| label: fig-evidence-p083
# //| fig-cap: "Visible inputs and the hidden prediction target for this controlled task."
# //| fig-alt: "Account records contain days, sessions, transactions, and a hidden largest-session total."
# #tree(node("account", kind: "root", children: (node("days", kind: "branch", children: (node("sessions", kind: "branch", children: (node("transactions", kind: "branch", children: (node("amount", type: "Number"),)),)),)), node("answer", kind: "target", type: "Number"),)))
# ```
#
# ## Comparison
#
# Generate paired accounts with the same amount multiset and identical collection lengths, but shuffle
# which transactions belong to each session. A comparison oracle that sees only the unordered amounts must
# give the same answer to both members; its optimal answer for this two-member pair is their target
# average. Session membership is therefore necessary to beat that reference. Evaluate new length
# combinations separately from familiar ones.
#
# ## Result
#
# {{< evidence P083 status >}}
#
# These checks define a new experiment. A full recorded run is required before claiming that the model
# meets them. Lower normalized error is better; a value of one matches the stated constant reference. The
# numerical gates are provisional and must be calibrated on separate validation seeds rather than adjusted
# after seeing acceptance-test outcomes.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P083 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P083 script >}}
#
# ### Process, controls, and held-out evaluation
#
# The schema has three unbounded collection axes. The generator nevertheless bounds each source: training
# uses one or two days, one or two sessions per day, and one through three transactions per session; the
# shifted test uses two or three days, one through three sessions, and two through four transactions.
# Amounts are independent Uniform(0,1) draws. Each pair reuses the same source amounts under two
# assignments; no summary or count field is supplied. Independent streams provide 4096/512
# training/validation rows and 1024 rows per test condition. A single xs model receives 600 AdamW updates,
# batch 64, learning rate 0.002, weight decay 0.01, FP32, without checkpoint selection. The paired oracle
# is deliberately optimistic about grouping-erased information, not a deployed baseline. All target
# differences, per-condition errors, actual steps, and frozen-state checks are retained. A second
# evaluation inserts null item placeholders, empty collections, and null child collections without adding
# any amount. Their synthetic contribution is defined as zero; this is an alignment and invariance
# intervention, not a claim about the value of unobserved real-world transactions. Expanded sources remain
# explicitly bounded by six slots per axis.
#
# ### Complete experiment code

# %%
"""P083: retain heterogeneous session membership through three ragged collection levels."""

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
import torch
from reporting import experiment, report

BUDGET = 600
TRAIN_ROWS = 4096
TEST_ROWS = 1024


def records(*, rows: int, seed: int, shifted: bool = False) -> Iterator[dict]:
    """Yield regrouping pairs with identical amount multisets and collection lengths."""
    if rows % 2:
        raise ValueError("P083 requires an even row count for complete regrouping pairs")
    rng = np.random.default_rng(seed)
    for _ in range(rows // 2):
        day_count = int(rng.integers(2, 4)) if shifted else int(rng.integers(1, 3))
        sizes = [
            rng.integers(2 if shifted else 1, 5 if shifted else 4, int(rng.integers(1, 4 if shifted else 3))).tolist()
            for _ in range(day_count)
        ]
        count = sum(sum(day) for day in sizes)
        values = rng.uniform(0.0, 1.0, count)
        for order in (np.arange(count), rng.permutation(count)):
            cursor = 0
            days = []
            totals = []
            for day in sizes:
                sessions = []
                for length in day:
                    amounts = values[order[cursor : cursor + length]]
                    cursor += length
                    sessions.append({"transactions": [{"amount": float(amount)} for amount in amounts]})
                    totals.append(float(amounts.sum()))
                days.append({"sessions": sessions})
            yield {"days": days, "answer": max(totals)}


def build() -> rf.Model:
    return rf.Model.xs(
        reduction=rf.Attention(),
        days=rf.Branch(
            length=None,
            overflow=None,
            reduction=rf.Attention(),
            sessions=rf.Branch(
                length=None,
                overflow=None,
                reduction=rf.Attention(),
                transactions=rf.Branch(length=None, overflow=None, reduction=rf.Attention(), amount=rf.Number),
            ),
        ),
        answer=rf.Number(mask=True, objective="mse"),
    )


def structural_holes(rows: list[dict]) -> list[dict]:
    """Insert empty collections and null placeholders without adding any amount."""
    result = []
    for row in rows:
        days = [None, {"sessions": []}, {"sessions": None}]
        for day in row["days"]:
            sessions = [None, {"transactions": []}, {"transactions": None}]
            sessions.extend({"transactions": [None, *session["transactions"], None]} for session in day["sessions"])
            days.append({"sessions": sessions})
        result.append({"days": days, "answer": row["answer"]})
    return result


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{"days": row["days"]} for row in rows]
    output = model.predict(pa.Table.from_pylist(visible), batch_size=64)["predictions"].to_pylist()
    return np.asarray([row["answer"]["content"] for row in output], dtype=np.float64)


def score(predicted: np.ndarray, rows: list[dict], mean: float) -> dict:
    target = np.asarray([row["answer"] for row in rows])
    paired = target.reshape(-1, 2)
    oracle = np.repeat(paired.mean(axis=1), 2)
    rmse = float(np.sqrt(np.mean((predicted - target) ** 2)))
    baseline = float(np.sqrt(np.mean((mean - target) ** 2)))
    oracle_rmse = float(np.sqrt(np.mean((oracle - target) ** 2)))
    delta = np.diff(paired, axis=1).ravel()
    delta_rmse = float(np.sqrt(np.mean((np.diff(predicted.reshape(-1, 2), axis=1).ravel() - delta) ** 2)))
    return {
        "rmse": rmse,
        "normalized_rmse": rmse / baseline,
        "flat_bag_oracle_rmse": oracle_rmse,
        "pair_delta_rmse": delta_rmse,
        "pair_delta_reference": float(np.sqrt(np.mean(delta**2))),
    }


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


@experiment("P083", seed=8301)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train = list(records(rows=TRAIN_ROWS, seed=seed * 100 + 1))
    validate = list(records(rows=512, seed=seed * 100 + 2))
    lit.seed_everything(seed, workers=True)
    model = build()
    completed = fit(
        model,
        train,
        validate,
        seed=seed,
        budget=BUDGET if steps is None else min(steps, BUDGET),
        accelerator=accelerator,
    )
    before = deepcopy(model.state_dict())
    mean = float(np.mean([row["answer"] for row in train]))
    metrics = {
        "steps": completed,
        "source_maxima": {"days": 3, "sessions_per_day": 3, "transactions_per_session": 4},
        "source_maxima_with_holes": {"days": 6, "sessions_per_day": 6, "transactions_per_session": 6},
    }
    checks = {"Full training budget completed": completed == BUDGET}
    for name, shifted in (("familiar", False), ("unseen_lengths", True)):
        rows = list(records(rows=TEST_ROWS, seed=seed * 100 + 3 + int(shifted), shifted=shifted))
        predicted = predict(model, rows)
        measured = score(predicted, rows, mean)
        with_holes = predict(model, structural_holes(rows))
        measured["structural_holes"] = score(with_holes, rows, mean)
        measured["structural_holes"]["mean_prediction_drift"] = float(np.mean(np.abs(with_holes - predicted)))
        metrics[name] = measured
        checks[f"{name}: predictions are finite"] = bool(np.isfinite(predicted).all())
        checks[f"{name}: normalized RMSE below 0.65"] = measured["normalized_rmse"] < 0.65
        checks[f"{name}: beats grouping-erased paired oracle by 20 percent"] = (
            measured["rmse"] < 0.8 * measured["flat_bag_oracle_rmse"]
        )
        checks[f"{name}: pair-difference error below 75 percent of zero-difference reference"] = (
            measured["pair_delta_rmse"] < 0.75 * measured["pair_delta_reference"]
        )
        checks[f"{name}: structural-hole normalized RMSE below 0.75"] = (
            measured["structural_holes"]["normalized_rmse"] < 0.75
        )
        checks[f"{name}: structural-hole predictions are finite"] = bool(np.isfinite(with_holes).all())
    checks["Evaluation preserves full model state"] = all(
        (
            torch.equal(value, model.state_dict()[name])
            if isinstance(value, torch.Tensor)
            else value == model.state_dict()[name]
        )
        for name, value in before.items()
    )
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
