# %% [markdown]
# ---
# title: Preserving Counts Across a Hierarchy
# categories:
# - Hierarchical statistics
# proof-id: P040
# description: Attention reductions can preserve enough local cardinality to predict
#   the largest session total from nested transactions.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# Six equal amounts can form three equal sessions or one large session between
# two small ones. Their global total stays the same, but their largest session
# total changes. This proof asks the hierarchy to preserve that distinction.
#
# {{< proof P040 status >}}
#
# ## Insights
#
# **How values are grouped can change the answer even when the values themselves stay the same.** The
# matched observations contain the same six amounts but divide them into different session sizes, changing
# the largest session total.
#
# Attention reduction includes an additive sum and count contribution alongside normalized attention. The
# model beats the best flattened-information prediction and recovers much of the true difference between the
# paired observations. The difference metric uses absolute magnitudes; the separate RMSE gate checks
# closeness to actual targets.
#
# This supports the tested route through two partitions of equal values. It does not establish exact nested
# arithmetic or arbitrary capacity. Varying amounts within sessions and testing more partitions remain
# necessary extensions.
#
# ## Setup

# %%
"""P040: retain local counts before selecting the largest session total.

Paired records contain the same six values partitioned as (2, 2, 2) and
(1, 4, 1). Flattening loses the answer. Attention must beat that erased-input
oracle and recover the difference between the paired targets.
"""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P040"

# %% [markdown]
# ## Examples
#
# `largest_session_total` is a hidden Number target. The generator also includes
# `global_total`, but this model does not define or use that field.
#
# ### One large session
#
# ```yaml
# sessions:
#   - transactions:
#       - amount: 0.5
#   - transactions:
#       - amount: 0.5
#       - amount: 0.5
#       - amount: 0.5
#       - amount: 0.5
#   - transactions:
#       - amount: 0.5
# global_total: 3.0
# largest_session_total: 2.0
# ```
#
# The middle session contains four amounts and has the largest subtotal.
#
# ### The same amounts in equal sessions
#
# ```yaml
# sessions:
#   - transactions:
#       - amount: 0.5
#       - amount: 0.5
#   - transactions:
#       - amount: 0.5
#       - amount: 0.5
#   - transactions:
#       - amount: 0.5
#       - amount: 0.5
# global_total: 3.0
# largest_session_total: 1.0
# ```
#
# Flattening either record gives the same six values. Only the session
# boundaries explain why this target is half the first record's target.
#
# ### The uneven partition at a larger scale
#
# ```yaml
# sessions:
#   - transactions:
#       - amount: 1.0
#   - transactions:
#       - amount: 1.0
#       - amount: 1.0
#       - amount: 1.0
#       - amount: 1.0
#   - transactions:
#       - amount: 1.0
# global_total: 6.0
# largest_session_total: 4.0
# ```
#
# The generator varies the common amount between pairs. The correct target
# must reflect both value and local count. The displayed totals are exact
# ground truth; the reported learned recovery is approximate.
#
# ## Synthetic data and controls


# %%
def records(*, pairs: int, seed: int) -> Iterator[dict]:
    """Yield adjacent regroupings of the same six equal transaction values."""
    rng = np.random.default_rng(seed)
    for _ in range(pairs):
        amount = float(rng.uniform(0.4, 1.6))
        for lengths in ((2, 2, 2), (1, 4, 1)):
            yield {
                "sessions": [{"transactions": [{"amount": amount} for _ in range(length)]} for length in lengths],
                "global_total": 6 * amount,
                "largest_session_total": max(lengths) * amount,
            }


def prediction(model: rf.Model, rows: list[dict], source: str, target: str) -> np.ndarray:
    """Predict one hidden scalar from its visible repeated context."""
    inputs = [{source: row[source]} for row in rows]
    output = model.predict(inputs).to_pylist()
    return np.asarray([row["predictions"][f"record/{target}"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-structure-attention-preserves-nested-cardinality
# //| fig-cap: "Root, sessions, and transactions each use one-output Attention reduction. Local capacities preserve the tested groupings; the largest total is hidden."
# //| fig-alt: "Record uses one-output Attention and contains up to three sessions, each containing up to four transactions with Number amounts; both repeated levels also use one-output Attention, and the Number largest session total target is always hidden from input."
# #tree(node("record", kind: "root", width: 150pt, body: [
#   - *Reduction:* Attention
#   - *Outputs:* 1
# ], children: (
#   node("sessions", kind: "branch", repeated: true, width: 150pt, body: [
#     - *Reduction:* Attention
#     - *Outputs:* 1
#     - *Capacity:* 3 sessions
#   ], children: (
#     node("transactions", kind: "branch", repeated: true, width: 150pt, body: [
#       - *Reduction:* Attention
#       - *Outputs:* 1
#       - *Capacity:* 4 items
#     ], children: (
#       node("amount", type: "Number"),
#     )),
#   )),
#   node("largest_session_total", width: 180pt, kind: "target", type: "Number", body: [
#     - *Input:* always hidden
#   ]),
# )))
# ```
#
# ## How it works
#
# The session, transaction, and root contexts all use Attention reduction.
# The reducer adds a projected sum and an explicit count contribution alongside
# normalized attention. This additional path lets equal tokens carry different
# mass when their repetition count changes.
#
# Each pair uses the same six amounts, with session sizes `(2, 2, 2)` versus
# `(1, 4, 1)`. The amount varies between pairs. The model trains on 128 pairs,
# validates on 32, and tests on 64 independently generated pairs, after 80
# deterministic steps.
#
# The comparison oracle sees only flattened values. It must predict the same
# answer for both members of a pair; their target average is its optimal
# squared-error prediction. Beating it requires retaining session structure.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    data_seed = (seed - 2601) % (2**32)
    train = partial(records, pairs=128, seed=data_seed + 44)
    validate = partial(records, pairs=32, seed=data_seed + 55)
    test = list(records(pairs=64, seed=data_seed + 66))
    flattened = [[item for session in row["sessions"] for item in session["transactions"]] for row in test]
    if not all(flattened[index] == flattened[index + 1] for index in range(0, len(flattened), 2)):
        raise ValueError("regrouping pairs must contain identical flattened values")

    model = rf.Model(
        d_model=24,
        n_layers=1,
        n_heads=4,
        reduction=rf.Attention(),
        batch_size=64,
        sessions=rf.Branch(
            length=3,
            reduction=rf.Attention(),
            transactions=rf.Branch(length=4, reduction=rf.Attention(), amount=rf.Number),
        ),
        largest_session_total=rf.Number(mask=True, objective="mse"),
    )
    model.optimizer = lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3)
    data = rf.SyntheticDataModule(model=model, train=train, validate=validate, seed=seed)
    trainer = lit.Trainer(
        accelerator=accelerator,
        max_epochs=-1,
        max_steps=steps if steps is not None else 80,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model=model, datamodule=data)

    targets = np.asarray([row["largest_session_total"] for row in test])
    predictions = prediction(model, test, "sessions", "largest_session_total")
    paired_targets = targets.reshape(-1, 2)
    flat_oracle = np.repeat(paired_targets.mean(axis=1), 2)
    oracle_rmse = rmse(targets, flat_oracle)
    test_rmse = rmse(targets, predictions)
    predicted_delta = float(np.abs(np.diff(predictions.reshape(-1, 2), axis=1)).mean())
    true_delta = float(np.abs(np.diff(paired_targets, axis=1)).mean())
    return {
        "steps": trainer.global_step,
        "flat_oracle_rmse": oracle_rmse,
        "test_rmse": test_rmse,
        "predicted_pair_delta": predicted_delta,
        "true_pair_delta": true_delta,
        "recovered_pair_delta_fraction": predicted_delta / true_delta,
    }, {
        "Attention improves over the flattened oracle by at least 25%": bool(
            np.isfinite(test_rmse) and test_rmse < oracle_rmse * 0.75
        ),
        "Attention recovers at least 75% of the pair delta": predicted_delta >= true_delta * 0.75,
    }


# %% [markdown]
# ## Evidence
#
# {{< proof P040 evidence >}}
#
# ## Remaining work
#
# Add nested global totals and largest-session averages. Vary amounts within
# sessions, test more partitions, and vary Attention output counts and nested
# capacities. Repeat the nested gates over three paired core seeds and at least
# ten calibration seeds without weakening either threshold.
#
# ## Reproduce
#
# {{< proof P040 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=2601)
