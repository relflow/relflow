"""Shared protocol and user guide for hierarchical-statistics proofs.

Claim
-----
With pre-reduction attention disabled, Mean is appropriate for a scalar average
but deliberately erases the count of repeated identical encoded tokens. One
fixed-width learned reduction can approximate a fixed-length sum. Attention's
additive mass lane retains enough local cardinality signal for the tested
``max(session sum)`` process.

What to do
----------
Choose Mean only when one normalized encoded-token summary is sufficient.
Remember that it is not a raw Number reducer, and that branch attention can
encode order or count before the mean. Verify that every reduction level
retains everything a parent-level target needs. Treat every learned reduction
as an empirical approximation and test that its output width preserves the
sufficient statistics required by its parent.

Omitting known subtotals is specific to this learned-hierarchy benchmark. If a
nested sum or maximum is deterministic business logic and must be exact,
calculate it in an :class:`relflow.Preprocessor` or the application; the
learned routes below are approximations and diagnostics, not accounting
primitives.

What not to do
--------------
Do not use the isolated ``attention=None`` plus Mean route when a parent also
needs count or distribution detail. Do not assume the normalized attention
path alone retains count: the passing Attention route relies on its separate
additive mass/count lane.

Why the nested counterexample is identifiable
---------------------------------------------
Matched rows have identical flattened amounts but session lengths
``(2, 2, 2)`` and ``(1, 4, 1)``. Their required answers differ even though a
cardinality-blind inner representation is identical.

Protocol and gates
------------------
The tests train deterministic CPU models on independently generated complete
groups. The Mean guide uses 300 steps, the one-level sum control uses 400, and
the nested Attention variant uses 80 steps. They compare test RMSE with a
train-mean baseline or the optimal oracle after cardinality and boundaries are
erased.

Mean average nRMSE must be below 0.25, and duplicate-value inputs of different
lengths must produce the same representation-dependent prediction. One-level
sum nRMSE must be below 0.25. One-output Attention must beat the flattened
oracle by 25% and recover at least 75% of the true pair delta.

Status
------
Provisional. Mean/count erasure, one-level sum, and nested Attention pass for
one seed.

Current evidence
----------------
* With item attention disabled, Mean learns the scalar average but produces
  identical downstream predictions for one versus six equal values,
  demonstrating deliberate count erasure for that controlled route.
* A one-level Attention branch learns the fixed-width six-value sum.
* One-output Attention recovers about 84% of the regrouping delta through its
  additive mass lane despite the normalized attention path.

Remaining work
--------------
* Add nested, regrouping-invariant ``global_total``; the current sum control is
  only one level.
* Add the specified ``largest_session_average`` target.
* Broaden value patterns and session partitions after retaining the isolated
  cardinality rung.
* Test multiple Attention output counts and nested capacity ranges.

Promotion criteria
------------------
Repeat the nested gates over three paired core seeds and at least ten
calibration seeds. Attention must continue to beat the flattened oracle by 25%
and recover at least 75% of the true pair delta.

Run all cases with::

    uv run pytest -n 0 proofs/structure/hierarchical_statistics -q
"""

from __future__ import annotations

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa

import relflow as rf

TRANSACTIONS = 6
PARTITIONS = ((2, 2, 2), (1, 4, 1))


def records(*, pairs: int, seed: int) -> pa.Table:
    """Create paired rows that differ only in their nested boundaries."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for _ in range(pairs):
        scale = float(rng.uniform(0.4, 1.6))
        flattened = [scale] * TRANSACTIONS
        for lengths in PARTITIONS:
            sessions: list[dict[str, object]] = []
            offset = 0
            for length in lengths:
                amounts = flattened[offset : offset + length]
                sessions.append({"transactions": [{"amount": amount} for amount in amounts]})
                offset += length
            rows.append(
                {
                    "sessions": sessions,
                    "global_total": TRANSACTIONS * scale,
                    "largest_session_total": max(lengths) * scale,
                }
            )
    return pa.Table.from_pylist(rows)


def sum_records(*, rows: int, seed: int) -> pa.Table:
    """Create fixed-width bags whose independently drawn values must all contribute."""
    rng = np.random.default_rng(seed)
    observations: list[dict[str, object]] = []
    for _ in range(rows):
        amounts = rng.uniform(-1.0, 1.0, size=TRANSACTIONS)
        observations.append(
            {
                "transactions": [{"amount": float(amount)} for amount in amounts],
                "global_total": float(amounts.sum()),
            }
        )
    return pa.Table.from_pylist(observations)


def mean_records(*, rows: int, seed: int) -> pa.Table:
    """Create variable-length bags where average survives but count does not."""
    rng = np.random.default_rng(seed)
    observations: list[dict[str, object]] = []
    for _ in range(rows):
        count = int(rng.integers(1, TRANSACTIONS + 1))
        amount = float(rng.uniform(0.5, 1.5))
        observations.append(
            {
                "transactions": [{"amount": amount}] * count,
                "mean_amount": amount,
                "global_total": count * amount,
            }
        )
    return pa.Table.from_pylist(observations)


def flatten(table: pa.Table) -> pa.Table:
    """Erase session boundaries while retaining ordered transactions."""
    rows = []
    for row in table.to_pylist():
        transactions = [transaction for session in row["sessions"] for transaction in session["transactions"]]
        rows.append({"transactions": transactions, "global_total": row["global_total"]})
    return pa.Table.from_pylist(rows)


def column(table: pa.Table, name: str) -> np.ndarray:
    """Return one Arrow column as a dense float array."""
    return np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64)


def prediction(model: rf.Model, table: pa.Table, source: str, target: str) -> np.ndarray:
    """Run inference and unwrap one numerical prediction field."""
    output = model.predict(table.select([source]))
    predicted = output["predictions"].combine_chunks().field(f"record/{target}").field("content")
    return np.asarray(predicted.to_numpy(zero_copy_only=False), dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    """Calculate root mean squared error."""
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def trainer(steps: int) -> lit.Trainer:
    """Build the deterministic CPU trainer shared by these cases."""
    return lit.Trainer(
        accelerator="cpu",
        max_epochs=-1,
        max_steps=steps,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )


def data(model: rf.Model, train: pa.Table, validate: pa.Table, seed: int) -> rf.ArrowDataModule:
    """Build a deterministic in-memory Arrow data module."""
    return rf.ArrowDataModule(
        model=model,
        train=train,
        validate=validate,
        seed=seed,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
