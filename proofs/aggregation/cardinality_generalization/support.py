"""Shared protocol and user guide for cardinality-generalization proofs.

Claim
-----
A collection reduction must preserve every statistic that a downstream answer
needs.  A mathematical arithmetic mean is invariant to duplicating a complete
bag, whereas sum is not.  A sum-capable route therefore has to preserve total
mass or cardinality as well as a normalized content statistic.

What to do
----------
Use :class:`relflow.Mean` when an average of encoded item tokens is an
appropriate inductive bias, and verify that the resulting model learns the raw
statistic the application needs. ``Mean`` averages encoded tokens; it does not
directly calculate the arithmetic mean of raw Number values.

Use :class:`relflow.Attention` for a learned variable-cardinality aggregate.
Branch Attention combines its normalized learned-query summary with two
unnormalized sufficient-statistic lanes: a masked sum of the evidence before
sequence attention and an explicit present-token count, both scaled by the
branch's fixed capacity. Those lanes are added after the query pool's final
normalization, so downstream branches can distinguish bags with the same
empirical distribution but different mass. This behavior is automatic; it
does not require a count field or another reduction parameter.

Exposing ``item_count`` as an ordinary visible :class:`relflow.Number` remains
a useful diagnostic for a Mean branch. Number embeddings retain a monotone
standardized scalar lane alongside their Fourier features. The count-visible
proof therefore isolates whether the ordinary root and decoder path can learn
``mean * count`` from colocated visible inputs without asking users to author a
query, join, or latent feature. If count or sum is deterministic business logic
and must be exact, calculate it in an :class:`relflow.Preprocessor` or the
application instead of replacing the accounting identity with a learned
approximation.

What not to do
--------------
Do not use Mean for a target that changes when a whole bag is duplicated unless
an earlier encoder is deliberately expected to retain cardinality.  In the
tested ``attention="none"`` route, duplicating identical encoded tokens leaves
the Mean summary exactly unchanged, so no decoder can recover the removed
multiplicity.  Do not infer support for arbitrary lengths from a low
random-split error: ordinary random splits contain the same length range on both
sides and can hide length-specific interpolation. Attention now preserves mass,
but it still learns a predictive approximation rather than executing exact
addition. More Attention outputs add representation capacity; they do not turn
the decoder into an accounting engine or remove the need for extrapolation and
metamorphic tests.

Why the controls are identifiable
----------------------------------
The mean proof draws independent random values and applies exact interventions:
joint permutation and whole-bag duplication.  With item attention disabled,
Mean sees the same average of encoded Number tokens after either intervention,
so prediction drift should be numerical noise.

The sum proof draws random variable-length bags.  A complete-bag duplicate has
the same empirical distribution but twice the required sum.  Equal-valued
probe pairs isolate cardinality still further: one value repeated ``a`` and
``b`` times has the same Mean representation but sums in ratio ``a:b``.

The visible-count control uses equal-valued bags, so its branch Mean contains
the same encoded-value summary at every cardinality and the root ``item_count``
contains the only changing factor.  This separates learned decoding of the
product from structural count recovery.  Its held-out test uses lengths seven
through ten after training only on one through six; that split measures genuine
length extrapolation rather than memorization of seen cardinalities.

Metamorphic semantics
---------------------
These identities are part of the proof contract:

* jointly permuting items leaves mean and sum unchanged;
* duplicating every item leaves mean unchanged;
* duplicating every item doubles sum; and
* repeating one equal value ``k`` times multiplies its sum by ``k``.

Protocol and gates
------------------
All models train deterministically on CPU from independent Arrow tables.
Training bags contain one through six items; configured capacity is twelve so
that duplication and unseen lengths fit without overflow.  The in-range sum
test uses independent values uniform on ``[-1, 1]``.  Mean and visible-count
controls use values bounded away from numerical extremes.  Accuracy is RMSE
divided by the RMSE of the training-target mean.

Mean must reach 0.25 nRMSE and retain its predictions under item permutation
and complete-bag duplication to within ``1e-6`` target standard deviations.
The ordinary one-output Attention interpolation rung must beat 0.35 nRMSE in
range. Its unseen-length case must also beat 0.30 nRMSE, keep complete-bag
duplication error below 0.20 target standard deviations, and keep the
equal-value cardinality probe below 0.20.
The count-visible control must learn the in-range product
below 0.25 nRMSE; on unseen counts it must stay below 0.40 nRMSE and 0.20
duplication error. All capability claims use ordinary hard assertions.

Status
------
Provisional established suite. All five deterministic cases currently clear
their hard gates at their fixed seeds: Mean invariance, Attention interpolation
and unseen-cardinality scaling, and the visible-count Mean control in and beyond
the trained length range. This is one-seed evidence for learned behavior, not
an exact-sum guarantee.

Current evidence
----------------
* Mean clears its 0.25 accuracy gate and preserves complete-item permutation
  and duplication to the required numerical precision at seed 3600.
* One-output Attention clears its seen-length accuracy and permutation gates at
  seed 3609. At seed 3615 it also clears every unseen-length accuracy,
  duplication, and equal-value cardinality gate.
* Visible count plus Mean clears the in-range product controls at seed 3605 and
  the unseen-count accuracy and duplication gates at seed 3621.

Remaining work
--------------
* Verify that the automatic Attention mass lanes remain stable across widths,
  capacities, output counts, and nested branches.
* Add an explicit semantic sum/count reduction if users need an architectural
  algebraic contract rather than a learned aggregate.
* Exercise empty collections, missing values, overflow policies, and nested
  collections separately; each changes the meaning of cardinality.
* Repeat unseen-length tests with signed, heavy-tailed, and highly duplicated
  values.
* Compare ``reduction=None`` with mass-preserving Attention at matched parameter
  and token budgets.

Promotion criteria
------------------
Repeat every gate over at least three core seeds and ten lightweight
calibration seeds.  Mean invariances in the no-item-attention route must remain
at numerical precision; sum must preserve item permutation, respond linearly
to duplication, and stay below 0.40 nRMSE for lengths beyond the training
maximum.  The explicit-count control must remain substantially easier than any
route that erased count.

Run all cases with::

    uv run pytest -n 0 proofs/aggregation/cardinality_generalization -q
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch

import relflow as rf

TRAIN_MAX = 6
CAPACITY = 12
Reduction = Literal["mean", "attention"]


def random_records(
    *,
    rows: int,
    seed: int,
    minimum: int = 1,
    maximum: int = TRAIN_MAX,
    repeated: bool = False,
    include_count: bool = False,
) -> pa.Table:
    """Draw variable-length numerical bags and their mean and sum."""

    if not (1 <= minimum <= maximum <= CAPACITY):
        raise ValueError(f"length range must satisfy 1 <= minimum <= maximum <= {CAPACITY}")
    rng = np.random.default_rng(seed)
    observations: list[dict[str, object]] = []
    for _ in range(rows):
        length = int(rng.integers(minimum, maximum + 1))
        if repeated:
            values = np.repeat(rng.uniform(-1.0, 1.0), length)
        else:
            values = rng.uniform(-1.0, 1.0, size=length)
        row: dict[str, object] = {
            "items": [{"amount": float(value)} for value in values],
            "mean_amount": float(values.mean()),
            "total": float(values.sum()),
        }
        if include_count:
            row["item_count"] = length
        observations.append(row)
    return pa.Table.from_pylist(observations)


def duplicate(table: pa.Table, *, include_count: bool) -> pa.Table:
    """Duplicate every complete bag and update its algebraic targets."""

    rows = table.to_pylist()
    if any(len(row["items"]) * 2 > CAPACITY for row in rows):
        raise ValueError(f"duplicated collection exceeds configured capacity {CAPACITY}")
    for row in rows:
        row["items"] = [*row["items"], *row["items"]]
        row["total"] *= 2.0
        if include_count:
            row["item_count"] *= 2
    return pa.Table.from_pylist(rows, schema=table.schema)


def permute(table: pa.Table, *, seed: int) -> pa.Table:
    """Jointly permute complete items without changing any target."""

    rng = np.random.default_rng(seed)
    rows = table.to_pylist()
    for row in rows:
        items = row["items"]
        row["items"] = [items[index] for index in rng.permutation(len(items))]
    return pa.Table.from_pylist(rows, schema=table.schema)


def equal_value_probes(*, value: float, lengths: tuple[int, ...], include_count: bool = False) -> pa.Table:
    """Create matched bags that differ only in repetition count."""

    if not lengths or min(lengths) < 1 or max(lengths) > CAPACITY:
        raise ValueError(f"probe lengths must be within 1..{CAPACITY}, got {lengths!r}")
    rows: list[dict[str, object]] = []
    for length in lengths:
        row: dict[str, object] = {
            "items": [{"amount": value}] * length,
            "mean_amount": value,
            "total": length * value,
        }
        if include_count:
            row["item_count"] = length
        rows.append(row)
    return pa.Table.from_pylist(rows)


def model(*, reduction: Reduction, include_count: bool, target: Literal["mean_amount", "total"]) -> rf.Model:
    """Build one public-schema model for a collection statistic."""

    if reduction == "mean":
        configured_reduction: rf.ReductionConfig = rf.Mean()
    else:
        configured_reduction = rf.Attention(n_layers=2)
    root_fields: dict[str, object] = {target: rf.Number(mask=True, objective="mse")}
    if include_count:
        root_fields["item_count"] = rf.Number
    return rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=1e-3),
        items=rf.Branch(
            length=CAPACITY,
            attention="mha" if reduction == "attention" else "none",
            n_layers=2,
            reduction=configured_reduction,
            amount=rf.Number,
        ),
        **root_fields,
    )


def data(configured: rf.Model, train: pa.Table, validate: pa.Table, *, seed: int) -> rf.ArrowDataModule:
    """Build the deterministic in-memory Arrow data module."""

    return rf.ArrowDataModule(
        model=configured,
        train=train,
        validate=validate,
        seed=seed,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )


def trainer(*, steps: int) -> lit.Trainer:
    """Build a quiet deterministic CPU trainer."""

    return lit.Trainer(
        accelerator="cpu",
        max_epochs=-1,
        max_steps=steps,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )


def prediction(configured: rf.Model, table: pa.Table, target: str) -> np.ndarray:
    """Run inference and unwrap one numerical prediction field."""

    columns = ["items"]
    if "item_count" in table.column_names:
        columns.append("item_count")
    output = configured.predict(table.select(columns))
    field = output["predictions"].combine_chunks().field(f"record/{target}")
    return np.asarray(field.field("content").to_numpy(zero_copy_only=False), dtype=np.float64)


def column(table: pa.Table, name: str) -> np.ndarray:
    """Return one Arrow column as a dense float array."""

    return np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    """Calculate root mean squared error."""

    return float(np.sqrt(np.mean(np.square(actual - predicted))))


@dataclass(frozen=True)
class Score:
    """Accuracy relative to the training-target mean."""

    rmse: float
    baseline_rmse: float
    nrmse: float


def score(*, train: pa.Table, test: pa.Table, predicted: np.ndarray, target: str) -> Score:
    """Normalize test RMSE by the constant training-mean baseline."""

    actual = column(test, target)
    baseline = rmse(actual, float(column(train, target).mean()))
    measured = rmse(actual, predicted)
    return Score(rmse=measured, baseline_rmse=baseline, nrmse=measured / baseline)


def diagnostics(label: str, measured: Score) -> str:
    """Render one compact assertion diagnostic."""

    return f"{label}: rmse={measured.rmse:.4f}, baseline={measured.baseline_rmse:.4f}, nrmse={measured.nrmse:.4f}"
