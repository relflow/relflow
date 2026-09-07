"""Shared protocol and user guide for weighted-aggregation proofs.

Claim
-----
For a bounded collection, RelFlow can infer the same-coordinate product of
each item's independently drawn ``value`` and ``weight`` and then aggregate
those products. The intended general contract is that users should not need
to derive ``contribution = value * weight`` merely to make the relationship
learnable.

The supplied-contribution case is an important diagnostic control, however:
it distinguishes failure to multiply and align sibling item fields from
failure to reduce already-computed contributions.

What to do
----------
Use an explicit learned reduction for a bounded weighted sum, and retain both
``value`` and ``weight`` on the same item branch. The one-seed proof establishes
that this schema can learn; require broader calibration before treating its
threshold as a production guarantee. A derived ``contribution`` in an
:class:`relflow.Preprocessor` remains a useful diagnostic rung when a harder
weighted task does not converge.

Omitting that derived field is a constraint of this modeling hypothesis, not
production advice against exact feature engineering. When a product or sum is
a contractual business calculation, compute it exactly in an
:class:`relflow.Preprocessor` or the application and validate that result; a
learned decoder remains an approximation.

:class:`relflow.Mean` averages encoded item tokens rather than raw Number
values. Use it only when that normalized representation is appropriate and a
held-out proof establishes the required target average. If a later target needs
a sum, count, denominator, or unnormalized mass, keep that information through
the reduction instead.

What not to do
--------------
Do not treat a one-seed, bounded one-output Attention result as a universal
multiplication guarantee simply because both fields are present. Do not use
Mean for a variable-cardinality sum when no earlier encoder retains
cardinality: in the no-item-attention counterexample, repeating an identical
encoded contribution leaves its Mean summary unchanged. Increasing
``Attention(n_outputs=...)`` adds capacity but does not by itself establish
coordinate binding or arithmetic semantics.

Why the controls are identifiable
----------------------------------
In the raw case every value and weight is independently random, so neither
field alone predicts the target. Permuting weights within each held-out row
preserves both field marginals while breaking only their item-wise pairing.

The decomposed control exposes the exact products and asks only for their
fixed-width sum. The cardinality counterexample repeats the same supplied
contribution either once or six times. With item attention disabled, Mean
produces the same average of identical encoded contribution tokens for both
inputs even though the required sums differ by a factor of six.

Metamorphic semantics
---------------------
The proof treats these identities as part of the behavioral contract:

* jointly permuting complete items must leave both weighted sum and weighted
  mean unchanged;
* multiplying every weight by the same positive constant must multiply a
  weighted sum by that constant but leave a weighted mean unchanged;
* duplicating every complete item must double a weighted sum but leave a
  weighted mean unchanged; and
* independently permuting weights must usually change either answer because
  it breaks same-coordinate value/weight binding.

The learned fixed-width sum exercises the first, second, and fourth
identities. The variable-cardinality weighted mean exercises all four except
the sum side of duplication. A variable-cardinality sum duplication gate is
listed as remaining work because the current fixed-width positive proof has no
unused branch capacity for duplicated inputs.

Protocol and gates
------------------
All examples use independent train, validation, and test draws and
deterministic CPU training. The fixed-width cases contain six items. Values
are uniform on ``[-1, 1]`` and positive weights are uniform on ``[0.15, 1.50]``.
The variable-cardinality weighted mean uses two through six items and
log-uniform weights on ``[0.03, 3.00]``. Performance is test RMSE divided by
the RMSE of the training-target mean.

The supplied-contribution control must reach nRMSE below 0.25. The desired
raw value/weight path must reach nRMSE below 0.30, and permuting held-out
weights must worsen nRMSE by at least 0.35. Its joint-permutation drift and
positive-scaling equivariance error must remain small relative to target
scale. The variable-cardinality weighted mean must reach 0.40 nRMSE and retain
its prediction under joint permutation, positive weight scaling, and complete
item duplication. The Mean-based model must learn the arithmetic mean target
from supplied contributions below 0.25 nRMSE while producing equal sum
predictions for one and six identical contributions.

Status
------
Provisional. Fixed-width raw weighted sum, variable-cardinality raw weighted
mean, reduction of supplied products, and the Mean cardinality counterexample
pass their gates for one deterministic seed. They still need the promotion
matrix below.

Current evidence
----------------
* A model given only supplied products learns their six-item sum below 0.25
  nRMSE.
* The same public schema learns the end-to-end six-item weighted sum below
  0.30 nRMSE, and permuting weights across items worsens nRMSE by more than
  0.35. Joint permutation and uniform positive weight scaling also satisfy
  their metamorphic gates. This establishes use of item-local pairing, not
  merely the marginals.
* A raw weighted mean over two through six items passes its accuracy,
  permutation, positive-weight-scaling, and duplication gates.
* Mean learns the supplied contribution average below 0.25 nRMSE and produces
  identical downstream sum predictions for one and six equal contributions.

Remaining work
--------------
* Repeat raw same-coordinate multiplication across the promotion seed matrix.
* Add a variable-cardinality weighted-sum duplication gate with explicit
  count-preserving reduction.
* Add group-conditioned weighted reductions over interleaved categories.
* Test zero, negative, missing, and extreme weights explicitly.
* Test extrapolation beyond trained collection lengths and duplicate items.
* Compare token-preserving ``reduction=None`` with explicit compression once
  record-local mixing is part of the architecture contract.

Promotion criteria
------------------
Repeat every accuracy and intervention gate over at least three core seeds
and ten lightweight calibration seeds. The raw model must remain sensitive to
pairing corruption, the supplied-product control must remain below 0.25
nRMSE, and the Mean invariance must hold to numerical precision.

Run all cases with::

    uv run pytest -n 0 proofs/aggregation/weighted_aggregation -q
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch

import relflow as rf

ITEMS = 6
Source = Literal["raw", "contribution"]
Target = Literal["weighted_sum", "weighted_mean"]


def weighted_records(*, rows: int, length: int, seed: int, source: Source) -> pa.Table:
    """Draw random value/weight pairs and expose raw fields or their products."""

    rng = np.random.default_rng(seed)
    observations: list[dict[str, object]] = []
    for _ in range(rows):
        values = rng.uniform(-1.0, 1.0, size=length)
        weights = rng.uniform(0.15, 1.50, size=length)
        contributions = values * weights
        if source == "raw":
            items = [
                {"value": float(value), "weight": float(weight)} for value, weight in zip(values, weights, strict=True)
            ]
        else:
            items = [{"contribution": float(contribution)} for contribution in contributions]
        observations.append(
            {
                "items": items,
                "weighted_sum": float(contributions.sum()),
                "weighted_mean": float(contributions.sum() / weights.sum()),
            }
        )
    return pa.Table.from_pylist(observations)


def variable_weighted_mean_records(*, rows: int, seed: int) -> pa.Table:
    """Draw variable-width rows with broad positive weights."""

    rng = np.random.default_rng(seed)
    observations: list[dict[str, object]] = []
    for _ in range(rows):
        length = int(rng.integers(2, ITEMS + 1))
        values = rng.uniform(-1.0, 1.0, size=length)
        weights = np.exp(rng.uniform(np.log(0.03), np.log(3.0), size=length))
        observations.append(
            {
                "items": [
                    {"value": float(value), "weight": float(weight)}
                    for value, weight in zip(values, weights, strict=True)
                ],
                "weighted_mean": float(np.dot(values, weights) / weights.sum()),
            }
        )
    return pa.Table.from_pylist(observations)


def repeated_contribution_records(*, rows: int, seed: int) -> pa.Table:
    """Draw variable counts of repeated products for the Mean counterexample."""

    rng = np.random.default_rng(seed)
    observations: list[dict[str, object]] = []
    for _ in range(rows):
        count = int(rng.integers(1, ITEMS + 1))
        contribution = float(rng.uniform(-1.25, 1.25))
        observations.append(
            {
                "items": [{"contribution": contribution}] * count,
                "mean_contribution": contribution,
                "weighted_sum": count * contribution,
            }
        )
    return pa.Table.from_pylist(observations)


def permute_weights(table: pa.Table, *, seed: int) -> pa.Table:
    """Break value/weight pairing within rows while preserving both marginals."""

    rng = np.random.default_rng(seed)
    rows = table.to_pylist()
    changed = 0
    for row in rows:
        items = row["items"]
        permutation = rng.permutation(len(items))
        if np.array_equal(permutation, np.arange(len(items))):
            permutation = np.roll(permutation, 1)
        shuffled = [items[index]["weight"] for index in permutation]
        changed += sum(item["weight"] != weight for item, weight in zip(items, shuffled, strict=True))
        row["items"] = [
            {"value": item["value"], "weight": weight} for item, weight in zip(items, shuffled, strict=True)
        ]
    if changed == 0:
        raise AssertionError("weight permutation did not change any item-local associations")
    return pa.Table.from_pylist(rows, schema=table.schema)


def permute_items(table: pa.Table, *, seed: int) -> pa.Table:
    """Jointly permute complete items without changing any target."""

    rng = np.random.default_rng(seed)
    rows = table.to_pylist()
    for row in rows:
        items = row["items"]
        row["items"] = [items[index] for index in rng.permutation(len(items))]
    return pa.Table.from_pylist(rows, schema=table.schema)


def scale_weights(table: pa.Table, *, factor: float) -> pa.Table:
    """Scale every raw weight and update targets according to their algebra."""

    if factor <= 0.0:
        raise ValueError(f"weight scale must be positive, got {factor}")
    rows = table.to_pylist()
    for row in rows:
        row["items"] = [{**item, "weight": factor * item["weight"]} for item in row["items"]]
        if "weighted_sum" in row:
            row["weighted_sum"] *= factor
    return pa.Table.from_pylist(rows, schema=table.schema)


def duplicate_items(table: pa.Table, *, maximum_original: int) -> pa.Table:
    """Duplicate every item on rows that remain within configured capacity."""

    rows = [row for row in table.to_pylist() if len(row["items"]) <= maximum_original]
    if not rows:
        raise ValueError(f"no rows contain at most {maximum_original} items")
    for row in rows:
        row["items"] = [*row["items"], *row["items"]]
        if "weighted_sum" in row:
            row["weighted_sum"] *= 2.0
    return pa.Table.from_pylist(rows, schema=table.schema)


def model(
    *,
    source: Source,
    target: Target = "weighted_sum",
) -> rf.Model:
    """Build one public-schema weighted-sum model."""

    item_fields: dict[str, object]
    if source == "raw":
        item_fields = {"value": rf.Number, "weight": rf.Number}
    else:
        item_fields = {"contribution": rf.Number}
    target_field = {target: rf.Number(mask=True, objective="mse")}
    return rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=1e-3),
        items=rf.Branch(
            length=ITEMS,
            n_layers=2,
            reduction=rf.Attention(n_layers=2),
            **item_fields,
        ),
        **target_field,
    )


def mean_model() -> rf.Model:
    """Build a model that intentionally erases contribution count with Mean."""

    return rf.Model(
        d_model=24,
        n_layers=1,
        n_heads=4,
        batch_size=64,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        items=rf.Branch(
            length=ITEMS,
            attention="none",
            reduction=rf.Mean(),
            contribution=rf.Number,
        ),
        mean_contribution=rf.Number(mask=True, objective="mse"),
        weighted_sum=rf.Number(mask=True, objective="mse"),
    )


def data(configured: rf.Model, train: pa.Table, validate: pa.Table, *, seed: int) -> rf.ArrowDataModule:
    """Build the deterministic in-memory data module shared by these proofs."""

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
    """Run inference and unwrap one numerical target."""

    output = configured.predict(table.select(["items"]))
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
    baseline_rmse = rmse(actual, float(column(train, target).mean()))
    measured = rmse(actual, predicted)
    return Score(rmse=measured, baseline_rmse=baseline_rmse, nrmse=measured / baseline_rmse)


def diagnostics(label: str, measured: Score) -> str:
    """Render one compact assertion diagnostic."""

    return f"{label}: rmse={measured.rmse:.4f}, baseline={measured.baseline_rmse:.4f}, nrmse={measured.nrmse:.4f}"
