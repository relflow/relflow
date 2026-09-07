"""Shared protocol and user guide for order-statistic proofs.

Claim
-----
A user should be able to place random :class:`relflow.Number` values on one
repeated branch, provide a visible requested rank, and reconstruct the value at
that rank.  The natural schema is deliberately short::

    model = rf.Model(
        items=rf.Branch(length=9, reduction=None, value=rf.Number),
        rank=rf.Category(size=5),
        answer=rf.Number(mask=True),
        ...,
    )

``reduction=None`` preserves the nine encoded Number field-coordinate slots
for the root instead of asking one learned query to compress the whole
empirical distribution.  The rank is data, not schema configuration: one
fitted model must answer ``minimum``, ``q25``, ``median``, ``q75``, and
``maximum`` requests.

What to do
----------
Keep raw Number values in one repeated branch and use ``reduction=None`` so
their encoded item tokens reach the context where the visible request can
condition the answer. ``None`` preserves evidence; it does not itself select a
rank. The scalar decoder uses the visible request to condition a
non-positional learned query over the preserved evidence. Randomize item order
whenever the domain is a bag.
Split by complete base bag. In particular, create every rank
request for a bag first and then assign that entire request family to train,
validation, or test.  Use an odd length whose requested quartiles land on
observed values when the first proof is meant to test ranking rather than
quantile interpolation semantics.

State the semantics of duplicates, nulls, even lengths, interpolation, and
empty collections before deploying a quantile model.  This proof uses nine
non-null values, inclusive zero-based ranks ``(0, 2, 4, 6, 8)``, and permits
ties.  A duplicate at the requested rank is therefore returned unchanged.

What not to do
--------------
Do not infer median support from a min/max result.  Attention can learn to
focus on a salient endpoint without retaining enough of the empirical
distribution to identify its middle.  Likewise, low aggregate error can hide
a failed interior rank, so each rank and data shape needs its own gate.

Do not sort values, attach their ordinal positions, calculate a histogram, or
precompute a quantile merely to claim that the model learned ranking. Those are
useful diagnostic rungs when a failure must be localized because they supply
part or all of the domain operation under test. If an exact quantile is itself
contractual business logic, compute it in an :class:`relflow.Preprocessor` or
the application rather than substituting a learned approximation. Do not use
accidental input position as rank. A complete-item permutation must preserve
the answer.

Why the controls are identifiable
----------------------------------
Every base bag is copied into all five requests in one split.  The item values
are consequently identical while the targets usually differ, so a constant
or bag-only prediction cannot solve the task.  Nulling ``rank`` makes the five
copies exactly identical to the model and must collapse their predictions.
Cycling rank labels preserves every value, every requested-rank frequency, and
every target marginal while breaking only the request-to-answer relation.

The test generator rotates among symmetric, left-skewed, right-skewed, and
duplicate-bearing bags.  Location and scale vary independently.  The model
cannot use a fixed value threshold for a rank, and the duplicate cases exercise
tie semantics without turning every bag into an easy repeated-value problem.

Metamorphic semantics
---------------------
The proof treats these invariances and interventions as part of the behavioral
contract:

* jointly permuting the nine complete items leaves every requested statistic
  unchanged;
* hiding the rank gives all five copies of one bag identical model inputs and
  therefore identical predictions;
* cycling rank requests while retaining the original targets must remove
  request-conditioned skill; and
* minimum and maximum accuracy provides no evidence about the three interior
  ranks unless those interior cells pass independently.

Protocol and gates
------------------
All data are generated independently for train, validation, and test and the
model is trained deterministically on CPU.  There are nine items per bag.
Quartiles use observed sorted positions rather than interpolation.  Accuracy
is RMSE divided by the RMSE of that rank's training-target mean.  Results are
reported for every rank and for the four data shapes.

Both endpoints must beat 0.30 nRMSE, and each interior rank must beat 0.35.
Every symmetric, skewed, and duplicate-bearing shape must beat 0.45 nRMSE.
Complete-item permutation drift must remain below 0.10 test-target standard
deviations.  Cycled rank labels must worsen overall nRMSE by at least 0.35 and
reach at least 0.80 nRMSE.  With rank hidden, predictions within each copied
five-request family must agree to numerical precision, both endpoint cells must
be worse than 0.65 nRMSE, and at least three of five rank cells must exceed that
threshold. Central order statistics can remain easier for an unconditional
estimate, so requiring four bad cells would conflate request dependence with
baseline difficulty. Dataset diversity and baseline strength are ordinary
hard assertions.

Status
------
Provisional established proof.  The natural raw-value schema clears every gate
for one deterministic seed at 800 updates under the defensible hidden-request
control above. It still needs the multi-seed promotion matrix below, and its
learned permutation stability is approximate rather than an architectural
invariance guarantee.

Current evidence
----------------
At seed 3700 with the current architecture, overall nRMSE is 0.0823. Per-rank
nRMSE is 0.0423 for ``minimum``, 0.0975 for ``q25``, 0.0882 for ``median``,
0.1243 for ``q75``, and 0.0550 for ``maximum``. Duplicate-bearing bags reach
0.0820 nRMSE; the symmetric, left-skewed, and right-skewed cells reach 0.0580,
0.0549, and 0.0522 respectively.

Cycling the visible rank raises overall nRMSE to 1.4380. Hiding it collapses
the within-bag prediction spread to ``2.384e-7``. The two endpoint cells and
three of five cells overall then exceed 0.65 nRMSE; the unconditional median
and upper-quartile estimates remain easier. A new complete-item permutation
changes predictions by 0.0367 target standard deviations.

Remaining work
--------------
* Repeat the complete gate over at least three core seeds and ten lightweight
  calibration seeds before promoting thresholds.
* Confirm across seeds that the hidden-request control continues to separate
  exact within-family collapse from the easier unconditional central ranks.
* Add variable odd lengths, then explicitly specify even-length and quantile
  interpolation behavior.
* Add empty collections, null values, all-equal bags, longer duplicate runs,
  extreme outliers, heavy tails, and magnitude extrapolation.
* Add rank requests by numeric fraction only after training-time and inference
  interpolation semantics are explicit.
* Compare token-preserving routing with learned ``Attention(n_outputs=k)`` and
  determine whether ``k`` has any stable distribution-retention meaning.
* Add a supplied-sort diagnostic only if the natural path fails and the
  diagnostic can distinguish ranking from request routing or numeric decoding.

Promotion criteria
------------------
All five rank cells, every data-shape cell, permutation invariance, hidden-rank
collapse, and rank-corruption sensitivity must pass for at least three core
seeds.  The public example must continue to contain only raw values and the
visible rank request; a diagnostic sorted position is not valid evidence for
the learned-ranking claim, though exact rank calculation may be the right
production implementation.

Run the proof with::

    uv run pytest -n 0 proofs/aggregation/order_statistics -q
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch

import relflow as rf

LENGTH = 9
RANKS = ("minimum", "q25", "median", "q75", "maximum")
RANK_INDEX = dict(zip(RANKS, (0, 2, 4, 6, 8), strict=True))
SHAPES = ("symmetric", "left_skewed", "right_skewed", "duplicates")


def values(rng: np.random.Generator, shape: str) -> np.ndarray:
    """Draw one independently located and scaled bag with a declared shape."""

    match shape:
        case "symmetric":
            standardized = rng.normal(size=LENGTH)
        case "left_skewed":
            standardized = 1.0 - rng.exponential(size=LENGTH)
        case "right_skewed":
            standardized = rng.exponential(size=LENGTH) - 1.0
        case "duplicates":
            standardized = rng.normal(size=LENGTH)
            standardized[2] = standardized[1]
            standardized[6] = standardized[5]
        case _:
            raise ValueError(f"unknown order-statistic shape: {shape!r}")

    standardized = np.clip(standardized, -3.0, 3.0)
    scale = float(rng.uniform(0.35, 1.35))
    location = float(rng.uniform(-1.25, 1.25))
    return location + scale * standardized


def records(*, bags: int, seed: int) -> pa.Table:
    """Expand each base bag into all rank requests before split assignment."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for bag in range(bags):
        shape = SHAPES[bag % len(SHAPES)]
        bag_values = values(rng, shape)
        ordered = np.sort(bag_values)
        permutation = rng.permutation(LENGTH)
        items = [{"value": float(bag_values[index])} for index in permutation]
        for rank in RANKS:
            rows.append(
                {
                    "bag": f"{seed}:{bag}",
                    "shape": shape,
                    "rank": rank,
                    "items": items,
                    "answer": float(ordered[RANK_INDEX[rank]]),
                }
            )
    return pa.Table.from_pylist(rows)


def permute_items(table: pa.Table, *, seed: int) -> pa.Table:
    """Randomize each complete bag once while retaining targets and requests."""

    rng = np.random.default_rng(seed)
    rows = table.to_pylist()
    reordered: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        bag = str(row["bag"])
        if bag not in reordered:
            items = row["items"]
            reordered[bag] = [items[index] for index in rng.permutation(len(items))]
        row["items"] = reordered[bag]
    return pa.Table.from_pylist(rows, schema=table.schema)


def validate_request_families(table: pa.Table) -> set[str]:
    """Require one identical five-request family for every complete base bag."""

    families: dict[str, list[dict[str, object]]] = {}
    for row in table.select(["bag", "rank", "items"]).to_pylist():
        families.setdefault(str(row["bag"]), []).append(row)
    for bag, family in families.items():
        ranks = {str(row["rank"]) for row in family}
        if len(family) != len(RANKS) or ranks != set(RANKS):
            raise AssertionError(f"bag {bag!r} does not contain exactly one request for every rank: {ranks!r}")
        if any(row["items"] != family[0]["items"] for row in family[1:]):
            raise AssertionError(f"bag {bag!r} changes item evidence between rank requests")
    return set(families)


def hide_rank(table: pa.Table) -> pa.Table:
    """Make every visible rank request unavailable without changing targets."""

    index = table.schema.get_field_index("rank")
    return table.set_column(index, "rank", pa.nulls(len(table), type=pa.string()))


def cycle_rank(table: pa.Table) -> pa.Table:
    """Break request/answer alignment while retaining every marginal exactly."""

    rows = table.to_pylist()
    successor = {rank: RANKS[(index + 1) % len(RANKS)] for index, rank in enumerate(RANKS)}
    for row in rows:
        row["rank"] = successor[row["rank"]]
    return pa.Table.from_pylist(rows, schema=table.schema)


def model() -> rf.Model:
    """Build the natural token-preserving public schema."""

    return rf.Model(
        name="request",
        d_model=64,
        n_layers=3,
        n_heads=4,
        reduction=None,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        items=rf.Branch(
            length=LENGTH,
            overflow="error",
            n_layers=3,
            n_heads=4,
            reduction=None,
            value=rf.Number,
        ),
        rank=rf.Category(size=len(RANKS), p_unavailable=0.0),
        answer=rf.Number(mask=True, objective="mse"),
    )


def fit(configured: rf.Model, train: pa.Table, validate: pa.Table, *, seed: int, steps: int) -> lit.Trainer:
    """Fit one deterministic CPU model from public Arrow data APIs."""

    data = rf.ArrowDataModule(
        model=configured,
        train=train,
        validate=validate,
        seed=seed,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    trainer = lit.Trainer(
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
    trainer.fit(configured, datamodule=data)
    return trainer


def prediction(configured: rf.Model, table: pa.Table) -> np.ndarray:
    """Run inference and unwrap the masked numerical answer."""

    output = configured.predict(table.select(["items", "rank"]))
    field = output["predictions"].combine_chunks().field("request/answer")
    return np.asarray(field.field("content").to_numpy(zero_copy_only=False), dtype=np.float64)


def column(table: pa.Table, name: str) -> np.ndarray:
    """Return one Arrow column as a dense array."""

    return np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False))


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    """Calculate root mean squared error."""

    return float(np.sqrt(np.mean(np.square(actual - predicted))))


@dataclass(frozen=True)
class Score:
    """Accuracy relative to a matched train-mean baseline."""

    rmse: float
    baseline_rmse: float
    nrmse: float


def scores(
    *,
    train: pa.Table,
    test: pa.Table,
    predicted: np.ndarray,
    keys: Iterable[str],
) -> dict[tuple[str, ...], Score]:
    """Calculate a normalized score for every requested metric cell."""

    train_rows = train.select([*keys, "answer"]).to_pylist()
    test_rows = test.select([*keys, "answer"]).to_pylist()
    cells = {tuple(str(row[key]) for key in keys) for row in test_rows}
    means: dict[tuple[str, ...], float] = {}
    for cell in cells:
        targets = [float(row["answer"]) for row in train_rows if tuple(str(row[key]) for key in keys) == cell]
        means[cell] = float(np.mean(targets))

    result: dict[tuple[str, ...], Score] = {}
    for cell in cells:
        indices = np.asarray(
            [index for index, row in enumerate(test_rows) if tuple(str(row[key]) for key in keys) == cell]
        )
        actual = np.asarray([float(test_rows[index]["answer"]) for index in indices])
        baseline_rmse = rmse(actual, means[cell])
        measured_rmse = rmse(actual, predicted[indices])
        result[cell] = Score(
            rmse=measured_rmse,
            baseline_rmse=baseline_rmse,
            nrmse=measured_rmse / baseline_rmse,
        )
    return result


def overall_score(*, train: pa.Table, test: pa.Table, predicted: np.ndarray) -> Score:
    """Calculate overall accuracy against per-rank training means."""

    train_rank = column(train, "rank")
    train_answer = column(train, "answer").astype(np.float64)
    test_rank = column(test, "rank")
    actual = column(test, "answer").astype(np.float64)
    means = {rank: float(train_answer[train_rank == rank].mean()) for rank in RANKS}
    baseline = np.asarray([means[rank] for rank in test_rank], dtype=np.float64)
    baseline_rmse = rmse(actual, baseline)
    measured_rmse = rmse(actual, predicted)
    return Score(measured_rmse, baseline_rmse, measured_rmse / baseline_rmse)


def diagnostics(label: str, result: dict[tuple[str, ...], Score] | Score) -> str:
    """Render compact score diagnostics for assertion failures."""

    if isinstance(result, Score):
        return f"{label}: rmse={result.rmse:.4f}, baseline={result.baseline_rmse:.4f}, nrmse={result.nrmse:.4f}"
    lines = [label]
    lines.extend(
        f"  {','.join(cell)}: rmse={score.rmse:.4f}, baseline={score.baseline_rmse:.4f}, nrmse={score.nrmse:.4f}"
        for cell, score in sorted(result.items())
    )
    return "\n".join(lines)
