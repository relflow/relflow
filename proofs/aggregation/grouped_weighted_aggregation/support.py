"""Shared protocol and user guide for grouped weighted-aggregation proofs.

Claim
-----
RelFlow should be able to select one category from an interleaved collection,
bind each selected item's independent ``value`` and positive ``weight``, and
reduce the resulting products.  A user should only have to describe the
natural schema::

    selected_group = Category(A, B, C)
    items[*] = {group: Category(A, B, C), value: Number, weight: Number}
    answer = masked Number

The final schema should not require the user to derive
``contribution = value * weight``.  That derived field is retained here only
as a diagnostic rung: if supplied contributions work and raw pairs do not,
the missing capability is local multiplication or coordinate binding rather
than category selection or reduction.

What to do
----------
Keep ``group``, ``value``, and ``weight`` as siblings on one repeated item so
their coordinate is explicit.  Put ``selected_group`` at the request root as an
ordinary visible Category field and use an explicit learned reduction for this
bounded conditional aggregate. The tested schema uses ``reduction=None`` on
the item branch so its field-coordinate tokens remain available to the root;
the root's ``Attention(n_outputs=6)`` then performs learned compression.

Record-local sibling mixing lets each item combine its group, value, and weight
before the preserved item tokens reach the root. The visible selected-group
request then conditions scalar decoding on those tokens, giving the natural
schema a learned route for category selection, item-local multiplication, and
aggregation. Root Attention also adds masked evidence-sum and present-count
lanes after normalized query pooling, so compression does not erase total
mass. When diagnosing a failure, start with the supplied-contribution control,
then restore raw value/weight pairs and use the two causal corruptions below.

What not to do
--------------
Do not prefilter the item list or create one branch per category merely to claim
that the model learned conditional selection.  Once inputs match this schema,
users should not need hand-written relational routing; structural ``query=``
paths and preprocessors remain appropriate for source selection and shaping.
Do not mistake a supplied ``contribution`` field for evidence that the model
learned multiplication.  It is still a valid production input when the product
is deterministic and must be exact.  The same distinction applies to the
filter and aggregate: prefiltering invalidates this learned-selection proof,
but exact contractual selection and summation belong in a
:class:`relflow.Preprocessor` or the application.  Do not accept low error
without interventions; a model can exploit a group prior, one marginal, or a
target-scale shortcut without learning the intended composition.

Diagnostic ladder
-----------------
The supplied-contribution test exposes ``value * weight`` and therefore asks
only for category-conditioned filtering plus summation.  Its group-label
rotation preserves every contribution and the A/B/C label counts while
breaking their association.

The raw-pair test removes that derived field.  In addition to the label
rotation it swaps weights between items *within each category*.  This keeps
the value multiset, weight multiset, per-group membership, and selected group
unchanged while breaking only the item-local value/weight pairing.  A genuine
grouped weighted reducer must become worse under both corruptions.

Why the task is identifiable
----------------------------
Every bag contains two independently sampled items from each of A, B, and C,
interleaved in random order.  Values are uniform on ``[-1, 1]`` and positive
weights are uniform on ``[0.20, 1.80]``.  The same bag is emitted once for each
selected group, so neither the bag nor its field marginals determine the
answer without the root request.  No individual value, weight, label, or
collection position predicts the selected weighted sum.

Metamorphic semantics
---------------------
Jointly permuting complete items must leave the answer unchanged.  Rotating
group labels must generally change which contributions belong to the request.
Swapping weights within each group must generally change the weighted sum but
not either univariate marginal.  Uniformly scaling all weights would scale a
weighted sum and leave a weighted mean invariant; that algebra is covered by
the adjacent ``weighted_aggregation`` proof and remains a promotion check for
this conditional composition.

Protocol and gates
------------------
All cases use public :class:`relflow.Model` and
:class:`relflow.ArrowDataModule` APIs, deterministic CPU training, and
independent train, validation, and test bags.  Accuracy is RMSE normalized by
the RMSE of the constant training-target mean.

The supplied-contribution control must reach nRMSE below 0.50.  Rotating item
group labels must worsen nRMSE by at least 0.20, and jointly permuting complete
items may add at most 0.10 target standard deviations of prediction drift.
The desired raw-pair path uses the same gates plus a within-group weight-swap
gap of at least 0.20. Finite, loosely bounded calibration assertions run before
the behavioral gates so schema, data, and numerical failures remain ordinary
test failures.

Status
------
Provisional established suite. Both the natural raw-pair path and the
supplied-contribution diagnostic clear every hard gate at their deterministic
seed. The two schemas have different effective pass-through widths, so their
scores should not be read as a capacity-matched architecture comparison.

Current evidence
----------------
* At seed 3600, the supplied-contribution path clears 0.50 nRMSE, the 0.20
  group-label corruption gap, and the 0.10 complete-item permutation gate.
* At seed 3605, the natural raw-pair path clears the same accuracy, group, and
  permutation gates, and its within-group weight swap worsens nRMSE by at least
  0.20. This one seed supports selection, item-local multiplication, and
  reduction together; it does not establish exact arithmetic or unseen-label
  generalization.
* The generator and interventions exactly preserve the advertised marginals;
  the tests do not infer success merely from output width.

Remaining work
--------------
* Repeat every supported gate over at least three core seeds and ten light
  calibration seeds.
* Add selected-group weighted mean, denominator recovery, and explicit tests
  of uniform weight scaling.
* Vary category counts and collection cardinality independently, including an
  absent selected group with an explicitly chosen empty-set convention.
* Cover zero, negative, missing, duplicated, and extreme weights.
* Hold out category labels to distinguish learned equality from memorized
  A/B/C routing; use ``Hash`` when unseen identity equality is the contract.
* Test nested owners, multiple simultaneous requested groups, and a request
  for both weighted sum and total weight.
* Ablate coordinate mixing, query context, and mass lanes separately; the
  current end-to-end proof does not identify which is necessary.
* Compare token-preserving ``reduction=None`` with compressed Attention at
  matched parameter and output-token budgets.

Promotion criteria
------------------
Promote only after all accuracy, group-corruption, pairing-corruption, and
permutation gates pass across the seed matrix.  The natural raw schema—not the
derived-contribution diagnostic—must be the promoted public example.

Run all cases with::

    uv run pytest -n 0 proofs/aggregation/grouped_weighted_aggregation -q
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch

import relflow as rf

GROUPS = ("A", "B", "C")
ITEMS_PER_GROUP = 2
ITEMS = len(GROUPS) * ITEMS_PER_GROUP
Source = Literal["raw", "contribution"]


def records(*, bags: int, seed: int, source: Source) -> pa.Table:
    """Draw interleaved groups and emit one request for every selected group."""

    rng = np.random.default_rng(seed)
    observations: list[dict[str, object]] = []
    for bag in range(bags):
        items: list[dict[str, object]] = []
        answers: dict[str, float] = {}
        for group in GROUPS:
            values = rng.uniform(-1.0, 1.0, size=ITEMS_PER_GROUP)
            weights = rng.uniform(0.20, 1.80, size=ITEMS_PER_GROUP)
            contributions = values * weights
            answers[group] = float(contributions.sum())
            for value, weight, contribution in zip(values, weights, contributions, strict=True):
                item: dict[str, object] = {"group": group}
                if source == "raw":
                    item.update(value=float(value), weight=float(weight))
                else:
                    item["contribution"] = float(contribution)
                items.append(item)
        interleaved = [items[index] for index in rng.permutation(len(items))]
        for selected_group in GROUPS:
            observations.append(
                {
                    "bag": bag,
                    "selected_group": selected_group,
                    "items": interleaved,
                    "answer": answers[selected_group],
                }
            )
    return pa.Table.from_pylist(observations)


def rotate_group_labels(table: pa.Table) -> pa.Table:
    """Break group/item association while preserving all group marginals."""

    successor = {group: GROUPS[(index + 1) % len(GROUPS)] for index, group in enumerate(GROUPS)}
    rows = table.to_pylist()
    for row in rows:
        row["items"] = [{**item, "group": successor[item["group"]]} for item in row["items"]]
    return pa.Table.from_pylist(rows, schema=table.schema)


def swap_weights_within_groups(table: pa.Table) -> pa.Table:
    """Break value/weight pairing without changing any per-group marginal."""

    rows = table.to_pylist()
    cached: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        bag = int(row["bag"])
        if bag not in cached:
            items = [dict(item) for item in row["items"]]
            for group in GROUPS:
                indices = [index for index, item in enumerate(items) if item["group"] == group]
                if len(indices) != ITEMS_PER_GROUP:
                    raise AssertionError(f"bag {bag} has {len(indices)} items in group {group!r}")
                weights = [items[index]["weight"] for index in indices]
                for index, weight in zip(indices, np.roll(weights, 1), strict=True):
                    items[index]["weight"] = float(weight)
            cached[bag] = items
        row["items"] = cached[bag]
    return pa.Table.from_pylist(rows, schema=table.schema)


def permute_items(table: pa.Table, *, seed: int) -> pa.Table:
    """Jointly permute complete items while retaining targets."""

    rng = np.random.default_rng(seed)
    rows = table.to_pylist()
    cached: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        bag = int(row["bag"])
        if bag not in cached:
            items = row["items"]
            cached[bag] = [items[index] for index in rng.permutation(len(items))]
        row["items"] = cached[bag]
    return pa.Table.from_pylist(rows, schema=table.schema)


def model(*, source: Source) -> rf.Model:
    """Build the natural raw schema or its supplied-contribution diagnostic."""

    fields: dict[str, object]
    if source == "raw":
        fields = {"value": rf.Number, "weight": rf.Number}
    else:
        fields = {"contribution": rf.Number}
    return rf.Model(
        name="request",
        d_model=48,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_outputs=ITEMS, n_layers=2),
        batch_size=96,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=1e-3),
        items=rf.Branch(
            length=ITEMS,
            n_layers=2,
            reduction=None,
            group=rf.Category(size=len(GROUPS), p_unavailable=0.0),
            **fields,
        ),
        selected_group=rf.Category(size=len(GROUPS), p_unavailable=0.0),
        answer=rf.Number(mask=True, objective="mse"),
    )


def data(configured: rf.Model, train: pa.Table, validate: pa.Table, *, seed: int) -> rf.ArrowDataModule:
    """Build the deterministic in-memory data module."""

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


def prediction(configured: rf.Model, table: pa.Table) -> np.ndarray:
    """Run inference and unwrap the numerical answer."""

    output = configured.predict(table.select(["selected_group", "items"]))
    field = output["predictions"].combine_chunks().field("request/answer")
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


def score(*, train: pa.Table, test: pa.Table, predicted: np.ndarray) -> Score:
    """Normalize test RMSE by the constant training-target mean."""

    actual = column(test, "answer")
    baseline_rmse = rmse(actual, float(column(train, "answer").mean()))
    measured = rmse(actual, predicted)
    return Score(rmse=measured, baseline_rmse=baseline_rmse, nrmse=measured / baseline_rmse)


def diagnostics(label: str, measured: Score) -> str:
    """Render one compact assertion diagnostic."""

    return f"{label}: rmse={measured.rmse:.4f}, baseline={measured.baseline_rmse:.4f}, nrmse={measured.nrmse:.4f}"
