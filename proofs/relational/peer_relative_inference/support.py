"""Shared protocol and user guide for peer-relative inference proofs.

Claim
-----
A repeated branch should be able to compute context from its members and route
that context back to every original coordinate.  The representative target is
``deviation = value - mean(values in the same group)``.  Each item exposes only
its own value and group; the group mean is deliberately absent.  Items from
groups A, B, and C are interleaved in a fresh random order on every row.

Expected user schema
--------------------
The natural public schema is a single repeated ``items`` branch with sibling
``group``, ``value``, and masked ``deviation`` fields.  Once the input already
matches that schema, no hand-written relational routing, attention Q/K/V, or
derived statistic should be necessary.  Structural ``query=`` paths and
preprocessors remain appropriate when source data needs selection or shaping.
The shared item coordinate states which group belongs to which value. Use
``reduction=None`` on the item branch when every token must remain globally
available, or ``Attention(n_outputs=1)`` when one learned collection summary
is enough. In either case, RelFlow retains the visible siblings aligned with
each repeated target as its query context. Keep the root unreduced when the
coordinate-shaped result must route back through it.

What to do
----------
Keep attributes that describe one entity as sibling fields on the same branch
coordinate. Randomize item order when order is not semantic. Preserve tokens
while diagnosing a difficult task; once that works, test whether a small
learned summary retains enough shared context. Test complete-item permutations
rather than permuting fields independently.

What not to do
--------------
Do not put groups and values into separate repeated branches: that erases their
pairing.  Do not encode an accidental item order as meaning.  Do not precompute
``group_mean`` merely to claim that the model learned peer selection; that
removes the relationship under test.  A precomputed mean is still the correct
production design when the statistic is deterministic and must be exact rather
than learned approximately. Do not assume that one output preserves arbitrary
collection information just because it works here: this target also receives
its own aligned group/value context, and the shared summary need only provide
peer context.
The supplied-mean case in this directory is only a diagnostic rung.  It uses
an ungrouped collection so it proves local subtraction and coordinate-shaped
decoding before asking the model to discover aggregation or peer selection.

Identifiability and interventions
---------------------------------
Group locations are independent and much wider than within-group deviations,
so an item's value alone is a poor estimate of its deviation.  Every target is
centered exactly within its row and group.  Three interventions guard against
shortcuts:

* corrupting group labels preserves every value and the per-row label counts,
  but changes which values are peers;
* independently translating all values in each group preserves every target,
  so predictions should be group-translation invariant;
* permuting complete item records preserves the problem and should permute the
  coordinate predictions by exactly the same order.

The ungrouped case removes peer selection while retaining aggregation and
broadcast-back.  The supplied-mean case removes both aggregation and peer
selection while retaining local arithmetic and repeated decoding.  Together,
the three rungs localize a failure rather than merely recording one score.

Protocol and gates
------------------
All rows have six items and use independently generated deterministic train,
validation, and test sets.  Accuracy is test RMSE divided by the constant
training-target baseline RMSE.  Dataset calibration and intervention strength
are ordinary assertions, so invalid or non-finite experiments cannot masquerade
as modeling evidence.

The supplied-mean control must reach nRMSE below 0.25.  Ungrouped raw values
must reach nRMSE below 0.35, remain below 0.45 after a common translation, and
have permutation drift below 0.20.  Grouped inference must reach nRMSE below
0.40, remain below 0.50 under independent group translations, worsen by at
least 0.25 when labels are corrupted, and have complete-item permutation drift
below 0.20. The one-output grouped route must reach nRMSE below 0.20 and worsen
to at least 0.90, with a gap of at least 0.75, when labels are corrupted.

Status
------
Provisional but supported for one deterministic core seed with either complete
token preservation or a one-output learned collection summary. The compressed
case relies on automatic coordinate-local mixing and aligned target queries;
it is not evidence that one vector losslessly represents every collection.

Current evidence
----------------
* The supplied-peer-mean control reaches 0.030 nRMSE with sibling
  ``reduction=None``.  In an exploratory contrast, compressing the same two
  numbers with one learned query stayed at 1.000 nRMSE.
* Raw ungrouped deviation reaches 0.091 nRMSE, 0.283 after a common translation,
  and 0.101 normalized complete-item permutation drift.
* Raw grouped deviation reaches 0.054 nRMSE with token preservation.  Independent
  group translations reach 0.358 nRMSE, corrupted labels worsen to 1.424, and
  complete-item permutation drift is 0.056.
* The label-corruption oracle moves by 1.801 nRMSE, confirming that the
  intervention strongly changes peer membership while preserving marginals.
* Reducing the coordinate-contextualized collection to one learned output now
  reaches 0.074 grouped nRMSE; corrupting group labels worsens it to 1.561.

Remaining work
--------------
* Repeat every passing rung across at least three core seeds and ten lightweight
  calibration seeds before promoting thresholds.
* Add variable cardinality, singleton groups, missing group labels, missing
  values, empty branches, and unseen group sizes.
* Add ``above_group_mean`` classification, group z-scores, within-group ranks,
  leave-one-out peer means, and robust peer statistics.
* Test multiple simultaneous peer-relative targets without duplicating the
  collection computation.
* Compare token-preserving routing with ``Attention(n_outputs=k)`` across more
  targets and characterize the minimum sufficient ``k`` without making it
  user folklore.
* Require exact or numerically tight set equivariance once unordered attention
  has an explicit architectural contract.

Promotion criteria
------------------
All accuracy and metamorphic gates must pass for at least three core seeds.  A
promotion must demonstrate that corrupting only peer membership hurts, while
group translations and complete-item permutations do not.  The public example
must continue to omit precomputed peer statistics and manual routing hints.

Run all cases with::

    uv run pytest -n 0 proofs/relational/peer_relative_inference -q
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
GROUPS = ("A", "B", "C")
Source = Literal["raw", "supplied_mean"]


def ungrouped_records(*, rows: int, seed: int, supplied_mean: bool = False) -> pa.Table:
    """Generate collections with independent locations and centered residuals."""

    rng = np.random.default_rng(seed)
    observations: list[dict[str, object]] = []
    for _ in range(rows):
        basis = rng.normal(size=ITEMS)
        basis -= basis.mean()
        basis /= np.sqrt(np.mean(np.square(basis)))
        scale = float(rng.uniform(0.25, 1.10))
        mean = float(rng.uniform(-3.5, 3.5))
        deviations = scale * basis
        values = mean + deviations
        items: list[dict[str, float]] = []
        for value, deviation in zip(values, deviations, strict=True):
            item = {"value": float(value), "deviation": float(deviation)}
            if supplied_mean:
                item["peer_mean"] = mean
            items.append(item)
        observations.append({"items": items})
    return pa.Table.from_pylist(observations)


def grouped_records(*, rows: int, seed: int, supplied_mean: bool = False) -> pa.Table:
    """Generate randomly interleaved, exactly centered two-member groups."""

    rng = np.random.default_rng(seed)
    observations: list[dict[str, object]] = []
    for _ in range(rows):
        items: list[dict[str, object]] = []
        for group in GROUPS:
            mean = float(rng.uniform(-3.5, 3.5))
            deviation = float(rng.uniform(0.20, 1.20))
            for signed in (-deviation, deviation):
                item: dict[str, object] = {
                    "group": group,
                    "value": mean + signed,
                    "deviation": signed,
                }
                if supplied_mean:
                    item["peer_mean"] = mean
                items.append(item)
        order = rng.permutation(ITEMS)
        observations.append({"items": [items[index] for index in order]})
    return pa.Table.from_pylist(observations)


def requests(table: pa.Table) -> pa.Table:
    """Remove repeated targets while preserving all visible item fields."""

    return pa.Table.from_pylist(
        [
            {"items": [{name: value for name, value in item.items() if name != "deviation"} for item in row["items"]]}
            for row in table.to_pylist()
        ]
    )


def target(table: pa.Table) -> np.ndarray:
    """Flatten item deviations in row-major coordinate order."""

    return np.asarray(
        [float(item["deviation"]) for row in table.to_pylist() for item in row["items"]],
        dtype=np.float64,
    )


def model(*, grouped: bool, source: Source, compress_siblings: bool = False) -> rf.Model:
    """Build the natural repeated-target schema for one diagnostic rung."""

    visible: dict[str, object] = {"value": rf.Number}
    if grouped:
        visible["group"] = rf.Category(size=len(GROUPS), p_unavailable=0.0)
    if source == "supplied_mean":
        visible["peer_mean"] = rf.Number
    item_reduction: rf.ReductionConfig | None
    if compress_siblings or (not grouped and source == "raw"):
        item_reduction = rf.Attention(n_outputs=1, n_layers=2)
    else:
        item_reduction = None
    return rf.Model(
        name="collection",
        d_model=48,
        n_layers=3,
        n_heads=4,
        reduction=None,
        batch_size=128,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=1e-3),
        items=rf.Branch(
            length=ITEMS,
            overflow="error",
            n_layers=2,
            reduction=item_reduction,
            **visible,
            deviation=rf.Number(mask=True, objective="mse", n_linear=2),
        ),
    )


def data(configured: rf.Model, train: pa.Table, validate: pa.Table, *, seed: int) -> rf.ArrowDataModule:
    """Build a deterministic in-memory Arrow data module."""

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
    """Predict and flatten repeated deviations in input coordinate order."""

    output = configured.predict(requests(table))
    rows = output["predictions"].combine_chunks().field("collection/items/deviation").to_pylist()
    return np.asarray(
        [float(coordinate["content"]) for row in rows for coordinate in row],
        dtype=np.float64,
    )


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    """Calculate root mean squared error."""

    return float(np.sqrt(np.mean(np.square(actual - predicted))))


@dataclass(frozen=True)
class Score:
    """Accuracy relative to a constant training-target prediction."""

    rmse: float
    baseline_rmse: float
    nrmse: float


def score(*, train: pa.Table, test: pa.Table, predicted: np.ndarray) -> Score:
    """Normalize target RMSE by the training-target mean baseline."""

    actual = target(test)
    baseline_rmse = rmse(actual, float(target(train).mean()))
    measured = rmse(actual, predicted)
    return Score(rmse=measured, baseline_rmse=baseline_rmse, nrmse=measured / baseline_rmse)


def validate_score(label: str, measured: Score) -> None:
    """Assert numerical calibration before evaluating behavioral gates."""

    assert measured.baseline_rmse > 1e-6, f"{label} baseline is degenerate: {measured.baseline_rmse}"
    assert np.isfinite(measured.rmse), f"{label} RMSE is not finite: {measured.rmse}"
    assert np.isfinite(measured.baseline_rmse), f"{label} baseline RMSE is not finite: {measured.baseline_rmse}"
    assert np.isfinite(measured.nrmse), f"{label} nRMSE is not finite: {measured.nrmse}"


def common_translation(table: pa.Table, *, seed: int) -> pa.Table:
    """Translate every value in a row while preserving all deviations."""

    rng = np.random.default_rng(seed)
    rows = table.to_pylist()
    for row in rows:
        offset = float(rng.uniform(-2.5, 2.5))
        for item in row["items"]:
            item["value"] += offset
            if "peer_mean" in item:
                item["peer_mean"] += offset
    return pa.Table.from_pylist(rows, schema=table.schema)


def group_translation(table: pa.Table, *, seed: int) -> pa.Table:
    """Translate each group independently while preserving its deviations."""

    rng = np.random.default_rng(seed)
    rows = table.to_pylist()
    for row in rows:
        offsets = {group: float(rng.uniform(-2.5, 2.5)) for group in GROUPS}
        for item in row["items"]:
            offset = offsets[str(item["group"])]
            item["value"] += offset
            if "peer_mean" in item:
                item["peer_mean"] += offset
    return pa.Table.from_pylist(rows, schema=table.schema)


def corrupt_groups(table: pa.Table) -> pa.Table:
    """Change peer membership while preserving values and label marginals."""

    rows = table.to_pylist()
    changed = 0
    for row in rows:
        labels = [item["group"] for item in row["items"]]
        shifted = labels[1:] + labels[:1]
        for item, label in zip(row["items"], shifted, strict=True):
            changed += int(item["group"] != label)
            item["group"] = label
    if changed == 0:
        raise AssertionError("group corruption changed no item memberships")
    return pa.Table.from_pylist(rows, schema=table.schema)


def implied_deviation(table: pa.Table) -> np.ndarray:
    """Compute the oracle deviations implied by the table's visible grouping."""

    values: list[float] = []
    for row in table.to_pylist():
        means = {
            group: float(np.mean([item["value"] for item in row["items"] if item["group"] == group]))
            for group in GROUPS
        }
        values.extend(float(item["value"] - means[item["group"]]) for item in row["items"])
    return np.asarray(values, dtype=np.float64)


def permute_items(table: pa.Table, *, seed: int) -> tuple[pa.Table, np.ndarray]:
    """Permute whole records and return flattened indices into original order."""

    rng = np.random.default_rng(seed)
    rows = table.to_pylist()
    flattened: list[int] = []
    for row_index, row in enumerate(rows):
        order = rng.permutation(ITEMS)
        row["items"] = [row["items"][index] for index in order]
        flattened.extend(row_index * ITEMS + order)
    return pa.Table.from_pylist(rows, schema=table.schema), np.asarray(flattened, dtype=np.int64)


def diagnostics(label: str, measured: Score) -> str:
    """Format one score for assertion output."""

    return f"{label}: rmse={measured.rmse:.4f}, baseline={measured.baseline_rmse:.4f}, nrmse={measured.nrmse:.3f}"
