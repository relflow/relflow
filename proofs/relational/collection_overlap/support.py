"""Shared protocol and guide for the deferred collection-overlap capability.

Claim
-----
RelFlow should eventually infer whether two bounded sibling collections of
previously unseen identities overlap from the natural schema alone. The
current architecture does not establish that behavior.

What to do
----------
Use ``Hash`` for observation-local identifiers that may be unseen at inference,
declare bounded branch lengths, and keep ``reduction=None`` while testing a
comparison that may need every candidate. If overlap is deterministic business
logic that must work today, compute it in a preprocessor or application and
treat it as an explicit feature. See ``RELATIONS.md`` for a deferred, opt-in
architectural proposal.

What not to do
--------------
Do not assume that exposing all child tokens with ``reduction=None`` makes a
scalar decoder compare every member of one collection with every member of the
other. Do not use independent ``Category`` vocabularies for fresh IDs: unseen
values collapse to unavailable tokens. Do not call a supplied intersection a
learned-overlap proof; it moves the operation outside the model.

Why the boundary is identifiable
--------------------------------
Every split uses a disjoint namespace and every identity belongs to one row.
Labels are balanced, side lengths are sampled independently of the label,
orders are random, and each side contains unique members. A positive row has
exactly one shared ID and a negative row has none. Frequency, position, length,
and vocabulary memorization therefore cannot predict the answer.

The flat-pair control verifies that two scalar ``Hash`` fields can represent
equality for unseen identities. Its success makes the collection result a
structural comparison boundary rather than a failure of the identity type.
Rename-plus-permute preserves the mathematical relation; breaking the one
shared right member removes it while retaining labels and unrelated members.

Protocol and gates
------------------
Train, validation, and test data are independent Arrow tables. Collections
contain two through five members per side. Training is deterministic on CPU
and metrics use probabilities from the public ``Model.predict`` API.

The flat equality control must reach at least 0.95 AUC on unseen IDs while an
all-unequal intervention remains at chance. The direct collection case records
the current limitation: intact, rename-plus-permute, and broken AUC must all
remain in ``[0.35, 0.65]``; invariant probability drift must be at most 0.15;
and removing positive overlaps must change their mean probability by at most
0.10 in absolute value.

Status and evidence
-------------------
Unresolved capability; established one-seed limitation proof. Flat unseen-Hash
equality reaches 0.9890 AUC, while the ordinary direct sibling-collection route
reaches 0.4972 intact AUC, 0.5153 after semantic rename/permutation, and 0.4940
after overlap removal. Its invariant drift is 0.0006 and its positive break
response rounds to 0.0000. The result describes this schema, seed, width,
optimizer, and 700-step budget; it does not prove that transformers cannot
learn set intersection.

Remaining work
--------------
* Repeat the boundary over the promotion seed matrix and larger budgets.
* Test purpose-built, bounded set-comparison designs before adding an API.
* Extend evaluation to intersection size, subset, duplicates, empty sets,
  payload retrieval, and lengths beyond training.
* Benchmark the time and memory cost of any proposed comparison mechanism.

Promotion criteria
------------------
Claim native collection overlap only when the natural sibling schema clears
the accuracy, intervention, and invariance gates across at least three core
seeds, without a precomputed intersection. A future mechanism must also meet
the scope, cost, extension, and documentation criteria in ``RELATIONS.md``.

Run both retained cases with::

    uv run pytest -n 0 proofs/relational/collection_overlap -q
"""

from __future__ import annotations

from dataclasses import dataclass

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch

import relflow as rf

MIN_LENGTH = 2
MAX_LENGTH = 5
TRAIN_ROWS = 3072
VALIDATE_ROWS = 768
TEST_ROWS = 1024
MAX_STEPS = 700


def collection_records(*, rows: int, seed: int, namespace: str) -> pa.Table:
    """Generate balanced, variable-width sibling collections with zero/one overlap."""

    if rows <= 0 or rows % 2:
        raise ValueError(f"collection rows must be a positive even integer, got {rows}")

    rng = np.random.default_rng(seed)
    labels = np.tile(np.asarray([False, True]), rows // 2)
    rng.shuffle(labels)
    observations: list[dict[str, object]] = []
    for row_index, has_overlap in enumerate(labels):
        left_length = int(rng.integers(MIN_LENGTH, MAX_LENGTH + 1))
        right_length = int(rng.integers(MIN_LENGTH, MAX_LENGTH + 1))
        left_ids = [f"{namespace}-{row_index:06d}-left-{index}" for index in range(left_length)]
        right_ids = [f"{namespace}-{row_index:06d}-right-{index}" for index in range(right_length)]
        if has_overlap:
            right_ids[int(rng.integers(0, right_length))] = left_ids[int(rng.integers(0, left_length))]
        rng.shuffle(left_ids)
        rng.shuffle(right_ids)
        observations.append(
            {
                "left": [{"entity_id": value} for value in left_ids],
                "right": [{"entity_id": value} for value in right_ids],
                "has_overlap": bool(has_overlap),
            }
        )
    table = pa.Table.from_pylist(observations)
    validate_collection_table(table)
    return table


def flat_pair_records(*, rows: int, seed: int, namespace: str, break_equal: bool = False) -> pa.Table:
    """Generate the unseen-Hash equality control used to calibrate the primitive."""

    if rows <= 0 or rows % 2:
        raise ValueError(f"pair rows must be a positive even integer, got {rows}")
    rng = np.random.default_rng(seed)
    labels = np.tile(np.asarray([False, True]), rows // 2)
    rng.shuffle(labels)
    observations: list[dict[str, object]] = []
    for row_index, equal in enumerate(labels):
        left = f"{namespace}-{row_index:06d}-left"
        right = left if equal and not break_equal else f"{namespace}-{row_index:06d}-right"
        observations.append({"left_id": left, "right_id": right, "equal": bool(equal)})
    return pa.Table.from_pylist(observations)


def rename_and_permute(table: pa.Table, *, seed: int) -> pa.Table:
    """Apply an equality-preserving unseen-ID bijection and reorder both sides."""

    rng = np.random.default_rng(seed)
    observations: list[dict[str, object]] = []
    for row_index, row in enumerate(table.to_pylist()):
        left_ids = [item["entity_id"] for item in row["left"]]
        right_ids = [item["entity_id"] for item in row["right"]]
        identities = list(dict.fromkeys([*left_ids, *right_ids]))
        replacements = [f"renamed-{seed}-{row_index:06d}-{index}" for index in range(len(identities))]
        rng.shuffle(replacements)
        mapping = dict(zip(identities, replacements, strict=True))
        renamed_left = [mapping[value] for value in left_ids]
        renamed_right = [mapping[value] for value in right_ids]
        rng.shuffle(renamed_left)
        rng.shuffle(renamed_right)
        observations.append(
            {
                "left": [{"entity_id": value} for value in renamed_left],
                "right": [{"entity_id": value} for value in renamed_right],
                "has_overlap": row["has_overlap"],
            }
        )
    transformed = pa.Table.from_pylist(observations, schema=table.schema)
    validate_collection_table(transformed)
    return transformed


def break_overlaps(table: pa.Table) -> pa.Table:
    """Remove the shared right member while retaining labels and all controls."""

    observations: list[dict[str, object]] = []
    for row_index, row in enumerate(table.to_pylist()):
        left_ids = [item["entity_id"] for item in row["left"]]
        left_set = set(left_ids)
        replacements = {
            value: f"broken-{row_index:06d}-{index}"
            for index, value in enumerate(item["entity_id"] for item in row["right"])
            if value in left_set
        }
        right_ids = [replacements.get(item["entity_id"], item["entity_id"]) for item in row["right"]]
        observations.append(
            {
                "left": row["left"],
                "right": [{"entity_id": value} for value in right_ids],
                "has_overlap": row["has_overlap"],
            }
        )
    return pa.Table.from_pylist(observations, schema=table.schema)


def labels(table: pa.Table, *, target: str) -> np.ndarray:
    """Read one Boolean target as a dense vector."""

    return np.asarray(table[target].to_pylist(), dtype=np.bool_)


def overlap_truth(table: pa.Table) -> np.ndarray:
    """Compute actual set overlap from the two natural sibling branches."""

    truth = []
    for row in table.to_pylist():
        left = {item["entity_id"] for item in row["left"]}
        right = {item["entity_id"] for item in row["right"]}
        truth.append(bool(left & right))
    return np.asarray(truth, dtype=np.bool_)


def validate_collection_table(table: pa.Table) -> None:
    """Assert the generator's balance, width, uniqueness, and exact-overlap contract."""

    target = labels(table, target="has_overlap")
    actual = overlap_truth(table)
    if target.mean() != 0.5:
        raise AssertionError(f"collection labels are not exactly balanced: positive rate={target.mean():.6f}")
    if not np.array_equal(target, actual):
        raise AssertionError("collection labels disagree with exact set overlap")
    for row in table.to_pylist():
        left = [item["entity_id"] for item in row["left"]]
        right = [item["entity_id"] for item in row["right"]]
        if not (MIN_LENGTH <= len(left) <= MAX_LENGTH and MIN_LENGTH <= len(right) <= MAX_LENGTH):
            raise AssertionError(f"collection length escaped [{MIN_LENGTH}, {MAX_LENGTH}]")
        if len(left) != len(set(left)) or len(right) != len(set(right)):
            raise AssertionError("collection side contains duplicate identities")
        if len(set(left) & set(right)) != int(row["has_overlap"]):
            raise AssertionError("positive rows must have exactly one overlap and negative rows none")


def collection_model() -> rf.Model:
    """Build the ordinary direct sibling-collection route."""

    return rf.Model(
        name="overlap",
        d_model=48,
        n_layers=3,
        n_heads=4,
        reduction=None,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        left=rf.Branch(
            length=MAX_LENGTH,
            n_layers=1,
            reduction=None,
            entity_id=rf.Hash(n_hashes=4, n_bands=8),
        ),
        right=rf.Branch(
            length=MAX_LENGTH,
            n_layers=1,
            reduction=None,
            entity_id=rf.Hash(n_hashes=4, n_bands=8),
        ),
        has_overlap=rf.Boolean(mask=True),
    )


def flat_pair_model() -> rf.Model:
    """Build the positive control for equality between two unseen Hash fields."""

    return rf.Model(
        name="pair",
        d_model=48,
        n_layers=2,
        n_heads=4,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        left_id=rf.Hash(n_hashes=4, n_bands=8),
        right_id=rf.Hash(n_hashes=4, n_bands=8),
        equal=rf.Boolean(mask=True),
    )


def train(configured: rf.Model, train: pa.Table, validate: pa.Table, *, seed: int, steps: int) -> rf.Model:
    """Fit one deterministic CPU proof model."""

    lit.seed_everything(seed, workers=True)
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
        max_steps=steps,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        check_val_every_n_epoch=5,
    )
    trainer.fit(configured, datamodule=data)
    return configured


def probabilities(configured: rf.Model, table: pa.Table, *, address: str) -> np.ndarray:
    """Read Boolean probabilities from the public Arrow prediction contract."""

    coordinates = configured.predict(table)["predictions"].combine_chunks().field(address).to_pylist()
    return np.asarray([float(value["content"]["probability"]) for value in coordinates], dtype=np.float64)


def auc(target: np.ndarray, predicted: np.ndarray) -> float:
    """Compute binary ROC AUC from all positive/negative score pairs."""

    positive = predicted[target]
    negative = predicted[~target]
    if not len(positive) or not len(negative):
        raise ValueError("AUC requires at least one positive and one negative example")
    comparisons = positive[:, None] - negative[None, :]
    return float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())


@dataclass(frozen=True)
class RouteScore:
    """Behavioral metrics for one collection-overlap route."""

    intact_auc: float
    invariant_auc: float
    broken_auc: float
    invariant_drift: float
    positive_break_drop: float


def route_score(
    configured: rf.Model,
    intact: pa.Table,
    invariant: pa.Table,
    broken: pa.Table,
) -> RouteScore:
    """Evaluate accuracy, corruption sensitivity, and rename/order invariance."""

    target = labels(intact, target="has_overlap")
    intact_probability = probabilities(configured, intact, address="overlap/has_overlap")
    invariant_probability = probabilities(configured, invariant, address="overlap/has_overlap")
    broken_probability = probabilities(configured, broken, address="overlap/has_overlap")
    return RouteScore(
        intact_auc=auc(target, intact_probability),
        invariant_auc=auc(target, invariant_probability),
        broken_auc=auc(target, broken_probability),
        invariant_drift=float(np.mean(np.abs(intact_probability - invariant_probability))),
        positive_break_drop=float(np.mean(intact_probability[target] - broken_probability[target])),
    )


def diagnostics(name: str, score: RouteScore) -> str:
    """Format all route metrics for assertion output."""

    return (
        f"{name}: intact_auc={score.intact_auc:.4f}, invariant_auc={score.invariant_auc:.4f}, "
        f"broken_auc={score.broken_auc:.4f}, invariant_drift={score.invariant_drift:.4f}, "
        f"positive_break_drop={score.positive_break_drop:.4f}"
    )
