"""Protect drift studies' generators, matched interventions, and fixed seed panels."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "research/distributions" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name",
    ["numeric_shifts", "typed_shifts", "policies_and_checkpoints", "enum_shifts", "hash_shifts", "cluster_shifts"],
)
def test_distribution_studies_have_independent_restartable_streams_and_fixed_budgets(name):
    proof = load(name)
    rows = list(proof.records(rows=32, seed=11))
    assert rows == list(proof.records(rows=32, seed=11))
    assert rows != list(proof.records(rows=32, seed=12))
    assert proof.SEED_OFFSETS == (0, 100, 200)
    assert (proof.TRAIN_ROWS, proof.VALIDATE_ROWS, proof.TEST_ROWS, proof.BUDGET) == (4096, 1024, 2048, 16)


@pytest.mark.parametrize("condition, multiplier, offset", [("location", 1, 4), ("spread", 8, 0)])
def test_numeric_shifts_share_the_same_latent_draws_and_targets(condition, multiplier, offset):
    proof = load("numeric_shifts")
    control = list(proof.records(rows=2048, seed=11))
    changed = list(proof.records(rows=2048, seed=11, condition=condition))
    np.testing.assert_allclose(
        [row["amount"] for row in changed], np.asarray([row["amount"] for row in control]) * multiplier + offset
    )
    assert [(row["partner"], row["target"]) for row in changed] == [(row["partner"], row["target"]) for row in control]


def test_correlation_boundary_changes_joint_structure_while_preserving_marginal_population():
    proof = load("numeric_shifts")
    original = list(proof.records(rows=8192, seed=11))
    changed = list(proof.records(rows=8192, seed=11, condition="correlation"))
    for population, expected in [(original, 1), (changed, -1)]:
        amount = np.asarray([row["amount"] for row in population])
        partner = np.asarray([row["partner"] for row in population])
        assert expected * np.corrcoef(amount, partner)[0, 1] > 0.98
        assert abs(partner.mean()) < 0.05
        assert 0.95 < partner.std() < 1.05
    assert [(row["amount"], row["target"]) for row in original] == [(row["amount"], row["target"]) for row in changed]


@pytest.mark.parametrize(
    "condition, field",
    [
        ("labels", "merchant"),
        ("boolean", "flag"),
        ("members", "tags"),
        ("set_size", "tags"),
        ("vector", "vector"),
        ("calendar", "happened"),
        ("nulls", "events"),
    ],
)
def test_typed_interventions_change_only_the_declared_field(condition, field):
    proof = load("typed_shifts")
    original = list(proof.records(rows=64, seed=11))
    changed = list(proof.records(rows=64, seed=11, condition=condition))
    assert [{key: value for key, value in row.items() if key != field} for row in changed] == [
        {key: value for key, value in row.items() if key != field} for row in original
    ]
    assert any(a[field] != b[field] for a, b in zip(original, changed, strict=True))


def test_policy_control_fields_have_identical_values():
    proof = load("policies_and_checkpoints")
    for row in proof.records(rows=256, seed=11, shift=8):
        assert row["monitored"] == row["tolerant"] == row["disabled"]


@pytest.mark.parametrize("name", ["enum_shifts", "hash_shifts", "cluster_shifts"])
@pytest.mark.parametrize("condition", ["frequencies", "nulls"])
def test_identity_drift_interventions_preserve_independent_targets(name, condition):
    proof = load(name)
    original = list(proof.records(rows=2048, seed=11))
    changed = list(proof.records(rows=2048, seed=11, condition=condition))
    assert [row["target"] for row in changed] == [row["target"] for row in original]
    if condition == "frequencies":
        assert len({row["value"] for row in changed if row["value"] is not None}) == 1
        assert len({row["value"] for row in original if row["value"] is not None}) == 4
        assert [row["value"] is None for row in original] == [row["value"] is None for row in changed]
    else:
        assert all(row["value"] is None for row in changed)


@pytest.mark.parametrize("name", ["hash_shifts", "cluster_shifts"])
def test_novel_population_changes_only_identity_spelling_or_value(name):
    proof = load(name)
    original = list(proof.records(rows=2048, seed=11))
    changed = list(proof.records(rows=2048, seed=11, condition="novel"))
    assert not {row["value"] for row in original if row["value"] is not None} & {
        row["value"] for row in changed if row["value"] is not None
    }
    assert [row["value"] is None for row in original] == [row["value"] is None for row in changed]
    assert [row["target"] for row in original] == [row["target"] for row in changed]


def test_enum_spelling_control_preserves_declared_canonical_labels():
    proof = load("enum_shifts")
    original = list(proof.records(rows=128, seed=11))
    folded = list(proof.records(rows=128, seed=11, condition="folded"))
    assert [None if row["value"] is None else row["value"].strip().lower() for row in folded] == [
        row["value"] for row in original
    ]
    assert set(row["value"] for row in original if row["value"] is not None) == set(proof.LABELS)


def test_hash_generator_preserves_large_integer_precision_and_within_row_equality():
    proof = load("hash_shifts")
    rows = list(proof.records(rows=128, seed=11))
    assert all(row["value"] == row["peer"] for row in rows)
    values = {row["value"] for row in rows if row["value"] is not None}
    assert len(values) == 4
    assert len({float(value) for value in values}) == 1


@pytest.mark.parametrize(
    "name, kind", [("enum_shifts", "enum"), ("hash_shifts", "hash"), ("cluster_shifts", "cluster")]
)
def test_identity_studies_keep_target_hidden_and_monitoring_enabled(name, kind):
    import relflow as rf

    model = load(name).build()
    assert model.drift is True
    assert model.schema.requests["/value"].type == kind
    assert model.schema.requests["/value"].drift == rf.Drift()
    assert all(mask.skip and mask.reconstruct for mask in model.schema.requests["/target"].mask)
