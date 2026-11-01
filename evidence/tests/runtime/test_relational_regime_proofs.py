"""Information-boundary tests for retrieval capacity and hidden-regime proofs."""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import relflow as rf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))


def load(directory, name):
    spec = importlib.util.spec_from_file_location(f"proof_{name}", ROOT / "experiments" / directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("hops", (1, 2))
def test_retrieval_oracle_requires_each_link_and_uses_fresh_names(hops):
    proof = load("relational", "retrieval_capacity")
    options = {"rows": 64, "seed": 31, "namespace": "train", "hops": hops}
    original = list(proof.records(**options))
    assert original == list(proof.records(**options))
    broken = list(proof.records(**options, broken=True))
    renamed = list(proof.records(**{**options, "namespace": "test"}))
    train_keys, test_keys = set(), set()
    sizes = Counter()
    for row, corrupted, unseen in zip(original, broken, renamed, strict=True):
        source = {item["entity_id"]: item["value"] for item in row["source"]}
        assert len(source) == len(row["source"])
        sizes[len(source)] += 1
        assert [item["value"] for item in row["target"]] == [item["value"] for item in corrupted["target"]]
        assert row["source"] == corrupted["source"]
        if hops == 1:
            assert all(source[item["entity_id"]] == item["value"] for item in row["target"])
            assert all(source[item["entity_id"]] != item["value"] for item in corrupted["target"])
        else:
            links = {item["entity_id"]: item["source_id"] for item in row["links"]}
            rotated = {item["entity_id"]: item["source_id"] for item in corrupted["links"]}
            assert set(links.values()) == set(source)
            assert set(rotated.values()) == set(source)
            assert all(source[links[item["entity_id"]]] == item["value"] for item in row["target"])
            assert all(source[rotated[item["entity_id"]]] != item["value"] for item in corrupted["target"])
        train_keys.update(source)
        test_keys.update(item["entity_id"] for item in unseen["source"])
    expected = proof.CAPACITIES if hops == 1 else proof.CAPACITIES[:2]
    assert sizes == dict.fromkeys(expected, 64 // len(expected))
    assert not train_keys & test_keys


def test_permutation_changes_positions_without_changing_keyed_relations():
    proof = load("relational", "retrieval_capacity")
    rows = list(proof.records(rows=32, seed=51, namespace="test", hops=2))
    permuted = proof.permute(rows, 52)
    assert rows != permuted
    for before, after in zip(rows, permuted, strict=True):
        for name in before:
            assert sorted(before[name], key=lambda item: item["entity_id"]) == sorted(
                after[name], key=lambda item: item["entity_id"]
            )


def test_regime_control_preserves_numeric_pairs_and_identity_counts():
    proof = load("cluster", "regime_recovery")
    options = {"rows": 4096, "seed": 21, "partition_seed": 20}
    stable = list(proof.records(**options))
    control = list(proof.records(**options, no_regime=True))
    validation = list(proof.records(**{**options, "seed": 22}))
    assert stable == list(proof.records(**options))
    assert stable != validation
    assert Counter(row["entity"] for row in stable) == Counter(row["entity"] for row in control)
    assert Counter(row["entity"] for row in stable) == Counter(row["entity"] for row in validation)
    assert [(row["x"], row["y"]) for row in stable] == [(row["x"], row["y"]) for row in control]
    assert set(stable[0]) == {"entity", "x", "y"}
    groups = proof.partition(20)
    assert Counter(groups) == dict.fromkeys(range(4), 16)
    errors = []
    wrong = []
    for row, shuffled in zip(stable, control, strict=True):
        group = groups[int(row["entity"].split("-")[1])]
        errors.append(row["y"] - proof.INTERCEPTS[group] - proof.SLOPES[group] * row["x"])
        wrong_group = groups[int(shuffled["entity"].split("-")[1])]
        wrong.append(shuffled["y"] - proof.INTERCEPTS[wrong_group] - proof.SLOPES[wrong_group] * shuffled["x"])
    assert 0.09 < np.std(errors) < 0.11
    assert np.sqrt(np.mean(np.square(wrong))) > 5.0


def test_adjusted_rand_ignores_label_names_and_penalizes_wrong_partitions():
    proof = load("cluster", "regime_recovery")
    actual = np.repeat(np.arange(4), 16)
    assert proof.adjusted_rand(actual, 10 - actual) == 1.0
    assert proof.adjusted_rand(actual, np.zeros(64)) == 0.0
    assert proof.adjusted_rand(actual, np.arange(64)) == 0.0
    assert abs(proof.adjusted_rand(actual, np.tile(np.arange(4), 16))) < 0.1


@pytest.mark.parametrize(
    "name,target,arguments",
    (
        ("aligned_pair_covariance", "covariance", ()),
        ("attention_sum_in_range", "total", ()),
        ("attention_sum_unseen_lengths", "total", ()),
        ("mean_duplicate_invariance", "mean_amount", ()),
        ("mean_erases_contribution_count", "total", ("total",)),
        ("raw_value_variance", "variance", ()),
        ("raw_value_weight_sum", "weighted_sum", ()),
        ("requested_rank_values", "answer", ()),
        ("selected_group_raw_weighted_sum", "answer", ()),
        ("selected_group_supplied_contribution_sum", "answer", ()),
        ("supplied_contribution_sum", "weighted_sum", ()),
        ("supplied_cross_deviations", "covariance", ()),
        ("supplied_squared_deviations", "variance", ()),
        ("variable_cardinality_weighted_mean", "weighted_mean", ()),
        ("visible_count_sum", "total", ()),
        ("visible_count_unseen_lengths", "total", ()),
    ),
)
def test_aggregation_readers_accept_named_scalar_outputs(name, target, arguments):
    proof = load("aggregation", name)
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        attention=None,
        fields={"items": rf.Branch(length=5, value=rf.Number), target: rf.Number(mask=True)},
    )
    rows = [
        {"items": [], "item_count": 0, "rank": "first", "selected_group": "A"},
        {"items": [{"value": 2.0}], "item_count": 1, "rank": "first", "selected_group": "A"},
    ]
    predicted = proof.prediction(model, rows, *arguments)
    assert predicted.shape == (2,)
    assert np.isfinite(predicted).all()


@pytest.mark.parametrize(
    "name",
    ("attention_learns_fixed_width_sum", "attention_preserves_nested_cardinality", "mean_erases_cardinality"),
)
def test_structure_readers_accept_named_scalar_outputs(name):
    proof = load("structure", name)
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        attention=None,
        items=rf.Branch(length=5, value=rf.Number),
        total=rf.Number(mask=True),
    )
    predicted = proof.prediction(model, [{"items": []}, {"items": [{"value": 2.0}]}], "items", "total")
    assert predicted.shape == (2,)
    assert np.isfinite(predicted).all()


@pytest.mark.parametrize(
    "name",
    (
        "ungrouped_peer_deviation",
        "grouped_peer_deviation",
        "single_query_preserves_group_membership",
        "supplied_peer_mean_control",
    ),
)
def test_peer_readers_score_real_items_without_cap_padding(name):
    proof = load("relational", name)
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        attention=None,
        items=rf.Branch(length=5, value=rf.Number, deviation=rf.Number(mask=True)),
    )
    rows = [{"items": []}, {"items": [{"value": 2.0, "deviation": 0.0}, {"value": 3.0, "deviation": 0.0}]}]
    predicted = proof.predict(model, rows)
    assert predicted.shape == (2,)
    assert np.isfinite(predicted).all()


@pytest.mark.parametrize("hops", (1, 2))
def test_retrieval_reader_preserves_target_collection_width(hops):
    proof = load("relational", "retrieval_capacity")
    rows = list(proof.records(rows=4, seed=31, namespace="test", hops=hops))
    predicted = proof.predict(proof.build(hops), rows)
    assert predicted.shape == (4, 2)
    assert np.isfinite(predicted).all()


def test_subtotal_reader_preserves_each_observed_length_and_inferred_flag():
    proof = load("relational", "repeated_subtotal_reconstruction")
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        attention=None,
        items=rf.Branch(
            length=proof.LENGTH,
            quantity=rf.Number,
            unit_price=rf.Number,
            subtotal=rf.Number(mask=True),
        ),
    )
    rows = list(proof.records(rows=proof.LENGTH + 1, seed=31))
    coordinates = proof.prediction(model, rows)
    assert [len(row) for row in coordinates] == list(range(proof.LENGTH + 1))
    assert all(value["inferred"] for row in coordinates for value in row)
    assert all(np.isfinite(value["content"]) for row in coordinates for value in row)
