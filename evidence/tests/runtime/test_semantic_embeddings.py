"""Protect unsupervised semantic experiments at their data and prediction boundaries."""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import relflow as rf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
EXPERIMENTS = ("semantic_clusters", "semantic_robustness", "nested_history_clusters")
FIELDS = {
    "semantic_clusters": {"merchant_pattern", "payment_rhythm", "cashflow"},
    "semantic_robustness": {"channel", "amount", "frequency", "nuisance"},
}


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"semantic_proof_{name}", ROOT / "experiments/representation" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encoded(row):
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


def assert_same(before, after):
    """Include tensor buffers and nested non-tensor extension state in the freeze contract."""
    assert type(before) is type(after)
    if isinstance(before, torch.Tensor):
        torch.testing.assert_close(before, after, atol=0.0, rtol=0.0, equal_nan=True)
    elif isinstance(before, np.ndarray):
        np.testing.assert_array_equal(before, after)
    elif isinstance(before, dict):
        assert before.keys() == after.keys()
        for key in before:
            assert_same(before[key], after[key])
    elif isinstance(before, (list, tuple)):
        assert len(before) == len(after)
        for left, right in zip(before, after, strict=True):
            assert_same(left, right)
    elif isinstance(before, float) and np.isnan(before):
        assert np.isnan(after)
    else:
        assert before == after


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_latent_labels_are_out_of_band_and_split_streams_are_fresh(name):
    proof = load(name)
    rows, labels = proof.sample(rows=128, seed=41)
    again, same_labels = proof.sample(rows=128, seed=41)
    held_out, _ = proof.sample(rows=128, seed=42)
    assert rows == again == list(proof.records(rows=128, seed=41))
    np.testing.assert_array_equal(labels, same_labels)
    np.testing.assert_array_equal(np.bincount(labels), np.full(4, 32))
    assert not {encoded(row) for row in rows} & {encoded(row) for row in held_out}
    if name in FIELDS:
        assert all(set(row) == FIELDS[name] for row in rows)
    assert not {"label", "target", "latent", "class", "cluster"} & set(rows[0])
    model = proof.build()
    assert model.schema.objectives
    assert all(
        str(address).split("/")[-1] not in {"label", "target", "latent", "class", "cluster"}
        for address in model.schema.requests
    )
    for address in model.schema.objectives:
        masks = model.schema.requests[address].mask
        assert any(mask.reconstruct and not mask.skip and mask.rate > 0 for mask in masks)


def test_independently_permuted_views_keep_all_marginals_but_break_their_label_alignment():
    proof = load("semantic_clusters")
    original, labels = proof.sample(rows=4096, seed=51)
    permuted, control_labels = proof.sample(rows=4096, seed=51, permuted=True)
    np.testing.assert_array_equal(labels, control_labels)
    for field in FIELDS["semantic_clusters"]:
        assert Counter(encoded(row[field]) for row in original) == Counter(encoded(row[field]) for row in permuted)
        assert np.mean([left[field] != right[field] for left, right in zip(original, permuted, strict=True)]) > 0.7
    signal = np.asarray([row["cashflow"][:2] for row in original])
    shuffled = np.asarray([row["cashflow"][:2] for row in permuted])
    centers = np.asarray(proof.CENTERS)
    aligned = np.argmin(((signal[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1), axis=1)
    broken = np.argmin(((shuffled[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1), axis=1)
    assert np.mean(aligned == labels) > 0.75
    assert 0.18 < np.mean(broken == labels) < 0.32


def test_robustness_conditions_change_only_the_declared_views_without_mutating_rows():
    proof = load("semantic_robustness")
    rows, _ = proof.sample(rows=256, seed=61)
    before = copy.deepcopy(rows)
    complete = proof.condition(rows, "complete", seed=62)
    missing = proof.condition(rows, "one_view_missing", seed=62)
    nuisance = proof.condition(rows, "nuisance_resampled", seed=62)
    absent = proof.condition(rows, "all_semantic_views_missing", seed=62)
    assert rows == before == complete
    assert nuisance == proof.condition(rows, "nuisance_resampled", seed=62)
    assert nuisance != proof.condition(rows, "nuisance_resampled", seed=63)
    for original, one, resampled, all_missing in zip(rows, missing, nuisance, absent, strict=True):
        assert set(one) == set(resampled) == set(all_missing) == set(original)
        assert one["channel"] is None
        assert {key: value for key, value in one.items() if key != "channel"} == {
            key: value for key, value in original.items() if key != "channel"
        }
        assert resampled["nuisance"] != original["nuisance"]
        assert len(resampled["nuisance"]) == len(original["nuisance"]) == 16
        assert {key: value for key, value in resampled.items() if key != "nuisance"} == {
            key: value for key, value in original.items() if key != "nuisance"
        }
        assert all(all_missing[field] is None for field in ("channel", "amount", "frequency"))
        assert all_missing["nuisance"] == original["nuisance"]
    complete[0]["nuisance"][0] += 1
    assert rows == before


def test_history_permutation_moves_whole_records_and_keeps_lengths_and_global_marginals():
    proof = load("nested_history_clusters")
    original, labels = proof.sample(rows=256, seed=71, lengths=(4, 8))
    shuffled, shuffled_labels = proof.sample(rows=256, seed=71, lengths=(4, 8), permuted=True)
    np.testing.assert_array_equal(labels, shuffled_labels)
    field = next(key for key, value in original[0].items() if isinstance(value, list))
    assert [len(row[field]) for row in original] == [len(row[field]) for row in shuffled]
    assert Counter(encoded(item) for row in original for item in row[field]) == Counter(
        encoded(item) for row in shuffled for item in row[field]
    )
    assert np.mean([left[field] != right[field] for left, right in zip(original, shuffled, strict=True)]) > 0.9
    assert all(4 <= len(row[field]) <= 8 for row in original)
    assert all(set(row) == {field} for row in original)
    assert set(original[0][field][0]) == {"merchant", "amount", "interval_days", "terminal"}


def test_history_length_and_reversal_controls_preserve_exact_paired_records():
    proof = load("nested_history_clusters")
    rows, _ = proof.sample(rows=32, seed=81, lengths=(12, 12))
    before = copy.deepcopy(rows)
    field = next(key for key, value in rows[0].items() if isinstance(value, list))
    for length in (4, 8, 12):
        cropped = proof.condition(rows, length)
        reversed_rows = proof.condition(rows, length, reverse=True)
        for original, short, reverse in zip(rows, cropped, reversed_rows, strict=True):
            assert short[field] == original[field][:length]
            assert reverse[field] == list(reversed(original[field][:length]))
    assert rows == before
    branch = proof.build().schema.branches[rf.Address(field)]
    assert branch.length >= 12


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_adjusted_rand_scores_partitions_independently_of_label_names(name):
    proof = load(name)
    actual = np.repeat(np.arange(4), 1024)
    assert proof.ari(actual, 20 - actual) == pytest.approx(1.0)
    assert proof.ari(actual, np.zeros(len(actual), dtype=int)) == pytest.approx(0.0)
    assert proof.ari(actual, np.arange(len(actual))) == pytest.approx(0.0)
    assert abs(proof.ari(actual, np.random.default_rng(91).permutation(actual))) < 0.02
    assert proof.ari(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])) == pytest.approx(-0.5)
    prediction = np.tile(np.arange(4), 1024)
    assert proof.ari(actual, prediction) == pytest.approx(proof.ari(prediction, actual))


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_kmeans_fits_only_unlabeled_training_geometry_and_is_deterministic(name):
    proof = load(name)
    assert list(inspect.signature(proof.cluster).parameters) == ["values", "seed"]
    rng = np.random.default_rng(101)
    truth = np.repeat(np.arange(4), 32)
    train = np.column_stack((np.array([-9.0, -3.0, 3.0, 9.0])[truth], np.zeros(len(truth))))
    train += rng.normal(0, 0.03, train.shape)
    before = train.copy()
    centers = proof.cluster(train, seed=102)
    np.testing.assert_array_equal(train, before)
    np.testing.assert_array_equal(centers, proof.cluster(train, seed=102))
    assert centers.shape == (4, 2)
    assigned = proof.assign(train, centers)
    assert proof.ari(truth, assigned) == pytest.approx(1.0)
    squared = proof.distances(train, centers)
    np.testing.assert_array_equal(assigned, squared.argmin(axis=1))
    assert squared.shape == (len(train), 4)
    np.testing.assert_allclose(squared[np.arange(len(train)), assigned], ((train - centers[assigned]) ** 2).sum(axis=1))
    relabeled = 30 - truth
    assert proof.ari(relabeled, assigned) == pytest.approx(1.0)
    held_out = np.array([[-9.01, 0.0], [-2.99, 0.0], [3.01, 0.0], [8.99, 0.0]])
    first = proof.assign(held_out, centers)
    proof.assign(np.full((64, 2), 1e9), centers)
    np.testing.assert_array_equal(centers, proof.cluster(before, seed=102))
    np.testing.assert_array_equal(first, proof.assign(held_out, centers))
    assert len(set(first)) == 4


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_embedding_reader_keeps_row_alignment_and_ignores_leaf_representation_noise(name):
    proof = load(name)
    expected = np.eye(4, 64, dtype=np.float32)

    class Model:
        def predict(self, table, *, batch_size):
            assert isinstance(table, pa.Table)
            assert batch_size == 2
            records = []
            for ordinal in table["ordinal"].to_pylist():
                leaf = {"path": ["noise"], "parents": [], "output_index": 0, "embedding": [999.0] * 64}
                root = {"path": [], "parents": [], "output_index": 0, "embedding": expected[ordinal].tolist()}
                records.append([leaf, root, {**leaf, "path": ["other"]}])
            return pa.table({"representations": records})

    actual = proof.embedding(Model(), [{"ordinal": index} for index in range(4)], batch_size=2)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("name", ("semantic_clusters", "nested_history_clusters"))
def test_real_frozen_exports_preserve_all_state_and_work_across_batch_sizes(name):
    proof = load(name)
    rows, _ = proof.sample(rows=8, seed=111)
    model = proof.build()
    leaf_type = "vector" if name == "semantic_clusters" else "number"
    model.update(rf.where("type") == leaf_type, embed=True)
    model.encode(pa.Table.from_pylist(rows), strata="train")
    model.eval()
    before = copy.deepcopy(model.state_dict())
    first = proof.embedding(model, rows, batch_size=1)
    second = proof.embedding(model, rows, batch_size=3)
    assert first.shape == second.shape == (len(rows), model.schema.d_model)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)
    after = model.state_dict()
    assert before.keys() == after.keys()
    assert_same(before, after)


@pytest.mark.parametrize("name", ("semantic_robustness", "nested_history_clusters"))
def test_condition_centroids_never_fit_test_variants_or_evaluator_labels(monkeypatch, name):
    proof = load(name)
    training = np.repeat(np.eye(4), 8, axis=0)
    labels = np.tile(np.arange(4), 4)
    queries = {"complete": np.tile(np.eye(4), (4, 1))}
    calls = []
    original = proof.cluster

    def fit(values, seed):
        np.testing.assert_array_equal(values, training)
        centroids = original(values, seed)
        calls.append(centroids.copy())
        return centroids

    monkeypatch.setattr(proof, "cluster", fit)
    first = proof.measure(training, queries, labels, seed=121)
    altered = proof.measure(
        training,
        {"complete": queries["complete"] * 1000, "removed": np.zeros_like(queries["complete"])},
        np.random.default_rng(122).permutation(labels),
        seed=121,
    )
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], calls[1])
    assert first["train_inertia"] == altered["train_inertia"]


@pytest.mark.parametrize("name", EXPERIMENTS)
@pytest.mark.parametrize("malformed", ("missing", "duplicate", "wrong_parent", "lost_row"))
def test_root_reader_rejects_ambiguous_or_misaligned_exports(name, malformed):
    proof = load(name)

    class Model:
        def predict(self, table, *, batch_size):
            root = {"path": [], "parents": [], "output_index": 0, "embedding": [0.125] * 64}
            leaf = {**root, "path": ["noise"]}
            first = [leaf, root]
            if malformed == "missing":
                first = [leaf]
            elif malformed == "duplicate":
                first = [root, root]
            elif malformed == "wrong_parent":
                first = [leaf, {**root, "parents": [0]}]
            rows = [first] if malformed == "lost_row" else [first, [root]]
            return pa.table({"representations": rows})

    with pytest.raises(ValueError):
        proof.embedding(Model(), [{"ordinal": 0}, {"ordinal": 1}], batch_size=2)


def test_semantic_evaluator_fits_centroids_without_training_or_test_labels(monkeypatch):
    proof = load("semantic_clusters")
    training_labels = np.repeat(np.arange(4), 8)
    training = np.repeat(np.eye(4), 8, axis=0)
    query_labels = np.tile(np.arange(4), 4)
    query = np.tile(np.eye(4), (4, 1))
    fitted = []
    original = proof.cluster

    def fit(values, seed):
        np.testing.assert_array_equal(values, training)
        centroids = original(values, seed)
        fitted.append(centroids.copy())
        return centroids

    monkeypatch.setattr(proof, "cluster", fit)
    first = proof.evaluate(training, training_labels, query, query_labels, seed=131)
    second = proof.evaluate(
        training,
        np.random.default_rng(132).permutation(training_labels),
        query * 1000,
        np.random.default_rng(133).permutation(query_labels),
        seed=131,
    )
    assert len(fitted) == 2
    np.testing.assert_array_equal(fitted[0], fitted[1])
    assert first["train_inertia"] == second["train_inertia"]


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_freeze_check_includes_buffers_and_extension_owned_nested_state(name):
    proof = load(name)
    equal = proof.same_state if name == "semantic_clusters" else proof.equal
    state = {
        "parameter": torch.tensor([1.0, 2.0]),
        "normalizer": {"count": torch.tensor(8), "moments": np.array([3.0, 4.0])},
        "extension": ({"vocabulary": ["first", "second"]},),
    }
    assert equal(state, copy.deepcopy(state))
    changed = copy.deepcopy(state)
    changed["normalizer"]["count"] += 1
    assert not equal(state, changed)
    changed = copy.deepcopy(state)
    changed["extension"][0]["vocabulary"].append("third")
    assert not equal(state, changed)
    changed = copy.deepcopy(state)
    changed["parameter"][0] += 1
    assert not equal(state, changed)


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_collapsed_geometry_is_reportable_without_inventing_distinct_clusters(name):
    proof = load(name)
    training = np.zeros((16, 5))
    centers = proof.cluster(training, seed=141)
    np.testing.assert_array_equal(centers, np.zeros((4, 5)))
    guessed = proof.assign(training, centers)
    assert len(set(guessed)) == 1
    assert proof.ari(np.tile(np.arange(4), 4), guessed) == pytest.approx(0.0)


@pytest.mark.parametrize("name", EXPERIMENTS)
@pytest.mark.parametrize("values", (np.zeros((3, 2)), np.ones(8), np.full((8, 2), np.nan)))
def test_centroid_fitting_rejects_invalid_training_geometry(name, values):
    proof = load(name)
    with pytest.raises(ValueError):
        proof.cluster(values, seed=151)
