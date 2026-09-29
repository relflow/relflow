"""Protect representation proofs against leakage and misleading controls."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

import relflow as rf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proofs"))


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"representation_proof_{name}", ROOT / "proofs/representation" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reconstruction_labels_never_enter_training_records_or_schema():
    proof = load("reconstruction_transfer")
    rows, labels = proof.sample(rows=128, seed=11)
    assert rows == proof.sample(rows=128, seed=11)[0]
    assert rows != proof.sample(rows=128, seed=12)[0]
    np.testing.assert_array_equal(np.bincount(labels), [32, 32, 32, 32])
    assert rows == list(proof.records(rows=128, seed=11))
    assert all(set(row) == {"first", "second", "geometry"} for row in rows)
    model = proof.build()
    assert set(model.schema.requests) == {rf.Address("first"), rf.Address("second"), rf.Address("geometry")}
    for request in model.schema.requests.values():
        assert len(request.mask) == 1
        assert request.mask[0].rate == 0.35
        assert request.mask[0].reconstruct


def test_permuted_reconstruction_views_preserve_marginals_and_destroy_common_signal():
    proof = load("reconstruction_transfer")
    original, labels = proof.sample(rows=8192, seed=11)
    shuffled, shuffled_labels = proof.sample(rows=8192, seed=11, permuted=True)
    np.testing.assert_array_equal(labels, shuffled_labels)
    for name in ("first", "second"):
        assert sorted(row[name] for row in original) == sorted(row[name] for row in shuffled)
    assert sorted(tuple(row["geometry"]) for row in original) == sorted(tuple(row["geometry"]) for row in shuffled)
    for rows, low, high in ((original, 0.78, 0.85), (shuffled, 0.22, 0.28)):
        for name in ("first", "second"):
            inferred = np.asarray([proof.CODES.index(row[name]) // 2 for row in rows])
            assert low < np.mean(inferred == labels) < high
    signal = np.asarray([row["geometry"][:2] for row in original])
    residual = signal - proof.CENTERS[labels]
    assert abs(residual.mean()) < 0.03
    assert 0.75 < residual.std() < 0.85


def test_frozen_probe_has_a_small_balanced_fit_set_and_uses_all_raw_coordinates():
    proof = load("reconstruction_transfer")
    rows, labels = proof.sample(rows=4 * proof.LABELS_PER_CLASS, seed=21)
    np.testing.assert_array_equal(np.bincount(labels), np.full(4, 8))
    raw = proof.raw_features(rows)
    assert raw.shape == (32, 32)
    np.testing.assert_array_equal(raw[:, :8].sum(axis=1), np.ones(32))
    np.testing.assert_array_equal(raw[:, 8:16].sum(axis=1), np.ones(32))
    np.testing.assert_array_equal(raw[:, 16:], [row["geometry"] for row in rows])
    assert proof.TRAIN_ROWS >= 4096 and proof.TEST_ROWS >= 2048


def test_sensor_missingness_is_independent_of_values_and_holds_out_one_pattern():
    proof = load("incomplete_evidence")
    natural = list(proof.records(rows=8192, seed=31))
    training = list(proof.records(rows=8192, seed=31, training=True))
    patterns = {tuple(row[f"hide_{name}"] for name in proof.SENSORS) for row in training}
    assert patterns == set(proof.PATTERNS) - {proof.UNSEEN}
    for name in (*proof.SENSORS, "nuisance", "target"):
        np.testing.assert_array_equal([row[name] for row in natural], [row[name] for row in training])
    for pattern in patterns:
        selected = [
            row["target"] for row in training if tuple(row[f"hide_{name}"] for name in proof.SENSORS) == pattern
        ]
        assert abs(np.mean(selected)) < 0.10
    targets = np.asarray([row["target"] for row in natural])
    errors = np.asarray([[row[name] - row["target"] for name in proof.SENSORS] for row in natural])
    np.testing.assert_allclose(errors.std(axis=0), proof.NOISE, rtol=0.04)
    np.testing.assert_allclose(np.corrcoef(np.column_stack((targets, errors)).T), np.eye(4), atol=0.04)


def test_gaussian_oracle_matches_empirical_conditional_error_for_all_patterns():
    proof = load("incomplete_evidence")
    rows = list(proof.records(rows=16384, seed=41))
    targets = np.asarray([row["target"] for row in rows])
    for pattern in proof.PATTERNS:
        prediction, variance = proof.oracle(rows, pattern)
        assert abs(np.mean((prediction - targets) ** 2) / variance - 1.0) < 0.05
    zero, variance = proof.oracle(rows, (True, True, True))
    np.testing.assert_array_equal(zero, np.zeros(len(rows)))
    assert variance == 1.0


def test_sensor_schema_hides_target_and_respects_each_explicit_selector():
    proof = load("incomplete_evidence")
    model = proof.build()
    for name in proof.SENSORS:
        mask = model.schema.requests[rf.Address(name)].mask
        assert len(mask) == 1 and mask[0].query == f"hide_{name}"
        assert mask[0].skip and not mask[0].reconstruct
    target = model.schema.requests[rf.Address("target")]
    assert all(mask.skip and mask.reconstruct for mask in target.mask)
    assert not model.schema.requests[rf.Address("nuisance")].mask
    rows = list(proof.records(rows=32, seed=11))
    selected = proof.select(rows, proof.UNSEEN)
    assert all(tuple(row[f"hide_{name}"] for name in proof.SENSORS) == proof.UNSEEN for row in selected)
    assert all(a["target"] == b["target"] for a, b in zip(rows, selected, strict=True))
