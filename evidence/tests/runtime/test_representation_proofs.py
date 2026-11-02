"""Protect representation proofs against leakage and misleading controls."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import relflow as rf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"representation_proof_{name}", ROOT / "experiments/representation" / f"{name}.py"
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


@pytest.mark.parametrize("outputs", (1, 3))
def test_frozen_probe_reads_every_root_representation_and_excludes_leaf_embeddings(outputs):
    proof = load("reconstruction_transfer")
    model = rf.Model(
        d_model=16,
        n_layers=1,
        n_heads=4,
        embed=True,
        reduction=rf.Attention(n_outputs=outputs),
        amount=rf.Number(embed=True),
    )
    rows = [{"amount": 1.0}, {"amount": 2.0}]
    records = model.predict(pa.Table.from_pylist(rows))["representations"].to_pylist()
    assert any(record["path"] == ["amount"] for record in records[0])
    root = [[record["embedding"] for record in row if record["path"] == []] for row in records]
    actual = proof.embedding(model, rows)
    assert actual.shape == (len(rows), outputs * 16)
    np.testing.assert_allclose(actual, np.asarray(root).reshape(len(rows), -1))
    np.testing.assert_allclose(np.linalg.norm(actual.reshape(len(rows), outputs, 16), axis=-1), 1.0, atol=1e-6)


@pytest.mark.parametrize(
    ("generator", "keys"),
    [("multiplicative", {"price", "rate", "ltv"}), ("additive", {"left", "right", "total"})],
)
def test_number_lanes_generators_restart_and_expose_only_visible_fields(generator, keys):
    proof = load("number_lanes")
    records = getattr(proof, generator)
    rows = list(records(rows=64, seed=11))
    assert rows == list(records(rows=64, seed=11))
    assert rows != list(records(rows=64, seed=12))
    assert all(set(row) == keys for row in rows)
    assert proof.TRAIN_ROWS >= 4096 and proof.TEST_ROWS >= 2048


def test_number_lanes_multiplicative_oracle_is_the_conditional_median_of_heavy_tails():
    proof = load("number_lanes")
    rows = list(proof.multiplicative(rows=16384, seed=21))
    price = np.asarray([row["price"] for row in rows])
    rate = np.asarray([row["rate"] for row in rows])
    ltv = np.asarray([row["ltv"] for row in rows])
    oracle = proof.median("heavy", rows)
    np.testing.assert_allclose(oracle, price * np.sqrt(rate))
    assert abs(np.median(ltv / oracle) - 1.0) < 0.03
    assert abs(np.mean(np.log(ltv / oracle))) < 0.02
    assert abs(np.std(np.log(ltv / oracle)) - proof.NOISE) < 0.02
    assert price.mean() > 1.5 * np.median(price)
    assert rate.max() > 5000 * np.median(rate)
    assert ltv.max() > 50 * np.median(ltv)
    moderate = np.asarray([row["rate"] for row in proof.RECORDS["moderate"](rows=16384, seed=21)])
    assert 50 * np.median(moderate) < moderate.max() < 2000 * np.median(moderate)


def test_number_lanes_additive_oracle_residual_has_the_declared_noise():
    proof = load("number_lanes")
    rows = list(proof.additive(rows=16384, seed=31))
    residual = np.asarray([row["total"] for row in rows]) - proof.median("additive", rows)
    assert abs(residual.mean()) < 0.1
    np.testing.assert_allclose(residual.std(), proof.ADDITIVE_NOISE, rtol=0.05)
    assert all(-proof.SPAN <= row["left"] <= proof.SPAN and -proof.SPAN <= row["right"] <= proof.SPAN for row in rows)


@pytest.mark.parametrize("inputs", ["number", "rank"])
@pytest.mark.parametrize("target", ["number", "rank"])
def test_number_lanes_arms_change_only_the_numeric_datatype(inputs, target):
    proof = load("number_lanes")
    model = proof.build("heavy", inputs, target)
    requests = model.schema.requests
    assert set(requests) == {rf.Address("price"), rf.Address("rate"), rf.Address("ltv")}
    for name in ("price", "rate"):
        request = requests[rf.Address(name)]
        assert not request.mask
        assert request.type == ("quantile" if inputs == "rank" else "number")
    hidden = requests[rf.Address("ltv")]
    assert all(mask.skip and mask.reconstruct for mask in hidden.mask)
    assert hidden.type == ("quantile" if target == "rank" else "number") and hidden.objective == "mae"


def test_number_lanes_narrow_task_is_the_additive_task_moved_away_from_zero():
    proof = load("number_lanes")
    centered = list(proof.RECORDS["additive"](rows=256, seed=5))
    moved = list(proof.RECORDS["narrow"](rows=256, seed=5))
    assert [row["left"] for row in centered] == [row["left"] for row in moved]
    np.testing.assert_allclose([row["total"] + proof.OFFSET for row in centered], [row["total"] for row in moved])
    np.testing.assert_allclose(proof.median("narrow", moved), proof.median("additive", centered) + proof.OFFSET)
