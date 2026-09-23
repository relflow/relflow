"""Protect the Quantile proofs' observability and paired evaluation controls."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import relflow as rf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proofs"))


def load(name):
    spec = importlib.util.spec_from_file_location(f"quantile_proof_{name}", ROOT / "proofs/quantile" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["rank_regression", "clipped_extrapolation"])
def test_quantile_proof_streams_restart_independently_without_latent_columns(name):
    proof = load(name)
    rows = list(proof.records(rows=32, seed=11))
    assert rows == list(proof.records(rows=32, seed=11))
    assert rows != list(proof.records(rows=32, seed=12))
    assert all(set(row) == {"amount", "cost"} for row in rows)
    assert proof.TRAIN_ROWS >= 4096
    assert proof.TEST_ROWS >= 2048


@pytest.mark.parametrize("name", ["rank_regression", "clipped_extrapolation"])
def test_quantile_proof_schema_keeps_the_only_target_hidden(name):
    model = load(name).build()
    assert model.schema.requests[rf.Address("amount")].type == "quantile"
    target = model.schema.requests[rf.Address("cost")]
    assert target.type == "quantile"
    assert target.objective == "mse"
    assert all(mask.skip and mask.reconstruct for mask in target.mask)
    assert set(model.schema.requests) == {rf.Address("amount"), rf.Address("cost")}


def test_rank_generator_has_skew_and_independent_source_unit_noise():
    rows = list(load("rank_regression").records(rows=8192, seed=11))
    amount = np.asarray([row["amount"] for row in rows])
    residual = np.asarray([row["cost"] for row in rows]) - 200.0 - 3.0 * amount
    assert amount.min() >= 0
    assert amount.max() <= 50.0 * np.expm1(3.0)
    assert amount.mean() > 1.25 * np.median(amount)
    assert 9.0 < np.sqrt(np.mean(residual**2)) < 11.0
    assert abs(np.corrcoef(amount, residual)[0, 1]) < 0.05


@pytest.mark.parametrize("shift", [-30.0, 30.0])
def test_shifted_control_retains_noise_but_moves_beyond_training_support(shift):
    proof = load("clipped_extrapolation")
    original = list(proof.records(rows=2048, seed=11))
    shifted = list(proof.records(rows=2048, seed=11, shift=shift))
    amount = np.asarray([row["amount"] for row in original])
    target = np.asarray([row["cost"] for row in original])
    changed_amount = np.asarray([row["amount"] for row in shifted])
    changed_target = np.asarray([row["cost"] for row in shifted])
    np.testing.assert_allclose(changed_amount, amount + shift, rtol=0, atol=1e-12)
    np.testing.assert_allclose(changed_target, target + 4.0 * shift, rtol=0, atol=1e-12)
    assert np.all((amount >= 10) & (amount <= 30))
    assert np.all(np.abs(target - 100.0 - 4.0 * amount) <= 1)
    assert np.all(changed_target < 139) if shift < 0 else np.all(changed_target > 221)
    clipping_bound = np.sqrt(np.mean((np.clip(changed_target, 139, 221) - changed_target) ** 2))
    assert clipping_bound > 50
