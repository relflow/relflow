"""Protect the consistency and numerical tradeoff experiments' information boundaries."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proofs"))


def load(family, name):
    spec = importlib.util.spec_from_file_location(f"proof_{name}", ROOT / "proofs" / family / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_anomalies_preserve_marginals_and_change_only_the_relationship():
    proof = load("anomaly", "contextual_consistency")
    clean = list(proof.records(rows=4096, seed=11))
    broken = list(proof.records(rows=4096, seed=11, shuffled=True))
    assert clean == list(proof.records(rows=4096, seed=11))
    assert clean != list(proof.records(rows=4096, seed=12))
    assert [(row["x"], row["group"]) for row in clean] == [(row["x"], row["group"]) for row in broken]
    assert sorted(row["y"] for row in clean) == sorted(row["y"] for row in broken)
    expected = np.asarray(
        [
            proof.SLOPES[proof.GROUPS.index(row["group"])] * row["x"] + proof.OFFSETS[proof.GROUPS.index(row["group"])]
            for row in clean
        ]
    )
    error = np.asarray([row["y"] for row in clean]) - expected
    changed = np.asarray([row["y"] for row in broken]) - expected
    assert 0.09 < np.sqrt(np.mean(error**2)) < 0.11
    assert np.sqrt(np.mean(changed**2)) > 1
    assert proof.auc(np.abs(error), np.abs(changed)) > 0.9


def test_valid_tail_control_stays_inside_support_and_preserves_the_rule():
    proof = load("anomaly", "contextual_consistency")
    tail = list(proof.records(rows=2048, seed=11, tail=True))
    assert all(1.8 <= abs(row["x"]) <= 2 for row in tail)
    errors = [
        row["y"]
        - proof.SLOPES[proof.GROUPS.index(row["group"])] * row["x"]
        - proof.OFFSETS[proof.GROUPS.index(row["group"])]
        for row in tail
    ]
    assert 0.09 < np.sqrt(np.mean(np.square(errors))) < 0.11


@pytest.mark.parametrize("anomaly, expected", [([3, 4], 1.0), ([0, 0], 0.0), ([1, 2], 0.5)])
def test_anomaly_auc_handles_ties(anomaly, expected):
    proof = load("anomaly", "contextual_consistency")
    assert proof.auc(np.array([1, 2]), np.array(anomaly)) == expected


def test_measurement_contamination_keeps_targets_and_clean_splits_paired():
    proof = load("quantile", "representation_tradeoffs")
    clean = list(proof.records(rows=8192, seed=11))
    corrupt = list(proof.records(rows=8192, seed=11, contaminated=True))
    assert clean == list(proof.records(rows=8192, seed=11))
    assert clean != list(proof.records(rows=8192, seed=12))
    assert [(row["rank"], row["amount"]) for row in clean] == [(row["rank"], row["amount"]) for row in corrupt]
    changed = np.array([a["x"] != b["x"] for a, b in zip(clean, corrupt, strict=True)])
    assert 0.01 < changed.mean() < 0.03
    assert all(np.isclose(b["x"], 100 * a["x"]) for a, b, diff in zip(clean, corrupt, changed, strict=True) if diff)
    assert all((row["rank"] == "upper") == (row["x"] > np.expm1(2)) for row in clean)


def test_population_percentile_and_source_unit_optima_are_distinct():
    proof = load("quantile", "representation_tradeoffs")
    rows = list(proof.noisy_records(rows=49152, seed=11))
    targets = proof.optima()
    for group, name in enumerate(proof.GROUPS):
        values = np.array([row["target"] for row in rows if row["group"] == name])
        assert abs(values.mean() / targets["number_mse"][group] - 1) < 0.04
        assert abs(np.median(values) / targets["number_mae"][group] - 1) < 0.04
        rank_optimum = proof.inverse(np.array([proof.cdf(values).mean()]))[0]
        assert abs(rank_optimum / targets["quantile_mse"][group] - 1) < 0.04
    assert np.allclose(targets["number_mae"], targets["quantile_mae"])
    assert not np.allclose(targets["number_mse"], targets["quantile_mse"], rtol=0.1)
    ranks = np.array([0.001, 0.1, 0.5, 0.9, 0.999])
    np.testing.assert_allclose(proof.cdf(proof.inverse(ranks)), ranks, atol=1e-10)
