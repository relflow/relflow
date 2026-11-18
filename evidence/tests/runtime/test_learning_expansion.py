"""Protect the data and control boundaries of new learning experiments without fitting."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))


def load(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location("expanded_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_label_efficiency_has_balanced_disjoint_unlabeled_streams_and_matched_raw_width():
    experiment = load("experiments/P085.py")
    rows, labels = experiment.sample(512, 11)
    held_out, _ = experiment.sample(512, 12)
    assert np.bincount(labels).tolist() == [128] * 4
    assert all(set(row) == {"first", "second", "geometry"} for row in rows)
    assert not {json.dumps(row, sort_keys=True) for row in rows} & {json.dumps(row, sort_keys=True) for row in held_out}
    assert experiment.raw(rows).shape == (512, 32)
    for count in experiment.COUNTS:
        selected = np.concatenate([np.flatnonzero(labels == label)[:count] for label in range(4)])
        assert np.bincount(labels[selected]).tolist() == [count] * 4


def test_cohort_targets_have_stable_compatible_rules_and_independent_test_rows():
    experiment = load("experiments/P086.py")
    for index, cohort in enumerate(experiment.COHORTS):
        rows = list(experiment.records(32, 11, cohort))
        held_out = list(experiment.records(32, 12, cohort))
        assert rows != held_out
        assert rows == list(experiment.records(32, 11, cohort))
        for row in rows:
            assert row["answer"] == (2.0, -2.0, 1.5)[index] * row["x"] + (-1.0, 1.0, 0.0)[index]


def test_group_shuffling_preserves_response_and_identity_marginals():
    experiment = load("experiments/P112.py")
    rows = list(experiment.records(512, 11, 19, skew=True))
    shuffled = list(experiment.records(512, 11, 19, skew=True, shuffled=True))
    assert [row["answer"] for row in rows] == [row["answer"] for row in shuffled]
    assert [row["x"] for row in rows] == [row["x"] for row in shuffled]
    assert sorted(row["entity"] for row in rows) == sorted(row["entity"] for row in shuffled)
    balanced = list(experiment.records(1024, 12, 19, balanced=True))
    assert len(set(row["entity"] for row in balanced)) == experiment.ENTITIES
    assert experiment.adjusted_rand(np.array([0, 0, 1, 1]), np.array([4, 4, 2, 2])) == 1.0


def test_anomaly_marginal_baseline_is_exactly_chance_and_noise_is_heterogeneous():
    experiment = load("experiments/P096.py")
    rows = list(experiment.records(8192, 11))
    actual = np.asarray([row["answer"] for row in rows])
    shuffled = actual[np.random.default_rng(19).permutation(len(rows))]
    assert experiment.auc(np.abs(actual), np.abs(shuffled)) == 0.5
    errors = {
        group: [
            row["answer"] - (2 * row["x"] + (-2 if group == "quiet" else 2)) for row in rows if row["group"] == group
        ]
        for group in experiment.GROUPS
    }
    assert np.std(errors["noisy"]) > 6 * np.std(errors["quiet"])
