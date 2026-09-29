"""Verify the Enum experiments' information controls without fitting models."""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import relflow as rf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proofs"))


def load(name):
    spec = importlib.util.spec_from_file_location(f"enum_proof_{name}", ROOT / "proofs/enum" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ("fixed_class_learning", "declared_without_examples"))
def test_enum_generators_restart_and_use_independent_streams(name):
    proof = load(name)
    first = list(proof.records(rows=64, seed=11))
    assert first == list(proof.records(rows=64, seed=11))
    assert first != list(proof.records(rows=64, seed=12))
    assert len(first) == 64


def test_shuffled_context_preserves_marginals_but_removes_target_information():
    proof = load("fixed_class_learning")
    original = list(proof.records(rows=4096, seed=11))
    shuffled = list(proof.records(rows=4096, seed=11, shuffled=True))
    assert [row["label"] for row in original] == [row["label"] for row in shuffled]
    assert Counter((row["route"], row["channel"]) for row in original) == Counter(
        (row["route"], row["channel"]) for row in shuffled
    )
    assert Counter(row["label"] for row in original) == dict.fromkeys(proof.LABELS, 1024)
    expected = [
        proof.LABELS[(proof.ROUTES.index(row["route"]) + 2 * proof.CHANNELS.index(row["channel"])) % 4]
        for row in original
    ]
    assert expected == [row["label"] for row in original]
    matches = [
        row["label"] == proof.LABELS[(proof.ROUTES.index(row["route"]) + 2 * proof.CHANNELS.index(row["channel"])) % 4]
        for row in shuffled
    ]
    assert 0.20 < np.mean(matches) < 0.30


def test_omission_and_rehearsal_change_exposure_without_changing_declared_support():
    proof = load("declared_without_examples")
    initial = list(proof.records(rows=60, seed=11))
    withheld = list(proof.records(rows=60, seed=12, phase="withheld"))
    mixed = list(proof.records(rows=60, seed=13, phase="mixed"))
    assert Counter(row["label"] for row in initial) == dict.fromkeys(proof.LABELS[:3], 20)
    assert {row["label"] for row in withheld} == {proof.LABELS[-1]}
    assert Counter(row["label"] for row in mixed) == dict.fromkeys(proof.LABELS, 15)
    for row in initial + withheld + mixed:
        assert abs(row["x"] - proof.CENTERS[proof.LABELS.index(row["label"])]) <= 0.15
    model = proof.build()
    storage = proof.storage(model)
    assert storage == {"labels": proof.LABELS, "embedding_rows": 4, "count_rows": 4, "decoder_rows": 4}
    cold = proof.measure(proof.prediction(model, withheld), withheld)
    assert cold["full_declared_support"]
    assert cold["probability_sum_error"] < 1e-5
    model.encode(pa.Table.from_pylist(initial), strata="train")
    counts = rf.Enum.counts(model, "/label")
    assert counts == {**dict.fromkeys(proof.LABELS[:3], 20), proof.LABELS[-1]: 0}
    proof.prediction(model, withheld)
    assert rf.Enum.counts(model, "/label") == counts
    model.encode(pa.Table.from_pylist(mixed), strata="train")
    assert rf.Enum.counts(model, "/label") == {label: value + 15 for label, value in counts.items()}
    assert proof.storage(model) == storage
