"""Check vocabulary experiment design and encoding boundaries without fitting."""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import torch

import relflow as rf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proofs"))
NAMES = (
    "unknown_input_fallback",
    "unknown_target_coverage",
    "capacity_admission_order",
    "unknown_set_members",
    "admission_and_continued_training",
)


def load(name):
    spec = importlib.util.spec_from_file_location(f"vocabulary_proof_{name}", ROOT / "proofs/vocabulary" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", NAMES)
def test_vocabulary_generators_are_restartable_with_independent_splits(name):
    proof = load(name)
    first = list(proof.records(rows=32, seed=11))
    assert first == list(proof.records(rows=32, seed=11))
    assert first != list(proof.records(rows=32, seed=12))
    assert len(first) == 32


def test_unknown_category_inputs_remain_valued_and_differ_from_null():
    proof = load("unknown_input_fallback")
    model = proof.build()
    model.encode(pa.Table.from_pylist(list(proof.records(rows=256, seed=11))), strata="train")
    rows = [{"x": 0.0, "code": code, "target": 0.0} for code in ("new-left", "new-right", None)]
    field = model.encode(pa.Table.from_pylist(rows), strata="test")["record/code"]
    assert field.state.reshape(-1)[:2].eq(rf.Tokens.valued).all()
    assert field.content.reshape(-1)[:2].eq(4).all()
    assert field.state.reshape(-1)[2] != rf.Tokens.valued


def test_target_coverage_shift_preserves_inputs_and_field_local_vocabularies():
    proof = load("unknown_target_coverage")
    model = proof.build()
    train = list(proof.records(rows=256, seed=11))
    model.encode(pa.Table.from_pylist(train), strata="train")
    assert set(rf.Category.vocabulary(model, "record/hint")) == set(proof.LABELS)
    assert set(rf.Category.vocabulary(model, "record/label")) == set(proof.LABELS[:2])
    original = list(proof.records(rows=64, seed=12))
    shifted = list(proof.records(rows=64, seed=12, shifted=True))
    assert [(row["x"], row["hint"]) for row in original] == [(row["x"], row["hint"]) for row in shifted]
    assert sum(row["label"] in proof.LABELS[:2] for row in shifted) == 32
    counts = rf.Category.counts(model, "record/label")
    targets = model.encode(pa.Table.from_pylist(shifted), strata="test")["record/label"].targets[rf.TensorKey.content]
    np.testing.assert_array_equal(targets.reshape(-1).numpy() == 8, np.arange(64) % 2 == 1)
    assert rf.Category.counts(model, "record/label") == counts


def test_capacity_controls_reorder_identical_records_without_duplication():
    proof = load("capacity_admission_order")
    rare = list(proof.records(rows=64, seed=11))
    common = list(proof.records(rows=64, seed=11, common_first=True))
    assert Counter((row["code"], row["target"]) for row in rare) == Counter(
        (row["code"], row["target"]) for row in common
    )
    assert [row["code"] for row in rare[:2]] == ["rare-left", "rare-right"]
    assert [row["code"] for row in common[:2]] == ["cold", "hot"]
    for observations, expected in ((rare, ("rare-left", "rare-right")), (common, ("cold", "hot"))):
        model = proof.build(size=2)
        model.encode(pa.Table.from_pylist(observations), strata="train")
        assert rf.Category.vocabulary(model, "record/code") == expected


def test_set_probes_distinguish_omitted_members_from_null_state():
    proof = load("unknown_set_members")
    model = proof.build()
    model.encode(pa.Table.from_pylist(list(proof.records(rows=256, seed=11))), strata="train")
    rows = [
        {"tags": tags, "labels": tags, "target": 0.0}
        for tags in ([], ["novel"], None, ["red"], ["red", "red", "novel"])
    ]
    fields = model.encode(pa.Table.from_pylist(rows), strata="test")
    inputs = fields["record/tags"]
    targets = fields["record/labels"].targets[rf.TensorKey.content]["membership"]
    assert torch.equal(inputs.content["membership"][0], inputs.content["membership"][1])
    assert torch.equal(inputs.content["membership"][3], inputs.content["membership"][4])
    assert torch.equal(targets[0], targets[1])
    assert torch.equal(targets[3], targets[4])
    assert inputs.state[:2].eq(rf.Tokens.valued).all()
    assert inputs.state[2] != rf.Tokens.valued
    assert set(rf.Set.vocabulary(model, "record/tags")) == {"red", "blue"}


def test_lifecycle_empty_prediction_admission_and_checkpoint(tmp_path):
    proof = load("admission_and_continued_training")
    model = proof.build()
    rows = list(proof.records(rows=32, seed=11))
    cold = proof.prediction(model, rows)
    assert all(row["value"] is None and row["probability"] == 0.0 for row in cold)
    assert rf.Category.vocabulary(model, "record/label") == ()
    model.encode(pa.Table.from_pylist(rows), strata="train")
    vocabulary = rf.Category.vocabulary(model, "record/label")
    assert set(vocabulary) == set(proof.LABELS[:2])
    model, matches = proof.roundtrip(model, rows, tmp_path / "vocabulary.pt")
    assert matches
    new = pa.Table.from_pylist(list(proof.records(rows=32, seed=12, phase="new")))
    counts = rf.Category.counts(model, "record/label")
    normalization = rf.Number.normalization(model, "record/x")
    for strata in ("validate", "test", "predict"):
        model.encode(new, strata=strata)
    assert rf.Category.vocabulary(model, "record/label") == vocabulary
    assert rf.Category.counts(model, "record/label") == counts
    assert rf.Number.normalization(model, "record/x") == normalization
    parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    model.encode(new, strata="train")
    admitted = rf.Category.vocabulary(model, "record/label")
    assert admitted[:2] == vocabulary
    assert set(admitted) == set(proof.LABELS)
    assert all(torch.equal(parameters[name], value) for name, value in model.named_parameters())


def test_continued_training_generator_rehearses_old_classes():
    proof = load("admission_and_continued_training")
    mixed = list(proof.records(rows=64, seed=11, phase="mixed"))
    assert Counter(row["label"] for row in mixed) == dict.fromkeys(proof.LABELS, 16)
    for row in mixed:
        assert abs(row["x"] - proof.CENTERS[proof.LABELS.index(row["label"])]) <= 0.15
