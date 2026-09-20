"""Keep the confidence experiments' split and observability boundaries explicit."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import relflow as rf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proofs"))


def load(family, name):
    spec = importlib.util.spec_from_file_location(f"confidence_{name}", ROOT / "proofs" / family / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "family,name",
    [
        ("calibration", "categorical_prior"),
        ("vocabulary", "cold_start_confidence"),
        ("vocabulary", "pretraining_coverage"),
        ("vocabulary", "set_reconstruction_coverage"),
        ("vocabulary", "set_unavailable_fallback"),
        ("vocabulary", "cluster_pretraining_coverage"),
        ("vocabulary", "rolling_input_admission"),
    ],
)
def test_confidence_generators_have_independent_restartable_streams(family, name):
    proof = load(family, name)
    rows = list(proof.records(rows=32, seed=11))
    assert rows == list(proof.records(rows=32, seed=11))
    assert rows != list(proof.records(rows=32, seed=12))


def test_entity_disjoint_split_changes_only_identity_spelling():
    proof = load("vocabulary", "cold_start_confidence")
    known = list(proof.records(rows=2048, seed=11))
    novel = list(proof.records(rows=2048, seed=11, novel=True))
    assert not {row["entity"] for row in known} & {row["entity"] for row in novel}
    assert [(row["x"], row["label"]) for row in known] == [(row["x"], row["label"]) for row in novel]
    assert set(known[0]) == {"x", "entity", "label"}
    module = proof.build(0.2)
    module.encode(pa.Table.from_pylist(known), strata="train")
    vocab = rf.Category.vocabulary(module, "record/entity")
    field = module.encode(pa.Table.from_pylist(novel), strata="test")["record/entity"]
    assert field.content.eq(16384).all()
    assert field.state.eq(rf.Tokens.valued).all()
    assert rf.Category.vocabulary(module, "record/entity") == vocab


def test_cold_start_oracle_distinguishes_ambiguous_and_decisive_context():
    proof = load("vocabulary", "cold_start_confidence")
    rows = list(proof.records(rows=8192, seed=11, novel=True))
    x = np.asarray([row["x"] for row in rows])
    oracle = (1 / (1 + np.exp(-(2 * x - 2.5))) + 1 / (1 + np.exp(-(2 * x + 2.5)))) / 2
    result = proof.score(oracle, rows, novel=True)
    assert result["oracle_rmse"] == 0
    assert result["ambiguous_confidence"] < 0.55
    assert result["decisive_confidence"] > 0.95


def test_rolling_admission_keeps_future_identities_unknown_and_masks_fixed():
    proof = load("vocabulary", "rolling_input_admission")
    module = proof.build()
    training = list(proof.records(rows=1024, seed=11, cohort=0))
    future = list(proof.records(rows=1024, seed=12, cohort=1))
    assert not {row["entity"] for row in training} & {row["entity"] for row in future}
    module.encode(pa.Table.from_pylist(training), strata="train")
    snapshot = rf.Category.vocabulary(module, "record/entity")
    fields = module.encode(pa.Table.from_pylist(future), strata="validate")
    assert fields["record/entity"].content.eq(16384).all()
    assert fields["record/label"].targets[rf.TensorKey.content].lt(2).all()
    np.testing.assert_array_equal(fields["record/label"].trainable.numpy().reshape(-1), [row["hide"] for row in future])
    assert rf.Category.vocabulary(module, "record/entity") == snapshot
    assert set(training[0]) == {"entity", "context", "label", "hide"}


@pytest.mark.parametrize("name", ["pretraining_coverage", "cluster_pretraining_coverage"])
def test_pretraining_queries_fix_selected_targets_across_validation_calls(name):
    proof = load("vocabulary", name)
    module = proof.build()
    source = pa.Table.from_pylist(list(proof.records(rows=128, seed=11)))
    module.encode(source, strata="train")
    expected = np.arange(128) % 2 == 0
    for _ in range(2):
        field = module.encode(source, strata="validate")["record/code"]
        np.testing.assert_array_equal(field.trainable.numpy().reshape(-1), expected)
        assert field.targets[rf.TensorKey.content].reshape(-1)[expected].lt(1024).all()


def test_set_fallback_split_changes_names_without_changing_observability():
    proof = load("vocabulary", "set_unavailable_fallback")
    known = list(proof.records(rows=256, seed=11))
    novel = list(proof.records(rows=256, seed=11, novel=True))
    assert not {label for row in known for label in row["tags"]} & {label for row in novel for label in row["tags"]}
    assert [row["nonempty"] for row in known] == [row["nonempty"] for row in novel]
    module = proof.build(0.5)
    module.encode(pa.Table.from_pylist(known), strata="train")
    field = module.encode(pa.Table.from_pylist(novel), strata="test")["record/tags"]
    np.testing.assert_array_equal(field.content["unavailable"].numpy().reshape(-1), [row["nonempty"] for row in novel])
