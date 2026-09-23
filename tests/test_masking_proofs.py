"""Check dynamic-masking experiment design without running learning budgets."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proofs"))
NAMES = ("sampled_reconstruction", "query_reconstruction", "branch_ablation", "prefix_generation")


def load(name):
    spec = importlib.util.spec_from_file_location(f"masking_proof_{name}", ROOT / "proofs/masking" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", NAMES)
def test_masking_generators_are_restartable_with_independent_splits(name):
    proof = load(name)
    first = list(proof.records(rows=32, seed=11))
    assert first == list(proof.records(rows=32, seed=11))
    assert first != list(proof.records(rows=32, seed=12))
    assert len(first) == 32


def test_sampled_proof_does_not_use_query_targets_during_training():
    proof = load("sampled_reconstruction")
    rows = list(proof.records(rows=256, seed=11))
    assert all(not row["hide_x"] and not row["hide_y"] for row in rows)
    model = proof.build()
    source = pa.Table.from_pylist(rows)
    training = model.encode(source, strata="train", seed=11)
    ordinary = model.encode(source, strata="predict")
    for name in ("x", "y"):
        field = training[f"/{name}"]
        assert field.trainable.any() and not field.trainable.all()
        assert field.present.all()
        assert not ordinary[f"/{name}"].inferred.any()


def test_nested_query_proof_selects_real_targets_not_padding():
    proof = load("query_reconstruction")
    rows = list(proof.records(rows=32, seed=11))
    expected = np.asarray(
        [[item["selected"] for item in row["items"]] + [False] * (proof.LENGTH - len(row["items"])) for row in rows]
    )
    assert any(len(row["items"]) < proof.LENGTH for row in rows)
    field = proof.build().encode(pa.Table.from_pylist(rows), strata="train")["/items/value"]
    np.testing.assert_array_equal(field.trainable.numpy().reshape(expected.shape), expected)
    assert not field.present.numpy().reshape(expected.shape)[expected].any()


def test_branch_proof_selectors_cover_all_visibility_patterns_and_compose():
    proof = load("branch_ablation")
    rows = list(proof.records(rows=128, seed=11))
    assert {tuple(view["hidden"] for view in row["views"]) for row in rows} == {
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    }
    fields = proof.build().encode(pa.Table.from_pylist(rows), strata="train")
    for name in ("reading", "backup"):
        expected = np.asarray(
            [
                [not (view["hidden"] or (name == "backup" and view["hide_backup"])) for view in row["views"]]
                for row in rows
            ]
        )
        np.testing.assert_array_equal(fields[f"/views/{name}"].present.numpy().reshape(expected.shape), expected)


def test_prefix_proof_hides_every_future_and_trains_one_next_value():
    proof = load("prefix_generation")
    rows = list(proof.records(rows=32, seed=11))
    model = proof.build()
    source = pa.Table.from_pylist(rows)
    field = model.encode(source, strata="train")["/events/value"]
    next_value = np.asarray([[event["next"] for event in row["events"]] for row in rows])
    visible = np.asarray([[not event["future"] for event in row["events"]] for row in rows])
    assert (next_value.sum(axis=1) == 1).all()
    assert set(visible.sum(axis=1)) == {2, 3, 4}
    np.testing.assert_array_equal(field.trainable.numpy().reshape(next_value.shape), next_value)
    np.testing.assert_array_equal(field.present.numpy().reshape(visible.shape), visible)


@pytest.mark.parametrize("name", ("query_reconstruction", "prefix_generation"))
def test_repeated_prediction_reads_public_output_without_learning(name):
    import relflow as rf

    proof = load(name)
    model = proof.build()
    address = "/items/value" if name == "query_reconstruction" else "/events/value"
    before = rf.Number.normalization(model, address)
    content, inferred = proof.prediction(model, list(proof.records(rows=4, seed=11)))
    assert content.shape == inferred.shape == (4, proof.LENGTH)
    assert np.isfinite(content).all()
    assert rf.Number.normalization(model, address) == before


def test_rollout_uses_generated_values_never_true_suffix(monkeypatch):
    proof = load("prefix_generation")
    rows = list(proof.records(rows=8, seed=11))
    calls = []

    def prediction(model, inputs):
        output = np.zeros((len(inputs), proof.LENGTH))
        inferred = np.zeros_like(output, dtype=bool)
        cut = next(index for index, event in enumerate(inputs[0]["events"]) if event["next"])
        for index, row in enumerate(inputs):
            events = row["events"]
            assert all(event["value"] == 0.0 for event in events[cut:])
            if cut > 2:
                assert events[cut - 1]["value"] == 100.0 + cut - 1
            output[index, cut] = 100.0 + cut
            inferred[index, cut] = True
        calls.append(cut)
        return output, inferred

    monkeypatch.setattr(proof, "prediction", prediction)
    generated = proof.rollout(None, rows)
    assert calls == [2, 3, 4]
    np.testing.assert_array_equal(generated[:, 2:], np.tile([102.0, 103.0, 104.0], (len(rows), 1)))
