"""Protect elapsed-time equivalence and withheld-rank information boundaries."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))


def load(name):
    spec = importlib.util.spec_from_file_location(f"datatype_proof_{name}", ROOT / "research/datatypes" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_duration_arms_share_targets_and_elapsed_values_across_epoch_offsets():
    proof = load("duration_equivalence")
    dynamic = list(proof.records(rows=128, seed=11))
    static = list(proof.records(rows=128, seed=11, static=True))
    assert dynamic == list(proof.records(rows=128, seed=11))
    assert dynamic != list(proof.records(rows=128, seed=12))
    assert [row["elapsed"] for row in dynamic] == [row["elapsed"] for row in static]
    assert [row["target"] for row in dynamic] == [row["target"] for row in static]
    for arm, rows in (("duration", dynamic), ("static", static), ("number", dynamic)):
        model = proof.build(arm)
        encoded = model.encode(proof.table(rows, arm=arm))["/elapsed"].content.flatten().numpy()
        np.testing.assert_allclose(encoded, [row["elapsed"] for row in rows], atol=1e-12)
        assert "target" not in proof.table(rows, arm=arm, targets=False).column_names


def test_shuffled_duration_control_preserves_marginals_and_breaks_only_training_alignment():
    proof = load("duration_equivalence")
    original = list(proof.records(rows=4096, seed=11))
    shuffled = list(proof.records(rows=4096, seed=11, shuffled=True))
    assert shuffled == list(proof.records(rows=4096, seed=11, shuffled=True))
    assert sorted(row["elapsed"] for row in original) == sorted(row["elapsed"] for row in shuffled)
    for name in ("started_at", "ended_at", "target"):
        assert [row[name] for row in original] == [row[name] for row in shuffled]
    assert abs(np.corrcoef([row["elapsed"] for row in shuffled], [row["target"] for row in shuffled])[0, 1]) < 0.05


@pytest.mark.parametrize("domain", ["integer", "labels"])
def test_ordinal_withheld_answers_are_inside_observed_range_and_disjoint(domain):
    proof = load("ordinal_generalization")
    populations = {}
    for arm in ("ordinal", "number", "category"):
        seen = list(proof.records(rows=2048, seed=11, domain=domain, novel=False, arm=arm))
        unseen = list(proof.records(rows=2048, seed=12, domain=domain, novel=True, arm=arm))
        familiar, withheld = proof.actual(seen, domain, arm), proof.actual(unseen, domain, arm)
        assert set(familiar).isdisjoint(withheld)
        assert familiar.min() < withheld.min() < withheld.max() < familiar.max()
        populations[arm] = [row["signal"] for row in seen + unseen]
    assert populations["ordinal"] == populations["number"] == populations["category"]


@pytest.mark.parametrize(
    "name,arm,field", [("duration_equivalence", "duration", "target"), ("ordinal_generalization", "ordinal", "rank")]
)
def test_prediction_readers_accept_polars_and_omit_answers(name, arm, field):
    proof = load(name)
    inputs = []

    class Model:
        def predict(self, table, *, batch_size):
            assert field not in table.column_names
            inputs.append(table)
            return pl.DataFrame({"predictions": [{field: {"content": value}} for value in (1, 2)]})

    if field == "target":
        rows = list(proof.records(rows=2, seed=11))
        result = proof.prediction(Model(), rows, arm)
    else:
        rows = [{"signal": 1.1, "rank": 1}, {"signal": 2.1, "rank": 2}]
        result = proof.prediction(Model(), rows, "integer", arm)
    np.testing.assert_array_equal(result, [1, 2])
    assert len(inputs) == 1


def test_ordinal_rank_readouts_keep_population_counts_and_errors(monkeypatch):
    proof = load("ordinal_generalization")
    monkeypatch.setattr(proof, "prediction", lambda *args: np.array([1, 3, 4]))
    rows = [{"signal": value + 0.02, "rank": value} for value in (1, 1, 3)]
    measured = proof.evaluate(None, rows, "integer", "ordinal")
    assert measured["exact_accuracy"] == pytest.approx(1 / 3)
    assert measured["rank_mae"] == 1
    assert measured["ranks"]["1"] == {"rows": 2, "exact_accuracy": 0.5, "rank_mae": 1.0}
    assert measured["ranks"]["3"] == {"rows": 1, "exact_accuracy": 0.0, "rank_mae": 1.0}
