"""Protect the new numeric evidence generators and their information boundaries."""

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import relflow as rf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))


def load(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(f"numeric_expansion_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_large_offsets_and_centering_keep_the_same_observable_residuals():
    proof = load("experiments/representation/large_numeric_offsets.py")
    wide = list(proof.records(rows=1024, seed=11, arm="wide"))
    centered = list(proof.records(rows=1024, seed=11, arm="centered"))
    collapsed = list(proof.records(rows=1024, seed=11, arm="float32"))
    np.testing.assert_array_equal([row["x"] - proof.OFFSET for row in wide], [row["x"] for row in centered])
    np.testing.assert_array_equal([row["target"] - proof.OFFSET for row in wide], [row["target"] for row in centered])
    assert len({row["x"] for row in wide}) > 900
    assert len({row["x"] for row in collapsed}) == 1
    assert proof.table(wide).schema.field("x").type.bit_width == 64
    np.testing.assert_allclose([row["target"] for row in centered], [3 * row["x"] for row in centered], atol=1e-4)


def test_zero_null_and_nan_have_three_different_arrow_observations():
    proof = load("experiments/state/zero_null_invalid.py")
    rows = list(proof.records(rows=300, seed=21))
    intact = proof.table(rows)["measurement"]
    assert intact.null_count == 100
    assert pc.sum(pc.fill_null(pc.is_nan(intact), False)).as_py() == 100
    assert pc.sum(pc.fill_null(pc.equal(intact, 0), False)).as_py() == 100
    coalesced = proof.table(rows, arm="coalesced")["measurement"]
    filled = proof.table(rows, arm="filled")["measurement"]
    assert coalesced.null_count == 200
    assert filled.null_count == 0 and pc.all(pc.equal(filled, 0)).as_py()
    assert sorted(row["target"] for row in rows) == sorted(proof.LABELS * 100)
    assert proof.table(rows, targets=False).column_names == ["measurement"]


def test_visibility_oracle_integrates_hidden_values_instead_of_leaking_them():
    proof = load("experiments/calibration/evidence_availability.py")
    hidden = [{"group": "rare", "signal": float(value), "hide_signal": True} for value in (-1, 0, 1)]
    expected = proof.PROBABILITY[1].mean()
    np.testing.assert_allclose(proof.oracle(hidden, "training"), expected)
    selection = float(proof.PROBABILITY[1] @ proof.POLICIES["selection"] / proof.POLICIES["selection"].sum())
    np.testing.assert_allclose(proof.oracle(hidden, "selection"), selection)
    assert abs(selection - 0.2242857142857143) < 1e-12
    observed = [{**row, "hide_signal": False} for row in hidden]
    np.testing.assert_array_equal(proof.oracle(observed, "selection"), proof.PROBABILITY[1])
    matched = list(proof.records(rows=1024, seed=31, policy="training"))
    shifted = list(proof.records(rows=1024, seed=31, policy="selection"))
    for left, right in zip(matched, shifted, strict=True):
        assert {key: value for key, value in left.items() if key != "hide_signal"} == {
            key: value for key, value in right.items() if key != "hide_signal"
        }
    model = rf.Model.xs(
        group=rf.Enum(values=proof.GROUPS),
        signal=rf.Number(mask=rf.Mask(query="hide_signal", skip=True, dropout=False)),
        target=rf.Boolean(mask=True),
    )
    assert set(model.schema.requests) == {rf.Address("group"), rf.Address("signal"), rf.Address("target")}
    assert model.schema.requests[rf.Address("signal")].mask[0].skip


def test_jitter_stress_changes_only_observations_and_keeps_pristine_targets():
    proof = load("experiments/representation/jitter_generalization.py")
    rows = list(proof.records(rows=4096, seed=41))
    clean = proof.perturb(rows, seed=42, scale=0)
    changed = proof.perturb(rows, seed=42, scale=0.25)
    assert rows == clean
    assert [row["target"] for row in rows] == [row["target"] for row in changed]
    noise = np.asarray([left["signal"] - right["signal"] for left, right in zip(changed, rows, strict=True)])
    assert abs(noise.mean()) < 0.02 and 0.23 < noise.std() < 0.27
    residual = np.asarray([row["target"] - np.sin(2 * row["signal"]) - 0.3 * row["signal"] for row in rows])
    assert abs(residual.mean()) < 0.003 and 0.027 < residual.std() < 0.033


def test_duration_generator_matches_fixed_elapsed_arithmetic_and_static_endpoint():
    proof = load("research/datatypes/duration_equivalence.py")
    pair = list(proof.records(rows=128, seed=51))
    static = list(proof.records(rows=128, seed=51, static=True))
    assert all(row["started_at"] == proof.ANCHOR for row in static)
    assert len({row["started_at"] for row in pair}) > 100
    np.testing.assert_array_equal([row["elapsed"] for row in pair], [row["elapsed"] for row in static])
    np.testing.assert_array_equal([row["target"] for row in pair], [row["target"] for row in static])
    for row in pair:
        assert row["elapsed"] == (row["ended_at"] - row["started_at"]).total_seconds() / 86400
        assert (row["ended_at"] + timedelta(days=371)) - (row["started_at"] + timedelta(days=371)) == row[
            "ended_at"
        ] - row["started_at"]
    assert set(proof.table(pair, arm="duration", targets=False).column_names) == {"started_at", "ended_at"}
    assert proof.table(pair, arm="number", targets=False).column_names == ["elapsed"]


def test_ordinal_protocol_withholds_ranks_without_extrapolating_the_range():
    proof = load("research/datatypes/ordinal_generalization.py")
    for domain in ("integer", "labels"):
        for arm in ("ordinal", "number", "category"):
            seen = list(proof.records(rows=1024, seed=61, domain=domain, novel=False, arm=arm))
            unseen = list(proof.records(rows=1024, seed=62, domain=domain, novel=True, arm=arm))
            known = set(proof.actual(seen, domain, arm))
            novel = set(proof.actual(unseen, domain, arm))
            assert not known & novel
            assert min(known) < min(novel) < max(novel) < max(known)
            if domain == "integer":
                assert known == set(range(-9, 10, 2))
                assert novel == set(range(-8, 9, 2))
            else:
                assert known == {0, 2, 4} and novel == {1, 3}
