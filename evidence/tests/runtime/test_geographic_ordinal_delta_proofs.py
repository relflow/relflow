"""Protect information boundaries and matched controls in the new datatype studies."""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))


def load(identifier):
    path = ROOT / "experiments" / f"{identifier}.py"
    spec = importlib.util.spec_from_file_location(f"datatype_evidence_{identifier}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spherical_input_study_has_independent_oracle_and_shuffled_positions():
    proof = load("P116")
    latitude, longitude = proof.REFERENCE
    assert proof.distance(np.array([latitude]), np.array([longitude]))[0] == pytest.approx(0, abs=1e-12)
    opposite = proof.distance(np.array([-latitude]), np.array([longitude - 180]))[0]
    assert opposite == pytest.approx(np.pi * proof.RADIUS_KM / 1000, rel=1e-8)
    train = list(proof.records(rows=512, seed=11602))
    heldout = list(proof.records(rows=512, seed=11603))
    assert {row["case"] for row in train}.isdisjoint(row["case"] for row in heldout)
    assert not set(map(tuple, (row["point"] for row in train))) & set(map(tuple, (row["point"] for row in heldout)))
    normal = proof.frame(train).to_dicts()
    shuffled = proof.frame(train, shuffle=True, seed=11605).to_dicts()
    assert [row["distance"] for row in normal] == [row["distance"] for row in shuffled]
    assert Counter(map(tuple, (row["point"] for row in normal))) == Counter(
        map(tuple, (row["point"] for row in shuffled))
    )
    assert sum(a["point"] != b["point"] for a, b in zip(normal, shuffled, strict=True)) > 450
    assert proof.frame(heldout, targets=False).columns == ["point"]


def test_geographic_target_study_preserves_dateline_and_hides_answers():
    proof = load("P117")
    rows = list(proof.records(rows=256, seed=11702))
    longitudes = [row["position"][1] for row in rows]
    assert min(longitudes) < -179 and max(longitudes) > 179
    assert proof.distance(np.array([[0.0, 179.999]]), np.array([[0.0, -179.999]]))[0] == pytest.approx(
        proof.RADIUS_M * np.pi / 180 * 0.002, rel=1e-8
    )
    assert proof.frame(rows, targets=False).columns == ["north", "east"]
    original = proof.frame(rows).to_dicts()
    shuffled = proof.frame(rows, shuffle=True, seed=11704).to_dicts()
    assert [row["position"] for row in original] == [row["position"] for row in shuffled]
    assert Counter((row["north"], row["east"]) for row in original) == Counter(
        (row["north"], row["east"]) for row in shuffled
    )


def test_delta_target_study_keeps_true_intervals_and_hides_timestamp_sources():
    proof = load("P118")
    rows = list(proof.records(rows=96, seed=11802))
    assert {len(row["events"]) for row in rows} == set(range(2, 7))
    for row in rows:
        events = row["events"]
        assert events[0]["prepared_gap"] is None and not events[0]["has_previous"]
        for previous, current in zip(events, events[1:]):
            assert (current["at"] - previous["at"]).total_seconds() == current["prepared_gap"]
    inputs = proof.frame(rows, targets=False).to_dicts()
    assert all("at" not in event and "prepared_gap" not in event for row in inputs for event in row["events"])
    duration = proof.DurationModel(preset=proof.CONTROL.name).encode(proof.frame(rows), strata="train")
    number = proof.NumericModel(preset=proof.CONTROL.name).encode(proof.frame(rows), strata="train")
    for key in ("state", "content"):
        torch.testing.assert_close(duration["/events/gap"].targets[key], number["/events/gap"].targets[key])
    torch.testing.assert_close(duration["/events/gap"].present, number["/events/gap"].present)


def test_delta_target_shuffle_preserves_retained_hints_states_and_targets():
    proof = load("P118")
    rows = list(proof.records(rows=128, seed=11802))
    original = proof.frame(rows).to_dicts()
    shuffled = proof.frame(rows, shuffle=True, seed=11805).to_dicts()
    before = [event for row in original for event in row["events"][-proof.CAP :]]
    after = [event for row in shuffled for event in row["events"][-proof.CAP :]]
    assert [event["prepared_gap"] for event in before] == [event["prepared_gap"] for event in after]
    assert [event["has_previous"] for event in before] == [event["has_previous"] for event in after]
    assert Counter(event["hint"] for event in before if event["has_previous"]) == Counter(
        event["hint"] for event in after if event["has_previous"]
    )
    assert sum(a["hint"] != b["hint"] for a, b in zip(before, after, strict=True)) > 200


def test_ordinal_target_study_declares_all_labels_and_preserves_control_marginals():
    proof = load("P119")
    rows = list(proof.records(rows=1024, seed=11902))
    assert {row["level"] for row in rows} == set(proof.LABELS)
    assert all(proof.LABELS[row["rank"]] == row["level"] for row in rows)
    original = proof.frame(rows).to_dicts()
    shuffled = proof.frame(rows, shuffle=True, seed=11905).to_dicts()
    assert [row["level"] for row in original] == [row["level"] for row in shuffled]
    assert Counter(row["signal"] for row in original) == Counter(row["signal"] for row in shuffled)
    assert proof.frame(rows, targets=False).columns == ["signal"]
    assert len(proof.OrdinalMae(preset=proof.CONTROL.name).schema.requests["/level"].values) == len(proof.LABELS)
    assert len(proof.EnumModel(preset=proof.CONTROL.name).schema.requests["/level"].values) == len(proof.LABELS)


def test_delta_input_study_aligns_prepared_values_and_shuffle_only_corrupts_gaps():
    proof = load("P120")
    rows = list(proof.records(rows=128, seed=12002))
    duration = proof.DurationModel(preset=proof.CONTROL.name).encode(proof.frame(rows), strata="train")
    numeric = proof.NumericModel(preset=proof.CONTROL.name).encode(proof.frame(rows), strata="train")
    for field in ("gap", "value"):
        for key in ("content", "state", "present"):
            torch.testing.assert_close(
                getattr(duration[f"/events/{field}"], key), getattr(numeric[f"/events/{field}"], key)
            )
    original = proof.frame(rows).to_dicts()
    shuffled = proof.frame(rows, shuffle=True, seed=12005).to_dicts()
    before = [event for row in original for event in row["events"][-proof.CAP :]]
    after = [event for row in shuffled for event in row["events"][-proof.CAP :]]
    assert [event["target"] for event in before] == [event["target"] for event in after]
    assert [event["value"] for event in before] == [event["value"] for event in after]
    assert [event["prepared_gap"] is None for event in before] == [event["prepared_gap"] is None for event in after]
    assert Counter(event["prepared_gap"] for event in before) == Counter(event["prepared_gap"] for event in after)
    assert sum(a["prepared_gap"] != b["prepared_gap"] for a, b in zip(before, after, strict=True)) > 200


def test_open_ordinal_study_separates_rank_ranges_and_preserves_shuffled_control():
    proof = load("P121")
    seen = list(proof.records(rows=512, seed=12102, condition="seen"))
    unseen = list(proof.records(rows=512, seed=12103, condition="unseen"))
    extrapolation = list(proof.records(rows=512, seed=12104, condition="extrapolation"))
    assert {row["rank"] for row in seen} == set(proof.RANKS["seen"])
    assert {row["rank"] for row in unseen} == set(proof.RANKS["unseen"])
    assert {row["rank"] for row in extrapolation} == set(proof.RANKS["extrapolation"])
    assert set(proof.RANKS["seen"]).isdisjoint(proof.RANKS["unseen"])
    assert all(abs(row["rank"]) > 8 for row in extrapolation)
    original = proof.frame(seen).to_dicts()
    shuffled = proof.frame(seen, shuffle=True, seed=12106).to_dicts()
    assert [row["rank"] for row in original] == [row["rank"] for row in shuffled]
    assert Counter(row["signal"] for row in original) == Counter(row["signal"] for row in shuffled)
    assert proof.frame(unseen, targets=False).columns == ["signal"]
    ordinal = proof.OrdinalMae(preset=proof.CONTROL.name).encode(proof.frame(seen), strata="train")
    numeric = proof.NumberMae(preset=proof.CONTROL.name).encode(proof.frame(seen), strata="train")
    for key in ("state", "content"):
        torch.testing.assert_close(ordinal["/rank"].targets[key], numeric["/rank"].targets[key])


def test_delta_target_poison_does_not_change_inferred_content_or_state():
    proof = load("P118")
    rows = list(proof.records(rows=8, seed=11806))
    model = proof.DurationModel(preset=proof.CONTROL.name)
    model.eval()
    values, inferred, states = proof.prediction(model, rows)
    poisoned_values, poisoned_inferred, poisoned_states = proof.prediction(model, rows, poison=True)
    np.testing.assert_allclose(values, poisoned_values, rtol=0, atol=0)
    np.testing.assert_allclose(states, poisoned_states, rtol=0, atol=0)
    assert all(inferred) and all(poisoned_inferred)
    assert states.shape == (len(values), 5)
