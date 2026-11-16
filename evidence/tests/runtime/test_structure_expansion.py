"""Information boundaries and exact oracles for the new structural experiments."""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import relflow as rf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
SCRIPTS = {
    "P083": "structure/ragged_hierarchy",
    "P084": "temporal/retained_recent_history",
    "P093": "temporal/irregular_intervals",
    "P094": "composition/requests_and_cardinality",
    "P097": "composition/context_extrapolation",
    "P098": "relational/lookup_multiplicity",
}


def load(identifier):
    path = ROOT / "experiments" / (SCRIPTS[identifier] + ".py")
    spec = importlib.util.spec_from_file_location(f"structure_expansion_{identifier}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("identifier", tuple(SCRIPTS))
def test_generators_restart_and_have_independent_streams(identifier):
    module = load(identifier)
    first = list(module.records(rows=64, seed=17))
    assert first == list(module.records(rows=64, seed=17))
    assert first != list(module.records(rows=64, seed=18))
    assert len(first) == 64


def test_hierarchy_pairs_preserve_values_and_lengths_but_change_membership():
    module = load("P083")
    rows = list(module.records(rows=128, seed=20))
    changed = []
    for left, right in zip(rows[::2], rows[1::2], strict=True):
        contents = []
        shapes = []
        for row in (left, right):
            sessions = [session for day in row["days"] for session in day["sessions"]]
            amounts = [item["amount"] for session in sessions for item in session["transactions"]]
            contents.append(sorted(amounts))
            shapes.append([[len(session["transactions"]) for session in day["sessions"]] for day in row["days"]])
            assert row["answer"] == pytest.approx(
                max(sum(item["amount"] for item in session["transactions"]) for session in sessions)
            )
        assert contents[0] == contents[1]
        assert shapes[0] == shapes[1]
        changed.append(left["answer"] != right["answer"])
    assert np.mean(changed) > 0.5
    with_holes = module.structural_holes(rows)
    for original, altered in zip(rows, with_holes, strict=True):
        assert altered["answer"] == original["answer"]
        amounts = [
            item["amount"]
            for day in altered["days"]
            if day is not None
            for session in (day["sessions"] or [])
            if session is not None
            for item in (session["transactions"] or [])
            if item is not None
        ]
        expected = [
            item["amount"]
            for day in original["days"]
            for session in day["sessions"]
            for item in session["transactions"]
        ]
        assert amounts == expected


def test_tail_crop_controls_do_not_change_the_retained_information():
    module = load("P084")
    rows = list(module.records(rows=64, seed=21))
    for row in rows:
        assert 8 <= len(row["events"]) <= 12
        assert row["answer"] == sum(item["amount"] for item in row["events"][-4:])
    for mode in ("trim", "discarded"):
        for original, changed in zip(rows, module.intervene(rows, mode), strict=True):
            assert changed["events"][-4:] == original["events"][-4:]
            assert changed["answer"] == original["answer"]
    for original, changed in zip(rows, module.intervene(rows, "retained"), strict=True):
        assert changed["answer"] == pytest.approx(original["answer"] * 0.5 + 0.8)
        assert changed["events"][:-4] == original["events"][:-4]
    assert all(13 <= len(row["events"]) <= 16 for row in module.records(rows=64, seed=22, long=True))


def test_irregular_time_oracle_uses_visible_time_and_control_erases_it():
    module = load("P093")
    for shifted in (False, True):
        rows = list(module.records(rows=128, seed=23, shifted=shifted))
        assert np.sqrt(np.mean((module.least_squares(rows) - [row["answer"] for row in rows]) ** 2)) < 0.06
        for original, changed in zip(rows, module.position_only(rows), strict=True):
            assert changed["answer"] == original["answer"]
            assert [item["value"] for item in changed["observations"]] == [
                item["value"] for item in original["observations"]
            ]
            assert [item["elapsed"] for item in changed["observations"]] == [-3.0, -2.0, -1.0, 0.0]
            assert changed["forecast_at"] == 1.0


def test_request_split_holds_out_pairs_and_length_shift_is_disjoint():
    module = load("P094")
    train = list(module.records(rows=192, seed=24))
    familiar = list(module.records(rows=288, seed=25, complete=True))
    shifted = list(module.records(rows=288, seed=26, complete=True, long=True))
    assert {(row["selected_group"], row["operation"]) for row in train} == set(module.CELLS) - set(module.WITHHELD)
    assert Counter((row["selected_group"], row["operation"]) for row in familiar) == dict.fromkeys(module.CELLS, 32)
    for rows, allowed in ((train, range(1, 4)), (familiar, range(1, 4)), (shifted, range(4, 7))):
        for row in rows:
            counts = Counter(item["group"] for item in row["items"])
            assert set(counts) == set(module.GROUPS)
            assert all(count in allowed for count in counts.values())
            selected = [item["value"] for item in row["items"] if item["group"] == row["selected_group"]]
            expected = {"mean": np.mean, "min": np.min, "max": np.max}[row["operation"]](selected)
            assert row["answer"] == pytest.approx(expected)


def test_context_extrapolation_oracle_and_geometry():
    module = load("P097")
    for count in (2, 4, 8):
        for outside in (False, True):
            rows = list(module.records(rows=128, seed=27, count=count, extrapolate=outside))
            assert np.sqrt(np.mean((module.least_squares(rows) - [row["answer"] for row in rows]) ** 2)) < 0.06
            for row in rows:
                assert len(row["examples"]) == count
                assert min(item["x"] for item in row["examples"]) == -1
                assert max(item["x"] for item in row["examples"]) == 1
                assert (abs(row["query"]) > 1) == outside


def test_lookup_match_semantics_and_fresh_namespaces():
    module = load("P098")
    rows = list(module.records(rows=128, seed=28))
    other = list(module.records(rows=128, seed=29))
    assert Counter(module.multiplicity(row) for row in rows) == {0: 32, 1: 32, 2: 32, 3: 32}

    def keys(batch):
        return {item["entity_id"] for row in batch for item in row["source"]} | {row["query_key"] for row in batch}

    assert not keys(rows) & keys(other)
    for row in rows:
        selected = [item["value"] for item in row["source"] if item["entity_id"] == row["query_key"]]
        assert row["found"] == bool(selected)
        assert row["answer"] == (float(np.mean(selected)) if selected else 0.0)
        assert 4 <= len(row["source"]) <= 8


@pytest.mark.parametrize("identifier", tuple(SCRIPTS))
def test_model_constructors_accept_current_public_api(identifier):
    module = load(identifier)
    if identifier == "P084":
        models = [module.build(overflow) for overflow in ("head", "tail")]
    elif identifier == "P097":
        models = [module.build(context) for context in (False, True)]
    else:
        models = [module.build()]
    for model in models:
        target = model.schema.requests[rf.Address("answer")]
        assert target.mask[0].skip and target.mask[0].reconstruct
