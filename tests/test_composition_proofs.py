"""Check held-out combinations and context-only information before empirical fitting."""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import relflow as rf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proofs"))


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"composition_proof_{name}", ROOT / "proofs/composition" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ("held_out_requests", "context_examples"))
def test_composition_generators_restart_and_keep_streams_independent(name):
    proof = load(name)
    rows = list(proof.records(rows=64, seed=17))
    assert rows == list(proof.records(rows=64, seed=17))
    assert rows != list(proof.records(rows=64, seed=18))
    assert len(rows) == 64


def test_instruction_training_excludes_pairs_but_retains_each_component():
    proof = load("held_out_requests")
    train = list(proof.records(rows=96, seed=19))
    test = list(proof.records(rows=144, seed=20, complete=True))
    observed = {(row["selected_group"], row["operation"]) for row in train}
    assert observed == set(proof.CELLS) - set(proof.WITHHELD)
    assert {group for group, _ in observed} == set(proof.GROUPS)
    assert {operation for _, operation in observed} == set(proof.OPERATIONS)
    assert Counter((row["selected_group"], row["operation"]) for row in test) == dict.fromkeys(proof.CELLS, 16)
    for row in train + test:
        assert Counter(item["group"] for item in row["items"]) == dict.fromkeys(proof.GROUPS, 2)
        values = [item["value"] for item in row["items"] if item["group"] == row["selected_group"]]
        expected = {"mean": np.mean, "min": np.min, "max": np.max}[row["operation"]](values)
        assert abs(expected - row["answer"]) < 0.10


def test_request_corruptions_change_exactly_one_instruction():
    proof = load("held_out_requests")
    rows = list(proof.records(rows=72, seed=20, complete=True))
    for field in ("selected_group", "operation"):
        corrupt = proof.corrupt(rows, field)
        assert Counter(row[field] for row in corrupt) == Counter(row[field] for row in rows)
        for original, altered in zip(rows, corrupt, strict=True):
            assert original[field] != altered[field]
            assert {key: value for key, value in original.items() if key != field} == {
                key: value for key, value in altered.items() if key != field
            }


def test_task_quadrant_split_and_example_oracle():
    proof = load("context_examples")
    familiar = list(proof.records(rows=600, seed=30))
    withheld = list(proof.records(rows=600, seed=31, withheld=True))
    for rows, expected in ((familiar, set(proof.QUADRANTS)), (withheld, {(1, 1)})):
        quadrants = set()
        for row in rows:
            assert set(row) == {"examples", "query", "answer"}
            x = np.asarray([item["x"] for item in row["examples"]])
            y = np.asarray([item["y"] for item in row["examples"]])
            slope, intercept = np.polyfit(x, y, 1)
            quadrants.add((int(np.sign(slope)), int(np.sign(intercept))))
            assert x.min() < row["query"] < x.max()
        assert quadrants == expected
        error = proof.least_squares(rows) - [row["answer"] for row in rows]
        assert np.sqrt(np.mean(error**2)) < 0.03


def test_context_swap_preserves_queries_targets_and_complete_example_sets():
    proof = load("context_examples")
    rows = list(proof.records(rows=120, seed=40))
    swapped = proof.swap(rows, seed=41)
    assert [(row["query"], row["answer"]) for row in swapped] == [(row["query"], row["answer"]) for row in rows]
    assert Counter(repr(row["examples"]) for row in swapped) == Counter(repr(row["examples"]) for row in rows)
    assert np.mean([a["examples"] != b["examples"] for a, b in zip(rows, swapped, strict=True)]) > 0.9


def test_query_name_is_a_child_field_and_query_only_has_no_example_branch():
    proof = load("context_examples")
    context, query_only = proof.build(), proof.build(context=False)
    assert rf.Address("query") in context.nodes
    assert rf.Address("query") in query_only.nodes
    assert rf.Address("examples") in context.nodes
    assert rf.Address("examples") not in query_only.nodes
    assert rf.Address("answer") in context.nodes
