"""Validate the required Group evidence generators and controls."""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))


@pytest.mark.parametrize("name", ["plain_input_is_dormant", "reconstructing_category_labels"])
def test_label_group_replacement_has_independent_context_and_known_identities(name):
    spec = importlib.util.spec_from_file_location("group_label_recovery", ROOT / f"experiments/group/{name}.py")
    proof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proof)
    train = list(proof.records(1090101))
    heldout = list(proof.records(1090103, repeats=4))
    assert len(train) == 1000 and len(heldout) == 200
    assert {row["merchant_id"] for row in train} == {row["merchant_id"] for row in heldout}
    assert {row["context"] for row in train}.isdisjoint(row["context"] for row in heldout)
    assert all(row["label"] == f"L{row['merchant_id'][1]}" for row in train + heldout)
    assert set(train[0]) == {"merchant_id", "context", "label"}


def test_nonlinear_group_recovery_control_preserves_observations():
    spec = importlib.util.spec_from_file_location(
        "nonlinear_group_recovery", ROOT / "experiments/group/hidden_regression_regimes.py"
    )
    proof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proof)
    options = {"rows": 4096, "seed": 1080101, "partition_seed": 1080100}
    train = list(proof.records(**options))
    control = list(proof.records(**options, no_regime=True))
    heldout = list(proof.records(**{**options, "seed": 1080103}))
    assert Counter(proof.partition(options["partition_seed"])) == dict.fromkeys(range(5), 20)
    assert [(row["x"], row["y"]) for row in train] == [(row["x"], row["y"]) for row in control]
    assert Counter(row["entity"] for row in train) == Counter(row["entity"] for row in control)
    assert any(a["entity"] != b["entity"] for a, b in zip(train, control, strict=True))
    assert {row["x"] for row in train}.isdisjoint(row["x"] for row in heldout)
