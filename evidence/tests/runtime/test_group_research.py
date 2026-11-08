"""Validate the independent crossed-rule research generator and controls."""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))


def test_crossing_regimes_keep_the_control_matched_and_hide_the_rule():
    spec = importlib.util.spec_from_file_location("crossing_regimes", ROOT / "research/group/crossing_regimes.py")
    proof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proof)
    options = {"rows": 4096, "seed": 1010101, "partition_seed": 1010100}
    stable = list(proof.records(**options))
    shuffled = list(proof.records(**options, no_regime=True))
    held_out = list(proof.records(**{**options, "seed": 1010103}))
    groups = proof.partition(options["partition_seed"])
    assert Counter(groups) == dict.fromkeys(range(4), 16)
    assert groups[1] == 1
    assert np.allclose(proof.INTERCEPTS + proof.SLOPES * -0.1, 0)
    assert set(stable[0]) == {"entity", "x", "y"}
    assert stable == list(proof.records(**options)) and stable != held_out
    assert Counter(row["entity"] for row in stable) == Counter(row["entity"] for row in held_out)
    assert Counter(row["entity"] for row in stable) == Counter(row["entity"] for row in shuffled)
    assert [(row["x"], row["y"]) for row in stable] == [(row["x"], row["y"]) for row in shuffled]
    assert any(left["entity"] != right["entity"] for left, right in zip(stable, shuffled, strict=True))
    errors = [
        row["y"]
        - proof.INTERCEPTS[groups[int(row["entity"][-4:])]]
        - proof.SLOPES[groups[int(row["entity"][-4:])]] * row["x"]
        for row in stable
    ]
    assert 0.09 < np.std(errors) < 0.11


@pytest.mark.parametrize(
    ("name", "seed", "example", "rule"),
    [
        ("rare_identities", 10201, 1, 3),
        ("group_capacity", 10301, 1, 1),
        ("unavailable_identities", 10401, 63, 0),
        ("assignment_intervention", 10501, 1, 2),
        ("behavior_drift", 10601, 0, 2),
        ("drift_confirmation", 10701, 0, 1),
    ],
)
def test_group_studies_keep_populations_and_controls_separate(name, seed, example, rule):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"research/group/{name}.py")
    proof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proof)
    ps = seed * 100
    assert proof.partition(ps)[example] == rule
    options = {"rows": 4096, "seed": ps + 1, "partition_seed": ps}
    stable = proof.records(**options)
    shuffled = proof.records(**options, shuffled=True)
    changed = proof.records(**options, phase="changed")
    assert set(stable[0]) == {"entity", "x", "y"}
    assert [(r["x"], r["y"]) for r in stable] == [(r["x"], r["y"]) for r in shuffled]
    assert Counter(r["entity"] for r in stable) == Counter(r["entity"] for r in shuffled)
    assert any(a["entity"] != b["entity"] for a, b in zip(stable, shuffled, strict=True))
    assert all(a == b for a, b in zip(stable, changed, strict=True) if int(a["entity"][-4:]) % 4)
    assert any(a["y"] != b["y"] for a, b in zip(stable, changed, strict=True))
    rare = proof.records(**options, population="rare")
    counts = Counter(r["entity"] for r in rare)
    assert all(counts[f"entity-{i:04d}"] == (120 if i < 32 else 8) for i in range(64))
    known = proof.records(**options, population="known")
    unseen = proof.records(**options, population="unseen")
    assert {r["entity"] for r in known}.isdisjoint(r["entity"] for r in unseen)
    assert proof.adjusted_rand(np.arange(64) % 4, (np.arange(64) % 4 + 1) % 4) == 1
