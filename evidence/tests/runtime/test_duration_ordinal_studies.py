"""Check experiment populations, controls, and matched tensorfield inputs."""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import relflow as rf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))


def load(name):
    path = ROOT / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"datatype_study_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_duration_histories_keep_signed_predecessors_before_cropping_and_hide_targets():
    proof = load("P113")
    rows = list(proof.records(rows=512, seed=11302))
    assert rows == list(proof.records(rows=512, seed=11302))
    assert {row["case"] for row in rows}.isdisjoint(row["case"] for row in proof.records(rows=512, seed=11303))
    assert {len(row["events"]) for row in rows} == {1, 2, 3, 4, 5}
    assert any(row["gap"] < 0 for row in rows)
    for row in rows:
        events = row["events"]
        assert (events[-1]["at"] - row["origin"]).total_seconds() / 3600 == row["age"]
        if len(events) > 1:
            assert (events[-1]["at"] - events[-2]["at"]).total_seconds() / 3600 == row["gap"]
        else:
            assert row["gap"] == 0 and events[-1]["prepared_gap"] is None
    prediction_rows = proof.frame(rows, targets=False).to_dicts()
    assert all("target" not in item for row in prediction_rows for item in row["events"])


def test_duration_and_prepared_number_encode_the_same_retained_values_and_states():
    proof = load("P113")
    rows = list(proof.records(rows=128, seed=11304))
    observations = proof.frame(rows, targets=False)
    elapsed = proof.build("duration").encode(observations)
    numeric = proof.build("number").encode(observations)
    for name in ("age", "gap"):
        left, right = elapsed[f"/events/{name}"], numeric[f"/events/{name}"]
        torch.testing.assert_close(left.content, right.content, rtol=0, atol=0)
        torch.testing.assert_close(left.state, right.state, rtol=0, atol=0)
        torch.testing.assert_close(left.present, right.present, rtol=0, atol=0)
    age = elapsed["/events/age"].content.reshape(-1).tolist()
    gap = elapsed["/events/gap"].content.reshape(-1).tolist()
    states = elapsed["/events/gap"].state.reshape(-1).tolist()
    assert age == [row["age"] for row in rows]
    assert gap == [row["gap"] for row in rows]
    assert states == [rf.Tokens.valued if len(row["events"]) > 1 else rf.Tokens.null for row in rows]


@pytest.mark.parametrize("feature", ["age", "gap"])
def test_duration_shuffles_preserve_feature_marginals_and_missingness(feature):
    proof = load("P113")
    rows = list(proof.records(rows=1024, seed=11302))
    baseline = proof.frame(rows).to_dicts()
    shuffled = proof.frame(rows, shuffle=feature, seed=11305).to_dicts()
    before = [row["events"][-1] for row in baseline]
    after = [row["events"][-1] for row in shuffled]
    assert [row["target"] for row in before] == [row["target"] for row in after]
    assert [row["prepared_gap"] is None for row in before] == [row["prepared_gap"] is None for row in after]
    name = f"prepared_{feature}"

    def values(events):
        return [row[name] for row in events]

    original, permuted = values(before), values(after)
    assert Counter(original) == Counter(permuted)
    valid = [index for index, value in enumerate(original) if value is not None]
    assert abs(np.corrcoef([permuted[i] for i in valid], [before[i]["target"] for i in valid])[0, 1]) < 0.1


def test_ordinal_populations_are_balanced_disjoint_and_leave_targets_hidden():
    proof = load("P114")
    for condition, ranks in proof.RANKS.items():
        rows = list(proof.records(rows=1000, seed=11402, condition=condition))
        assert rows == list(proof.records(rows=1000, seed=11402, condition=condition))
        assert set(row["rank"] for row in rows) == set(ranks)
        counts = Counter(row["rank"] for row in rows)
        assert max(counts.values()) - min(counts.values()) <= 1
        for arm in ("ordinal_ranks", "number"):
            assert proof.frame(rows, arm=arm, targets=False).columns == ["level"]
    assert set(proof.RANKS["seen"]).isdisjoint(proof.RANKS["unseen"])
    assert set(proof.RANKS["extrapolation"]).isdisjoint((*proof.RANKS["seen"], *proof.RANKS["unseen"]))
    extrapolation = list(proof.records(rows=20, seed=11407, condition="extrapolation"))
    for arm in ("ordinal_labels", "category"):
        with pytest.raises(ValueError, match="declared label positions"):
            proof.frame(extrapolation, arm=arm)


def test_ordinal_input_arms_encode_the_same_seen_and_withheld_ranks():
    proof = load("P114")
    for condition in ("seen", "unseen"):
        rows = list(proof.records(rows=80, seed=11405, condition=condition))
        encoded = {
            arm: proof.build(arm).encode(proof.frame(rows, arm=arm, targets=False))["/level"]
            for arm in ("ordinal_labels", "ordinal_ranks", "number")
        }
        assert encoded["ordinal_labels"].content.reshape(-1).tolist() == [float(row["rank"]) for row in rows]
        for arm in ("ordinal_ranks", "number"):
            torch.testing.assert_close(encoded[arm].content, encoded["ordinal_labels"].content, rtol=0, atol=0)
            torch.testing.assert_close(encoded[arm].state, encoded["ordinal_labels"].state, rtol=0, atol=0)


def test_ordinal_shuffled_control_breaks_alignment_without_changing_marginals():
    proof = load("P114")
    rows = list(proof.records(rows=4096, seed=11402, condition="seen"))
    original = proof.frame(rows, arm="number")
    shuffled = proof.frame(rows, arm="shuffled", shuffle=True, seed=11407)
    assert original["target"].to_list() == shuffled["target"].to_list()
    assert Counter(original["level"].to_list()) == Counter(shuffled["level"].to_list())
    assert abs(np.corrcoef(shuffled["level"].to_numpy(), shuffled["target"].to_numpy())[0, 1]) < 0.05


def test_duration_target_populations_preserve_actual_elapsed_values_and_hide_sources():
    proof = load("P115")
    rows = list(proof.records(rows=256, seed=11502))
    assert rows == list(proof.records(rows=256, seed=11502))
    assert {row["case"] for row in rows}.isdisjoint(row["case"] for row in proof.records(rows=256, seed=11503))
    assert any(row["target"] < 0 for row in rows)
    assert any(row["target"] > 0 for row in rows)
    for row in rows:
        assert (row["ended_at"] - row["started_at"]).total_seconds() / 3600 == row["target"]
        assert (row["fixed_end"] - proof.ANCHOR).total_seconds() / 3600 == row["target"]
    for arm in proof.ARMS:
        assert proof.frame(rows, arm=arm, targets=False).columns == ["signal"]


def test_duration_target_arms_tensorize_the_same_supervision_before_masking():
    proof = load("P115")
    rows = list(proof.records(rows=128, seed=11502))
    expected = torch.tensor([row["target"] for row in rows], dtype=torch.float64)
    for arm in ("duration", "translated", "static", "number"):
        model = proof.build(arm)
        field = model.encode(proof.frame(rows, arm=arm), strata="train")["/target"]
        # Arrow and Python can round the same microsecond interval a few float64 ULPs apart.
        torch.testing.assert_close(field.targets[rf.TensorKey.content].reshape(-1), expected, rtol=0, atol=1e-13)
        assert field.targets[rf.TensorKey.state].eq(rf.Tokens.valued).all()
        assert not field.present.any()


def test_duration_target_shuffled_control_preserves_target_and_clue_distributions():
    proof = load("P115")
    rows = list(proof.records(rows=4096, seed=11502))
    original = proof.frame(rows, arm="number")
    shuffled = proof.frame(rows, arm="shuffled", shuffle=True, seed=11505)
    assert original["target"].to_list() == shuffled["target"].to_list()
    assert Counter(original["signal"].to_list()) == Counter(shuffled["signal"].to_list())
    assert abs(np.corrcoef(shuffled["signal"].to_numpy(), shuffled["target"].to_numpy())[0, 1]) < 0.05
