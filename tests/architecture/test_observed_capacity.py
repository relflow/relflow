"""Unbounded branches divide their lanes by the largest count observed while training."""

import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.architecture.pool import LearnedQueryCrossAttention, MeanPool

EVENTS = rf.Address("/events")


def events(*lengths: int) -> pa.Table:
    return pa.Table.from_pylist(
        [{"events": [{"amount": float(index)} for index in range(length)], "total": 1.0} for length in lengths]
    )


@pytest.mark.parametrize("reduction", [rf.Mean(), rf.Attention()], ids=["mean", "attention"])
def test_training_raises_the_observed_capacity_and_evaluation_keeps_it(reduction, tmp_path):
    model = rf.Model.xs(d_model=8, events=rf.Branch(reduction=reduction, amount=rf.Number), total=rf.Number(mask=True))
    pool = model.nodes[EVENTS].encoder.pool
    assert pool.mass_capacity is None
    assert pool.mass_observed is not None and pool.mass_observed.item() == 1

    model(model.encode(events(4, 2), strata="train"), strata="train")
    assert pool.mass_observed.item() == 4
    model(model.encode(events(3), strata="train"), strata="train")
    assert pool.mass_observed.item() == 4

    model.predict(events(40))
    assert pool.mass_observed.item() == 4

    model.save(tmp_path / "model.pt")
    assert rf.Model.load(tmp_path / "model.pt").nodes[EVENTS].encoder.pool.mass_observed.item() == 4


def test_bounded_branches_keep_their_static_capacity():
    model = rf.Model.xs(d_model=8, events=rf.Branch(length=4, amount=rf.Number), total=rf.Number(mask=True))
    pool = model.nodes[EVENTS].encoder.pool
    assert pool.mass_capacity == 4
    assert pool.mass_observed is None
    assert "mass_observed" not in {name for name, _ in pool.named_buffers()}


def test_mean_lane_scales_by_the_observed_count_rather_than_the_raw_count():
    pool = MeanPool(n_context=1, d_model=4).train()
    with torch.no_grad():
        pool.mass_direction.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    pool.observe(torch.tensor([[True, True, True, True, False], [True, False, False, False, False]]))
    pool.eval()

    short = pool(torch.zeros(1, 4, 4))
    long = pool(torch.zeros(1, 40, 4))

    torch.testing.assert_close(short[0, 0], pool.mass_direction)
    torch.testing.assert_close(long - short, 9 * pool.mass_direction.expand(1, 1, 4))


def test_attention_lanes_match_the_bounded_form_at_the_observed_count():
    torch.manual_seed(5)
    observed = LearnedQueryCrossAttention(1, 8, 2, 0.0, position=False, mass=True).train()
    with torch.no_grad():
        observed.mass_projection.weight.normal_(std=0.1)
        observed.mass_direction.normal_(std=0.1)
    bounded = LearnedQueryCrossAttention(1, 8, 2, 0.0, position=False, mass_capacity=4, mass=True).eval()
    assert bounded.load_state_dict(observed.state_dict(), strict=False).unexpected_keys == ["mass_observed"]
    observed.observe(torch.ones(2, 4, dtype=torch.bool))
    observed.eval()
    assert observed.mass_observed.item() == 4

    memory = torch.randn(2, 4, 8)
    torch.testing.assert_close(observed(memory), bounded(memory))

    short = observed(torch.zeros(1, 4, 8))
    long = observed(torch.zeros(1, 40, 8))
    torch.testing.assert_close(long - short, 9 * observed.mass_direction.unsqueeze(0))


def test_lanes_require_an_explicit_switch():
    with pytest.raises(ValueError, match="mass=True"):
        LearnedQueryCrossAttention(1, 8, 2, 0.0, mass_capacity=4)
    with pytest.raises(ValueError, match="d_model"):
        MeanPool(n_context=1, mass_capacity=4)
    assert LearnedQueryCrossAttention(1, 8, 2, 0.0).mass_projection is None
    assert MeanPool(n_context=2).mass_direction is None
