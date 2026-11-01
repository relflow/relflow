import pytest
import torch
from torch.utils.module_tracker import ModuleTracker

from relflow.architecture.pool import LearnedQueryCrossAttention, MeanPool


def test_learned_query_pooling_supports_module_tracker_with_grad_disabled():
    model = LearnedQueryCrossAttention(n_context=2, d_model=8, nhead=2, dropout=0.0)
    memory = torch.randn(3, 4, 8)

    with ModuleTracker():
        with torch.no_grad():
            output = model(memory)

    assert output.shape == (3, 2, 8)
    assert not output.requires_grad


def test_learned_query_pooling_ignores_absent_memory_and_zeros_empty_rows():
    model = LearnedQueryCrossAttention(n_context=2, d_model=8, nhead=2, dropout=0.0).eval()
    memory = torch.randn(2, 4, 8)
    present = torch.tensor([[True, False, True, False], [False, False, False, False]])

    first = model(memory, present=present)
    changed = memory.clone()
    changed[0, ~present[0]] = torch.nan
    changed[1] = torch.nan
    second = model(changed, present=present)

    assert torch.allclose(first[0], second[0])
    assert torch.equal(first[1], torch.zeros_like(first[1]))
    assert torch.equal(second[1], torch.zeros_like(second[1]))


def test_position_free_learned_query_pool_is_permutation_invariant():
    torch.manual_seed(31)
    model = LearnedQueryCrossAttention(
        n_context=2,
        d_model=8,
        nhead=2,
        dropout=0.0,
        position=False,
    ).eval()
    memory = torch.randn(3, 5, 8)
    present = torch.tensor(
        [
            [True, True, True, False, False],
            [True, False, True, True, False],
            [True, True, True, True, True],
        ]
    )
    order = torch.tensor([3, 0, 4, 1, 2])

    original = model(memory, present=present)
    permuted = model(memory[:, order], present=present[:, order])

    torch.testing.assert_close(original, permuted, atol=1e-6, rtol=1e-6)


def test_learned_query_context_and_evidence_validate_shape():
    model = LearnedQueryCrossAttention(
        n_context=2,
        d_model=8,
        nhead=2,
        dropout=0.0,
        mass_capacity=3,
        mass=True,
    )
    memory = torch.randn(4, 3, 8)

    with pytest.raises(ValueError, match="query context"):
        model(memory, context=torch.randn(4, 1, 8))
    with pytest.raises(ValueError, match="additive evidence"):
        model(memory, evidence=torch.randn(4, 2, 8))


def test_mean_pooling_uses_only_present_memory():
    model = MeanPool(n_context=2)
    memory = torch.tensor(
        [
            [[1.0, 2.0], [torch.nan, torch.nan], [3.0, 4.0]],
            [[torch.nan, torch.nan], [torch.nan, torch.nan], [torch.nan, torch.nan]],
        ]
    )
    present = torch.tensor([[True, False, True], [False, False, False]])

    output = model(memory, present=present)

    assert torch.equal(output[0], torch.tensor([[2.0, 3.0], [2.0, 3.0]]))
    assert torch.equal(output[1], torch.zeros_like(output[1]))


def test_mean_pooling_count_lane_starts_as_the_plain_mean_and_scales_with_presence():
    plain = MeanPool(n_context=1)
    counted = MeanPool(n_context=1, d_model=2, mass_capacity=4)
    memory = torch.tensor([[[2.0, 4.0]] + [[0.0, 0.0]] * 3, [[2.0, 4.0]] * 4])
    present = torch.tensor([[True, False, False, False], [True, True, True, True]])

    assert torch.equal(counted(memory, present=present), plain(memory, present=present))

    with torch.no_grad():
        counted.mass_direction.copy_(torch.tensor([1.0, -1.0]))
    output = counted(memory, present=present)
    # One item and four equal items share a mean but now differ by their count over capacity.
    assert torch.allclose(output[0, 0], torch.tensor([2.25, 3.75]))
    assert torch.allclose(output[1, 0], torch.tensor([3.0, 3.0]))


@pytest.mark.parametrize("kwargs", [{"mass_capacity": 4}, {"d_model": 2, "mass_capacity": 0}])
def test_mean_pooling_count_lane_rejects_a_capacity_without_a_direction_or_below_one(kwargs):
    with pytest.raises(ValueError):
        MeanPool(n_context=1, **kwargs)
