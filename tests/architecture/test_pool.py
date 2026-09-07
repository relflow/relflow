import pytest
import torch
from torch.utils.module_tracker import ModuleTracker

from relflow.architecture.pool import LearnedQueryCrossAttention, MeanPool


def test_pool():
    n_context = 5
    d_model = 16
    nhead = 4
    dropout = 0.1
    batch_size = 2
    seq_length = 10

    model = LearnedQueryCrossAttention(n_context, d_model, nhead, dropout, n_linear=2)
    memory = torch.randn(batch_size, seq_length, d_model)

    output = model(memory)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, n_context, d_model)


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


def test_learned_query_mass_lane_preserves_additive_evidence():
    model = LearnedQueryCrossAttention(
        n_context=1,
        d_model=4,
        nhead=2,
        dropout=0.0,
        position=False,
        mass_capacity=4,
    ).eval()
    assert model.mass_projection is not None
    assert model.mass_direction is not None
    with torch.no_grad():
        model.mass_projection.weight.copy_(torch.eye(4))
        model.mass_direction.zero_()

    value = torch.tensor([[[0.5, -1.0, 0.25, 2.0]]]).expand(1, 4, 4).clone()
    one = torch.tensor([[True, False, False, False]])
    three = torch.tensor([[True, True, True, False]])

    one_output = model(value, present=one, evidence=value)
    three_output = model(value, present=three, evidence=value)

    assert torch.allclose(three_output - one_output, value[:, :1], atol=1e-5, rtol=1e-5)


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
