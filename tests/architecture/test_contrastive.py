import torch

from relflow.architecture.contrastive import hierarchical


def test_hierarchical_shares_sum_to_a_finite_loss_with_gradients():
    generator = torch.Generator().manual_seed(0)
    first = torch.randn(4, 8, 16, generator=generator, requires_grad=True)
    second = (first + 0.01 * torch.randn(4, 8, 16, generator=generator)).detach().requires_grad_()
    present = torch.ones(4, 8, dtype=torch.bool)
    present[3, 2:] = False
    shares = hierarchical(first, second, present=present)
    assert shares.shape == (4,) and torch.isfinite(shares).all() and shares.sum() > 0
    shares.sum().backward()
    assert first.grad is not None and torch.count_nonzero(first.grad) > 0


def test_matching_views_score_lower_than_shuffled_views():
    generator = torch.Generator().manual_seed(1)
    first = torch.randn(8, 16, 32, generator=generator)
    present = torch.ones(8, 16, dtype=torch.bool)
    aligned = hierarchical(first, first.clone(), present=present).sum()
    shuffled = hierarchical(first, first[torch.randperm(8, generator=generator)], present=present).sum()
    assert aligned < shuffled


def test_hierarchical_with_nothing_present_is_zero_and_connected():
    first = torch.randn(2, 4, 8, requires_grad=True)
    shares = hierarchical(first, first * 2, present=torch.zeros(2, 4, dtype=torch.bool))
    assert torch.equal(shares, torch.zeros(2)) and shares.requires_grad
