"""Per-column online statistics shared by tensorfields that standardize values."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from relflow.tensorfields.shared.normalizer import GlobalOnlineNormalizer, moments

NAN = float("nan")


def test_moments_are_per_column():
    stats = moments(torch.tensor([[1.0, 10.0], [3.0, NAN], [5.0, 30.0]], dtype=torch.float64), width=2)
    assert stats["count"].tolist() == [3, 2]
    torch.testing.assert_close(stats["mean"], torch.tensor([3.0, 20.0], dtype=torch.float64))
    torch.testing.assert_close(stats["var"], torch.tensor([8 / 3, 100.0], dtype=torch.float64))


def test_moments_match_var_mean_for_one_column():
    values = torch.randn(64, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    values[::7] = NAN
    finite = values[torch.isfinite(values)]
    expected_var, expected_mean = torch.var_mean(finite, correction=0)
    stats = moments(values)
    assert stats["count"].item() == finite.numel()
    torch.testing.assert_close(stats["mean"], expected_mean.reshape(1))
    torch.testing.assert_close(stats["var"], expected_var.reshape(1))


def test_moments_without_finite_values_are_zero():
    stats = moments(torch.tensor([NAN, float("inf")], dtype=torch.float64))
    assert stats["count"].tolist() == [0]
    assert stats["mean"].tolist() == [0.0]
    assert stats["var"].tolist() == [0.0]


def test_vector_normalizer_merges_each_channel_independently():
    normalizer = GlobalOnlineNormalizer(width=2)
    normalizer.learn(moments(torch.tensor([[1.0, 100.0], [3.0, 300.0]]), width=2))
    normalizer.learn(moments(torch.tensor([[5.0, NAN]]), width=2))
    assert normalizer.count.tolist() == [3, 2]
    torch.testing.assert_close(normalizer.mean, torch.tensor([3.0, 200.0], dtype=torch.float64))


def test_zero_count_channel_keeps_previous_statistics():
    normalizer = GlobalOnlineNormalizer(width=2)
    normalizer.learn(moments(torch.tensor([[1.0, 2.0]]), width=2))
    before = normalizer.mean.clone()
    normalizer.learn(moments(torch.tensor([[3.0, NAN]]), width=2))
    assert torch.isfinite(normalizer.mean).all() and normalizer.mean[1] == before[1]


def test_width_one_matches_scalar_semantics():
    normalizer = GlobalOnlineNormalizer()
    normalizer.learn(moments(torch.tensor([1.0, 3.0, float("inf")])))
    assert normalizer.mean.shape == (1,) and normalizer.count.tolist() == [2]


def test_update_path_rejects_vector_width():
    with pytest.raises(ValueError, match="width"):
        GlobalOnlineNormalizer(width=2).train()(torch.ones(4, 2), torch.ones(4, 2, dtype=torch.bool))


def test_width_one_merges_pool_exactly_at_large_offset():
    rng = np.random.default_rng(0)
    sizes = (64, 64, 128)
    batches = [1e9 + rng.integers(-40, 40, size=n) / 16 * scale for scale, n in zip((1, 3, 1), sizes)]
    normalizer = GlobalOnlineNormalizer()
    for batch in batches:
        normalizer.merge(moments(torch.from_numpy(batch)))
    pooled = np.concatenate(batches)
    assert normalizer.count.tolist() == [len(pooled)]
    np.testing.assert_allclose(normalizer.mean.numpy(), [pooled.mean()], rtol=1e-9)
    np.testing.assert_allclose(normalizer.var.numpy(), [pooled.var()], rtol=1e-9)


def test_width_one_ema_follows_exponential_update():
    normalizer = GlobalOnlineNormalizer(alpha=0.1)
    old_mean, old_var = 0.0, 1.0
    for mean, var in ((10.0, 4.0), (20.0, 9.0)):
        normalizer.merge(
            TensorDict({"count": torch.tensor([5]), "mean": torch.tensor([mean]), "var": torch.tensor([var])})
        )
        old_mean, old_var = 0.9 * old_mean + 0.1 * mean, 0.9 * old_var + 0.1 * var
    assert old_mean == pytest.approx(2.9) and old_var == pytest.approx(1.0 * 0.81 + 0.9 * 0.4 + 0.9)
    np.testing.assert_allclose(normalizer.mean.numpy(), [2.9], rtol=1e-12)
    np.testing.assert_allclose(normalizer.var.numpy(), [1.0 * 0.81 + 0.9 * 0.4 + 0.9], rtol=1e-12)


@pytest.mark.parametrize("alpha", [None, 0.1])
def test_width_one_zero_count_observation_changes_nothing(alpha):
    normalizer = GlobalOnlineNormalizer(alpha=alpha)
    normalizer.merge(moments(torch.tensor([1.0, 2.0, 6.0])))
    before = (normalizer.mean.clone(), normalizer.var.clone(), normalizer.count.clone())
    normalizer.merge(moments(torch.tensor([NAN])))
    for kept, now in zip(before, (normalizer.mean, normalizer.var, normalizer.count)):
        assert torch.equal(kept, now)
