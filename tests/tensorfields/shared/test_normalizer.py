"""Per-column online statistics shared by tensorfields that standardize values."""

import pytest
import torch

from relflow.tensorfields.shared.normalizer import GlobalOnlineNormalizer, moments

NAN = float("nan")


def test_moments_are_per_column():
    stats = moments(torch.tensor([[1.0, 10.0], [3.0, NAN], [5.0, 30.0]], dtype=torch.float64), width=2)
    assert stats["count"].tolist() == [3, 2]
    torch.testing.assert_close(stats["mean"], torch.tensor([3.0, 20.0], dtype=torch.float64))
    torch.testing.assert_close(stats["var"], torch.tensor([8 / 3, 100.0], dtype=torch.float64))


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
