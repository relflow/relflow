from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from relflow.architecture.attention import RotaryMultiheadAttention


def test_rotary_attention_rejects_kv_heads_that_do_not_divide_query_heads():
    with pytest.raises(ValueError, match="nhead must be divisible by n_kv_heads"):
        RotaryMultiheadAttention(d_model=16, nhead=4, n_kv_heads=3, dropout=0.0)


def test_rotary_attention_passes_grouped_heads_to_sdpa(monkeypatch: pytest.MonkeyPatch):
    seen = {}

    def sdpa(query, key, value, attn_mask, dropout_p, enable_gqa):
        seen["query_heads"] = query.shape[1]
        seen["key_heads"] = key.shape[1]
        seen["value_heads"] = value.shape[1]
        seen["attn_mask"] = attn_mask
        seen["dropout_p"] = dropout_p
        seen["enable_gqa"] = enable_gqa
        return torch.zeros_like(query)

    monkeypatch.setattr("relflow.architecture.attention.F.scaled_dot_product_attention", sdpa)
    attention = RotaryMultiheadAttention(d_model=16, nhead=4, n_kv_heads=1, dropout=0.2)
    inputs = torch.randn(3, 5, 16)

    attention(query=inputs, key=inputs, value=inputs)

    assert seen == {
        "query_heads": 4,
        "key_heads": 1,
        "value_heads": 1,
        "attn_mask": None,
        "dropout_p": 0.2,
        "enable_gqa": True,
    }


def test_rotary_attention_padding_mask_uses_sdpa_keep_mask(monkeypatch: pytest.MonkeyPatch):
    seen = {}

    def sdpa(query, key, value, attn_mask, dropout_p, enable_gqa):
        seen["attn_mask"] = attn_mask
        return torch.zeros_like(query)

    monkeypatch.setattr("relflow.architecture.attention.F.scaled_dot_product_attention", sdpa)
    attention = RotaryMultiheadAttention(d_model=16, nhead=4, dropout=0.0)
    inputs = torch.randn(2, 3, 16)
    padding = torch.tensor(
        [
            [False, True, False],
            [True, True, True],
        ]
    )
    memory = inputs.masked_fill(padding.unsqueeze(-1), torch.nan)

    output = attention(
        query=inputs,
        key=memory,
        value=memory,
        key_padding_mask=padding,
    )

    assert torch.equal(
        seen["attn_mask"],
        torch.tensor(
            [
                [[[True, False, True]]],
                [[[True, False, False]]],
            ]
        ),
    )
    assert torch.equal(output[1], torch.zeros_like(output[1]))


def test_rotary_attention_zeros_all_absent_rows_after_sdpa(monkeypatch: pytest.MonkeyPatch):
    called = False

    def sdpa(*args, **kwargs):
        nonlocal called
        called = True
        return torch.zeros_like(kwargs["query"])

    monkeypatch.setattr("relflow.architecture.attention.F.scaled_dot_product_attention", sdpa)
    attention = RotaryMultiheadAttention(d_model=16, nhead=4, dropout=0.0)
    inputs = torch.randn(2, 3, 16)

    output = attention(
        query=inputs,
        key=inputs,
        value=inputs,
        key_padding_mask=torch.ones(2, 3, dtype=torch.bool),
    )

    assert called
    assert torch.equal(output, torch.zeros_like(output))


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
def test_attention_normalizes_each_head_before_rotary_and_learns_its_scale(monkeypatch, n_kv_heads):
    torch.manual_seed(41)
    attention = RotaryMultiheadAttention(16, 4, 0, n_kv_heads=n_kv_heads).double()
    with torch.no_grad():
        attention.q_norm.weight.copy_(torch.tensor([0.5, 1, 2, 3]))
        attention.k_norm.weight.copy_(torch.tensor([2, 0.5, 3, 1]))
        attention.head_scale.copy_(torch.tensor([[0.5], [1], [2], [3]]))
    query = torch.randn(2, 3, 16, dtype=torch.float64)
    key = torch.randn(2, 5, 16, dtype=torch.float64)
    value = torch.randn_like(key)
    query_positions = torch.tensor([[0, 2, 6], [1, 4, 8]])
    key_positions = torch.tensor([[0, 3, 5, 9, 12], [1, 2, 7, 10, 11]])
    projected_q = attention.q_proj(query).unflatten(-1, (4, 4)).transpose(1, 2)
    projected_k = attention.k_proj(key).unflatten(-1, (n_kv_heads, 4)).transpose(1, 2)
    expected_q = (
        attention.rotate(F.rms_norm(projected_q, (4,), attention.q_norm.weight, eps=1e-5), query_positions)
        * attention.head_scale[None, :, None, :]
    )
    expected_k = attention.rotate(F.rms_norm(projected_k, (4,), attention.k_norm.weight, eps=1e-5), key_positions)
    sdpa = F.scaled_dot_product_attention
    seen = []

    def observe(**kwargs):
        torch.testing.assert_close(kwargs["query"], expected_q)
        torch.testing.assert_close(kwargs["key"], expected_k)
        seen.append(True)
        return sdpa(**kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", observe)
    output = attention(query, key, value, query_positions=query_positions, key_positions=key_positions)
    output.square().sum().backward()
    assert seen == [True]
    assert attention.head_scale.grad is not None and torch.count_nonzero(attention.head_scale.grad) == 4
    assert torch.isfinite(attention.head_scale.grad).all()


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
def test_attention_resists_projected_magnitude_changes(n_kv_heads):
    torch.manual_seed(43)
    attention = RotaryMultiheadAttention(32, 4, 0, n_kv_heads=n_kv_heads)
    scaled = deepcopy(attention)
    with torch.no_grad():
        scaled.q_proj.weight.mul_(10_000)
        scaled.q_proj.bias.mul_(10_000)
        scaled.k_proj.weight.mul_(1_000)
        scaled.k_proj.bias.mul_(1_000)
    query = torch.randn(3, 4, 32)
    key = torch.randn(3, 7, 32)
    value = torch.randn_like(key)

    expected = attention(query, key, value)
    actual = scaled(query, key, value)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    actual.square().sum().backward()
    for parameter in scaled.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_normalized_heads_keep_projection_dtype_under_bfloat16_autocast():
    attention = RotaryMultiheadAttention(16, 4, 0, n_kv_heads=2)
    inputs = torch.randn(2, 3, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        q, k, v = attention.project(inputs, inputs, inputs)
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        output = attention(inputs, inputs, inputs)
    output.float().square().sum().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in attention.parameters()
    )
