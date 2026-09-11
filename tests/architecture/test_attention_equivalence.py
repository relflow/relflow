from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from relflow.architecture.attention import RotaryMultiheadAttention

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
]


def selected_attention(attention, query, key, value, padding):
    """Attend to each row's selected keys, preserving their original rotary positions."""
    if key.shape[1] == 0:
        return query.new_zeros(query.shape) + sum(parameter.sum() * 0 for parameter in attention.parameters())

    present = torch.ones(key.shape[:2], dtype=torch.bool, device=key.device) if padding is None else ~padding
    q = attention.rotate(attention.splitheads(attention.q_proj(query), attention.nhead))
    k = attention.rotate(
        attention.splitheads(attention.k_proj(key.masked_fill(~present[..., None], 0)), attention.n_kv_heads)
    )
    v = attention.splitheads(attention.v_proj(value.masked_fill(~present[..., None], 0)), attention.n_kv_heads)
    rows = []
    for row, selected in enumerate(present):
        if selected.any():
            context = F.scaled_dot_product_attention(
                q[row : row + 1],
                k[row : row + 1, :, selected],
                v[row : row + 1, :, selected],
                enable_gqa=attention.n_kv_heads != attention.nhead,
            )
            context = context.transpose(1, 2).reshape(1, query.shape[1], attention.d_model)
            rows.append(attention.out_proj(context))
        else:
            # An empty row participates in backward with zero gradients.
            anchor = q[row].sum() * 0 + k[row].sum() * 0 + v[row].sum() * 0
            anchor = anchor + sum(parameter.sum() * 0 for parameter in attention.out_proj.parameters())
            rows.append(query.new_zeros(1, query.shape[1], attention.d_model) + anchor)
    return torch.cat(rows)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "mode,heads,query_length,key_length,mask,position",
    [
        ("self", 4, 5, 5, "mixed", True),
        ("self", 2, 5, 5, "mixed", True),
        ("self", 1, 5, 5, "mixed", True),
        ("shared", 1, 2, 5, "mixed", True),
        ("separate", 2, 2, 5, "mixed", True),
        ("separate", 4, 2, 5, "none", False),
        ("shared", 4, 1, 1, "mixed", False),
        ("separate", 2, 2, 5, "empty", True),
        ("shared", 1, 2, 0, "empty", True),
        ("separate", 4, 0, 5, "mixed", True),
    ],
)
def test_attention_matches_selected_keys_and_gradients(device, mode, heads, query_length, key_length, mask, position):
    torch.manual_seed(31)
    attention = RotaryMultiheadAttention(16, 4, 0, n_kv_heads=heads, position=position).to(device)
    reference = deepcopy(attention)
    padding = torch.zeros(3, key_length, dtype=torch.bool, device=device)
    if mask == "mixed":
        padding[0, 1::2] = True
        padding[1] = True
    elif mask == "empty":
        padding[:] = True

    query = torch.randn(3, query_length, 16, device=device, requires_grad=True)
    if mode == "self":
        key = value = query
    else:
        key = torch.randn(3, key_length, 16, device=device).masked_fill(padding[..., None], torch.nan)
        key.requires_grad_()
        value = key
        if mode == "separate":
            value = torch.randn_like(key).masked_fill(padding[..., None], torch.nan).requires_grad_()
    inputs = (query, key, value)
    reference_inputs = deepcopy(inputs)
    original_inputs = tuple(item.detach().clone() for item in inputs)
    original_padding = padding.clone()
    supplied_padding = None if mask == "none" else padding

    actual = attention(*inputs, key_padding_mask=supplied_padding)
    expected = selected_attention(reference, *reference_inputs, supplied_padding)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert torch.isfinite(actual).all()
    if supplied_padding is not None:
        assert torch.count_nonzero(actual[padding.all(dim=1)]) == 0
    for item, original in zip(inputs, original_inputs, strict=True):
        torch.testing.assert_close(item, original, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(padding, original_padding)

    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    for actual_parameter, expected_parameter in zip(attention.parameters(), reference.parameters(), strict=True):
        assert actual_parameter.grad is not None
        assert expected_parameter.grad is not None
        assert torch.isfinite(actual_parameter.grad).all()
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=3e-5, atol=3e-6)
    for actual_input, expected_input in zip(inputs, reference_inputs, strict=True):
        if expected_input.grad is None:
            assert actual_input.grad is None
        else:
            assert actual_input.grad is not None
            assert torch.isfinite(actual_input.grad).all()
            torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=3e-5, atol=3e-6)
    if mode != "self" and key_length:
        assert torch.count_nonzero(key.grad[padding]) == 0
        assert torch.count_nonzero(value.grad[padding]) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_attention_dropout_preserves_rng_boundary_and_eval_is_deterministic(device, monkeypatch):
    torch.manual_seed(32)
    attention = RotaryMultiheadAttention(16, 4, 0.25, n_kv_heads=1).to(device)
    query = torch.randn(3, 2, 16, device=device)
    memory = torch.randn(3, 5, 16, device=device)
    padding = torch.tensor([[False, True, False, True, False], [True] * 5, [False] * 5], device=device)
    original_sdpa = F.scaled_dot_product_attention
    seen = []

    def rng():
        return torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()

    def sdpa(*args, **kwargs):
        seen.append((kwargs["dropout_p"], rng().clone()))
        return original_sdpa(*args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", sdpa)
    torch.manual_seed(33)
    before = rng().clone()
    first = attention(query, memory, memory, key_padding_mask=padding)
    after = rng().clone()
    assert seen[-1][0] == 0.25
    assert torch.equal(seen[-1][1], before)
    assert not torch.equal(after, before)

    torch.manual_seed(33)
    second = attention(query, memory, memory, key_padding_mask=padding)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(rng(), after)

    attention.eval()
    before = rng().clone()
    first = attention(query, memory, memory, key_padding_mask=padding)
    second = attention(query, memory, memory, key_padding_mask=padding)
    assert seen[-1][0] == 0
    assert torch.equal(rng(), before)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_attention_state_dict_retains_projection_parameters_only(device):
    attention = RotaryMultiheadAttention(16, 4, 0, n_kv_heads=2).to(device)
    restored = RotaryMultiheadAttention(16, 4, 0, n_kv_heads=2).to(device)
    state = deepcopy(attention.state_dict())
    expected_keys = {
        f"{projection}.{parameter}"
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj")
        for parameter in ("weight", "bias")
    }
    assert set(state) == expected_keys
    restored.load_state_dict(state, strict=True)
    query = torch.randn(2, 3, 16, device=device)
    memory = torch.randn(2, 5, 16, device=device)
    output = attention(query, memory, memory)
    torch.testing.assert_close(output, restored(query, memory, memory), rtol=0, atol=0)
    assert set(attention.state_dict()) == expected_keys
