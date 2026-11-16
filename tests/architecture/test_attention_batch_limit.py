from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from relflow.architecture import packed
from relflow.architecture.attention import RotaryMultiheadAttention
from relflow.architecture.encoder import RotaryTransformerEncoderLayer


@pytest.mark.parametrize("kv_heads", [4, 2, 1])
@pytest.mark.parametrize("group_rows", [1, 2, 3, 5])
def test_packed_attention_chunks_whole_rows_and_preserves_values_and_gradients(monkeypatch, kv_heads, group_rows):
    torch.manual_seed(67)
    attention = RotaryMultiheadAttention(20, 4, 0.0, n_kv_heads=kv_heads).double()
    reference = deepcopy(attention)
    patterns = [[True, False, True, False, False], [False, True, False, True, True]]
    rows = patterns * group_rows
    rows.insert(2, [False] * 5)
    present = torch.tensor(rows)
    inputs = torch.randn(len(rows), 5, 20, dtype=torch.float64, requires_grad=True)
    original = inputs.detach().clone().requires_grad_()
    positions = torch.arange(5).square()[None, :] + torch.arange(len(rows))[:, None] * 11
    weights = torch.randn_like(inputs)
    calls = []
    sdpa = F.scaled_dot_product_attention

    def observe(query, key, value, **options):
        calls.append((tuple(query.shape), tuple(key.shape), options))
        return sdpa(query, key, value, **options)

    with monkeypatch.context() as patch:
        patch.setattr(packed, "SDPA_BATCH_MAX", 2)
        patch.setattr(F, "scaled_dot_product_attention", observe)
        values, packing = packed.pack(inputs, present, positions)
        attended = attention(values, values, values, packing=packing)
        actual = packed.unpack(attended, packing, inputs, present, ())
    expected = reference(
        original,
        original,
        original,
        key_padding_mask=~present,
        query_positions=positions,
        key_positions=positions,
    ).masked_fill(~present.unsqueeze(-1), 0.0)

    expected_shapes = [
        (min(2, group_rows - start), 4, length, 5) for length in (2, 3) for start in range(0, group_rows, 2)
    ]
    assert [query for query, _, _ in calls] == expected_shapes
    assert all(key[1] == kv_heads for _, key, _ in calls)
    assert all(options == {"dropout_p": 0.0, "enable_gqa": kv_heads != 4} for _, _, options in calls)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    assert not actual[~present].any()

    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(inputs.grad, original.grad, rtol=1e-10, atol=1e-10)
    assert not inputs.grad[~present].any()
    for (name, parameter), (other, original_parameter) in zip(
        attention.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == other
        torch.testing.assert_close(parameter.grad, original_parameter.grad, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("kv_heads", [4, 1])
@pytest.mark.parametrize("capacity", [0, 3])
def test_packed_attention_accepts_no_live_groups(monkeypatch, kv_heads, capacity):
    attention = RotaryMultiheadAttention(32, 4, 0.0, n_kv_heads=kv_heads).double()
    inputs = torch.randn(4, capacity, 32, dtype=torch.float64, requires_grad=True)
    values, packing = packed.pack(inputs, torch.zeros(4, capacity, dtype=torch.bool))

    def unexpected_attention(*args, **kwargs):
        pytest.fail("an empty packed sequence must not launch attention")

    monkeypatch.setattr(F, "scaled_dot_product_attention", unexpected_attention)
    output = attention(values, values, values, packing=packing)

    assert not packing.groups
    assert output.shape == (0, 32)
    query_gradient, weight_gradient, bias_gradient = torch.autograd.grad(
        output.sum(), (values, attention.out_proj.weight, attention.out_proj.bias), allow_unused=True
    )
    assert query_gradient is None
    assert weight_gradient is not None and not weight_gradient.any()
    assert bias_gradient is not None and not bias_gradient.any()


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_packed_flash_attention_exceeds_cuda_batch_grid_with_finite_gradients(monkeypatch):
    if not torch.cuda.is_bf16_supported() or not torch.backends.cuda.is_flash_attention_available():
        pytest.skip("requires CUDA BF16 FlashAttention support")
    torch.manual_seed(71)
    layer = RotaryTransformerEncoderLayer(32, 4, 4, 0.0).cuda().bfloat16()
    reference = deepcopy(layer)
    rows = packed.SDPA_BATCH_MAX + 1
    inputs = torch.randn(rows, 3, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    present = torch.tensor([True, False, True], device="cuda").expand(rows, -1)
    positions = torch.tensor([0, 3, 7], device="cuda").expand(rows, -1)
    calls = []
    sdpa = F.scaled_dot_product_attention

    def observe(query, key, value, **options):
        calls.append(query.shape[0])
        return sdpa(query, key, value, **options)

    with monkeypatch.context() as patch, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        patch.setattr(F, "scaled_dot_product_attention", observe)
        values, packing = packed.pack(inputs, present, positions)
        encoded = layer(values, present, packing=packing)
        actual = packed.unpack(encoded, packing, inputs, present, (layer,))
        actual.float().square().mean().backward()
        torch.cuda.synchronize()

    selected = torch.tensor([0, rows - 1], device="cuda")
    with torch.no_grad():
        expected = reference(
            inputs.index_select(0, selected),
            present.index_select(0, selected),
            positions=positions.index_select(0, selected),
        )
    assert calls == [packed.SDPA_BATCH_MAX, 1]
    torch.testing.assert_close(actual.index_select(0, selected), expected, rtol=0.05, atol=0.05)
    assert not actual[~present].any()
    assert torch.isfinite(actual).all() and torch.isfinite(inputs.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in layer.parameters())
