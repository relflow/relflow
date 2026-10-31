from copy import deepcopy

import pytest
import torch

import relflow as rf
from relflow.architecture import compiler
from relflow.architecture.blocks import feed_forward, norm
from relflow.architecture.encoder import BranchEncoder, RotaryTransformerEncoderLayer
from relflow.architecture.packed import customized, pack, unpack
from relflow.architecture.pool import LearnedQueryCrossAttention


@pytest.mark.parametrize("width", [16, 96, 128])
@pytest.mark.parametrize("ffnx", [4, 8, 12])
def test_feed_forward_uses_node_channels_tanh_gelu_and_stable_parameter_keys(width, ffnx):
    module = feed_forward(width, 0.2, ffnx)
    assert module[0].in_features == width
    assert module[0].out_features == ffnx * width
    assert module[1].approximate == "tanh"
    assert module[2].p == module[4].p == 0.2
    assert module[3].in_features == ffnx * width
    assert module[3].out_features == width
    assert sum(parameter.numel() for parameter in module.parameters()) == 2 * ffnx * width**2 + (ffnx + 1) * width
    assert list(module.state_dict()) == ["0.weight", "0.bias", "3.weight", "3.bias"]


def test_feed_forward_defaults_to_eightfold_channels():
    module = feed_forward(16, 0.0)
    assert module[0].out_features == module[3].in_features == 8 * 16


def test_branch_multiplier_reaches_sequence_coordinate_and_pool_blocks_independently():
    schema = rf.Schema.from_tree(
        d_model=16,
        n_layers=2,
        n_heads=4,
        ffnx=6,
        group=rf.Branch(length=4, ffnx=3, n_layers=2, left=rf.Number, right=rf.Number),
        answer=rf.Boolean(mask=True),
    )
    root = BranchEncoder(schema, rf.Address())
    group = BranchEncoder(schema, rf.Address("group"))
    for encoder, expected in ((root, 6), (group, 3)):
        layers = [*encoder.encoder, *encoder.pool.blocks]
        if encoder.coordinate_encoder is not None:
            layers.append(encoder.coordinate_encoder)
        assert layers
        for layer in layers:
            assert layer.ffn[0].out_features == layer.ffn[3].in_features == expected * 16
            assert layer.ffn[1].approximate == "tanh"
            assert isinstance(layer.attention_norm, torch.nn.RMSNorm)
            assert isinstance(layer.ffn_norm, torch.nn.RMSNorm)


def test_feed_forward_tanh_formula_and_weight_reload_preserve_outputs_and_gradients():
    torch.manual_seed(37)
    module = feed_forward(16, 0.0).double()
    restored = feed_forward(16, 0.0).double()
    restored.load_state_dict(module.state_dict(), strict=True)
    inputs = torch.linspace(-4, 4, 48, dtype=torch.float64).reshape(3, 16).requires_grad_()
    copied = inputs.detach().clone().requires_grad_()
    expected = module[3](torch.nn.functional.gelu(module[0](inputs), approximate="tanh"))
    actual = restored(copied)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(copied.grad, inputs.grad, rtol=0, atol=0)
    for parameter, reference in zip(restored.parameters(), module.parameters(), strict=True):
        assert parameter.grad is not None and reference.grad is not None
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)


def test_rms_normalization_has_explicit_epsilon_and_width_local_axes():
    module = norm(8).double()
    assert isinstance(module, torch.nn.RMSNorm)
    assert module.eps == 1e-5
    assert module.normalized_shape == (8,)
    assert list(module.state_dict()) == ["weight"]
    inputs = (torch.randn(3, 4, 8, dtype=torch.float64) + 2).requires_grad_()
    outputs = module(inputs)
    expected = inputs * torch.rsqrt(inputs.square().mean(dim=-1, keepdim=True) + 1e-5)
    torch.testing.assert_close(outputs, expected, rtol=1e-12, atol=1e-12)
    assert outputs.mean(dim=-1).abs().sum() > 0
    outputs.square().sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


def test_rmsnorm_uses_explicit_epsilon_for_bfloat16_zero_and_small_inputs():
    module = norm(8).to(dtype=torch.bfloat16)
    inputs = torch.tensor([[0.0] * 8, [1e-4] * 8], dtype=torch.bfloat16, requires_grad=True)
    output = module(inputs)
    assert torch.isfinite(output).all()
    assert not output[0].any()
    expected = inputs.float() * torch.rsqrt(inputs.float().square().mean(dim=-1, keepdim=True) + 1e-5)
    torch.testing.assert_close(output.float(), expected, rtol=0.01, atol=1e-5)
    output.float().square().sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize("n_kv_heads", [4, 2, 1])
@pytest.mark.parametrize("ffnx", [2, 8, 12])
def test_node_blocks_preserve_dense_and_packed_outputs_and_gradients(n_kv_heads, ffnx):
    torch.manual_seed(43)
    dense = RotaryTransformerEncoderLayer(16, 4, n_kv_heads, 0.0, ffnx=ffnx).double()
    packed = deepcopy(dense)
    assert not customized((packed,))
    assert isinstance(dense.attention_norm, torch.nn.RMSNorm)
    assert isinstance(dense.ffn_norm, torch.nn.RMSNorm)
    present = torch.tensor(
        [[True, False, True, False, True], [False, False, False, False, False], [False, True, False, True, False]]
    )
    values = torch.randn(3, 5, 16, dtype=torch.float64).masked_fill(~present[..., None], torch.nan)
    dense_inputs = values.clone().requires_grad_()
    packed_inputs = values.clone().requires_grad_()
    expected = dense(dense_inputs, present)
    selected, packing = pack(packed_inputs, present)
    actual = unpack(packed(selected, present, packing=packing), packing, packed_inputs, present, (packed,))
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(packed_inputs.grad, dense_inputs.grad, rtol=1e-10, atol=1e-10)
    assert not actual[~present].any()
    assert not packed_inputs.grad[~present].any()
    for (name, parameter), (other, reference) in zip(packed.named_parameters(), dense.named_parameters(), strict=True):
        assert name == other
        assert parameter.grad is not None and reference.grad is not None
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("n_kv_heads", [4, 2, 1])
@pytest.mark.parametrize("training", [True, False])
def test_packed_node_passes_training_dropout_to_each_attention_group(monkeypatch, n_kv_heads, training):
    torch.manual_seed(47)
    layer = RotaryTransformerEncoderLayer(16, 4, n_kv_heads, 0.2).double().train(training)
    assert not customized((layer,))
    present = torch.tensor(
        [[True, False, True, False, True], [False, False, False, False, False], [False, True, False, True, False]]
    )
    inputs = torch.randn(3, 5, 16, dtype=torch.float64).masked_fill(~present[..., None], torch.nan).requires_grad_()
    values, packing = pack(inputs, present)
    calls = []
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def observe(query, key, value, **kwargs):
        calls.append((query.shape, key.shape, kwargs))
        return sdpa(query, key, value, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", observe)
    output = unpack(layer(values, present, packing=packing), packing, inputs, present, (layer,))
    output.square().sum().backward()

    assert len(calls) == 2
    assert {shape[-2] for shape, _, _ in calls} == {2, 3}
    for query, key, options in calls:
        assert query[1] == 4 and key[1] == n_kv_heads
        assert options["dropout_p"] == (0.2 if training else 0.0)
        assert options["enable_gqa"] is (n_kv_heads != 4)
        assert options.get("attn_mask") is None
    assert torch.isfinite(output).all() and not output[~present].any()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert not inputs.grad[~present].any()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in layer.parameters())


def test_learned_pool_preserves_additive_evidence_and_count_after_rms_normalization():
    pool = LearnedQueryCrossAttention(1, 4, 2, 0.0, position=False, mass_capacity=4, mass=True, ffnx=3).double()
    assert isinstance(pool.norm, torch.nn.RMSNorm)
    with torch.no_grad():
        pool.mass_projection.weight.copy_(torch.eye(4))
        pool.mass_direction.fill_(0.25)
    memory = torch.tensor([[[0.5, -1.0, 0.25, 2.0]]], dtype=torch.float64).expand(1, 4, 4).clone()
    one = torch.tensor([[True, False, False, False]])
    three = torch.tensor([[True, True, True, False]])
    first = pool(memory, present=one, evidence=memory)
    second = pool(memory, present=three, evidence=memory)
    torch.testing.assert_close(second - first, memory[:, :1] + 0.125, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("boundary", ["normalization", "projection", "activation"])
def test_hooks_added_after_compiler_warmup_force_eager_and_resume(boundary):
    pool = LearnedQueryCrossAttention(1, 8, 2, 0.0)
    observed, executions = [], []

    def backend(graph, inputs):
        def execute(*args):
            executions.append(graph)
            return graph.forward(*args)

        return execute

    descendant = {
        "normalization": pool.blocks[0].ffn_norm,
        "projection": pool.blocks[0].ffn[0],
        "activation": pool.blocks[0].ffn[1],
    }[boundary]
    memory = torch.randn(2, 3, 8)
    present = torch.ones(2, 3, dtype=torch.bool)

    def hook(owner, inputs, output):
        observed.append(owner)
        return output + 1

    handle = None
    torch._dynamo.reset()
    try:
        compiler.compile(pool, encoders=False, pools=True, backend=backend, dynamic=None, options=None)
        first = pool.compute(memory, present, None, None)
        assert len(executions) == 1
        handle = descendant.register_forward_hook(hook)
        second = pool.compute(memory, present, None, None)
        assert observed == [descendant]
        assert len(executions) == 1
        assert not torch.equal(second, first)
        handle.remove()
        handle = None
        restored = pool.compute(memory, present, None, None)
        assert len(executions) == 2
        torch.testing.assert_close(restored, first, rtol=0, atol=0)
    finally:
        if handle is not None:
            handle.remove()
        torch._dynamo.reset()


def test_rmsnorm_over_multiple_axes_remains_outside_packing_contract():
    layer = RotaryTransformerEncoderLayer(8, 2, 2, 0.0)
    assert not customized((layer,))
    layer.ffn_norm = torch.nn.RMSNorm((3, 8), eps=1e-5)
    assert customized((layer,))


def test_replaced_layernorm_retains_dense_module_contract():
    layer = RotaryTransformerEncoderLayer(8, 2, 2, 0.0)
    layer.ffn_norm = torch.nn.LayerNorm(8)
    assert customized((layer,))
