import pytest
import torch

from relflow.architecture.rotary import RotaryEmbedding

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [7, 16])
def test_rotary_preserves_head_axis_outputs_and_gradients(device, dtype, width):
    rotary = RotaryEmbedding(width).to(device=device)
    torch.manual_seed(31)

    for batch, length in ((2, 5), (2, 0), (2, 1), (2, 9), (2, 3), (0, 5)):
        inputs = torch.randn(batch, length, 3, width, device=device, dtype=dtype).transpose(1, 2).requires_grad_()
        original = inputs.detach().clone()
        flattened = inputs.detach().reshape(batch * 3, length, width).requires_grad_()
        weight = torch.randn_like(inputs)

        output = rotary(inputs)
        expected = rotary(flattened).reshape_as(inputs)
        gradient = torch.autograd.grad(output, inputs, weight)[0]
        expected_gradient = torch.autograd.grad(expected, flattened, weight)[0].reshape_as(inputs)

        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
        torch.testing.assert_close(inputs, original, rtol=0, atol=0)
        assert output.dtype == dtype
        assert output.device == inputs.device
        if batch and length > 1:
            assert not inputs.is_contiguous()
        if width % 2:
            torch.testing.assert_close(output[..., -1], inputs[..., -1], rtol=0, atol=0)
            torch.testing.assert_close(gradient[..., -1], weight[..., -1], rtol=0, atol=0)


def test_rotary_preserves_position_zero_and_pair_lengths():
    rotary = RotaryEmbedding(7).double()
    inputs = torch.randn(2, 3, 5, 7, dtype=torch.float64)

    output = rotary(inputs)

    torch.testing.assert_close(output[..., 0, :], inputs[..., 0, :], rtol=0, atol=0)
    input_pairs = inputs[..., :6].unflatten(-1, (3, 2))
    output_pairs = output[..., :6].unflatten(-1, (3, 2))
    torch.testing.assert_close(output_pairs.square().sum(-1), input_pairs.square().sum(-1))


@pytest.mark.parametrize("device", DEVICES)
def test_rotary_keeps_checkpoint_state_and_device_dtype_movement(device):
    rotary = RotaryEmbedding(7)
    assert set(dict(rotary.named_buffers())) == {"inv_freq"}
    assert rotary.state_dict() == {}

    for dtype in (torch.float64, torch.float16, torch.bfloat16, torch.float32):
        rotary.to(device=device, dtype=dtype)
        restored = RotaryEmbedding(7).to(device=device, dtype=dtype)
        restored.inv_freq.copy_(rotary.inv_freq)
        restored.load_state_dict(rotary.state_dict(), strict=True)
        inputs = torch.randn(2, 3, 5, 7, device=device, dtype=dtype)
        with torch.inference_mode():
            rotary(inputs)

        output = rotary(inputs.requires_grad_())
        expected = restored(inputs)

        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert rotary.inv_freq.device == inputs.device
        assert rotary.inv_freq.dtype == dtype
        assert torch.isfinite(torch.autograd.grad(output.sum(), inputs)[0]).all()
        assert rotary.state_dict() == {}


def test_rotary_compiles_with_dynamic_sequence_lengths():
    rotary = RotaryEmbedding(7)
    compiled = torch.compile(rotary, backend="eager", fullgraph=True, dynamic=True)

    for length in (5, 9, 0, 1):
        inputs = torch.randn(2, length, 3, 7).transpose(1, 2).requires_grad_()
        weight = torch.randn_like(inputs)

        output = compiled(inputs)
        expected = rotary(inputs)

        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.autograd.grad(output, inputs, weight)[0],
            torch.autograd.grad(expected, inputs, weight)[0],
            rtol=0,
            atol=0,
        )
