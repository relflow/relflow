from copy import deepcopy
from types import MethodType

import pytest
import torch
import torch.nn.functional as F

import relflow as rf
from relflow.architecture import compiler
from relflow.architecture.encoder import BranchEncoder
from relflow.structs.packages import Parcel

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
DEVICES = ["cpu", pytest.param("cuda", marks=CUDA)]


def branch(*fields, attention="mha", dropout=0.0, device="cpu", dtype=torch.float32):
    schema = rf.Schema.from_tree(
        fields={name: rf.Number for name in fields or ("value",)},
        d_model=32,
        n_layers=2,
        n_heads=4,
        attention=attention,
        dropout=dropout,
        reduction=None,
    )
    return BranchEncoder(schema, "/").to(device=device, dtype=dtype)


def parcel(values, present, name="value"):
    return Parcel(
        payload=values,
        present=present,
        origin=f"/{name}",
        destination="/",
        batch_size=values.shape[0],
    )


def masked(encoder, scope):
    """Keep the reference on the original dense, tensor-presence protocol."""
    if scope == "stack":

        def compute(inputs, present):
            assert isinstance(present, torch.Tensor)
            return BranchEncoder.compute(encoder, inputs, present)

        encoder.compute = compute
    else:
        layer = encoder.coordinate_encoder
        forward = layer.forward

        def compute(inputs, present):
            assert isinstance(present, torch.Tensor)
            return forward(inputs, present)

        layer.forward = compute


def invoke(encoder, scope, values, present):
    if scope == "stack":
        return encoder([parcel(values[0], present)]).payload
    parcels = [parcel(value, present, name) for value, name in zip(values, ("left", "middle", "right"), strict=True)]
    return torch.stack([value.payload for value in encoder.contextualize(parcels)], dim=-2)


@CUDA
@pytest.mark.parametrize("scope", ["stack", "coordinate"])
@pytest.mark.parametrize("attention", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_presence_removes_sdpa_mask_and_preserves_training(monkeypatch, scope, attention, dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("requires CUDA BF16 support")
    torch.manual_seed(101)
    fields = ("left", "middle", "right") if scope == "coordinate" else ("value",)
    encoder = branch(*fields, attention=attention, device="cuda", dtype=dtype)
    reference = deepcopy(encoder)
    masked(reference, scope)
    values = [torch.randn(3, 5, 32, device="cuda", dtype=dtype, requires_grad=True) for _ in fields]
    originals = [value.detach().clone().requires_grad_() for value in values]
    present = torch.ones(3, 5, dtype=torch.bool, device="cuda")
    masks = []
    sdpa = F.scaled_dot_product_attention

    def observe(*args, **kwargs):
        masks.append(kwargs.get("attn_mask"))
        return sdpa(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(F, "scaled_dot_product_attention", observe)
        actual = invoke(encoder, scope, values, present)
    expected = invoke(reference, scope, originals, present)
    assert len(masks) == (2 if scope == "stack" else 1)
    assert all(mask is None for mask in masks)
    tolerance = (0.05, 0.05) if dtype == torch.bfloat16 else (2e-5, 2e-5)
    torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])

    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    for value, original in zip(values, originals, strict=True):
        assert value.grad is not None and original.grad is not None
        torch.testing.assert_close(value.grad, original.grad, rtol=tolerance[0], atol=tolerance[1])
    actual_gradients, expected_gradients = [], []
    for (name, parameter), (other, original) in zip(
        encoder.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == other
        if parameter.grad is None or original.grad is None:
            assert parameter.grad is None and original.grad is None
            continue
        torch.testing.assert_close(parameter.grad, original.grad, rtol=tolerance[0], atol=tolerance[1])
        actual_gradients.append(parameter.grad.flatten().float())
        expected_gradients.append(original.grad.flatten().float())
    actual_gradient = torch.cat(actual_gradients)
    expected_gradient = torch.cat(expected_gradients)
    relative_error = (actual_gradient - expected_gradient).norm() / expected_gradient.norm()
    assert relative_error < (0.02 if dtype == torch.bfloat16 else 2e-5)


@CUDA
@pytest.mark.parametrize("scope", ["stack", "coordinate"])
@pytest.mark.parametrize("boundary", ["dropout", "hook"])
def test_full_presence_keeps_masks_for_dropout_and_observers(monkeypatch, scope, boundary):
    torch.manual_seed(103)
    fields = ("left", "middle", "right") if scope == "coordinate" else ("value",)
    encoder = branch(*fields, dropout=0.2 if boundary == "dropout" else 0.0, device="cuda")
    reference = deepcopy(encoder)
    masked(reference, scope)
    values = [torch.randn(3, 5, 32, device="cuda") for _ in fields]
    present = torch.ones(3, 5, dtype=torch.bool, device="cuda")
    masks, observed = [], []
    sdpa = F.scaled_dot_product_attention

    def observe(*args, **kwargs):
        masks.append(kwargs["attn_mask"])
        return sdpa(*args, **kwargs)

    def hook(module, args, kwargs):
        observed.append(kwargs["present"])

    layer = encoder.coordinate_encoder if scope == "coordinate" else encoder.encoder[0]
    handle = layer.register_forward_pre_hook(hook, with_kwargs=True) if boundary == "hook" else None
    try:
        torch.manual_seed(107)
        with monkeypatch.context() as patch:
            patch.setattr(F, "scaled_dot_product_attention", observe)
            actual = invoke(encoder, scope, values, present)
        actual_rng = torch.cuda.get_rng_state()
        torch.manual_seed(107)
        expected = invoke(reference, scope, values, present)
        assert torch.equal(torch.cuda.get_rng_state(), actual_rng)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert len(masks) == (2 if scope == "stack" else 1)
        assert all(mask is not None and mask.all() for mask in masks)
        if boundary == "hook":
            assert len(observed) == 1
            expected_shape = (15, 3) if scope == "coordinate" else (3, 5)
            assert observed[0].shape == expected_shape and observed[0].dtype == torch.bool and observed[0].all()
    finally:
        if handle is not None:
            handle.remove()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["plain", "method", "borrowed", "subclass"])
def test_custom_compute_preserves_dense_tensor_protocol(device, kind):
    encoder = branch(device=device)
    observed = []

    def compute(inputs, present):
        assert isinstance(present, torch.Tensor) and present.dtype == torch.bool
        assert inputs.ndim == 3 and present.shape == inputs.shape[:2]
        observed.append((tuple(inputs.shape), present.clone()))
        return inputs * 2

    def method(self, inputs, present):
        return compute(inputs, present)

    if kind == "plain":
        encoder.compute = compute
    elif kind == "method":
        encoder.compute = MethodType(method, encoder)
    elif kind == "borrowed":
        donor = branch(device=device)
        original = donor.encoder[0].forward

        def forward(inputs, present):
            compute(inputs, present)
            return original(inputs, present)

        donor.encoder[0].forward = forward
        encoder.compute = donor.compute
    else:

        class Encoder(BranchEncoder):
            def compute(self, inputs, present):
                return compute(inputs, present)

        schema = rf.Schema.from_tree(value=rf.Number, d_model=32, n_layers=2, n_heads=4, dropout=0.0, reduction=None)
        encoder = Encoder(schema, "/").to(device)

    present = torch.ones(3, 5, dtype=torch.bool, device=device)
    values = torch.randn(3, 5, 32, device=device)
    original_presence = present.clone()
    result = encoder([parcel(values, present)])
    selected = present[present.any(dim=1)]
    assert observed and all(shape == (*selected.shape, 32) for shape, mask in observed)
    assert all(torch.equal(mask, selected) for shape, mask in observed)
    assert torch.equal(result.present, original_presence)
    assert torch.equal(present, original_presence)
    assert torch.isfinite(result.payload).all()
    assert not result.payload[~present].any()


@CUDA
@pytest.mark.parametrize("scope", ["stack", "coordinate"])
def test_large_prepared_batches_retain_masks_and_run_on_cuda(monkeypatch, scope):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("requires CUDA BF16 support")
    fields = ("left", "middle", "right") if scope == "coordinate" else ("value",)
    encoder = branch(*fields, device="cuda", dtype=torch.bfloat16).eval()
    values = [torch.randn(65536, 1, 32, device="cuda", dtype=torch.bfloat16) for _ in fields]
    present = torch.ones(65536, 1, device="cuda", dtype=torch.bool)
    masks = []
    sdpa = F.scaled_dot_product_attention

    def observe(*args, **kwargs):
        masks.append(kwargs["attn_mask"])
        return sdpa(*args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", observe)
    with torch.inference_mode():
        result = invoke(encoder, scope, values, present)
    assert masks and all(mask is not None for mask in masks)
    assert result.shape[0] == 65536 and torch.isfinite(result).all()


@CUDA
def test_compiled_full_presence_falls_back_for_new_hooks_and_resumes():
    encoder = branch(device="cuda").eval()
    graphs, executions, observed = [], [], []

    def backend(graph, inputs):
        graphs.append(graph)

        def execute(*args):
            executions.append(graph)
            return graph.forward(*args)

        return execute

    compiler.compile(encoder, encoders=True, pools=False, backend=backend, dynamic=None, options=None)
    values = torch.randn(3, 5, 32, device="cuda")
    present = torch.ones(3, 5, dtype=torch.bool, device="cuda")

    def hook(module, args, kwargs):
        observed.append((tuple(args[0].shape), kwargs["present"]))

    handle = None
    torch._dynamo.reset()
    try:
        with torch.no_grad():
            first = encoder([parcel(values, present)]).payload
            assert len(graphs) == len(executions) == 1
            attention = [node for node in graphs[0].graph.nodes if node.target is F.scaled_dot_product_attention]
            assert len(attention) == 2 and all(node.kwargs["attn_mask"] is None for node in attention)
            handle = encoder.encoder[0].register_forward_pre_hook(hook, with_kwargs=True)
            changed = encoder([parcel(values, present)]).payload
            assert len(graphs) == len(executions) == 1
            assert len(observed) == 1
            assert observed[0][0] == (3, 5, 32) and torch.equal(observed[0][1], present)
            torch.testing.assert_close(changed, first, rtol=2e-5, atol=2e-5)
            handle.remove()
            handle = None
            restored = encoder([parcel(values, present)]).payload
            assert len(graphs) == 1 and len(executions) == 2
            torch.testing.assert_close(restored, first, rtol=0, atol=0)
    finally:
        if handle is not None:
            handle.remove()
        torch._dynamo.reset()
