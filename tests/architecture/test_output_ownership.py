import pytest
import torch

import relflow as rf
from relflow.architecture.encoder import BranchEncoder
from relflow.architecture.pool import LearnedQueryCrossAttention
from relflow.structs.packages import Parcel


class ReusableOutput(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_buffer("value", torch.zeros(shape))
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        self.value.add_(1)
        return self.value


def check_output_ownership(invoke, producer):
    first = invoke()
    retained = first.clone()
    second = invoke()
    torch.testing.assert_close(first, retained, rtol=0, atol=0)
    private = producer.value.clone()
    second.fill_(99)
    torch.testing.assert_close(producer.value, private, rtol=0, atol=0)
    torch.testing.assert_close(first, retained, rtol=0, atol=0)
    assert first.is_contiguous()
    assert second.is_contiguous()


@pytest.mark.parametrize("mode", [torch.no_grad, torch.inference_mode], ids=["no_grad", "inference"])
@pytest.mark.parametrize("active_rows", [2, 1, 0], ids=["complete", "partial", "empty"])
@pytest.mark.parametrize("boundary", ["compute", "norm_hook", "norm_subclass"])
def test_pool_owns_outputs_from_reusable_custom_buffers(monkeypatch, mode, active_rows, boundary):
    pool = LearnedQueryCrossAttention(1, 8, 2, 0.0).eval()
    producer = ReusableOutput((active_rows, 1, 8))
    if boundary == "compute":
        monkeypatch.setattr(pool, "compute", producer.forward)
    elif boundary == "norm_hook":
        pool.norm.register_forward_hook(producer.forward)
    else:

        class ReusableNorm(torch.nn.LayerNorm):
            def forward(self, inputs):
                return producer(inputs)

        pool.norm = ReusableNorm(8)
    memory = torch.randn(2, 3, 8)
    present = torch.arange(2).unsqueeze(1).expand(2, 3) < active_rows
    original_memory = memory.clone()
    original_present = present.clone()
    with mode():
        check_output_ownership(lambda: pool(memory, present), producer)
    assert producer.calls == (2 if active_rows else 0)
    torch.testing.assert_close(memory, original_memory, rtol=0, atol=0)
    assert torch.equal(present, original_present)


@pytest.mark.parametrize("mode", [torch.no_grad, torch.inference_mode], ids=["no_grad", "inference"])
@pytest.mark.parametrize("active_rows", [2, 1, 0], ids=["complete", "partial", "empty"])
@pytest.mark.parametrize("boundary", ["compute", "layer_hook", "pool", "pool_hook"])
def test_branch_owns_outputs_from_reusable_custom_buffers(monkeypatch, mode, active_rows, boundary):
    pooled = boundary in {"pool", "pool_hook"}
    schema = rf.Schema.from_tree(
        x=rf.Number(), d_model=8, n_heads=2, n_layers=1, dropout=0.0, reduction=rf.Attention() if pooled else None
    )
    encoder = BranchEncoder(schema, "record").eval()
    producer = ReusableOutput((active_rows, 1 if pooled else 3, 8))
    if boundary == "compute":
        monkeypatch.setattr(encoder, "compute", producer.forward)
    elif boundary == "layer_hook":
        encoder.encoder[0].register_forward_hook(producer.forward)
    elif boundary == "pool":
        encoder.pool = producer
    else:
        encoder.pool.register_forward_hook(producer.forward)
    parcel = Parcel(
        payload=torch.randn(2, 3, 8),
        present=torch.arange(2).unsqueeze(1).expand(2, 3) < active_rows,
        origin="record/x",
        destination="record",
        batch_size=2,
    )
    original_payload = parcel.payload.clone()
    original_present = parcel.present.clone()
    with mode():
        check_output_ownership(lambda: encoder([parcel]).payload, producer)
    assert producer.calls == (2 if active_rows else 0)
    torch.testing.assert_close(parcel.payload, original_payload, rtol=0, atol=0)
    assert torch.equal(parcel.present, original_present)


@pytest.mark.parametrize("active_rows", [2, 1, 0], ids=["complete", "partial", "empty"])
def test_branch_presence_does_not_alias_compute_inputs(monkeypatch, active_rows):
    schema = rf.Schema.from_tree(x=rf.Number(), d_model=8, n_heads=2, n_layers=1, dropout=0.0, reduction=None)
    encoder = BranchEncoder(schema, "record")
    captured = []

    def compute(inputs, present):
        captured.append(present)
        return inputs

    monkeypatch.setattr(encoder, "compute", compute)
    present = torch.arange(2).unsqueeze(1).expand(2, 3) < active_rows
    parcel = Parcel(
        payload=torch.randn(2, 3, 8),
        present=present,
        origin="record/x",
        destination="record",
        batch_size=2,
    )
    original = present.clone()
    with torch.inference_mode():
        first = encoder([parcel])
        second = encoder([parcel])
        for selected in captured:
            selected.fill_(False)
        assert torch.equal(first.present, original)
        assert torch.equal(second.present, original)
        second.present.fill_(True)
        assert torch.equal(first.present, original)
        assert torch.equal(parcel.present, original)


@pytest.mark.parametrize("branch", [False, True], ids=["pool", "branch"])
@pytest.mark.parametrize("active_rows", [2, 1, 0], ids=["complete", "partial", "empty"])
def test_parameter_outputs_are_owned_without_detaching_gradients(monkeypatch, branch, active_rows):
    memory = torch.randn(2, 3, 8)
    present = torch.arange(2).unsqueeze(1).expand(2, 3) < active_rows
    if branch:
        schema = rf.Schema.from_tree(x=rf.Number(), d_model=8, n_heads=2, n_layers=1, dropout=0.0, reduction=None)
        module = BranchEncoder(schema, "record")
        module.register_parameter("result", torch.nn.Parameter(torch.randn(3, 8)))
        parameter = module.result
        parcel = Parcel(
            payload=memory,
            present=present,
            origin="record/x",
            destination="record",
            batch_size=2,
        )

        def invoke():
            return module([parcel]).payload
    else:
        module = LearnedQueryCrossAttention(1, 8, 2, 0.0)
        parameter = module.queries

        def invoke():
            return module(memory, present)

    def compute(inputs, *args, **kwargs):
        return parameter.unsqueeze(0).expand(inputs.shape[0], -1, -1)

    monkeypatch.setattr(module, "compute", compute)
    before = parameter.detach().clone()
    output = invoke()
    assert torch.count_nonzero(output[active_rows:]) == 0
    output.sum().backward()
    torch.testing.assert_close(parameter.grad, torch.full_like(parameter, active_rows), rtol=0, atol=0)
    for mode in (torch.no_grad, torch.inference_mode):
        with mode():
            invoke().add_(10)
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)


@pytest.mark.parametrize("branch", [False, True], ids=["pool", "branch"])
@pytest.mark.parametrize("active_rows", [2, 1], ids=["complete", "partial"])
@pytest.mark.parametrize("invalid", ["shape", "dtype"])
def test_custom_outputs_preserve_shape_and_dtype_contract(monkeypatch, branch, active_rows, invalid):
    memory = torch.randn(2, 3, 8)
    present = torch.arange(2).unsqueeze(1).expand(2, 3) < active_rows
    if branch:
        schema = rf.Schema.from_tree(x=rf.Number(), d_model=8, n_heads=2, n_layers=1, dropout=0.0, reduction=None)
        module = BranchEncoder(schema, "record")
        parcel = Parcel(
            payload=memory,
            present=present,
            origin="record/x",
            destination="record",
            batch_size=2,
        )

        def invoke():
            return module([parcel])
    else:
        module = LearnedQueryCrossAttention(1, 8, 2, 0.0)

        def invoke():
            return module(memory, present)

    def compute(inputs, *args, **kwargs):
        width = 9 if invalid == "shape" else 8
        dtype = torch.float64 if invalid == "dtype" else torch.float32
        return torch.zeros(inputs.shape[0], 3 if branch else 1, width, dtype=dtype)

    monkeypatch.setattr(module, "compute", compute)
    with pytest.raises(RuntimeError):
        invoke()


@pytest.mark.parametrize("active_rows", [2, 1], ids=["complete", "partial"])
@pytest.mark.parametrize("invalid", ["shape", "dtype"])
def test_branch_rejects_custom_mutation_of_prepared_presence(monkeypatch, active_rows, invalid):
    schema = rf.Schema.from_tree(x=rf.Number(), d_model=8, n_heads=2, n_layers=1, dropout=0.0, reduction=None)
    encoder = BranchEncoder(schema, "record")

    def compute(inputs, present):
        if invalid == "shape":
            present.resize_(inputs.shape[0], inputs.shape[1] + 1)
        else:
            present.data = present.float()
        return inputs

    monkeypatch.setattr(encoder, "compute", compute)
    present = torch.arange(2).unsqueeze(1).expand(2, 3) < active_rows
    parcel = Parcel(
        payload=torch.randn(2, 3, 8),
        present=present,
        origin="record/x",
        destination="record",
        batch_size=2,
    )
    with pytest.raises(RuntimeError):
        encoder([parcel])
    assert present.dtype == torch.bool and present.shape == (2, 3)


@pytest.mark.parametrize("active_rows", [2, 1], ids=["complete", "partial"])
def test_pool_rejects_dtype_change_through_prepared_memory(monkeypatch, active_rows):
    pool = LearnedQueryCrossAttention(1, 8, 2, 0.0)

    def compute(memory, *args):
        # Mutating a private prepared tensor cannot change the caller's output contract.
        memory.data = memory.double()
        return memory.new_zeros(memory.shape[0], 1, 8)

    monkeypatch.setattr(pool, "compute", compute)
    memory = torch.randn(2, 3, 8)
    present = torch.arange(2).unsqueeze(1).expand(2, 3) < active_rows
    with pytest.raises(RuntimeError):
        pool(memory, present)
    assert memory.dtype == torch.float32
