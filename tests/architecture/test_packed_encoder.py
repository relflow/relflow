from copy import deepcopy

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import relflow as rf
from relflow.architecture.encoder import BranchEncoder
from relflow.structs.packages import Parcel


def branch(*fields, attention="mha", dropout=0.0):
    schema = rf.Schema.from_tree(
        fields={name: rf.Number for name in fields or ("value",)},
        d_model=20,
        n_layers=2,
        n_heads=4,
        attention=attention,
        dropout=dropout,
        reduction=None,
    )
    return BranchEncoder(schema, "/").double()


def parcel(values, present, name="value"):
    return Parcel(
        payload=values,
        present=present,
        origin=f"/{name}",
        destination="/",
        batch_size=values.shape[0],
    )


def presence():
    return torch.tensor(
        [
            [True, False, True, False, False, True, False],
            [False, False, False, False, False, False, False],
            [False, True, False, True, True, False, False],
            [True, False, False, False, False, False, True],
        ]
    )


class Projections(TorchDispatchMode):
    """Observe actual matrix operands without changing module callback behavior."""

    def __init__(self, encoder):
        super().__init__()
        self.weights = {
            module.weight.untyped_storage().data_ptr(): name
            for name, module in encoder.named_modules()
            if isinstance(module, torch.nn.Linear)
        }
        self.observed = {}

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function == torch.ops.aten.addmm.default:
            name = self.weights.get(args[2].untyped_storage().data_ptr())
            if name is not None:
                self.observed.setdefault(name, []).append(tuple(args[1].shape))
        return function(*args, **(kwargs or {}))


def test_packing_precedes_every_projection_and_feedforward():
    torch.manual_seed(31)
    encoder = branch()
    present = presence()
    values = torch.randn(4, 7, 20, dtype=torch.float64).masked_fill(~present[..., None], torch.nan)
    original = values.clone()
    parameters = {name: id(value) for name, value in encoder.named_parameters()}
    state = deepcopy(encoder.state_dict())
    observer = Projections(encoder)
    with observer:
        output = encoder([parcel(values, present)])

    assert observer.observed.keys() == set(observer.weights.values())
    assert all(len(shapes) == 1 for shapes in observer.observed.values())
    assert all(shape[0] == int(present.sum()) for shapes in observer.observed.values() for shape in shapes)
    assert {name: id(value) for name, value in encoder.named_parameters()} == parameters
    assert encoder.state_dict().keys() == state.keys()
    for name, value in encoder.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    torch.testing.assert_close(values, original, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(output.present, present)
    assert not output.payload[~present].any()


@pytest.mark.parametrize("scope", ["layer", "attention", "projection", "normalization"])
def test_module_hooks_keep_dense_shapes_and_can_replace_outputs(scope):
    torch.manual_seed(33)
    encoder = branch()
    reference = deepcopy(encoder)
    present = presence()
    values = torch.randn(4, 7, 20, dtype=torch.float64)
    layer = encoder.encoder[0]
    module = {
        "layer": layer,
        "attention": layer.attention,
        "projection": layer.attention.v_proj,
        "normalization": layer.ffn_norm,
    }[scope]
    observed = []

    def observe(owner, inputs, output):
        observed.append(tuple(inputs[0].shape))

    handle = module.register_forward_hook(observe)
    try:
        actual = encoder([parcel(values, present)]).payload
    finally:
        handle.remove()
    assert observed == [(3, 7, 20)]
    expected = torch.zeros_like(values)
    active = present.any(dim=1)
    expected[active] = reference.compute(values[active], present[active])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def erase(owner, inputs, output):
        return torch.zeros_like(output)

    handle = module.register_forward_hook(erase)
    try:
        changed = encoder([parcel(values, present)]).payload
    finally:
        handle.remove()
    assert not torch.allclose(changed, actual)


@pytest.mark.parametrize("scope", ["stack", "coordinate"])
def test_custom_layer_forward_keeps_original_arity_and_dense_shapes(scope):
    encoder = branch("left", "right") if scope == "coordinate" else branch()
    layer = encoder.coordinate_encoder if scope == "coordinate" else encoder.encoder[0]
    original = layer.forward
    observed = []

    def forward(inputs, present):
        observed.append(tuple(inputs.shape))
        return original(inputs, present)

    layer.forward = forward
    if scope == "coordinate":
        values = [torch.randn(3, 20, dtype=torch.float64) for _ in range(2)]
        present = [torch.tensor([True, False, True]), torch.tensor([False, False, True])]
        output = encoder.contextualize(
            [parcel(value, mask, name) for value, mask, name in zip(values, present, ["left", "right"], strict=True)]
        )
        assert observed == [(3, 2, 20)]
        assert all(torch.isfinite(value.payload).all() for value in output)
    else:
        encoder([parcel(torch.randn(4, 7, 20, dtype=torch.float64), presence())])
        assert observed == [(3, 7, 20)]


def test_replaced_module_subclass_keeps_dense_input_contract():
    class Linear(torch.nn.Linear):
        def forward(self, inputs):
            assert inputs.ndim == 3
            return super().forward(inputs)

    encoder = branch()
    replacement = Linear(20, 20).double()
    replacement.load_state_dict(encoder.encoder[0].attention.q_proj.state_dict())
    encoder.encoder[0].attention.q_proj = replacement
    output = encoder([parcel(torch.randn(4, 7, 20, dtype=torch.float64), presence())])
    assert torch.isfinite(output.payload).all()


@pytest.mark.parametrize("scope", ["stack", "coordinate"])
def test_custom_layer_needs_only_original_forward_protocol(scope):
    class Layer(torch.nn.Module):
        def forward(self, inputs, present):
            assert inputs.ndim == 3
            return (inputs * 2).masked_fill(~present[..., None], 0)

    encoder = branch("left", "right") if scope == "coordinate" else branch()
    if scope == "coordinate":
        encoder.coordinate_encoder = Layer()
        values = torch.randn(3, 20, dtype=torch.float64)
        present = torch.tensor([True, False, True])
        result = encoder.contextualize([parcel(values, present, name) for name in ["left", "right"]])
        for value in result:
            torch.testing.assert_close(value.payload, (values * 2).masked_fill(~present[..., None], 0))
    else:
        encoder.encoder = torch.nn.ModuleList([Layer()])
        values = torch.randn(4, 7, 20, dtype=torch.float64)
        present = presence()
        result = encoder([parcel(values, present)])
        torch.testing.assert_close(result.payload, (values * 2).masked_fill(~present[..., None], 0))


@pytest.mark.parametrize("attention", ["mha", "gqa", "mqa"])
def test_packed_training_preserves_original_rotary_holes_and_independent_rows(attention):
    torch.manual_seed(37)
    encoder = branch(attention=attention)
    reference = deepcopy(encoder)
    present = presence()
    values = torch.randn(4, 7, 20, dtype=torch.float64).masked_fill(~present[..., None], torch.nan)
    inputs = values.clone().requires_grad_()
    dense_inputs = values.clone().requires_grad_()
    weights = torch.randn_like(values)

    actual = encoder([parcel(inputs, present)]).payload
    expected = reference.compute(dense_inputs, present)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(inputs.grad, dense_inputs.grad, rtol=1e-10, atol=1e-10)
    for (name, parameter), (other, original) in zip(
        encoder.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == other
        assert parameter.grad is not None and original.grad is not None
        torch.testing.assert_close(parameter.grad, original.grad, rtol=1e-10, atol=1e-10)
    assert not inputs.grad[~present].any()

    changed = values.clone()
    changed[0] += 3
    isolated = encoder([parcel(changed, present)]).payload
    torch.testing.assert_close(isolated[1:], actual[1:], rtol=0, atol=0)


def test_empty_branch_routing_keeps_parameter_anchors_and_unused_payload():
    encoder = branch()
    values = torch.full((2, 3, 20), torch.nan, dtype=torch.float64, requires_grad=True)
    present = torch.zeros(2, 3, dtype=torch.bool)
    output = encoder([parcel(values, present)])
    output.payload.sum().backward()
    assert values.grad is None
    assert not output.payload.any()
    assert not output.present.any()
    for parameter in encoder.parameters():
        assert parameter.grad is not None
        assert not parameter.grad.any()


def test_zero_length_compute_keeps_attention_norm_gradients_unallocated():
    encoder = branch()
    values = torch.empty(2, 0, 20, dtype=torch.float64, requires_grad=True)
    encoder.compute(values, torch.empty(2, 0, dtype=torch.bool)).sum().backward()
    assert values.grad is not None and values.grad.shape == values.shape
    for name, parameter in encoder.named_parameters():
        if ".attention_norm." in name:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            assert not parameter.grad.any()


def test_empty_coordinate_routing_keeps_zero_payload_gradients():
    encoder = branch("left", "right")
    values = [torch.full((2, 20), torch.nan, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    present = torch.zeros(2, dtype=torch.bool)
    outputs = encoder.contextualize([parcel(value, present, name) for value, name in zip(values, ["left", "right"])])
    sum(output.payload.sum() for output in outputs).backward()
    for value in values:
        assert value.grad is not None
        assert not value.grad.any()
    for parameter in encoder.coordinate_encoder.parameters():
        assert parameter.grad is not None
        assert not parameter.grad.any()
    assert all(parameter.grad is None for parameter in encoder.encoder.parameters())


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_original_dense_compute_remains_fullgraph_with_changing_presence(dropout):
    torch.manual_seed(41)
    eager = branch(dropout=dropout)
    compiled = deepcopy(eager)
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    compiled.compute = torch.compile(compiled.compute, backend=backend, fullgraph=True)
    values = torch.randn(4, 7, 20, dtype=torch.float64)
    masks = [presence(), torch.ones(4, 7, dtype=torch.bool), torch.zeros(4, 7, dtype=torch.bool)]
    runs = []
    for encoder in [eager, compiled]:
        torch.manual_seed(43)
        records = []
        for present in masks:
            encoder.zero_grad(set_to_none=True)
            inputs = values.masked_fill(~present[..., None], torch.nan).requires_grad_()
            output = encoder.compute(inputs, present)
            output.square().sum().backward()
            records.append(
                (
                    output.detach().clone(),
                    inputs.grad.clone(),
                    [parameter.grad.clone() for parameter in encoder.parameters()],
                    torch.get_rng_state(),
                )
            )
        runs.append(records)
    assert len(graphs) == 1
    for actual, expected in zip(runs[1], runs[0], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("training,dropout", [(True, 0.2), (True, 0.0), (False, 0.2)])
def test_nested_dropout_keeps_dense_masks_and_rng_only_when_active(training, dropout):
    torch.manual_seed(47)
    encoder = branch()
    encoder.encoder[0].ffn[2] = torch.nn.Sequential(torch.nn.Dropout(dropout))
    encoder.train(training)
    reference = deepcopy(encoder)
    present = presence()
    values = torch.randn(4, 7, 20, dtype=torch.float64)
    active = present.any(dim=1)
    observer = Projections(encoder)

    torch.manual_seed(53)
    with observer:
        actual = encoder([parcel(values, present)]).payload
    actual_rng = torch.get_rng_state()
    torch.manual_seed(53)
    expected = torch.zeros_like(values)
    expected[active] = reference.compute(values[active], present[active])
    expected_rng = torch.get_rng_state()

    stochastic = training and dropout > 0
    tolerance = 0 if stochastic else 1e-10
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert torch.equal(actual_rng, expected_rng)
    expected_tokens = int(active.sum()) * values.shape[1] if stochastic else int(present.sum())
    assert observer.observed.keys() == set(observer.weights.values())
    assert all(shape[0] == expected_tokens for shapes in observer.observed.values() for shape in shapes)


def test_sequence_layernorm_keeps_dense_normalization_axes():
    torch.manual_seed(59)
    encoder = branch()
    encoder.encoder[0].attention_norm = torch.nn.LayerNorm((7, 20)).double()
    reference = deepcopy(encoder)
    present = presence()
    values = torch.randn(4, 7, 20, dtype=torch.float64)
    active = present.any(dim=1)

    actual = encoder([parcel(values, present)]).payload
    expected = torch.zeros_like(values)
    expected[active] = reference.compute(values[active], present[active])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
