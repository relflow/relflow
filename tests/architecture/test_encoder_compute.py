import copy

import pytest
import torch

import relflow as rf
from relflow.architecture.encoder import BranchEncoder
from relflow.structs.packages import Parcel


@pytest.mark.integration
@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_branch_compute_compiles_without_breaks_and_preserves_compacted_training(dropout):
    schema = rf.Schema.from_tree(value=rf.Number, d_model=16, n_layers=2, n_heads=4, dropout=dropout, reduction=None)
    eager = BranchEncoder(schema, "/")
    compiled = copy.deepcopy(eager)
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    compiled.compute = torch.compile(compiled.compute, backend=backend, fullgraph=True)
    outputs, gradients, random_states = [], [], []
    for encoder in (eager, compiled):
        torch.manual_seed(1729)
        payload = torch.randn(3, 2, 16, requires_grad=True)
        present = torch.tensor([[True, False], [False, False], [True, True]])
        parcel = Parcel(
            payload=payload.masked_fill(~present.unsqueeze(-1), torch.nan),
            present=present,
            origin="/value",
            destination="/",
            batch_size=3,
        )
        retained = []
        for _ in range(2):
            output = encoder([parcel])
            retained.append(output.payload.detach().clone())
            output.payload.square().mean().backward(retain_graph=True)
        outputs.append(retained)
        gradients.append([parameter.grad for parameter in encoder.parameters()] + [payload.grad])
        random_states.append(torch.get_rng_state())

    assert len(graphs) == 1
    for actual, expected in zip(outputs[1], outputs[0], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.isfinite(actual).all()
        assert not actual[1].any()
    for actual, expected in zip(gradients[1], gradients[0], strict=True):
        assert actual is not None and expected is not None
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(random_states[0], random_states[1])


def structured(schema, present):
    """Two leaf parcels whose structural positions leave capacity gaps between fields."""
    rows, width = present.shape
    return [
        Parcel(
            payload=torch.randn(rows, width, schema.d_model, dtype=torch.float64).masked_fill(
                ~present.unsqueeze(-1), torch.nan
            ),
            present=present,
            origin=f"/{name}",
            destination="/",
            positions=torch.arange(width).expand(rows, width),
            spans=torch.full((rows,), width + 2),
            slots=present,
            batch_size=rows,
        )
        for name in ("value", "other")
    ]


class Traced(BranchEncoder):
    def __init__(self, schema, address):
        super().__init__(schema, address)
        self.positioned = []

    def compute(self, inputs, present, *, packing=None, positions=None):
        self.positioned.append(packing is not None or positions is not None)
        return super().compute(inputs, present, packing=packing, positions=positions)


def test_subclass_compute_receives_structural_positions_on_packed_and_dense_paths():
    torch.manual_seed(3)
    schema = rf.Schema.from_tree(
        fields={"value": rf.Number, "other": rf.Number}, d_model=20, n_layers=1, n_heads=4, reduction=None
    )
    reference = BranchEncoder(schema, "/").double().eval()
    traced = Traced(schema, "/").double().eval()
    traced.load_state_dict(reference.state_dict())
    present = torch.tensor([[True, True, False, False], [False, False, False, False], [True, False, True, True]])
    parcels = structured(schema, present)

    expected = reference(parcels).payload
    packed = traced(parcels).payload
    handle = traced.encoder[0].register_forward_hook(lambda module, inputs, output: None)
    dense = traced(parcels).payload
    handle.remove()

    assert traced.positioned == [True, True]
    torch.testing.assert_close(packed, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(dense, expected, rtol=1e-10, atol=1e-10)


def test_custom_layers_and_pools_receive_structural_positions():
    from relflow.architecture.encoder import RotaryTransformerEncoderLayer
    from relflow.architecture.pool import LearnedQueryCrossAttention

    class Layer(RotaryTransformerEncoderLayer):
        def forward(self, inputs, present, *, packing=None, positions=None):
            self.positioned = positions is not None
            return super().forward(inputs, present, packing=packing, positions=positions)

    class Pool(LearnedQueryCrossAttention):
        def compute(self, memory, present, context, evidence, *, queries=None, positions=None, memory_positions=None):
            self.memory_positions = memory_positions
            return super().compute(
                memory,
                present,
                context,
                evidence,
                queries=queries,
                positions=positions,
                memory_positions=memory_positions,
            )

    torch.manual_seed(4)
    schema = rf.Schema.from_tree(
        fields={"value": rf.Number, "other": rf.Number}, d_model=20, n_layers=1, n_heads=4, reduction=rf.Attention()
    )
    reference = BranchEncoder(schema, "/").double().eval()
    present = torch.tensor([[True, True, False, False], [True, False, True, True]])
    parcels = structured(schema, present)
    expected = reference(parcels).payload

    custom = copy.deepcopy(reference)
    layer = Layer(d_model=20, nhead=4, n_kv_heads=4, dropout=0.0).double()
    layer.load_state_dict(reference.encoder[0].state_dict())
    custom.encoder[0] = layer
    pool = Pool(1, 20, 4, 0.0, mass_capacity=reference.pool.mass_capacity, mass=True).double()
    pool.load_state_dict(reference.pool.state_dict())
    custom.pool = pool

    actual = custom(parcels).payload

    assert layer.positioned
    assert pool.memory_positions is not None and pool.memory_positions.shape == (2, 8)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
