import copy

import pytest
import torch

import relflow as rf
from relflow.architecture.encoder import BranchEncoder
from relflow.structs.packages import Parcel


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_branch_compute_compiles_without_breaks_and_preserves_compacted_training(dropout):
    schema = rf.Schema.from_tree(value=rf.Number(), d_model=16, n_layers=2, n_heads=4, dropout=dropout, reduction=None)
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
