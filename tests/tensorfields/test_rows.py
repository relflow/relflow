"""Checkpoint shape restoration retains existing optimizer parameter references."""

from copy import deepcopy

import pytest
import torch

from relflow.tensorfields.shared.rows import Embedding, Linear


@pytest.mark.parametrize("kind", ["embedding", "linear", "linear_without_bias"])
@pytest.mark.parametrize("before,after", [(1, 4), (4, 1), (4, 4)])
@pytest.mark.parametrize("algorithm", [torch.optim.AdamW, torch.optim.SGD])
def test_checkpoint_restores_row_shape_in_place_then_optimizer_training_continues(kind, before, after, algorithm):
    def build(rows):
        return Embedding(rows, 3) if kind == "embedding" else Linear(3, rows, bias=kind == "linear")

    def step(layer, optimizer):
        optimizer.zero_grad(set_to_none=True)
        value = layer(torch.tensor([0, 0])) if kind == "embedding" else layer(torch.ones(2, 3))
        value.square().mean().backward()
        optimizer.step()

    restored, reference = build(before), build(after)
    options = {"momentum": 0.9} if algorithm is torch.optim.SGD else {"amsgrad": True}
    optimizer = algorithm(restored.parameters(), lr=0.01, **options)
    expected_optimizer = algorithm(reference.parameters(), lr=0.01, **options)
    step(restored, optimizer)
    step(reference, expected_optimizer)
    identities = [id(parameter) for parameter in restored.parameters()]
    groups = optimizer.param_groups
    parameters = groups[0]["params"]

    restored.load_state_dict(deepcopy(reference.state_dict()))
    optimizer.load_state_dict(deepcopy(expected_optimizer.state_dict()))

    assert [id(parameter) for parameter in restored.parameters()] == identities
    assert [id(parameter) for parameter in optimizer.param_groups[0]["params"]] == identities
    # Existing outside references still point to the restored objects.
    assert [id(parameter) for parameter in parameters] == identities
    if before != after:
        assert all(parameter.grad is None for parameter in restored.parameters())
    assert (restored.num_embeddings if kind == "embedding" else restored.out_features) == after

    step(restored, optimizer)
    step(reference, expected_optimizer)

    for parameter, expected in zip(restored.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        torch.testing.assert_close(parameter.grad, expected.grad, rtol=0, atol=0)
        for key, value in optimizer.state[parameter].items():
            torch.testing.assert_close(value, expected_optimizer.state[expected][key], rtol=0, atol=0)
