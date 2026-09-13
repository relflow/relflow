from copy import deepcopy

import pytest
import torch

from relflow.architecture.pool import LearnedQueryCrossAttention


def test_pool_compacts_and_sanitizes_inputs_before_compute(monkeypatch):
    model = LearnedQueryCrossAttention(2, 8, 2, 0.0, mass_capacity=3)
    memory = torch.randn(3, 3, 8)
    present = torch.tensor([[True, False, True], [False, False, False], [False, True, False]])
    memory[~present] = torch.nan
    evidence = memory * 2
    context = torch.randn(3, 2, 8)
    original = [value.clone() for value in (memory, evidence, context)]
    captured = []
    compute = model.compute

    def observe(memory, present, context, evidence):
        captured.append((memory.clone(), present.clone(), context.clone(), evidence.clone()))
        return compute(memory, present, context, evidence)

    monkeypatch.setattr(model, "compute", observe)
    result = model(memory, present, context, evidence)

    assert len(captured) == 1
    selected = torch.tensor([0, 2])
    clean = memory.masked_fill(~present.unsqueeze(-1), 0.0).index_select(0, selected)
    for actual, expected in zip(captured[0], (clean, present[selected], context[selected], clean * 2), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert result.shape == (3, 2, 8)
    assert torch.count_nonzero(result[1]) == 0
    for value, expected in zip((memory, evidence, context), original, strict=True):
        torch.testing.assert_close(value, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_pool_compute_captures_seeded_dropout_and_accumulated_gradients(monkeypatch, grad_enabled):
    torch.manual_seed(1729)
    eager = LearnedQueryCrossAttention(2, 8, 2, 0.2, n_linear=2, mass_capacity=4).train()
    with torch.no_grad():
        eager.mass_projection.weight.normal_(std=0.1)
        eager.mass_direction.normal_(std=0.1)
    candidate = deepcopy(eager)
    identities = {name: id(parameter) for name, parameter in candidate.named_parameters()}
    keys = list(candidate.state_dict())
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    monkeypatch.setattr(candidate, "compute", torch.compile(candidate.compute, backend=backend, fullgraph=True))
    for step in range(2):
        present = torch.tensor([[True, False, True, False], [False, False, False, False], [False, True, True, True]])
        memory = torch.randn(3, 4, 8).masked_fill(~present.unsqueeze(-1), torch.nan)
        context = torch.randn(3, 2, 8)
        records = []
        for model in (eager, candidate):
            values = [value.detach().clone().requires_grad_(grad_enabled) for value in (memory, context, memory * 2)]
            torch.manual_seed(2718 + step)
            with torch.set_grad_enabled(grad_enabled):
                output = model(values[0], present, values[1], values[2])
                if grad_enabled:
                    output.square().mean().backward()
            records.append((output.detach().clone(), [value.grad for value in values], torch.get_rng_state()))
        torch.testing.assert_close(records[1], records[0], rtol=0, atol=0, equal_nan=True)
        for (name, left), (other, right) in zip(eager.named_parameters(), candidate.named_parameters(), strict=True):
            assert name == other
            torch.testing.assert_close(right.grad, left.grad, rtol=0, atol=0)
    assert graphs, "the tensor compute boundary must actually reach the backend"
    assert identities == {name: id(parameter) for name, parameter in candidate.named_parameters()}
    assert keys == list(candidate.state_dict())


def test_empty_pool_preserves_validation_order_and_skips_compute(monkeypatch):
    model = LearnedQueryCrossAttention(2, 8, 2, 0.2, mass_capacity=3)
    memory = torch.full((2, 3, 8), torch.nan)
    present = torch.zeros(2, 3, dtype=torch.bool)

    def unexpected(*args):
        raise AssertionError("empty rows must bypass compute")

    monkeypatch.setattr(model, "compute", unexpected)
    # Context validation has always followed the all-empty return; evidence
    # geometry is validated before deciding whether there is anything to pool.
    output = model(memory, present, context=torch.empty(1))
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0 for parameter in model.parameters()
    )
    with pytest.raises(ValueError, match="additive evidence"):
        model(memory, present, evidence=torch.empty(1))
    with pytest.raises(ValueError, match="query context"):
        model(memory, torch.ones_like(present), context=torch.empty(1))
