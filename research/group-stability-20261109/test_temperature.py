"""The experimental surrogate preserves hard routing and the unit baseline."""

import pytest
import torch
from soft import forward as soft_forward
from temperature import forward

from relflow.structs.enums import Tokens
from tests.tensorfields.extensions.test_group import build_embedder, build_schema, input_field


@pytest.mark.parametrize("training", [True, False])
def test_temperature_preserves_hard_route_and_unit_gradient(training):
    embedder = build_embedder(build_schema())
    embedder.train(training)
    field = input_field(
        torch.tensor([[Tokens.valued.value, Tokens.null.value, Tokens.valued.value]]),
        torch.tensor([[0, 0, -1]]),
    )
    inputs = field.take(None)
    original = embedder.forward(inputs).payload
    original_gradient = None
    if training:
        original_gradient = torch.autograd.grad(original.sum(), embedder.embeddings["group"].weight)[0]
    for temperature in (0.5, 1.0, 2.0):
        candidate = forward(embedder, inputs, temperature).payload
        torch.testing.assert_close(candidate, original)
        if training and temperature == 1.0:
            gradient = torch.autograd.grad(candidate.sum(), embedder.embeddings["group"].weight)[0]
            torch.testing.assert_close(gradient, original_gradient)


def test_soft_training_changes_only_the_training_route():
    embedder = build_embedder(build_schema())
    field = input_field(torch.tensor([[Tokens.valued.value]]), torch.tensor([[0]]))
    inputs = field.take(None)
    with torch.no_grad():
        embedder.embeddings["group"].weight[0].copy_(torch.tensor([0.2, 1.0, -0.7, 0.4]))
        embedder.embeddings["content"].weight.zero_()
        embedder.embeddings["content"].weight[0] = torch.tensor([0.0, 1.0, 3.0, 7.0])
    embedder.eval()
    torch.testing.assert_close(soft_forward(embedder, inputs).payload, embedder.forward(inputs).payload)
    embedder.train()
    soft = soft_forward(embedder, inputs).payload
    assert not torch.allclose(soft, embedder.forward(inputs).payload)
    gradient = torch.autograd.grad(soft.sum(), embedder.embeddings["group"].weight)[0]
    assert torch.isfinite(gradient).all() and gradient[0].abs().sum() > 0
