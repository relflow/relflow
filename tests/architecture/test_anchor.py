import torch

from relflow.architecture.anchor import anchor


def test_anchor_is_zero_and_gives_every_parameter_a_zero_gradient():
    module = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    value = anchor(module)
    assert isinstance(value, torch.Tensor) and value.item() == 0.0
    value.backward()
    assert all(parameter.grad is not None and not parameter.grad.any() for parameter in module.parameters())
