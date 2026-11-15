"""Zero-valued terms that keep every parameter of a module in the autograd graph."""

import torch

__all__ = ["anchor"]


def anchor(module: torch.nn.Module) -> torch.Tensor | int:
    """Return a zero scalar that depends on every parameter of ``module``.

    A batch with no live token still has to backpropagate through the same parameters
    on every distributed rank, so its empty result adds this term.

    Args:
        module: Module whose parameters must receive a gradient.

    Returns:
        A zero scalar tensor on the device of the parameters, or ``0`` for a module without parameters.
    """
    return sum(parameter.sum() * 0.0 for parameter in module.parameters())
