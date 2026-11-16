"""Extension-owned state compatibility when reconstructing runtime resources."""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import torch

__all__ = ["Rebuild", "compatible"]


@runtime_checkable
class Rebuild(Protocol):
    """Prepare a fresh resource and select state for its entire module subtree.

    The resource owns the distinction between its construction configuration and
    learned runtime shape. It may resize itself before returning loadable state,
    but must not change the previous resource. Checkpoint loading is independent.
    """

    def rebuild_state(self, previous: torch.nn.Module) -> Mapping[str, Any]: ...


def compatible(current: Mapping[str, Any], previous: Mapping[str, Any]) -> dict[str, Any]:
    """Retain named entries with matching types and tensor shapes."""
    retained = {}
    for name, value in previous.items():
        if name not in current:
            continue
        target = current[name]
        if isinstance(target, torch.Tensor) and isinstance(value, torch.Tensor):
            if target.shape != value.shape:
                continue
        elif type(target) is not type(value):
            continue
        retained[name] = value
    return retained
