"""Direction rules that decide which steps of a branch sequence one token may see."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Protocol

import torch

from relflow.structs.enums import Direction


class Rule(Protocol):
    """Turn step indices into an attention mask, fixed once per branch at build time."""

    def __call__(self, *, steps: torch.Tensor | None) -> torch.Tensor | None:
        """Return ``None`` for padding only, else a Boolean ``(rows, L, L)`` visibility mask.

        Args:
            steps: Row index of every token within its branch instance, shape ``(rows, L)``.
        """
        ...


def bidirectional(*, steps: torch.Tensor | None) -> None:
    """Let every token see every present token, as the standard encoder does.

    Args:
        steps: Ignored; visibility depends on padding only.

    Returns:
        ``None``, meaning padding only.
    """
    return None


rules: Mapping[Direction, Rule] = MappingProxyType({Direction.bidirectional: bidirectional})

__all__ = ["Rule", "bidirectional", "rules"]
