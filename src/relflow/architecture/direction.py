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


def causal(*, steps: torch.Tensor | None) -> torch.Tensor:
    """Let a token see the tokens whose step is at or before its own.

    Tokens are numbered item-major, so token order says nothing about steps and the
    rule reads the step ids.

    Args:
        steps: Row index of every token within its branch instance, shape ``(rows, L)``.

    Returns:
        A Boolean ``(rows, L, L)`` mask where ``[r, query, key]`` is true when the key
        step is not after the query step.

    Raises:
        ValueError: If ``steps`` is ``None``.
    """
    if steps is None:
        message = "direction 'causal' needs the step ids of every token, but the encoder passed none"
        raise ValueError(message)
    return steps.unsqueeze(-2) <= steps.unsqueeze(-1)


rules: Mapping[Direction, Rule] = MappingProxyType({Direction.bidirectional: bidirectional, Direction.causal: causal})

# Whether a rule reads step ids; the encoder computes them only when one does.
stepped: Mapping[Direction, bool] = MappingProxyType({Direction.bidirectional: False, Direction.causal: True})

# Whether a rule on one token per row is the lower-triangular mask that SDPA builds with ``is_causal``.
triangular: Mapping[Direction, bool] = MappingProxyType({Direction.bidirectional: False, Direction.causal: True})

__all__ = ["Rule", "bidirectional", "causal", "rules", "stepped", "triangular"]
