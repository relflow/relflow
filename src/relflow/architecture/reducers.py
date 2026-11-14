"""Reducers that summarize a branch sequence, picked once from a table by reduction setting."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

import torch

from relflow.architecture.pool import LearnedQueryCrossAttention, MeanPool
from relflow.structs.reduction import Attention, Mean

if TYPE_CHECKING:
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch


class Reducer(Protocol):
    """Reduce a mixed branch sequence to ``n_context`` tokens per row."""

    @property
    def n_context(self) -> int:
        """Number of output tokens per row."""
        ...

    def observe(self, present: torch.Tensor) -> None:
        """Record the present token counts of a training batch."""
        ...

    def __call__(
        self,
        memory: torch.Tensor,
        *,
        present: torch.Tensor,
        context: torch.Tensor | None,
        evidence: torch.Tensor | None,
        memory_positions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return reduced tokens of shape ``(rows, n_context, d_model)``."""
        ...


class ReducerBuilder[S](Protocol):
    """Build the reducer for one branch from its reduction setting."""

    def __call__(
        self, reduction: S, *, branch: Branch, schema: Schema, capacity: int | None, dropout: float
    ) -> Reducer: ...


def mean(reduction: Mean, *, branch: Branch, schema: Schema, capacity: int | None, dropout: float) -> MeanPool:
    """Build the mean reducer.

    Args:
        reduction: The mean setting.
        branch: Branch being reduced.
        schema: Schema that owns the branch.
        capacity: Static visible width of the branch, or ``None`` when unbounded.
        dropout: Dropout inherited from the branch.

    Returns:
        A mean pool with presence-mass lanes.
    """
    return MeanPool(n_context=1, d_model=schema.d_model, mass_capacity=capacity)


def cross(
    reduction: Attention, *, branch: Branch, schema: Schema, capacity: int | None, dropout: float
) -> LearnedQueryCrossAttention:
    """Build the learned-query cross-attention reducer.

    Args:
        reduction: The attention setting.
        branch: Branch being reduced.
        schema: Schema that owns the branch.
        capacity: Static visible width of the branch, or ``None`` when unbounded.
        dropout: Dropout inherited from the branch unless the setting overrides it.

    Returns:
        A cross-attention pool with presence-mass lanes.
    """
    return LearnedQueryCrossAttention(
        n_context=reduction.n_outputs,
        d_model=schema.d_model,
        nhead=reduction.n_heads or branch.n_heads,
        dropout=float(reduction.dropout if reduction.dropout is not None else dropout),
        n_linear=reduction.n_layers,
        position=reduction.position,
        mass_capacity=capacity,
        mass=True,
        ffnx=branch.ffnx,
    )


reducers: Mapping[type, ReducerBuilder[Any]] = MappingProxyType({Mean: mean, Attention: cross})

__all__ = ["Reducer", "ReducerBuilder", "cross", "mean", "reducers"]
