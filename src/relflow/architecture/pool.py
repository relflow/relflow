import math
from typing import TYPE_CHECKING, cast

import torch

from relflow.architecture.attention import RotaryMultiheadAttention
from relflow.architecture.blocks import feed_forward, norm


class CrossAttentionBlock(torch.nn.Module):
    """Update query embeddings against present memory coordinates."""

    def __init__(self, d_model: int, nhead: int, dropout: float, position: bool = True, *, ffnx: int = 8) -> None:
        super().__init__()

        self.attention_norm = norm(d_model)
        self.ffn_norm = norm(d_model)
        self.attention = RotaryMultiheadAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            position=position,
        )

        self.ffn = feed_forward(d_model, dropout, ffnx)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        present: torch.Tensor | None = None,
        *,
        positions: torch.Tensor | None = None,
        memory_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        padding = None if present is None else ~present
        attended = self.attention(
            self.attention_norm(queries),
            memory,
            memory,
            key_padding_mask=padding,
            query_positions=positions,
            key_positions=memory_positions,
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))

    if TYPE_CHECKING:
        __call__ = forward


class LearnedQueryCrossAttention(torch.nn.Module):
    """Pool nonempty memory rows into learned queries and optional additive evidence."""

    def __init__(
        self,
        n_context: int,
        d_model: int,
        nhead: int,
        dropout: float,
        n_linear: int = 1,
        position: bool = True,
        mass_capacity: int | None = None,
        *,
        ffnx: int = 8,
    ) -> None:
        super().__init__()

        if mass_capacity is not None and mass_capacity < 1:
            raise ValueError("mass_capacity must be >= 1")

        self.queries = torch.nn.Parameter(torch.normal(mean=0.0, std=1e-2, size=(n_context, d_model)))
        self.blocks = torch.nn.ModuleList()
        for _ in range(n_linear):
            self.blocks.append(
                CrossAttentionBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    position=position,
                    ffnx=ffnx,
                )
            )
        self.norm = norm(d_model)
        self.mass_capacity = mass_capacity
        self.mass_projection = (
            torch.nn.Linear(d_model, n_context * d_model, bias=False) if mass_capacity is not None else None
        )
        self.mass_direction = torch.nn.Parameter(torch.zeros(n_context, d_model)) if mass_capacity is not None else None
        if self.mass_projection is not None:
            torch.nn.init.zeros_(self.mass_projection.weight)

    def forward(
        self,
        memory: torch.Tensor,
        present: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        evidence: torch.Tensor | None = None,
        *,
        queries: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        memory_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from relflow.architecture.compiler import canonical

        N, _, _ = memory.shape
        if present is None:
            present = torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
        if present.dtype != torch.bool or tuple(present.shape) != tuple(memory.shape[:2]):
            raise ValueError(
                f"pool presence must have bool shape {tuple(memory.shape[:2])}, "
                f"got {tuple(present.shape)} with dtype {present.dtype}"
            )
        if evidence is not None and tuple(evidence.shape) != tuple(memory.shape):
            raise ValueError(
                f"pool additive evidence must have shape {tuple(memory.shape)}, got {tuple(evidence.shape)}"
            )

        active = present.any(dim=1)
        indices = active.nonzero(as_tuple=False).reshape(-1)
        shape = (N, self.queries.shape[0], self.queries.shape[1]) if queries is None else tuple(queries.shape)
        dtype, device = memory.dtype, memory.device
        if not indices.numel():
            pooled = memory.new_zeros(shape)
            if torch.is_grad_enabled():
                for parameter in self.parameters():
                    pooled = pooled + parameter.sum() * 0.0
            return pooled

        complete = indices.numel() == N
        memory = memory.contiguous() if complete else memory.index_select(0, indices)
        present = present.index_select(0, indices)
        memory = memory.masked_fill(~present.unsqueeze(-1), 0.0)
        if evidence is not None:
            evidence = evidence.contiguous() if complete else evidence.index_select(0, indices)
            evidence = evidence.masked_fill(~present.unsqueeze(-1), 0.0)
        if context is not None and tuple(context.shape) != shape:
            raise ValueError(f"pool query context must have shape {shape}, got {tuple(context.shape)}")
        if context is not None:
            context = context.index_select(0, indices)
        if queries is not None:
            queries = queries.index_select(0, indices)
        if positions is not None and positions.ndim > 1:
            positions = positions.index_select(0, indices)
        if memory_positions is not None and memory_positions.ndim > 1:
            memory_positions = memory_positions.index_select(0, indices)

        original = type(self).compute is LearnedQueryCrossAttention.compute and canonical(self, self.compute)
        if not original or (queries is None and positions is None and memory_positions is None):
            queries = self.compute(memory, present, context, evidence)
        else:
            queries = self.compute(
                memory,
                present,
                context,
                evidence,
                queries=queries,
                positions=positions,
                memory_positions=memory_positions,
            )
        expected = (indices.numel(), *shape[1:])
        if tuple(queries.shape) != expected:
            raise RuntimeError(
                f"custom pool compute must return query shape {expected}, got {tuple(queries.shape)}; "
                "use the base compute implementation with module hooks, or adapt the pool forward method "
                "to the runtime query geometry"
            )
        if complete and queries.shape == shape and queries.dtype == dtype and queries.device == device:
            return queries.clone(memory_format=torch.contiguous_format)
        return memory.new_zeros(shape, dtype=dtype, device=device).index_copy(0, indices, queries)

    def compute(
        self,
        memory: torch.Tensor,
        present: torch.Tensor,
        context: torch.Tensor | None,
        evidence: torch.Tensor | None,
        *,
        queries: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        memory_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool compacted nonempty rows with absent memory and evidence already zeroed."""
        if queries is None:
            queries = self.queries.unsqueeze(0).expand(memory.shape[0], -1, -1)

        if not torch.is_grad_enabled():
            queries = queries.detach()
            memory = memory.detach()
            if evidence is not None:
                evidence = evidence.detach()
            if context is not None:
                context = context.detach()

        if context is not None:
            queries = queries + context

        for block in self.blocks:
            queries = block(
                queries=queries,
                memory=memory,
                present=present,
                positions=positions,
                memory_positions=memory_positions,
            )

        result = cast(torch.Tensor, self.norm(queries))
        if self.mass_projection is not None and self.mass_direction is not None:
            additive = memory if evidence is None else evidence
            assert self.mass_capacity is not None
            total = additive.sum(dim=1) / math.sqrt(self.mass_capacity)
            mass = self.mass_projection(total).reshape_as(result)
            count = present.sum(dim=1).to(dtype=result.dtype) / self.mass_capacity
            result = result + mass + count[:, None, None] * self.mass_direction.unsqueeze(0)

        return result

    if TYPE_CHECKING:
        __call__ = forward


class MeanPool(torch.nn.Module):
    """Average present memory coordinates and repeat the result across queries."""

    def __init__(self, n_context: int) -> None:
        super().__init__()
        self.n_context = n_context

    def forward(self, memory: torch.Tensor, present: torch.Tensor | None = None) -> torch.Tensor:
        if present is None:
            present = torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
        if present.dtype != torch.bool or tuple(present.shape) != tuple(memory.shape[:2]):
            raise ValueError(
                f"pool presence must have bool shape {tuple(memory.shape[:2])}, "
                f"got {tuple(present.shape)} with dtype {present.dtype}"
            )

        weights = present.unsqueeze(-1).to(dtype=memory.dtype)
        pooled = memory.masked_fill(~present.unsqueeze(-1), 0.0).sum(dim=1, keepdim=True)
        pooled = pooled / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return pooled.expand(-1, self.n_context, -1)

    if TYPE_CHECKING:
        __call__ = forward
