import math
from typing import TYPE_CHECKING, cast

import torch

from relflow.architecture.attention import RotaryMultiheadAttention
from relflow.architecture.blocks import feed_forward, norm
from relflow.distributed import all_reduce_max


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


class Pool(torch.nn.Module):
    """Own the count and evidence lanes that restore multiplicity after pooling.

    A bounded branch divides its lanes by a static visible capacity. An
    unbounded branch has no cap, so its lanes divide by the largest present
    count seen while training: a persisted buffer that every rank raises
    together, keeping DDP replicas identical. Without ``mass`` a pool has no
    lanes.
    """

    def __init__(self, *, mass: bool, mass_capacity: int | None) -> None:
        super().__init__()
        if mass_capacity is not None and not mass:
            raise ValueError(
                f"{type(self).__name__} mass_capacity={mass_capacity} requires its lanes; "
                "construct MeanPool with d_model or LearnedQueryCrossAttention with mass=True"
            )
        if mass_capacity is not None and mass_capacity < 1:
            raise ValueError(f"{type(self).__name__} mass_capacity must be >= 1, got {mass_capacity}")
        self.mass_capacity = mass_capacity
        self.mass_observed: torch.Tensor | None
        self.register_buffer(
            "mass_observed", torch.ones((), dtype=torch.int64) if mass and mass_capacity is None else None
        )

    def observe(self, present: torch.Tensor) -> None:
        """Raise the observed capacity to this training batch's largest present count.

        Every rank calls this on every training step, so the reduction is a
        uniform collective.
        """
        if self.mass_observed is None or not self.training:
            return
        counts = present.sum(dim=1)
        largest = counts.max() if counts.numel() else counts.new_zeros(())
        self.mass_observed.copy_(torch.maximum(self.mass_observed, all_reduce_max(largest.to(self.mass_observed))))

    def divisors(self, dtype: torch.dtype) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """Return the evidence and count divisors for the current capacity."""
        if self.mass_capacity is not None:
            return math.sqrt(self.mass_capacity), float(self.mass_capacity)
        observed = cast(torch.Tensor, self.mass_observed).to(dtype)
        return observed.sqrt(), observed


class LearnedQueryCrossAttention(Pool):
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
        mass: bool = False,
        ffnx: int = 8,
    ) -> None:
        super().__init__(mass=mass, mass_capacity=mass_capacity)

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
        self.mass_projection = torch.nn.Linear(d_model, n_context * d_model, bias=False) if mass else None
        self.mass_direction = torch.nn.Parameter(torch.zeros(n_context, d_model)) if mass else None
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

        # Geometry reaches compute whenever a caller supplies it; pools without it keep the plain protocol.
        geometry = {
            name: value
            for name, value in (("queries", queries), ("positions", positions), ("memory_positions", memory_positions))
            if value is not None
        }
        queries = self.compute(memory, present, context, evidence, **geometry)
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
            root, capacity = self.divisors(result.dtype)
            mass = self.mass_projection(additive.sum(dim=1) / root).reshape_as(result)
            count = present.sum(dim=1).to(dtype=result.dtype) / capacity
            result = result + mass + count[:, None, None] * self.mass_direction.unsqueeze(0)

        return result

    if TYPE_CHECKING:
        __call__ = forward


class MeanPool(Pool):
    """Average present memory coordinates, keep their count, and repeat the result across queries.

    A mean erases multiplicity: one item and six equal items pool identically.
    With ``d_model`` the pool adds a learned direction scaled by the present
    count over its capacity, starting at zero so an untrained pool is still the
    plain mean. ``mass_capacity`` fixes that capacity; omitting it tracks the
    largest count observed while training.
    """

    def __init__(self, n_context: int, d_model: int | None = None, mass_capacity: int | None = None) -> None:
        super().__init__(mass=d_model is not None, mass_capacity=mass_capacity)
        self.n_context = n_context
        self.mass_direction = torch.nn.Parameter(torch.zeros(d_model)) if d_model is not None else None

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
        if self.mass_direction is not None:
            count = present.sum(dim=1).to(dtype=pooled.dtype) / self.divisors(pooled.dtype)[1]
            pooled = pooled + count[:, None, None] * self.mass_direction.to(dtype=pooled.dtype)
        return pooled.expand(-1, self.n_context, -1)

    if TYPE_CHECKING:
        __call__ = forward
