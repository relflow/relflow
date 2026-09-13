"""Live-token encoder blocks with attention grouped by sequence length.

Packing is shared across an entire encoder stack. Equal-length grouping supports
standard SDPA without padded queries. Nonzero training dropout retains the
original dense RNG schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Sequence

    from relflow.architecture.attention import RotaryMultiheadAttention
    from relflow.architecture.encoder import RotaryTransformerEncoderLayer


@dataclass(frozen=True)
class Group:
    """Equal-length rows addressed within the packed token axis."""

    rows: int
    length: int
    indices: torch.Tensor


@dataclass(frozen=True)
class Packed:
    """Original token coordinates and attention groups shared by an encoder stack."""

    indices: torch.Tensor
    positions: torch.Tensor
    groups: tuple[Group, ...]


def stochastic(layers: Sequence[RotaryTransformerEncoderLayer]) -> bool:
    """Whether active dropout requires the original dense RNG schedule."""
    return any(
        (layer.attention.training and layer.attention.dropout_p > 0)
        or any(
            isinstance(module, torch.nn.Dropout) and module.training and module.p > 0 for module in layer.ffn.modules()
        )
        for layer in layers
    )


def customized(layers: Sequence[torch.nn.Module], *, additional: tuple[type[torch.nn.Module], ...] = ()) -> bool:
    """Retain dense module shapes for callbacks and replaced implementations.

    Packing changes nested module inputs from [batch, length, width] to
    [live_tokens, width]. Only the canonical, unobserved encoder modules opt in;
    hooks and user replacements continue to receive the original dense contract.
    """
    from relflow.architecture.attention import RotaryMultiheadAttention
    from relflow.architecture.encoder import RotaryTransformerEncoderLayer
    from relflow.architecture.rotary import RotaryEmbedding

    canonical = {
        *additional,
        RotaryTransformerEncoderLayer,
        RotaryMultiheadAttention,
        RotaryEmbedding,
        torch.nn.Linear,
        torch.nn.LayerNorm,
        torch.nn.Sequential,
        torch.nn.GELU,
        torch.nn.Dropout,
    }
    hooks = ("_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks")
    if any(getattr(torch.nn.modules.module, "_global" + name) for name in hooks):
        return True
    return any(
        type(module) not in canonical
        or (isinstance(module, torch.nn.LayerNorm) and len(module.normalized_shape) != 1)
        or "forward" in module.__dict__
        or any(getattr(module, name) for name in hooks)
        for layer in layers
        for module in layer.modules()
    )


def pack(inputs: torch.Tensor, present: torch.Tensor) -> tuple[torch.Tensor, Packed]:
    """Select live values once, retaining their original positions and row ownership."""
    if present.dtype != torch.bool or tuple(present.shape) != tuple(inputs.shape[:2]):
        raise ValueError(
            f"encoder presence must have bool shape {tuple(inputs.shape[:2])}, "
            f"got {tuple(present.shape)} with dtype {present.dtype}"
        )
    batch, capacity, width = inputs.shape
    indices = present.reshape(-1).nonzero(as_tuple=False).reshape(-1)
    positions = indices.remainder(capacity) if capacity else indices
    counts = present.sum(dim=1, dtype=torch.int32)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0, dtype=torch.int32)))
    groups = []
    # One host transfer groups rows by length for every layer in the stack.
    grouped_rows: dict[int, list[int]] = {}
    for row, length in enumerate(counts.detach().cpu().tolist()):
        if length:
            grouped_rows.setdefault(length, []).append(row)
    for length, rows in sorted(grouped_rows.items()):
        selected_rows = torch.tensor(rows, dtype=torch.int64, device=inputs.device)
        starts = offsets.index_select(0, selected_rows).to(torch.int64)
        tokens = (starts[:, None] + torch.arange(length, device=inputs.device)[None, :]).reshape(-1)
        groups.append(Group(len(rows), length, tokens))
    return inputs.reshape(batch * capacity, width).index_select(0, indices), Packed(indices, positions, tuple(groups))


def attend(
    attention: RotaryMultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    packing: Packed,
) -> torch.Tensor:
    q = attention.q_proj(query).reshape(-1, attention.nhead, attention.head_dim)
    k = attention.k_proj(key).reshape(-1, attention.n_kv_heads, attention.head_dim)
    v = attention.v_proj(value).reshape(-1, attention.n_kv_heads, attention.head_dim)
    if attention.rotary is not None:
        q = attention.rotary(q.transpose(0, 1), positions=packing.positions).transpose(0, 1)
        k = attention.rotary(k.transpose(0, 1), positions=packing.positions).transpose(0, 1)
    context = torch.zeros_like(q)
    for group in packing.groups:
        selected_q = (
            q.index_select(0, group.indices)
            .reshape(group.rows, group.length, attention.nhead, attention.head_dim)
            .transpose(1, 2)
        )
        selected_k = (
            k.index_select(0, group.indices)
            .reshape(group.rows, group.length, attention.n_kv_heads, attention.head_dim)
            .transpose(1, 2)
        )
        selected_v = (
            v.index_select(0, group.indices)
            .reshape(group.rows, group.length, attention.n_kv_heads, attention.head_dim)
            .transpose(1, 2)
        )
        attended = F.scaled_dot_product_attention(
            selected_q, selected_k, selected_v, dropout_p=0.0, enable_gqa=attention.nhead != attention.n_kv_heads
        )
        attended = attended.transpose(1, 2).reshape(-1, attention.nhead, attention.head_dim)
        context = context.index_copy(0, group.indices, attended)
    return attention.out_proj(context.reshape(-1, attention.d_model))


def block(layer: RotaryTransformerEncoderLayer, inputs: torch.Tensor, packing: Packed) -> torch.Tensor:
    normed = layer.attention_norm(inputs)
    inputs = inputs + layer.attention(normed, normed, normed, packing=packing)
    return inputs + layer.ffn(layer.ffn_norm(inputs))


def unpack(
    values: torch.Tensor,
    packing: Packed,
    inputs: torch.Tensor,
    present: torch.Tensor,
    layers: Sequence[RotaryTransformerEncoderLayer],
) -> torch.Tensor:
    """Restore structural holes and retain zero gradients for entirely empty blocks."""
    output = inputs.masked_fill(~present.unsqueeze(-1), 0.0) * 0.0
    if packing.indices.numel():
        return output.reshape(-1, inputs.shape[-1]).index_copy(0, packing.indices, values).reshape(inputs.shape)
    if torch.is_grad_enabled():
        for layer in layers:
            for parameter in layer.parameters():
                output = output + parameter.sum() * 0.0
    return output
