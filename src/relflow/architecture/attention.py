from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from relflow.architecture.blocks import norm
from relflow.architecture.packed import Packed, attend
from relflow.architecture.rotary import RotaryEmbedding


class RotaryMultiheadAttention(torch.nn.Module):
    """Project and attend dense sequences or preselected live tokens."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        n_kv_heads: int | None = None,
        position: bool = True,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        n_kv_heads = nhead if n_kv_heads is None else n_kv_heads
        if n_kv_heads < 1:
            raise ValueError("n_kv_heads must be >= 1")
        if nhead % n_kv_heads != 0:
            raise ValueError("nhead must be divisible by n_kv_heads")

        self.d_model = d_model
        self.nhead = nhead
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // nhead

        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = torch.nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.q_norm = norm(self.head_dim)
        self.k_norm = norm(self.head_dim)
        self.head_scale = torch.nn.Parameter(torch.ones(nhead, 1))

        self.rotary = RotaryEmbedding(d_model=self.head_dim) if position else None
        self.dropout_p = dropout

    def project(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize projected heads before position rotations in either token layout."""
        q = self.q_proj(query).unflatten(-1, (self.nhead, self.head_dim))
        k = self.k_proj(key).unflatten(-1, (self.n_kv_heads, self.head_dim))
        v = self.v_proj(value).unflatten(-1, (self.n_kv_heads, self.head_dim))
        # RMSNorm's float32 parameters can promote mixed-precision activations.
        # Attention kernels and packed scatter require the projection dtype.
        q = (self.q_norm(q) * self.head_scale).to(q)
        k = self.k_norm(k).to(k)
        return q, k, v

    def rotate(self, inputs: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        if self.rotary is None:
            return inputs
        return self.rotary(inputs, positions=positions)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        *,
        packing: Packed | None = None,
        query_positions: torch.Tensor | None = None,
        key_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if packing is not None:
            return attend(self, query, key, value, packing)
        active: torch.Tensor | None = None
        attn_mask: torch.Tensor | None = None
        if key_padding_mask is not None:
            expected = (key.shape[0], key.shape[1])
            if key_padding_mask.dtype != torch.bool or tuple(key_padding_mask.shape) != expected:
                raise ValueError(
                    f"attention key padding mask must have bool shape {expected}, "
                    f"got {tuple(key_padding_mask.shape)} with dtype {key_padding_mask.dtype}"
                )

            attn_mask = ~key_padding_mask
            active = attn_mask.any(dim=1)
            if key.shape[1] == 0:
                output = query.new_zeros(query.shape)
                if torch.is_grad_enabled():
                    for parameter in self.parameters():
                        output = output + parameter.sum() * 0.0
                return output

            # Give an empty row one safe placeholder key for SDPA, then erase its
            # result below. This stays entirely on-device and avoids a host sync.
            shared = key is value
            padding = key_padding_mask.unsqueeze(-1)
            key = key.masked_fill(padding, 0.0)
            value = key if shared else value.masked_fill(padding, 0.0)
            attn_mask[:, 0] |= ~active
            attn_mask = attn_mask[:, None, None, :]

        q, k, v = self.project(query, key, value)
        q = self.rotate(q.transpose(1, 2), query_positions)
        k = self.rotate(k.transpose(1, 2), key_positions)
        v = v.transpose(1, 2)

        context = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            enable_gqa=self.n_kv_heads != self.nhead,
        )
        context = context.transpose(1, 2).reshape(query.shape[0], query.shape[1], self.d_model)

        output = self.out_proj(context)
        if active is not None:
            output = output.masked_fill(~active[:, None, None], 0.0)
        return output

    if TYPE_CHECKING:
        __call__ = forward
