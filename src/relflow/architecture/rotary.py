import math
from typing import TYPE_CHECKING

import torch


def flatten(
    positions: torch.Tensor,
    spans: torch.Tensor,
    *,
    capacities: tuple[int | None, ...],
    lengths: tuple[torch.Tensor | None, ...],
    capacity: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten nested structural coordinates while preserving finite-cap gaps."""
    shape = tuple(positions.shape)
    for axis in range(len(capacities) - 1, -1, -1):
        offsets = spans.cumsum(dim=-1) - spans
        positions = positions + offsets.reshape(*offsets.shape, *([1] * (len(shape) - offsets.ndim)))
        outer_capacity = capacities[axis]
        if capacity is not None and outer_capacity is not None:
            capacity *= outer_capacity
            spans = torch.full(shape[: axis + 1], capacity, device=positions.device, dtype=torch.int64)
        else:
            retained = lengths[axis]
            if retained is not None:
                slots = torch.arange(shape[axis + 1], device=positions.device) < retained.unsqueeze(-1)
                spans = spans.masked_fill(~slots, 0)
            spans = spans.sum(dim=-1)
            capacity = None
    return positions.reshape(shape[0], math.prod(shape[1:])), spans


def coordinates(
    shape: tuple[int, ...],
    capacities: tuple[int | None, ...],
    lengths: tuple[torch.Tensor | None, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Number structural coordinates using record-local spans, never batch maxima."""
    positions = torch.arange(shape[-1], device=device).expand(shape)
    capacity = capacities[-1]
    spans = torch.full(shape[:-1], capacity, device=device, dtype=torch.int64) if capacity is not None else lengths[-1]
    if spans is None:
        spans = torch.full(shape[:-1], shape[-1], device=device, dtype=torch.int64)
    positions, _ = flatten(positions, spans, capacities=capacities[:-1], lengths=lengths[:-1], capacity=capacity)
    return positions.reshape(shape)


def signal(positions: torch.Tensor, width: int, *, dtype: torch.dtype) -> torch.Tensor:
    """Give shared queries fixed coordinate features without growing parameters."""
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=positions.device, dtype=torch.float32) * (-math.log(10_000.0) / width)
    )
    angles = positions.unsqueeze(-1).to(dtype=frequencies.dtype) * frequencies
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)[..., :width].to(dtype=dtype)


class RotaryEmbedding(torch.nn.Module):
    """Rotate paired channels while preserving an optional final odd channel."""

    def __init__(self, d_model: int, base: float = 10_000.0) -> None:
        super().__init__()

        if d_model < 2:
            raise ValueError("d_model must be at least 2 for rotary embeddings")

        self.d_model = d_model
        self.rotary_dim = d_model - (d_model % 2)
        self.base = base

        index = torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
        self.inv_freq: torch.Tensor
        self.register_buffer("inv_freq", base ** (-index / self.rotary_dim), persistent=False)

    def forward(self, inputs: torch.Tensor, *, positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = inputs.shape[-2]

        if positions is None:
            positions = torch.arange(seq_len, device=inputs.device, dtype=self.inv_freq.dtype)
        else:
            positions = positions.to(device=inputs.device, dtype=self.inv_freq.dtype)
        freqs = positions.unsqueeze(-1) * self.inv_freq
        while freqs.ndim < inputs.ndim:
            freqs = freqs.unsqueeze(-3)
        cos = freqs.cos().to(dtype=inputs.dtype)
        sin = freqs.sin().to(dtype=inputs.dtype)

        rotated = inputs[..., : self.rotary_dim]
        passthrough = inputs[..., self.rotary_dim :]

        even = rotated[..., ::2]
        odd = rotated[..., 1::2]

        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        rotated = torch.stack((rotated_even, rotated_odd), dim=-1).flatten(start_dim=-2)

        if passthrough.shape[-1] == 0:
            return rotated

        return torch.cat((rotated, passthrough), dim=-1)

    if TYPE_CHECKING:
        __call__ = forward
