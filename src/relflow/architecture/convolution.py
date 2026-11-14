"""Dilated convolution blocks over ordered steps, shared by mixers and embedders."""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import override

import torch
import torch.nn.functional as F

from relflow.architecture.packed import Packed
from relflow.structs.enums import Direction

type Padding = Callable[[int], tuple[int, tuple[int, int]]]

# For one dilation: the symmetric padding of the convolution and the extra
# (left, right) padding applied before it. A causal convolution pads only the
# past, so no output reads a later step.
padding: Mapping[Direction, Padding] = MappingProxyType(
    {
        Direction.bidirectional: lambda dilation: (dilation, (0, 0)),
        Direction.causal: lambda dilation: (0, (2 * dilation, 0)),
    }
)


class Block(torch.nn.Module):
    """Pre-norm residual block of two dilated convolutions over observed steps.

    Unobserved steps are zero on entry to every convolution, so a sequence
    encodes the same whatever the batch pads it with.

    Attributes:
        direction: Whether a step may read later steps.
    """

    def __init__(self, *, width: int, dilation: int, direction: Direction) -> None:
        super().__init__()
        self.direction = direction
        same, self.extra = padding[direction](dilation)
        self.norm = torch.nn.RMSNorm(width)
        self.first = torch.nn.Conv1d(width, width, kernel_size=3, padding=same, dilation=dilation)
        self.second = torch.nn.Conv1d(width, width, kernel_size=3, padding=same, dilation=dilation)
        self.activation = torch.nn.GELU(approximate="tanh")

    @override
    def forward(self, x: torch.Tensor, *, observed: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: Steps of shape ``(batch, steps, width)``.
            observed: Boolean ``(batch, steps)`` mask of steps that exist.

        Returns:
            Tensor shaped like ``x`` with unobserved steps zero.
        """
        gone = ~observed.unsqueeze(-1)
        hidden = self.activation(self.norm(x)).transpose(1, 2)
        hidden = self.activation(self.first(F.pad(hidden, self.extra)).transpose(1, 2)).masked_fill(gone, 0.0)
        hidden = self.second(F.pad(hidden.transpose(1, 2), self.extra)).transpose(1, 2)
        return (x + hidden).masked_fill(gone, 0.0)


class ConvMixer(torch.nn.Module):
    """Mix the tokens of a branch, one per row, with dilated convolution blocks.

    The dilation doubles with depth, so ``n_blocks`` blocks see
    ``1 + 4 * (2**n_blocks - 1)`` rows.
    """

    def __init__(self, *, width: int, n_blocks: int, direction: Direction) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [Block(width=width, dilation=2**depth, direction=direction) for depth in range(n_blocks)]
        )

    @override
    def forward(
        self,
        tokens: torch.Tensor,
        *,
        present: torch.Tensor | None,
        positions: torch.Tensor | None,
        steps: torch.Tensor | None,
        packing: Packed | None,
    ) -> torch.Tensor:
        """Convolve over rows; absent rows stay zero and are never read.

        Args:
            tokens: Tokens of shape ``(rows, L, d_model)``, one per row of the branch.
            present: Boolean ``(rows, L)`` presence, or ``None`` when every row is live.
            positions: Ignored; the convolution reads row order.
            steps: Ignored; each token is already one row.
            packing: Ignored; the mixer always receives dense tokens.

        Returns:
            Mixed tokens shaped like ``tokens``.
        """
        observed = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device) if present is None else present
        hidden = tokens.masked_fill(~observed.unsqueeze(-1), 0.0)
        for block in self.blocks:
            hidden = block(hidden, observed=observed)
        return hidden


__all__ = ["Block", "ConvMixer", "padding"]
