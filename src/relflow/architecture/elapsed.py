"""Learned embedding of the time elapsed between the rows of a timed branch."""

from typing import TYPE_CHECKING, override

import torch
import torch.nn.functional as F


def gaps(times: torch.Tensor, *, slots: torch.Tensor) -> torch.Tensor:
    """Measure the time from the previous valid row of the same record to each row.

    Args:
        times: Row times of shape ``(..., L)``.
        slots: Boolean row validity of shape ``(..., L)``.

    Returns:
        Gaps of the same shape as ``times``: zero for an invalid row and for the first valid row.
    """
    length = times.shape[-1]
    index = torch.where(slots, torch.arange(length, device=times.device), -1).cummax(dim=-1).values
    previous = F.pad(index[..., :-1], (1, 0), value=-1)
    return torch.where(slots & (previous >= 0), times - times.gather(-1, previous.clamp_min(0)), 0.0)


class Elapsed(torch.nn.Module):
    """Embed a gap measured in ``time_step`` units.

    The gap passes through ``log1p`` so a rare long gap does not saturate the
    layer. Regular data, where every gap is one, adds one shared vector to
    every row after the first, so evenly spaced rows keep their meaning.
    """

    def __init__(self, *, d_model: int) -> None:
        super().__init__()
        self.first = torch.nn.Linear(1, d_model)
        self.activation = torch.nn.GELU(approximate="tanh")
        self.second = torch.nn.Linear(d_model, d_model)

    @override
    def forward(self, gaps: torch.Tensor) -> torch.Tensor:
        """Embed gaps.

        Args:
            gaps: Non-negative gaps of any shape.

        Returns:
            Embeddings of shape ``(*gaps.shape, d_model)`` in the parameter dtype.
        """
        scaled = gaps.log1p().to(dtype=self.first.weight.dtype)
        return self.second(self.activation(self.first(scaled.unsqueeze(-1))))

    if TYPE_CHECKING:
        __call__ = forward


__all__ = ["Elapsed", "gaps"]
