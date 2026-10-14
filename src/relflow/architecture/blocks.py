"""Node-sized feed-forward updates with fixed tanh GELU and RMS normalization."""

from __future__ import annotations

import torch

__all__ = ["feed_forward", "norm"]


def norm(width: int) -> torch.nn.RMSNorm:
    """Normalize channels with the same epsilon in every dtype and tensor region."""
    return torch.nn.RMSNorm(width, eps=1e-5)


def feed_forward(width: int, dropout: float, ffnx: int = 8) -> torch.nn.Sequential:
    """Expand channels by the node's multiplier for a tanh-GELU residual update."""
    hidden = ffnx * width
    return torch.nn.Sequential(
        torch.nn.Linear(width, hidden),
        torch.nn.GELU(approximate="tanh"),
        torch.nn.Dropout(p=dropout),
        torch.nn.Linear(hidden, width),
        torch.nn.Dropout(p=dropout),
    )
