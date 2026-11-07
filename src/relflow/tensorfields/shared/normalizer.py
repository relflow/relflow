"""Online per-column statistics for tensorfields that standardize source values."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import cast

import torch
from tensordict import TensorDict

from relflow.distributed import all_gather

__all__ = ["GlobalOnlineNormalizer", "moments", "precise", "wide"]


def precise(device: torch.device) -> AbstractContextManager[object]:
    """Disable mixed precision for source-value arithmetic; bf16 would quantize lanes and readouts."""
    if device.type in {"cuda", "cpu", "mps"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


def wide(device: torch.device) -> torch.dtype:
    """The widest float the device holds.

    Source values and their statistics stay float64 so that values far from
    zero keep their spread; MPS has no float64 and keeps float32.
    """
    return torch.float32 if device.type == "mps" else torch.float64


def moments(values: torch.Tensor, *, width: int = 1) -> TensorDict:
    """Summarize finite values per column with an exact count and centered moments in the widest float of the device.

    ``values`` reshapes to ``(-1, width)``; each column ignores its own
    non-finite entries. ``count``, ``mean`` and ``var`` have shape ``(width,)``.
    """

    columns = values.reshape(-1, width).to(dtype=wide(values.device))
    finite = torch.isfinite(columns)
    count = finite.sum(dim=0)
    safe = count.clamp_min(1).to(dtype=columns.dtype)
    kept = torch.where(finite, columns, 0.0)
    mean = kept.sum(dim=0) / safe
    variance = torch.where(finite, columns - mean, 0.0).square().sum(dim=0) / safe
    return TensorDict({"count": count.to(dtype=torch.int64), "mean": mean, "var": variance}, batch_size=[])


class GlobalOnlineNormalizer(torch.nn.Module):
    """Running mean and variance of one lane, or of ``width`` independent channels.

    Statistics stay float64 where the device allows, so values far from
    zero keep their spread. ``epsilon`` floors the standardizing scale at that
    fraction of ``|mean| + 1``: a constant lane standardizes to zero instead of
    dividing by zero, and a narrow lane keeps its spread rather than meeting an
    absolute floor. Each channel merges independently; a channel that sees no
    finite value keeps its statistics.
    """

    def __init__(self, alpha: float | None = None, epsilon: float = 1e-12, *, width: int = 1):
        super().__init__()

        self.epsilon: float = epsilon
        self.alpha: float | None = alpha
        self.width: int = width

        self.register_buffer("mean", torch.zeros(width, dtype=torch.float64))
        self.register_buffer("var", torch.ones(width, dtype=torch.float64))
        self.register_buffer("count", torch.zeros(width, dtype=torch.int64))

    @property
    def scale(self) -> torch.Tensor:
        """Standard deviation floored at ``epsilon`` of the magnitude of the mean."""
        return torch.sqrt(self.var).clamp_min(self.epsilon * (self.mean.abs() + 1.0))

    def _apply(self, fn, recurse=True):
        # MPS cannot hold float64: keep the statistics in float32 there.
        try:
            fn(self.mean)
        except TypeError:
            self.mean = self.mean.to(dtype=torch.float32)
            self.var = self.var.to(dtype=torch.float32)
        return super()._apply(fn, recurse)

    @torch.no_grad()
    def update(self, values: torch.Tensor):
        self.merge(moments(values, width=self.width))

    @torch.no_grad()
    def learn(self, observation: TensorDict) -> None:
        """Synchronize and apply one pristine numeric observation.

        Ranks exchange their (count, mean, var) triples in one collective and
        merge them centered on the shared mean, so small variances survive
        large offsets. Counts travel in the statistics' float: exact below
        2^53 on float64 devices and below 2^24 on MPS, which has no DDP.
        """

        if not isinstance(observation, TensorDict):
            raise TypeError(f"normalizer observation must be a TensorDict, got {type(observation).__name__}")
        dtype = self.mean.dtype
        local = torch.stack(
            [
                cast(torch.Tensor, observation[name]).reshape(self.width).to(dtype=dtype)
                for name in ("count", "mean", "var")
            ]
        ).to(device=self.mean.device)
        counts, means, variances = all_gather(local).unbind(dim=1)
        count = counts.sum(dim=0)
        weights = counts / count.clamp_min(1)
        mean = (weights * means).sum(dim=0)
        # Empty ranks contribute zero to the centered spread.
        delta = torch.where(counts > 0, means - mean, 0.0)
        variance = (weights * (variances + delta.square())).sum(dim=0)
        self.merge(TensorDict({"count": count.to(dtype=torch.int64), "mean": mean, "var": variance}, batch_size=[]))

    @torch.no_grad()
    def merge(self, observation: TensorDict) -> None:
        """Merge centered moments into running statistics without retaining values."""

        count = cast(torch.Tensor, observation["count"])
        mean = cast(torch.Tensor, observation["mean"])
        variance = cast(torch.Tensor, observation["var"])
        if not all(torch.isfinite(value).all() for value in (count, mean, variance)):
            raise ValueError("normalizer observation must contain only finite values")
        if (count < 0).any():
            raise ValueError("normalizer observation count cannot be negative")
        if (variance < 0).any():
            raise ValueError("normalizer observation variance cannot be negative")

        batch_count = count.to(device=self.count.device, dtype=self.count.dtype).reshape(self.width)
        batch_mean = mean.to(self.mean).reshape(self.width)
        batch_var = variance.to(self.var).reshape(self.width)

        if self.alpha is not None:
            alpha: float = self.alpha
            active = batch_count > 0
            self.mean = torch.where(active, (1 - alpha) * self.mean + alpha * batch_mean, self.mean)
            self.var = torch.where(active, (1 - alpha) * self.var + alpha * batch_var, self.var)
            return

        new_count = self.count + batch_count
        weight = batch_count.to(dtype=self.mean.dtype) / new_count.clamp_min(1)
        delta = batch_mean - self.mean
        correction = delta * torch.sqrt((self.count.to(dtype=self.mean.dtype) / new_count.clamp_min(1)) * weight)
        self.mean = torch.lerp(self.mean, batch_mean, weight)
        self.var = torch.lerp(self.var, batch_var, weight) + correction.square()
        self.count = new_count

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor, update=True) -> torch.Tensor:
        if self.training and update and self.width != 1:
            message = (
                f"normalizer with width={self.width} cannot update from flattened inputs; "
                "call learn(moments(values, width=width)) or pass update=False"
            )
            raise ValueError(message)
        finite_mask = mask & torch.isfinite(inputs)
        if self.training and update:
            self.update(inputs[finite_mask])

        normalized = (torch.atleast_1d(inputs) - self.mean) / self.scale
        return torch.where(finite_mask, normalized.reshape_as(inputs), inputs)
