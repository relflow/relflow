"""Hierarchical contrastive loss over two aligned views of the same records."""

import torch
import torch.nn.functional as F

__all__ = ["hierarchical"]


def candidates(z: torch.Tensor, valid: torch.Tensor, *, target: torch.Tensor) -> torch.Tensor:
    """Score every valid vector against the others in its group.

    Args:
        z: Vectors, `(groups, size, d_model)`.
        valid: Real vectors, `(groups, size)`.
        target: Index of the positive vector for each position, `(size,)`.

    Returns:
        Cross-entropy of the positive per position, zero where the vector is absent.
    """
    groups, size = valid.shape
    logits = z @ z.transpose(1, 2)
    # A finite fill keeps a fully blocked query from turning into NaN.
    blocked = torch.eye(size, dtype=torch.bool, device=z.device) | ~valid.unsqueeze(1)
    logits = logits.masked_fill(blocked, torch.finfo(logits.dtype).min)
    nll = F.cross_entropy(logits.flatten(0, 1), target.repeat(groups), reduction="none")
    return nll.view_as(valid) * valid


def instance(first: torch.Tensor, second: torch.Tensor, *, present: torch.Tensor) -> torch.Tensor:
    """Contrast each row with the same row of the other view against other records.

    Returns:
        Per-record shares, `(rows,)`.
    """
    rows = first.shape[0]
    z = torch.cat((first, second)).transpose(0, 1)
    valid = torch.cat((present, present)).transpose(0, 1)
    target = (torch.arange(2 * rows, device=z.device) + rows) % (2 * rows)
    per_view = candidates(z, valid, target=target).sum(dim=0)
    return (per_view[:rows] + per_view[rows:]) / valid.sum().clamp_min(1)


def temporal(first: torch.Tensor, second: torch.Tensor, *, present: torch.Tensor) -> torch.Tensor:
    """Contrast each row with the same row of the other view against other rows of its record.

    Returns:
        Per-record shares, `(rows,)`.
    """
    steps = first.shape[1]
    z = torch.cat((first, second), dim=1)
    valid = torch.cat((present, present), dim=1)
    target = (torch.arange(2 * steps, device=z.device) + steps) % (2 * steps)
    return candidates(z, valid, target=target).sum(dim=1) / valid.sum().clamp_min(1)


def coarsen(
    first: torch.Tensor, second: torch.Tensor, *, present: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Halve the time axis by max-pooling pairs of rows, ignoring absent rows."""
    kept = torch.max_pool1d(present.unsqueeze(1).float(), kernel_size=2, ceil_mode=True).squeeze(1) > 0

    def pool(values: torch.Tensor) -> torch.Tensor:
        floor = values.masked_fill(~present.unsqueeze(-1), torch.finfo(values.dtype).min)
        pooled = torch.max_pool1d(floor.transpose(1, 2), kernel_size=2, ceil_mode=True).transpose(1, 2)
        return pooled.masked_fill(~kept.unsqueeze(-1), 0.0)

    return pool(first), pool(second), kept


def hierarchical(first: torch.Tensor, second: torch.Tensor, *, present: torch.Tensor) -> torch.Tensor:
    """Contrast two views of the same records at every time scale.

    Matching rows of the two views are positives. Rows of other records and
    other rows of the same record are negatives. The comparison repeats after
    each max-pooling of pairs of rows until one row is left.

    Args:
        first: First view, `(rows, steps, d_model)`.
        second: Second view aligned on the same source rows.
        present: Rows both views hold, `(rows, steps)`.

    Returns:
        Per-record shares, `(rows,)`; their sum is the loss.
    """
    shares = []
    while True:
        shares.append(instance(first, second, present=present) + temporal(first, second, present=present))
        if first.shape[1] == 1:
            break
        first, second, present = coarsen(first, second, present=present)
    return torch.stack(shares).mean(dim=0)
