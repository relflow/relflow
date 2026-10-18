"""Group-local membership proposals from the field's own identity likelihood."""

from __future__ import annotations

import torch

from relflow.distributed import all_reduce_sum


@torch.no_grad()
def propose(
    assignments: torch.Tensor,
    group_log_probabilities: torch.Tensor,
    identities: torch.Tensor,
    available: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose unique per-identity minima with all other identity rows held fixed.

    Inputs contain only known identity IDs and contextual group log probabilities.
    Nonretired columns participate in both candidate selection and likelihood.
    Every proposal retains the original row's total mass across all columns.
    Rows without global evidence or with tied minima retain their complete rows.
    All DDP ranks must share assignments and availability; empty local evidence
    still participates in the count and candidate-error reductions.

    This is an own-objective Group update, distinct from supervised reassignment.
    """
    if assignments.ndim != 2 or not assignments.is_floating_point():
        raise ValueError("Group assignments must be a floating tensor shaped [identities, groups]")
    size, groups = assignments.shape
    if groups < 1:
        raise ValueError("Group assignments require at least one group column")
    if group_log_probabilities.ndim != 2 or group_log_probabilities.shape[-1] != groups:
        raise ValueError(f"Group context log probabilities must have shape [observations, {groups}]")
    if not group_log_probabilities.is_floating_point():
        raise TypeError("Group context log probabilities must use a floating dtype")
    if identities.dtype != torch.int64 or identities.shape != (group_log_probabilities.shape[0],):
        raise ValueError("Group identity IDs must be an int64 vector matching the context observation count")
    if available.dtype != torch.bool or available.shape != (groups,):
        raise ValueError(f"Group availability must be a Boolean vector with shape [{groups}]")
    if any(value.device != assignments.device for value in (group_log_probabilities, identities, available)):
        raise ValueError("Group assignments, context, identity IDs, and availability must use the same device")
    if ((identities < 0) | (identities >= size)).any():
        raise IndexError(f"Group identity IDs must refer to known rows between 0 and {size - 1}")
    if not torch.isfinite(assignments).all():
        raise ValueError("Group assignment log weights must be finite")

    replacement = assignments.detach().clone()
    empty = torch.empty(0, dtype=torch.int64, device=assignments.device)
    if not size or not available.any():
        return replacement, empty
    context = group_log_probabilities[:, available]
    if torch.isnan(context).any() or torch.isposinf(context).any() or not torch.isfinite(context.logsumexp(-1)).all():
        raise ValueError("Group context must provide finite positive probability mass over available groups")

    dtype = torch.float32 if assignments.device.type == "mps" else torch.float64
    weights = assignments.to(dtype=dtype)
    context = group_log_probabilities.to(dtype=dtype).masked_fill(~available, -torch.inf)
    prefix = torch.logcumsumexp(weights, dim=0)
    suffix = torch.logcumsumexp(weights.flip(0), dim=0).flip(0)
    absent = weights.new_full((1, groups), -torch.inf)
    other_mass = torch.logaddexp(torch.cat((absent, prefix[:-1])), torch.cat((suffix[1:], absent)))
    row_mass = weights.logsumexp(-1, keepdim=True)
    margin = weights.new_tensor(1e-8).log()
    candidates = weights.new_zeros((size, groups))
    for candidate in available.nonzero(as_tuple=True)[0].tolist():
        hard = margin.expand(groups).clone()
        hard[candidate] = 0
        proposal = row_mass + hard - hard.logsumexp(0)
        conditional = proposal - torch.logaddexp(other_mass, proposal)
        errors = -(context + conditional.index_select(0, identities)).logsumexp(-1)
        candidates[:, candidate].scatter_add_(0, identities, errors)

    counts = all_reduce_sum(torch.bincount(identities, minlength=size))
    candidates = all_reduce_sum(candidates)
    if not torch.isfinite(candidates[counts.gt(0)][:, available]).all():
        raise ValueError("Group membership candidate errors must remain finite after global reduction")
    candidates.masked_fill_(~available, torch.inf)
    minimum, chosen = candidates.min(-1)
    tolerance = 8 * torch.finfo(dtype).eps
    tied = torch.isclose(candidates, minimum.unsqueeze(-1), atol=tolerance, rtol=tolerance).sum(-1)
    rows = (counts.gt(0) & tied.eq(1)).nonzero(as_tuple=True)[0]
    hard = margin.expand(rows.numel(), groups).clone()
    hard.scatter_(1, chosen[rows].unsqueeze(-1), 0)
    replacement[rows] = (row_mass[rows] + hard - hard.logsumexp(-1, keepdim=True)).to(assignments.dtype)
    return replacement, rows
