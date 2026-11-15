"""Fixed-width crops of branch tokens for the contrastive objective."""

from dataclasses import dataclass

import torch

__all__ = ["Views", "crops", "pairs"]


@dataclass(frozen=True, kw_only=True)
class Views:
    """Two crops of every record, stacked along the record axis.

    Attributes:
        tokens: View one rows, then view two rows, `(2 * rows, width * per_row, d_model)`.
        present: Slots holding a real token, `(2 * rows, width * per_row)`.
        positions: Positions of the gathered tokens, or `None` when the branch has none.
        steps: Source row of every gathered token, `(2 * rows, width * per_row)`.
        shift: Rows between the first crop start and the second, `(rows,)`.
        width: Rows per crop.
    """

    tokens: torch.Tensor
    present: torch.Tensor
    positions: torch.Tensor | None
    steps: torch.Tensor
    shift: torch.Tensor
    width: int


def crops(
    tokens: torch.Tensor,
    *,
    present: torch.Tensor,
    positions: torch.Tensor | None,
    steps: torch.Tensor,
    starts: torch.Tensor,
    extent: int,
    width: int,
    hidden: torch.Tensor | None = None,
) -> Views:
    """Gather two crops of `width` rows from every record.

    Every row owns the same number of tokens, padding included, so a crop has a
    fixed shape whatever the record length. Slots past the end of the record
    repeat the last token and are marked absent. A hidden source row has its
    tokens zeroed in that crop only; the tokens stay present, so the crop keeps
    its shape and the row still pairs with the other crop.

    Args:
        tokens: Branch tokens, `(rows, L, d_model)`.
        present: Real tokens, `(rows, L)`.
        positions: Token positions, `(rows, L)`, or `None`.
        steps: Source row of every token, `(rows, L)`.
        starts: First row of each crop, `(rows, 2)`.
        extent: Rows of the branch axis.
        width: Rows per crop.
        hidden: Source rows each crop hides, `(rows, 2, extent)`, or `None` to hide nothing.

    Returns:
        The stacked crops.

    Raises:
        ValueError: The token axis is not a whole number of tokens per row.
    """
    rows, size, depth = tokens.shape
    if size % extent:
        message = (
            f"contrastive crops need the same number of tokens per row, but {size} tokens do not divide "
            f"into {extent} rows; give child branches a reduction"
        )
        raise ValueError(message)
    per_row = size // extent
    # Tokens of row s sit in block [s * per_row, (s + 1) * per_row) after a stable sort by row.
    order = torch.sort(steps, dim=1, stable=True).indices
    wanted = torch.arange(width * per_row, device=tokens.device)
    index = torch.cat([starts[:, view].unsqueeze(1) * per_row + wanted for view in range(2)])
    inside = index < size
    source = order.repeat(2, 1).gather(1, index.clamp(max=size - 1))

    def take(values: torch.Tensor) -> torch.Tensor:
        return values.repeat(2, *[1] * (values.ndim - 1)).gather(1, source)

    gathered = tokens.repeat(2, 1, 1).gather(1, source.unsqueeze(-1).expand(-1, -1, depth))
    cut_steps = take(steps)
    if hidden is not None:
        drawn = hidden.transpose(0, 1).reshape(2 * rows, extent)
        gathered = gathered.masked_fill(drawn.gather(1, cut_steps.clamp(max=extent - 1)).unsqueeze(-1), 0.0)

    return Views(
        tokens=gathered,
        present=take(present) & inside,
        positions=None if positions is None else take(positions),
        steps=cut_steps,
        shift=starts[:, 1] - starts[:, 0],
        width=width,
    )


def pairs(mixed: torch.Tensor, *, views: Views) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average the tokens of each row and align the two crops on the same source row.

    Args:
        mixed: Mixed tokens of the crops, `(2 * rows, width * per_row, d_model)`.
        views: The crops that produced `mixed`.

    Returns:
        Both views as `(rows, width, d_model)` and a `(rows, width)` mask that is true
        where both views hold the source row.
    """
    total, _, depth = mixed.shape
    rows = total // 2
    held = views.present.reshape(total, views.width, -1)
    values = mixed.reshape(total, views.width, -1, depth) * held.unsqueeze(-1)
    means = values.sum(dim=2) / held.sum(dim=2, keepdim=True).clamp_min(1)
    seen = held.any(dim=2)
    ahead = torch.arange(views.width, device=mixed.device) + views.shift.unsqueeze(1)
    inside = ahead < views.width
    ahead = ahead.clamp(max=views.width - 1)
    first = means[:rows].gather(1, ahead.unsqueeze(-1).expand(-1, -1, depth))
    return first, means[rows:], inside & seen[:rows].gather(1, ahead) & seen[rows:]
