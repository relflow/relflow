import pytest
import torch

from relflow.architecture.views import crops, pairs


def gathered(*, positions: bool = True):
    rows, extent, per_row, d = 3, 5, 2, 4
    steps = torch.arange(extent).repeat(per_row).expand(rows, -1)  # field-major storage
    present = steps < torch.tensor([5, 2, 0]).unsqueeze(1)  # lengths 5, 2, 0
    tokens = torch.randn(rows, extent * per_row, d, generator=torch.Generator().manual_seed(0))
    return crops(
        tokens,
        present=present,
        positions=steps.clone() if positions else None,
        steps=steps,
        starts=torch.tensor([[0, 2], [0, 0], [0, 0]]),
        extent=extent,
        width=4,
    )


def test_crops_keep_a_fixed_width_and_mask_short_records():
    views = gathered()
    assert views.tokens.shape == (6, 8, 4) and views.present.shape == (6, 8)
    assert views.present.sum(dim=1).tolist() == [8, 4, 0, 6, 4, 0]
    assert views.steps[3].tolist() == [2, 2, 3, 3, 4, 4, 4, 4]  # clamped tail slots are absent
    assert views.shift.tolist() == [2, 0, 0]
    assert views.positions is not None
    assert gathered(positions=False).positions is None


def test_crops_reject_a_width_that_rows_do_not_divide():
    with pytest.raises(ValueError, match="tokens per row"):
        crops(
            torch.zeros(1, 7, 4),
            present=torch.ones(1, 7, dtype=torch.bool),
            positions=None,
            steps=torch.zeros(1, 7, dtype=torch.int64),
            starts=torch.zeros(1, 2, dtype=torch.int64),
            extent=5,
            width=2,
        )


def test_pairs_align_views_on_the_same_source_row():
    views = gathered()
    mixed = views.steps.unsqueeze(-1).float().expand(-1, -1, 4)
    first, second, present = pairs(mixed, views=views)
    assert first.shape == (3, 4, 4) and present.shape == (3, 4)
    assert torch.equal(first[present], second[present])
    assert present.sum(dim=1).tolist() == [2, 2, 0]
