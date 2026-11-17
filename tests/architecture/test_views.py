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


def masked(hidden: torch.Tensor | None):
    rows, extent, per_row, d = 2, 5, 2, 4
    steps = torch.arange(extent).repeat(per_row).expand(rows, -1)
    return crops(
        torch.ones(rows, extent * per_row, d),
        present=torch.ones(rows, extent * per_row, dtype=torch.bool),
        positions=None,
        steps=steps,
        starts=torch.tensor([[0, 1], [0, 1]]),
        extent=extent,
        width=4,
        hidden=hidden,
    )


def test_crops_without_hidden_match_crops_with_nothing_hidden():
    none = torch.zeros(2, 2, 5, dtype=torch.bool)
    assert torch.equal(masked(None).tokens, masked(none).tokens)


def test_crops_zero_the_hidden_source_rows_of_each_view_and_keep_presence():
    hidden = torch.zeros(2, 2, 5, dtype=torch.bool)
    hidden[0, 0, 1] = True  # record 0, view one hides source row 1
    hidden[0, 1, 2] = True  # record 0, view two hides source row 2
    views = masked(hidden)
    zeroed = (views.tokens == 0).all(dim=-1)
    assert zeroed[0].nonzero().flatten().tolist() == [2, 3]  # source row 1 holds slots 2 and 3 of view one
    assert zeroed[2].nonzero().flatten().tolist() == [2, 3]  # view two starts at row 1, so row 2 is crop row 1
    assert zeroed[1].sum() == 0 and zeroed[3].sum() == 0
    assert views.present.all()
