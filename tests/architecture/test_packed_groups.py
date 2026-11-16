"""Packing groups rows by length on the device and transfers only the distinct lengths."""

import torch

from relflow.architecture.packed import pack


def test_groups_are_ascending_slices_of_row_major_live_tokens():
    present = torch.tensor(
        [
            [False, False, False],
            [True, True, True],
            [True, True, True],
            [True, False, False],
            [False, False, False],
            [True, True, True],
        ]
    )
    inputs = torch.arange(18, dtype=torch.float32).reshape(6, 3, 1)
    positions = torch.tensor([[0, 1, 2]]).expand(6, 3) * 10

    values, packing = pack(inputs, present, positions)

    assert values.flatten().tolist() == [3, 4, 5, 6, 7, 8, 9, 15, 16, 17]
    assert packing.indices.tolist() == [3, 4, 5, 6, 7, 8, 9, 15, 16, 17]
    assert packing.positions.tolist() == [0, 10, 20, 0, 10, 20, 0, 0, 10, 20]
    assert [(group.rows, group.length) for group in packing.groups] == [(1, 1), (3, 3)]
    assert packing.groups[0].indices.tolist() == [6]
    assert packing.groups[1].indices.tolist() == [0, 1, 2, 3, 4, 5, 7, 8, 9]
    assert all(group.indices.dtype == torch.int64 for group in packing.groups)


def test_empty_batches_and_absent_rows_pack_to_no_groups():
    for rows in (0, 2):
        values, packing = pack(torch.zeros(rows, 4, 3), torch.zeros(rows, 4, dtype=torch.bool))
        assert values.shape == (0, 3)
        assert packing.indices.numel() == 0
        assert packing.groups == ()
