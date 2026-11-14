import torch

import relflow as rf
from relflow.architecture.elapsed import Elapsed, gaps
from relflow.architecture.encoder import BranchEncoder
from relflow.structs.tree import Address


def test_gaps_are_differences_to_the_previous_valid_row():
    times = torch.tensor([[0.0, 1.0, 3.0, 0.0]], dtype=torch.float64)
    slots = torch.tensor([[True, True, True, False]])
    assert gaps(times, slots=slots).tolist() == [[0.0, 1.0, 2.0, 0.0]]


def test_gaps_skip_null_rows_and_zero_empty_records():
    times = torch.tensor([[0.0, 0.0, 5.0], [1.0, 2.0, 3.0]], dtype=torch.float64)
    slots = torch.tensor([[True, False, True], [False, False, False]])
    assert gaps(times, slots=slots).tolist() == [[0.0, 0.0, 5.0], [0.0, 0.0, 0.0]]


def test_gaps_start_at_zero_for_every_instance_of_a_repeated_parent():
    times = torch.tensor([[[10.0, 11.0, 14.0], [100.0, 101.0, 105.0]]], dtype=torch.float64)
    slots = torch.ones(1, 2, 3, dtype=torch.bool)
    assert gaps(times, slots=slots).tolist() == [[[0.0, 1.0, 3.0], [0.0, 1.0, 4.0]]]


def test_elapsed_maps_gaps_to_the_parameter_dtype():
    result = Elapsed(d_model=8)(torch.tensor([[0.0, 2.0]], dtype=torch.float64))
    assert result.shape == (1, 2, 8) and result.dtype == torch.float32


def test_elapsed_parameter_count_is_independent_of_length():
    def timed(length):
        schema = rf.Schema.from_tree(
            d_model=16,
            n_layers=1,
            n_heads=4,
            ett=rf.Branch(length=length, time_index="ts", ts=rf.Number, value=rf.Number),
        )
        return BranchEncoder(schema=schema, address=Address("/ett"))

    counts = {length: sum(p.numel() for p in timed(length).elapsed.parameters()) for length in (8, 512)}
    assert counts[8] == counts[512] == 16 * 16 + 3 * 16
