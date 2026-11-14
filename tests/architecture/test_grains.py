import typing

import pytest
import torch

import relflow as rf
from relflow.architecture.encoder import BranchEncoder
from relflow.architecture.grain import StepStage, grains
from relflow.structs.grain import GrainConfig, Steps
from relflow.structs.packages import Parcel
from relflow.structs.tree import Address


def test_every_grain_setting_has_a_builder():
    union = typing.get_args(typing.get_args(GrainConfig)[0])
    assert set(union) == set(grains)


def test_coordinates_grain_keeps_parcels_and_adds_no_state():
    schema = rf.Schema.from_tree(d_model=16, n_layers=1, n_heads=4, a=rf.Number, b=rf.Number)
    encoder = BranchEncoder(schema=schema, address="/")
    parcels = [object()]
    assert encoder.grain(parcels) is parcels
    assert not any(name.startswith("grain") for name, _ in encoder.named_modules())


def test_steps_and_conv_build_in_a_model():
    rf.Model(
        d_model=8, n_layers=1, n_heads=2, events=rf.Branch(grain=rf.Steps(), mixer=rf.Conv(n_blocks=1), a=rf.Number)
    )


def ett(**options):
    return rf.Schema.from_tree(
        d_model=16,
        n_layers=1,
        n_heads=4,
        ett=rf.Branch(length=4, grain=Steps(**options), reduction=None, a=rf.Number, b=rf.Number, c=rf.Number),
    )


def leaves(present):
    generator = torch.Generator().manual_seed(0)
    return [
        Parcel(
            origin=f"/ett/{name}",
            destination="/ett",
            payload=torch.randn(1, 1, 4, 16, generator=generator),
            present=present.clone().reshape(1, 1, 4),
            positions=torch.arange(4).expand(1, 1, 4),
            spans=torch.full((1, 1), 4),
            slots=torch.ones(1, 1, 4, dtype=torch.bool),
            batch_size=1,
        )
        for name in "abc"
    ]


@pytest.mark.parametrize("pooling", ["query", "mean"])
def test_steps_pools_each_row_to_one_token(pooling):
    stage = BranchEncoder(schema=ett(pooling=pooling), address="/ett").grain
    (row,) = stage(leaves(torch.tensor([True, True, False, True])))
    assert row.payload.shape == (1, 1, 4, 16)
    assert row.present.tolist() == [[[True, True, False, True]]]
    assert not row.payload[:, :, 2].any()
    assert row.positions.tolist() == [[[0, 1, 2, 3]]] and row.spans.tolist() == [[4]]


def test_steps_row_present_when_any_child_token_present():
    stage = BranchEncoder(schema=ett(), address="/ett").grain
    parcels = leaves(torch.tensor([True, False, False, False]))
    parcels[1].present[..., 1] = True
    (row,) = stage(parcels)
    assert row.present.tolist() == [[[True, True, False, False]]]


def test_steps_row_is_absent_without_its_slot():
    stage = BranchEncoder(schema=ett(), address="/ett").grain
    parcels = leaves(torch.ones(4, dtype=torch.bool))
    for parcel in parcels:
        parcel.slots[..., 3] = False
    (row,) = stage(parcels)
    assert row.present.tolist() == [[[True, True, True, False]]]


def test_steps_stage_is_registered_only_for_steps_branches():
    assert isinstance(BranchEncoder(schema=ett(), address="/ett").grain, StepStage)


def test_steps_stage_does_not_rerun_coordinate_attention():
    encoder = BranchEncoder(schema=ett(), address="/ett")
    calls = []
    assert encoder.coordinate_encoder is not None
    encoder.coordinate_encoder.register_forward_hook(lambda *_: calls.append(1))
    encoder(leaves(torch.ones(4, dtype=torch.bool)))
    assert len(calls) == 1


def test_steps_routes_row_positions_through_a_repeated_parent():
    schema = rf.Schema.from_tree(
        d_model=16,
        n_layers=1,
        n_heads=4,
        stations=rf.Branch(
            length=2,
            reduction=rf.Mean(),
            readings=rf.Branch(length=3, grain=Steps(), reduction=None, value=rf.Number),
        ),
    )
    encoder = BranchEncoder(schema=schema, address="/stations/readings")
    parcel = Parcel(
        origin="/stations/readings/value",
        destination="/stations/readings",
        payload=torch.randn(1, 1, 2, 3, 16),
        present=torch.ones(1, 1, 2, 3, dtype=torch.bool),
        positions=torch.arange(3).expand(1, 1, 2, 3),
        spans=torch.full((1, 1, 2), 3),
        slots=torch.ones(1, 1, 2, 3, dtype=torch.bool),
        batch_size=1,
    )
    routed = encoder([parcel])
    assert routed.positions.tolist() == [[[0, 1, 2, 3, 4, 5]]]


def test_steps_sequence_tokens_measured_at_336_rows_and_7_channels():
    children = {f"c{index}": rf.Number for index in range(7)}
    seen = {}
    for name, grain in (("plain", rf.Coordinates()), ("steps", Steps())):
        schema = rf.Schema.from_tree(
            d_model=16,
            n_layers=1,
            n_heads=4,
            ett=rf.Branch(length=336, grain=grain, reduction=None, **children),
        )
        encoder = BranchEncoder(schema=schema, address="/ett")
        encoder.encoder.register_forward_pre_hook(
            lambda module, args, kwargs, name=name: seen.__setitem__(name, args[0].shape[1]), with_kwargs=True
        )
        generator = torch.Generator().manual_seed(0)
        parcels = [
            Parcel(
                origin=f"/ett/c{index}",
                destination="/ett",
                payload=torch.randn(1, 1, 336, 16, generator=generator),
                present=torch.ones(1, 1, 336, dtype=torch.bool),
                positions=torch.arange(336).expand(1, 1, 336),
                spans=torch.full((1, 1), 336),
                slots=torch.ones(1, 1, 336, dtype=torch.bool),
                batch_size=1,
            )
            for index in range(7)
        ]
        encoder(parcels)
    assert seen["plain"] == 336 * 7 == 2352
    assert seen["steps"] == 336


def test_steps_branch_widths_are_length():
    schema = ett()
    assert schema.branch_outputs[Address("/ett")] == 4
    assert schema.branch_widths[Address("/ett")] == 4
