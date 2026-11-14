import typing

import pytest

import relflow as rf
from relflow.architecture.encoder import BranchEncoder
from relflow.architecture.grain import grains
from relflow.structs.grain import GrainConfig, Steps


def test_every_grain_setting_has_a_builder():
    union = typing.get_args(typing.get_args(GrainConfig)[0])
    assert set(union) == set(grains)


def test_coordinates_grain_keeps_parcels_and_adds_no_state():
    schema = rf.Schema.from_tree(d_model=16, n_layers=1, n_heads=4, a=rf.Number, b=rf.Number)
    encoder = BranchEncoder(schema=schema, address="/")
    parcels = [object()]
    assert encoder.grain(parcels) is parcels
    assert not any(name.startswith("grain") for name, _ in encoder.named_modules())


def test_steps_grain_is_rejected_at_build_until_implemented():
    with pytest.raises(ValueError, match=r"branch '/events' sets grain=Steps"):
        rf.Model(d_model=8, n_layers=1, n_heads=2, events=rf.Branch(grain=Steps(), a=rf.Number))
