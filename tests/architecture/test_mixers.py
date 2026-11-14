import typing

import pytest
import torch

import relflow as rf
from relflow.architecture.encoder import BranchEncoder, RotaryTransformerEncoderLayer
from relflow.architecture.mixers import TransformerMixer, mixers
from relflow.structs.mixer import Conv, MixerConfig


def test_every_mixer_setting_has_a_builder():
    union = typing.get_args(typing.get_args(MixerConfig)[0])
    assert set(mixers) == {*union, type(None)}


def test_plain_branch_builds_todays_transformer_stack():
    schema = rf.Schema.from_tree(d_model=16, n_layers=2, n_heads=4, a=rf.Number, b=rf.Number)
    encoder = BranchEncoder(schema=schema, address="/")
    assert type(encoder.encoder) is TransformerMixer and len(encoder.encoder) == 2
    assert all(type(layer) is RotaryTransformerEncoderLayer for layer in encoder.encoder)


def test_attention_none_keeps_both_stacks_empty_under_the_transformer_mixer():
    schema = rf.Schema.from_tree(d_model=16, n_layers=1, n_heads=4, attention=None, a=rf.Number, b=rf.Number)
    encoder = BranchEncoder(schema=schema, address="/")
    assert len(encoder.encoder) == 0 and encoder.coordinate_encoder is None


def test_mixer_none_disables_sequence_mixing_but_keeps_coordinate_attention():
    schema = rf.Schema.from_tree(
        d_model=16, n_layers=1, n_heads=4, events=rf.Branch(length=2, mixer=None, a=rf.Number, b=rf.Number)
    )
    encoder = BranchEncoder(schema=schema, address=rf.Address("events"))
    assert len(encoder.encoder) == 0 and encoder.coordinate_encoder is not None


def test_conv_mixer_is_rejected_at_build_until_implemented():
    with pytest.raises(ValueError, match=r"branch '/events' sets mixer=Conv"):
        rf.Model(d_model=8, n_layers=1, n_heads=2, events=rf.Branch(mixer=Conv(), a=rf.Number))


def test_transformer_mixer_matches_layer_by_layer_application():
    generator = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    layers = [RotaryTransformerEncoderLayer(d_model=8, nhead=2, n_kv_heads=2, dropout=0.0) for _ in range(2)]
    mixer = TransformerMixer(layers).eval()
    tokens = torch.randn(3, 5, 8, generator=generator)
    present = torch.rand(3, 5, generator=generator) > 0.3
    expected = tokens
    for layer in layers:
        expected = layer(expected, present=present)
    actual = mixer(tokens, present=present, positions=None, steps=None, packing=None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
