"""Explicit preset classmethods retain subclass and checkpoint behavior."""

import pytest
import torch

import relflow as rf
from relflow import presets


def profile() -> presets.Preset:
    return presets.Preset(
        name="Merchant small",
        revision=3,
        d_model=16,
        root=presets.BranchDefaults(n_layers=2, n_heads=2, reduction=rf.Attention(n_outputs=2)),
        branch=presets.BranchDefaults(n_layers=1, n_heads=4, reduction=rf.Attention(n_outputs=2)),
        leaf=presets.LeafDefaults(n_heads=4),
    )


def test_custom_factory_preserves_subclass_and_node_overrides():
    selected = profile()

    class MerchantModel(rf.Model):
        compact = presets.factory(selected)

    model = MerchantModel.compact(
        n_layers=1,
        attention=None,
        reduction=None,
        batch_size=3,
        events=rf.Branch(length=3, amount=rf.Number),
        target=rf.Category(mask=True),
        preset=rf.Number,
        profile=rf.Number,
        cls=rf.Number,
    )

    assert type(model) is MerchantModel
    assert model.preset == selected
    assert model.batch_size == 3
    assert model.schema.fields.n_layers == 1
    assert model.schema.fields.attention is None
    assert model.schema.fields.reduction is None
    events = model.schema.branches[rf.Address("events")]
    assert (events.n_layers, events.n_heads, events.reduction.n_outputs) == (1, 4, 2)
    assert model.schema.requests[rf.Address("target")].n_heads == 4
    assert {rf.Address("preset"), rf.Address("profile"), rf.Address("cls")} <= set(model.schema.requests)


def test_custom_policy_survives_checkpoint_and_defaults_new_nodes(tmp_path):
    class MerchantModel(rf.Model):
        compact = presets.factory(profile())

    original = MerchantModel.compact(
        events=rf.Branch(length=2, amount=rf.Number),
        label=rf.Category(mask=True),
    )
    original.update(rf.where("address") == rf.Address("events"), n_layers=2)
    loaded = rf.Model.load(original.save(tmp_path / "custom.ckpt"))

    assert loaded.preset == original.preset
    assert loaded.schema.model_dump() == original.schema.model_dump()
    for key, weight in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], weight, rtol=0, atol=0)
    loaded.extend(rf.where("address") == rf.Address(), purchases=rf.Branch(length=2, amount=rf.Number))
    assert loaded.schema.branches[rf.Address("events")].n_layers == 2
    assert loaded.schema.branches[rf.Address("purchases")].n_heads == 4
    assert loaded.schema.branches[rf.Address("purchases")].reduction.n_outputs == 2


@pytest.mark.parametrize("selection", ["sm", 7, None])
def test_factory_requires_a_preset_object(selection):
    with pytest.raises(TypeError, match="Preset"):
        presets.factory(selection)


def test_factory_validates_copied_policy_before_creating_a_method():
    invalid = profile().model_copy(update={"revision": 0})
    with pytest.raises(ValueError, match="revision"):
        presets.factory(invalid)
