"""Saved construction policies survive restoration and apply only to new nodes."""

from copy import deepcopy

import pytest
import torch

import relflow as rf
from relflow import presets


def configured() -> rf.Model:
    return rf.Model.sm(
        d_model=16,
        n_layers=1,
        n_heads=2,
        amount=rf.Number,
        events=rf.Branch(length=2, amount=rf.Number),
        target=rf.Number(mask=True),
    )


def test_checkpoint_preserves_policy_and_resolved_overrides_despite_new_defaults(tmp_path, monkeypatch):
    original = configured()
    original.update(rf.where("address") == "/events", n_layers=3)
    path = original.save(tmp_path / "preset.ckpt")
    saved = torch.load(path, weights_only=False, map_location="cpu")
    assert saved["preset"] == original.preset.model_dump(mode="json")

    monkeypatch.setattr(
        presets,
        "SM",
        presets.SM.model_copy(
            update={
                "revision": 2,
                "branch": presets.SM.branch.model_copy(update={"n_layers": 4, "n_heads": 2}),
                "leaf": presets.SM.leaf.model_copy(update={"n_heads": 2}),
            }
        ),
    )
    restored = rf.Model.load(path)
    assert restored.preset == original.preset
    assert restored.schema.model_dump(mode="json") == original.schema.model_dump(mode="json")
    for name, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)

    for model in (original, restored):
        model.extend(
            rf.where("address") == "/",
            purchases=rf.Branch(length=3, price=rf.Number),
            score=rf.Number(mask=True, n_heads=2),
        )
        added = model.schema.branches[rf.Address("purchases")]
        assert added.n_layers == 1
        assert added.n_heads == 4
        assert added.reduction.n_outputs == 2
        assert model.schema.requests[rf.Address("purchases", "price")].n_heads == 4
        assert model.schema.requests[rf.Address("score")].n_heads == 2
        assert model.schema.branches[rf.Address("events")].n_layers == 3
        assert model.schema.fields.n_layers == 1
        assert model.schema.fields.n_heads == 2
    assert restored.schema.model_dump(mode="json") == original.schema.model_dump(mode="json")


def test_lightning_metadata_restores_and_clears_construction_policy():
    original = configured()
    checkpoint = {}
    original.on_save_checkpoint(checkpoint)
    restored = rf.Model(d_model=16, n_layers=1, n_heads=2, value=rf.Number)
    restored.on_load_checkpoint(checkpoint)
    assert restored.preset == original.preset

    restored.on_load_checkpoint({"version": restored.version})
    assert restored.preset is None
    restored.extend(rf.where("address") == "/", child=rf.Branch(value=rf.Number))
    assert restored.schema.branches[rf.Address("child")].reduction.n_outputs == 1


@pytest.mark.parametrize("failure", ["weights", "policy"])
def test_failed_restore_preserves_previous_policy_and_graph(failure):
    original = configured()
    checkpoint = {"state_dict": original.state_dict()}
    original.on_save_checkpoint(checkpoint)
    checkpoint["version"] = "other-version"
    checkpoint = deepcopy(checkpoint)
    if failure == "weights":
        checkpoint["state_dict"].pop(next(iter(checkpoint["state_dict"])))
    else:
        checkpoint["preset"]["revision"] = 0

    target = rf.Model.xs(d_model=16, amount=rf.Number).eval()
    policy, schema, nodes, version = target.preset, target.schema, target.nodes, target.version
    with pytest.raises((RuntimeError, ValueError)):
        target.restore_checkpoint_state(checkpoint)
    assert target.preset is policy
    assert target.schema is schema
    assert target.nodes is nodes
    assert target.version == version
    assert not target.training


def test_in_place_restore_of_ordinary_model_clears_policy():
    original = rf.Model(d_model=16, n_layers=1, n_heads=2, value=rf.Number)
    checkpoint = {"state_dict": original.state_dict()}
    original.on_save_checkpoint(checkpoint)
    target = configured()
    target.restore_checkpoint_state(checkpoint)
    assert target.preset is None
    assert target.schema.model_dump(mode="json") == original.schema.model_dump(mode="json")


def test_failed_extension_does_not_change_policy_or_existing_schema():
    model = configured()
    policy = model.preset
    schema = model.schema.model_dump(mode="json")
    with pytest.raises(ValueError, match="duplicate field"):
        model.extend(rf.where("address") == "/", amount=rf.Number)
    assert model.preset is policy
    assert model.schema.model_dump(mode="json") == schema


def test_presets_apply_only_to_new_nodes_during_update_and_override():
    model = configured()
    selector = rf.where("address") == "/events"
    model.update(selector, n_layers=3, reduction=None)
    assert model.schema.branches[rf.Address("events")].reduction is None
    with model.override(selector, n_layers=2):
        assert model.schema.branches[rf.Address("events")].n_layers == 2
        assert model.schema.branches[rf.Address("events")].reduction is None
    assert model.schema.branches[rf.Address("events")].n_layers == 3
    assert model.schema.branches[rf.Address("events")].reduction is None
