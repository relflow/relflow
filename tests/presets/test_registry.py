"""Named preset registration retains subclass, YAML, and checkpoint behavior."""

import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

import relflow as rf
from relflow import presets
from relflow.presets import registry


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, "PRESETS", dict(registry.PRESETS))


def profile() -> presets.Preset:
    return presets.Preset(
        name="merchant-small",
        revision=3,
        d_model=16,
        root=presets.BranchDefaults(n_layers=2, n_heads=2, reduction=rf.Attention(n_outputs=2)),
        branch=presets.BranchDefaults(n_layers=1, n_heads=4, reduction=rf.Attention(n_outputs=2)),
        leaf=presets.LeafDefaults(n_heads=4),
    )


def test_custom_registration_preserves_subclass_and_node_overrides():
    class MerchantModel(rf.Model):
        pass

    selected = profile()
    registered = presets.register(selected)
    assert registered is not selected
    assert registered == selected
    model = MerchantModel(
        preset=selected.name,
        n_layers=1,
        attention=None,
        reduction=None,
        batch_size=3,
        events=rf.Branch(length=3, amount=rf.Number),
        target=rf.Category(mask=True),
        profile=rf.Number,
        cls=rf.Number,
        fields={"preset": rf.Number},
        name=rf.Category,
    )

    assert type(model) is MerchantModel
    assert model.preset == selected
    assert model.batch_size == 3
    assert model.schema.fields.name is None
    assert model.schema.fields.n_layers == 1
    assert model.schema.fields.attention is None
    assert model.schema.fields.reduction is None
    events = model.schema.branches["/events"]
    assert (events.n_layers, events.n_heads, events.reduction.n_outputs) == (1, 4, 2)
    assert model.schema.requests["/target"].n_heads == 4
    assert {"/name", "/preset", "/profile", "/cls"} <= set(model.schema.requests)


def test_custom_policy_survives_checkpoint_without_registration_and_defaults_new_nodes(tmp_path):
    class MerchantModel(rf.Model):
        pass

    selected = presets.register(profile())
    original = MerchantModel(
        preset=selected.name, events=rf.Branch(length=2, amount=rf.Number), label=rf.Category(mask=True)
    )
    original.update(rf.where("address") == "/events", n_layers=2)
    pathname = original.save(tmp_path / "custom.ckpt")
    del registry.PRESETS[selected.name]
    loaded = rf.Model.load(pathname)

    assert loaded.preset == original.preset
    assert loaded.schema.model_dump() == original.schema.model_dump()
    for key, weight in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], weight, rtol=0, atol=0)
    loaded.extend(rf.where("address") == "/", purchases=rf.Branch(length=2, amount=rf.Number))
    assert loaded.schema.branches["/events"].n_layers == 2
    assert loaded.schema.branches["/purchases"].n_heads == 4
    assert loaded.schema.branches["/purchases"].reduction.n_outputs == 2


def test_yaml_selects_a_registered_application_policy(tmp_path):
    selected = presets.register(profile())
    path = tmp_path / "merchant.yaml"
    path.write_text(f"preset: {selected.name}\nfields:\n  amount:\n    type: number\n")
    model = rf.Model.from_yaml(path)
    assert model.preset == selected
    assert model.schema.d_model == selected.d_model


@pytest.mark.parametrize("selection", ["sm", 7, None])
def test_registration_requires_a_preset_object(selection):
    with pytest.raises(TypeError, match="Preset"):
        presets.register(selection)


def test_registration_revalidates_copied_policies_and_protects_existing_names():
    invalid = profile().model_copy(update={"revision": 0})
    with pytest.raises(ValueError, match="revision"):
        presets.register(invalid)
    selected = presets.register(profile())
    with pytest.raises(ValueError, match="already registered"):
        presets.register(selected)
    with pytest.raises(ValueError, match="already registered"):
        presets.register(presets.SMALL)


@pytest.mark.parametrize("name", ["missing", "", "SMALL", "xs", "sm", "md", "lg", "xl", "large", "xsmall", "xlarge"])
def test_unknown_names_are_rejected(name):
    with pytest.raises(ValueError, match="unknown preset"):
        rf.Model(preset=name, value=rf.Number)


@pytest.mark.parametrize("name", [3, presets.SMALL])
def test_preset_selection_requires_a_name_string(name):
    with pytest.raises(BeartypeCallHintParamViolation, match="preset"):
        rf.Model(preset=name, value=rf.Number)


def test_declared_name_field_is_independent_of_preset_selection():
    class NamedModel(rf.Model):
        name = rf.Category
        preset = rf.Number
        result = rf.Boolean(mask=True)

    model = NamedModel(preset="small", d_model=8)
    assert model.schema.fields.name is None
    assert "/name" in model.schema.requests
    assert "/preset" in model.schema.requests
    assert model.preset == presets.SMALL


@pytest.mark.parametrize(
    "name,selected",
    [
        ("small", presets.SMALL),
        ("medium", presets.MEDIUM),
    ],
)
def test_readable_names_select_the_existing_bundled_policies(name, selected):
    model = rf.Model(preset=name, d_model=16, value=rf.Number)
    assert presets.get(name) is selected
    assert model.preset == selected
    assert model.schema.fields.n_layers == selected.root.n_layers
    assert model.schema.requests["/value"].n_heads == selected.leaf.n_heads


def test_none_preset_keeps_explicit_architecture_construction():
    with pytest.raises(TypeError, match="d_model, n_layers, n_heads"):
        rf.Model(preset=None, value=rf.Number)
    model = rf.Model(preset=None, d_model=8, n_layers=1, n_heads=2, value=rf.Number)
    assert model.preset is None
