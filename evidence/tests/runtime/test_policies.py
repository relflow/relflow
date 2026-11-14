"""Frozen Evidence configuration preserves architectures, weights and mutation defaults."""

import importlib
import sys
from pathlib import Path

import pytest
import relflow as rf
import torch
from pydantic import ValidationError
from relflow.presets.base import resolve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))


def historical():
    """The complete policy previously repeated inside each XS experiment."""
    return rf.presets.Preset(
        name="xs",
        d_model=64,
        root=rf.presets.BranchDefaults(n_layers=1, n_heads=2, reduction=rf.Attention()),
        branch=rf.presets.BranchDefaults(n_layers=1, n_heads=2, reduction=rf.Attention()),
        leaf=rf.presets.LeafDefaults(n_heads=2),
    )


def test_internal_registration_is_unique_and_leaves_public_policies_unchanged():
    public = {name: rf.presets.get(name).model_dump(mode="json") for name in ("small", "medium")}
    module = importlib.import_module("policies")
    assert importlib.import_module("policies").CONTROL is module.CONTROL
    assert rf.presets.get("evidence-control-v1") is module.CONTROL
    assert module.CONTROL.name != "xs"
    assert module.CONTROL.fit is None
    assert public == {name: rf.presets.get(name).model_dump(mode="json") for name in public}
    assert module.CONTROL.model_dump(exclude={"name"}) == historical().model_dump(exclude={"name"})
    for policy, field, value in (
        (module.CONTROL, "d_model", 128),
        (module.CONTROL.root, "n_heads", 4),
        (module.CONTROL.branch.reduction, "n_outputs", 2),
        (module.CONTROL.leaf, "n_linear", 2),
    ):
        with pytest.raises(ValidationError, match="frozen"):
            setattr(policy, field, value)


def test_class_policy_matches_historical_schema_seeded_weights_and_mutation_defaults():
    from policies import CONTROL

    class Item(rf.Branch):
        amount = rf.Number
        merchant = rf.Category

    class Model(rf.Model):
        items = Item(length=4, overflow="tail")
        label = rf.Boolean(mask=True)

    previous = historical()
    torch.manual_seed(901)
    old = rf.Model(
        schema=resolve(
            previous,
            items=rf.Branch(length=4, overflow="tail", amount=rf.Number, merchant=rf.Category),
            label=rf.Boolean(mask=True),
        )
    )
    old.preset = previous
    torch.manual_seed(901)
    current = Model(preset=CONTROL.name)
    assert current.schema.model_dump() == old.schema.model_dump()
    torch.testing.assert_close(current.state_dict(), old.state_dict(), rtol=0, atol=0)

    for model in (old, current):
        torch.manual_seed(902)
        model.extend(rf.where("address") == "/items", risk=rf.Number(mask=True))
    assert current.schema.model_dump() == old.schema.model_dump()
    torch.testing.assert_close(current.state_dict(), old.state_dict(), rtol=0, atol=0)
    assert current.schema.requests["/items/risk"].n_heads == 2


def test_repeated_class_construction_owns_independent_state():
    from policies import CONTROL

    class Model(rf.Model):
        merchant = rf.Category(topk=[2])
        label = rf.Boolean(mask=True)

    first = Model(preset=CONTROL.name)
    second = Model(preset=CONTROL.name)
    first.schema.requests["/merchant"].topk.append(3)
    assert second.schema.requests["/merchant"].topk == [2]
    assert CONTROL.leaf.n_heads == 2
    assert next(first.parameters()).data_ptr() != next(second.parameters()).data_ptr()
