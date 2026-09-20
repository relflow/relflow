"""Schema options are declared configuration; descriptive notes use description."""

from typing import Literal

import pydantic
import pytest

import relflow as rf

REQUESTS = [
    (rf.Boolean, {}),
    (rf.Category, {}),
    (rf.Cluster, {"bounds": 4}),
    (rf.DateParts, {"dateparts": ["day_of_week"]}),
    (rf.Hash, {}),
    (rf.Number, {}),
    (rf.Set, {}),
    (rf.Text, {}),
    (rf.Vector, {"n_dim": 4}),
]


@pytest.mark.parametrize("constructor, required", REQUESTS, ids=lambda value: getattr(value, "__module__", None))
@pytest.mark.parametrize("option", ["notes", "n_heeds"])
def test_tensorfields_reject_undeclared_options(constructor, required, option):
    with pytest.raises(pydantic.ValidationError, match=option) as error:
        constructor(**required, **{option: "unexpected"})
    assert error.value.errors()[0]["type"] == "extra_forbidden"
    with pytest.raises(pydantic.ValidationError, match=option) as error:
        constructor.model_validate({"name": "value", **required, option: "unexpected"})
    assert error.value.errors()[0]["type"] == "extra_forbidden"


def test_declared_options_and_descriptions_survive_checkpoint_round_trip(tmp_path):
    options = {"n_bands": 4, "description": "  Amount in dollars  "}
    model = rf.Model(d_model=16, n_layers=1, n_heads=4, amount=rf.Number(**options))
    model.update(rf.where("name") == "amount", description="  Amount in USD  ")
    restored = rf.Model.load(model.save(tmp_path / "model.ckpt"))
    amount = restored.schema.requests["/amount"]
    assert amount.description == "Amount in USD"
    assert amount.n_bands == 4
    assert amount.model_extra is None


def test_declared_extension_options_remain_supported():
    extension = rf.Extension(name="declared_options", types=(float,))
    try:

        @extension.register
        class Request(rf.RequestBase):
            type: Literal["declared_options"] = "declared_options"
            scale: float = 1.0

        field = Request(scale=2.0, description="Custom units")
        assert field.scale == 2.0
        assert Request.model_validate(field.model_dump()).description == "Custom units"
        with pytest.raises(pydantic.ValidationError, match="note"):
            Request(scale=2.0, note="Custom units")
    finally:
        rf.TENSORFIELDS.pop("declared_options", None)


def test_checkpoint_loading_rejects_undeclared_field_options(tmp_path):
    import torch

    model = rf.Model(d_model=16, n_layers=1, n_heads=4, amount=rf.Number)
    path = model.save(tmp_path / "model.ckpt")
    state = torch.load(path, weights_only=False)
    state["schema"]["fields"]["fields"][0]["notes"] = "Undeclared option"
    torch.save(state, path)
    with pytest.raises(ValueError, match="notes"):
        rf.Model.load(path)


@pytest.mark.parametrize("schema_only", [False, True])
def test_mutations_reject_metadata_even_when_validation_is_disabled(schema_only):
    model = rf.Model(d_model=16, n_layers=1, n_heads=4, amount=rf.Number)
    target = model.schema if schema_only else model
    selector = rf.where("name") == "amount"
    before = model.schema.model_dump()
    for strict in (False, True):
        with pytest.raises(AttributeError, match="notes"):
            target.update(selector, notes="Undeclared option", strict=strict, validate=False)
        with pytest.raises(AttributeError, match="notes"):
            with target.override(selector, notes="Undeclared option", strict=strict, validate=False):
                pass
    assert model.schema.model_dump() == before


@pytest.mark.parametrize("schema_only", [False, True])
@pytest.mark.parametrize(
    "field, attribute, value, canonical",
    [
        (rf.Category(), "topk", [2, 100], "topk"),
        (rf.Set(), "threshold", 0.8, "threshold"),
        (rf.Cluster(bounds=4), "n_clusters", (4, 8), "n_clusters"),
    ],
)
def test_mutations_accept_declared_options_with_serialization_aliases(schema_only, field, attribute, value, canonical):
    model = rf.Model(d_model=16, n_layers=1, n_heads=4, value=field)
    target = model.schema if schema_only else model
    selector = rf.where("name") == "value"
    original = getattr(model.schema.requests["/value"], canonical)
    with target.override(selector, **{attribute: value}):
        assert getattr(model.schema.requests["/value"], canonical) == value
    assert getattr(model.schema.requests["/value"], canonical) == original
    target.update(selector, **{attribute: value})
    assert getattr(model.schema.requests["/value"], canonical) == value


def test_partial_updates_still_accept_options_supported_by_some_selected_nodes():
    model = rf.Model(d_model=16, n_layers=1, n_heads=4, amount=rf.Number, label=rf.Category())
    model.update(n_bands=4, strict=False)
    assert model.schema.requests["/amount"].n_bands == 4
    assert not hasattr(model.schema.requests["/label"], "n_bands")
