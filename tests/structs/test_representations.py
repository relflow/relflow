from pprint import pformat
from typing import Literal

import pyarrow as pa
import torch

import relflow as rf


def test_schema_representations_include_extension_options_without_parent_cycles() -> None:
    extension = rf.Extension(name="representation", types=(float,))
    try:

        @extension.register(rf.Component.Request)
        class Request(rf.RequestBase):
            type: Literal["representation"] = "representation"
            units: str = "USD"

        schema = rf.Schema.from_tree(
            d_model=8,
            n_layers=1,
            n_heads=2,
            transactions=rf.Branch(length=4, amount=Request(units="EUR")),
        )
        field = schema.requests["/transactions/amount"]
        before = schema.model_dump(mode="python")
        assert field.address == "/transactions/amount"
        assert field.heritage

        for value in (schema, schema.fields, field):
            for output in (str(value), repr(value), pformat(value)):
                assert "units='EUR'" in output
                assert "type='representation'" in output
                assert "_NodeMixin" not in output

        assert schema.model_dump(mode="python") == before
        assert field.parent is schema.branches["/transactions"]
        assert rf.Schema.model_validate_json(schema.model_dump_json()).model_dump() == before
    finally:
        rf.TENSORFIELDS.pop("representation", None)


def test_schema_selection_returns_independent_plain_lists() -> None:
    schema = rf.Schema.from_tree(d_model=8, n_layers=1, n_heads=2, amount=rf.Number)
    predicate = rf.where("name") == "amount"
    selected = schema.select(predicate)

    assert type(selected) is list
    assert "name='amount'" in pformat(selected)
    selected.clear()

    assert schema.select(predicate) == [schema.requests["/amount"]]


def test_model_and_tensorfield_use_framework_representations() -> None:
    model = rf.Model(amount=rf.Number, label=rf.Boolean(mask=True), d_model=8, n_layers=1, n_heads=2)
    field = model.encode(pa.Table.from_pylist([{"amount": 1.0, "label": True}]), strata=rf.Strata.train)["/amount"]
    state = field.state.clone()

    model_output = repr(model)
    assert model_output.startswith("Model(")
    assert "(nodes): ModuleDict(" in model_output
    assert "(/amount): NodeModule(" in model_output
    assert str(model) == model_output

    for output in (str(field), repr(field), pformat(field)):
        assert output.startswith("TensorField(")
        assert "content=Tensor(" in output
        assert "state=Tensor(" in output
        assert "dtype=torch.int64" in output
        assert "counts V=" not in output
    torch.testing.assert_close(field.state, state)

    for value in (model, model.schema, model.schema.fields, model.schema.requests["/amount"], field):
        for hook in ("__rich_console__", "_repr_html_", "_repr_mimebundle_", "_mime_"):
            assert not hasattr(value, hook)
