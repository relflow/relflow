from pprint import pformat
from typing import Literal

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
