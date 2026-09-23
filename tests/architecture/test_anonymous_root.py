"""Anonymous roots preserve graph behavior without a user-supplied namespace."""

import inspect

import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.structs.tree import Node


def test_root_and_reserved_child_names_have_distinct_absolute_addresses():
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        name=rf.Number,
        keys=rf.Number,
        forward=rf.Number,
        record=rf.Number,
        root=rf.Number,
        items=rf.Branch(length=2, name=rf.Number),
    )

    assert "name" not in inspect.signature(rf.Model).parameters
    assert "name" not in inspect.signature(rf.Schema.from_tree).parameters
    assert model.schema.fields.name is None
    assert model.schema.address == ""
    assert model.schema.fields.address == "/"
    assert set(model.nodes) == {"/", "/name", "/keys", "/forward", "/record", "/root", "/items", "/items/name"}
    assert model.schema.shapes["/items/name"] == (1, 2)
    assert model.schema.requests["/items/name"].heritage == ["/", "/items", "/items/name"]
    address = rf.Address("items", "name")
    assert model.select(rf.where("address") == address) == [model.schema.requests[address]]
    assert model.nodes[address] is model.nodes["/items/name"]
    assert model.select(rf.where("address") == rf.Address()) == [model.schema.fields]
    assert model.select(rf.where("parent") == "/items") == [model.schema.requests["/items/name"]]
    assert model.select(rf.where("address") == "/") == [model.schema.fields]
    assert model.select(rf.where("address") == "/", include_root=False) == []
    assert all(node.parent is model.schema.fields for node in model.select(rf.where("parent") == "/"))


@pytest.mark.parametrize("constructor", [rf.Model, rf.Schema.from_tree])
def test_root_name_is_not_a_constructor_option(constructor):
    with pytest.raises(TypeError, match="tree field 'name'.*got str"):
        constructor(d_model=8, n_layers=1, n_heads=2, name="record", amount=rf.Number)


def test_schema_serializes_only_child_names():
    schema = rf.Schema.from_tree(
        d_model=8,
        n_layers=1,
        n_heads=2,
        embed=True,
        query="payload",
        mask=rf.Mask(query="selected", reconstruct=True),
        items=rf.Branch(length=2, value=rf.Number),
    )
    payload = schema.model_dump(mode="json", round_trip=True)
    assert "name" not in payload["fields"]
    assert payload["fields"]["fields"][0]["name"] == "items"
    restored = rf.Schema.model_validate_json(schema.model_dump_json(round_trip=True))
    assert restored.model_dump(mode="json") == payload
    assert restored.fields.name is None
    assert restored.embed == ["/"]
    assert restored.requests["/items/value"].heritage == ["/", "/items", "/items/value"]


def test_binding_invalidates_addresses_cached_on_unbound_nodes():
    branch = rf.Branch(items=rf.Branch(length=2, value=rf.Number))
    assert branch.address == ""
    assert branch.fields[0].address == ""
    assert branch.fields[0].fields[0].address == ""
    schema = rf.Schema(d_model=8, fields=branch)
    assert schema.fields.address == "/"
    assert list(schema.branches) == ["/", "/items"]
    assert list(schema.requests) == ["/items/value"]


def test_anonymous_root_options_can_be_updated_and_temporarily_overridden():
    model = rf.Model(d_model=8, n_layers=1, n_heads=2, amount=rf.Number)
    root = rf.where("address") == "/"
    model.update(root, embed=True, description="Whole observation")
    assert model.schema.embed == ["/"]
    assert model.schema.fields.description == "Whole observation"
    with model.override(root, embed=False):
        assert model.schema.embed == []
        assert model.schema.fields.name is None
    assert model.schema.embed == ["/"]


@pytest.mark.parametrize("validate", [True, False])
def test_root_cannot_be_named_by_mutation_and_failed_edit_is_atomic(validate):
    model = rf.Model(d_model=8, n_layers=1, n_heads=2, amount=rf.Number)
    before = model.schema.model_dump()
    nodes = model.nodes
    with pytest.raises(ValueError, match="model root must be anonymous"):
        model.update(rf.where("address") == "/", name="event", validate=validate)
    assert model.schema.model_dump() == before
    assert model.nodes is nodes
    assert model.schema.fields.name is None
    assert model.schema.fields.address == "/"


def test_root_extension_and_deletion_keep_surviving_state():
    model = rf.Model(d_model=8, n_layers=1, n_heads=2, items=rf.Branch(value=rf.Number), keep=rf.Number)
    before = {name: value.clone() for name, value in model.nodes["/keep"].state_dict().items()}
    model.extend(rf.where("address") == "/", added=rf.Number)
    model.delete(rf.where("address") == "/items")
    model.delete(rf.where("address") == "/added")
    assert set(model.nodes) == {"/", "/keep"}
    for name, value in before.items():
        torch.testing.assert_close(model.nodes["/keep"].state_dict()[name], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="cannot remove the root"):
        model.delete(rf.where("address") == "/", include_root=True)
    with pytest.raises(ValueError, match="would remove every request"):
        model.delete(rf.where("address") == "/keep")


@pytest.mark.parametrize("root_name", ["record", "custom"])
def test_named_root_checkpoint_is_rejected_without_changing_the_live_model(tmp_path, root_name):
    model = rf.Model(d_model=8, n_layers=1, n_heads=2, amount=rf.Number)
    checkpoint = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(checkpoint)
    checkpoint["schema"]["fields"]["name"] = root_name
    path = tmp_path / "named.ckpt"
    torch.save(checkpoint, path)
    nodes = model.nodes
    schema = model.schema

    with pytest.raises(ValueError, match="model root must be anonymous.*named-root checkpoints"):
        rf.Model.load(path)
    with pytest.raises(ValueError, match="model root must be anonymous"):
        model.restore_checkpoint_state(checkpoint)
    assert model.nodes is nodes
    assert model.schema is schema
    assert model.schema.fields.name is None


def test_root_query_prediction_envelope_and_checkpoint_round_trip(tmp_path):
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        query="payload",
        embed=True,
        items=rf.Branch(length=2, value=rf.Number(mask=True)),
    )
    table = pa.Table.from_pylist([{"payload": {"items": [{"value": 3.0}, {"value": 5.0}]}}])
    expected = model.predict(table)
    predictions = expected["predictions"].combine_chunks()
    assert [field.name for field in predictions.type] == ["/", "/items/value"]
    assert len(predictions.field("/").field("embedding")[0]) == 8
    assert len(predictions.field("/items/value")[0]) == 2
    restored = rf.Model.load(model.save(tmp_path / "anonymous.ckpt"))
    assert restored.predict(table).equals(expected)
    assert set(restored.nodes) == {"/", "/items", "/items/value"}


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (("/Amount", "validate", "loss"), ".Amount/validate.loss"),
        (("/items/Amount", "validate", "loss"), ".items.Amount/validate.loss"),
        (("/items/Amount/", "Validate", "Loss/Content"), ".items.Amount/validate.loss.content"),
        ((rf.Address("returned"), "validate", "loss", "state"), ".returned/validate.loss.state"),
        (
            (rf.Address("carriers", "cxr"), rf.Strata.validate, "accuracy", "content"),
            ".carriers.cxr/validate.accuracy.content",
        ),
        ((rf.Address(), "validate", "loss"), "./validate.loss"),
        ((rf.Address("loss"), "train"), ".loss/train"),
        (("loss", "train"), "loss/train"),
        ((rf.Address("throughput"), "predict"), ".throughput/predict"),
        (("throughput", "predict"), "throughput/predict"),
    ],
)
def test_metric_names_use_dotted_addresses_and_preserve_case(monkeypatch, names, expected):
    logged = []
    monkeypatch.setattr(rf.Model, "log", lambda self, **kwargs: logged.append(kwargs["name"]))
    model = rf.Model(d_model=8, n_layers=1, n_heads=2, Amount=rf.Number)
    model.track(names, value=torch.tensor(1.0))
    assert logged == [expected]


def test_absolute_routing_preserves_forward_values_and_gradients(monkeypatch):
    """Compare identical weights with an explicit test-only root namespace."""

    def run():
        torch.manual_seed(418)
        model = rf.Model(
            d_model=8,
            n_layers=1,
            n_heads=2,
            embed=True,
            amount=rf.Number,
            items=rf.Branch(length=2, embed=True, value=rf.Number),
            answer=rf.Number(mask=True),
        ).eval()
        inputs = model.encode(
            pa.Table.from_pylist([{"amount": 2.0, "items": [{"value": 3.0}, {"value": 5.0}], "answer": 4.0}]),
            strata="predict",
        )
        predictions = model(inputs, strata=rf.Strata.predict)
        objective = sum(
            value.square().sum()
            for prediction in predictions
            for value in prediction.payload.values()
            if isinstance(value, torch.Tensor) and value.requires_grad
        )
        objective.backward()
        outputs = [
            {key: value.detach().clone() for key, value in prediction.payload.items()} for prediction in predictions
        ]
        gradients = [None if value.grad is None else value.grad.clone() for value in model.parameters()]
        return outputs, gradients

    def named_address(node):
        if node.parent is None or node.root.type != "schema":
            return rf.Address("")
        return rf.Address("reference", *(part.name for part in node.path[2:]))

    with monkeypatch.context() as reference:
        reference.setattr(Node, "address", property(named_address))
        old_outputs, old_gradients = run()
    outputs, gradients = run()

    assert len(outputs) == len(old_outputs) == 3
    for actual, expected in zip(outputs, old_outputs, strict=True):
        assert actual.keys() == expected.keys()
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert any(gradient is not None and gradient.count_nonzero() for gradient in gradients)
    for actual, expected in zip(gradients, old_gradients, strict=True):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
