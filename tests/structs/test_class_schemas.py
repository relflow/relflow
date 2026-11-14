import pickle
from typing import Literal

import pyarrow as pa
import pytest
import torch

import relflow as rf


class Transaction(rf.Branch):
    amount = rf.Number
    merchant = rf.Category(topk=[2, 3])


class FraudDetector(rf.Model):
    account = rf.Hash
    transactions = Transaction(length=4, overflow="tail")
    fraudulent = rf.Boolean(mask=True)


def test_class_declarations_build_the_same_schema_as_constructor_fields():
    declared = FraudDetector(d_model=16, n_layers=1, n_heads=2)
    ordinary = rf.Model(
        d_model=16,
        n_layers=1,
        n_heads=2,
        account=rf.Hash,
        transactions=rf.Branch(length=4, overflow="tail", amount=rf.Number, merchant=rf.Category(topk=[2, 3])),
        fraudulent=rf.Boolean(mask=True),
    )

    assert declared.schema.model_dump() == ordinary.schema.model_dump()
    assert type(declared.schema.branches["/transactions"]) is rf.Branch
    assert declared.schema.reconstruct == ["/fraudulent"]
    assert declared.schema.fields.name is None
    assert declared.schema.fields.length == 1
    assert list(Transaction.model_fields) == list(rf.Branch.model_fields)


@pytest.mark.parametrize("profile", ["small", "medium"])
def test_preset_factories_resolve_declared_children(profile):
    model = FraudDetector(preset=profile, d_model=16, n_layers=1, n_heads=2)
    ordinary = rf.Model(
        preset=profile,
        d_model=16,
        n_layers=1,
        n_heads=2,
        account=rf.Hash,
        transactions=rf.Branch(length=4, overflow="tail", amount=rf.Number, merchant=rf.Category(topk=[2, 3])),
        fraudulent=rf.Boolean(mask=True),
    )

    assert isinstance(model, FraudDetector)
    assert model.schema.model_dump() == ordinary.schema.model_dump()
    assert model.preset.name == profile
    assert list(model.schema.requests) == ["/account", "/transactions/amount", "/transactions/merchant", "/fraudulent"]
    assert model.schema.branches["/transactions"].n_heads == model.preset.branch.n_heads
    assert model.schema.requests["/transactions/merchant"].topk == [2, 3]


def test_child_names_are_separate_from_configuration_and_runtime_attributes():
    class CollisionBranch(rf.Branch):
        length = rf.Number
        fields = rf.Category
        type = rf.Category
        name = rf.Category
        description = rf.Category

    class CollisionModel(rf.Model):
        d_model = rf.Number
        schema = rf.Category
        predict = rf.Category
        nodes = rf.Category
        items = CollisionBranch(length=3, description="collection options")
        result = rf.Boolean(mask=True)

    model = CollisionModel(d_model=16, n_layers=1, n_heads=2, description="root options")

    assert model.schema.d_model == 16
    assert model.schema.fields.description == "root options"
    assert model.schema.branches["/items"].length == 3
    assert model.schema.branches["/items"].description == "collection options"
    assert "/d_model" in model.schema.requests
    assert "/schema" in model.schema.requests
    assert "/predict" in model.schema.requests
    assert [child.name for child in model.schema.branches["/items"].fields] == [
        "length",
        "fields",
        "type",
        "name",
        "description",
    ]
    assert callable(model.predict)
    assert isinstance(model.nodes, torch.nn.ModuleDict)


def test_inheritance_replaces_whole_children_and_keeps_declaration_order():
    class InternationalTransaction(Transaction):
        merchant = rf.Hash
        country = rf.Category

    class InternationalFraud(FraudDetector):
        transactions = InternationalTransaction(length=2)
        device = rf.Hash

        def label(self):
            return "international"

    branch = InternationalTransaction(length=8)
    model = InternationalFraud(preset="small", d_model=16, n_layers=1, n_heads=2)

    assert [field.name for field in branch.fields] == ["amount", "merchant", "country"]
    assert branch.fields[1].type == "hash"
    assert [field.name for field in model.schema.fields.fields] == ["account", "transactions", "fraudulent", "device"]
    assert model.label() == "international"
    assert model.schema.branches["/transactions"].length == 2
    assert Transaction(length=1).fields[1].type == "category"
    assert FraudDetector(preset="small", d_model=16, n_layers=1, n_heads=2).schema.branches["/transactions"].length == 4


def test_repeated_bindings_and_model_instances_own_independent_nodes_and_state():
    class Comparison(rf.Model):
        recent = Transaction(length=2)
        historical = Transaction(length=8)
        result = rf.Boolean(mask=True)

    first = Comparison(d_model=16, n_layers=1, n_heads=2)
    second = Comparison(d_model=16, n_layers=1, n_heads=2)
    recent = first.schema.requests["/recent/merchant"]
    historical = first.schema.requests["/historical/merchant"]
    recent.topk.append(4)

    assert recent.parent is first.schema.branches["/recent"]
    assert historical.parent is first.schema.branches["/historical"]
    assert historical.topk == [2, 3]
    assert second.schema.requests["/recent/merchant"].topk == [2, 3]
    assert Transaction(length=1).fields[1].topk == [2, 3]
    assert next(first.parameters()).data_ptr() != next(second.parameters()).data_ptr()
    assert pickle.loads(pickle.dumps(first.schema)).model_dump() == first.schema.model_dump()


def test_nested_declarations_lower_to_portable_branches():
    class Item(rf.Branch):
        sku = rf.Hash
        quantity = rf.Number

    class Basket(Transaction):
        items = Item(length=3, overflow="tail")

    class NestedModel(rf.Model):
        baskets = Basket(length=2)
        result = rf.Boolean(mask=True)

    model = NestedModel(d_model=16, n_layers=1, n_heads=2)
    schema = rf.Schema.model_validate_json(model.schema.model_dump_json())

    assert schema.model_dump() == model.schema.model_dump()
    assert schema.shapes["/baskets/items/quantity"] == (1, 2, 3)
    assert all(type(branch) is rf.Branch for branch in model.schema.branches.values())
    assert schema.requests["/baskets/items/sku"].parent is schema.branches["/baskets/items"]


@pytest.mark.parametrize("preset", [False, True])
def test_bare_branch_classes_match_empty_calls_and_preset_defaults(preset):
    class BareModel(rf.Model):
        transactions = Transaction
        result = rf.Boolean(mask=True)

    class CalledModel(rf.Model):
        transactions = Transaction()
        result = rf.Boolean(mask=True)

    options = {"d_model": 16, "n_layers": 1, "n_heads": 2}
    bare = BareModel(preset="small", **options) if preset else BareModel(**options)
    called = CalledModel(preset="small", **options) if preset else CalledModel(**options)

    assert bare.schema.model_dump() == called.schema.model_dump()
    assert bare.schema.branches["/transactions"].length is None
    assert bare.schema.branches["/transactions"].overflow is None
    assert rf.presets.SMALL.defaults(Transaction) == rf.presets.SMALL.defaults(Transaction())


def test_bare_branch_classes_bind_in_keywords_mappings_and_mutations():
    nested = rf.Branch(length=2, fields={"history": Transaction})
    model = rf.Model(
        preset="small", d_model=16, histories=nested, transactions=Transaction, result=rf.Boolean(mask=True)
    )
    model.extend(rf.where("address") == "/", extra=Transaction)

    assert "/histories/history/merchant" in model.schema.requests
    assert "/transactions/merchant" in model.schema.requests
    assert "/extra/merchant" in model.schema.requests
    assert model.schema.branches["/extra"].n_heads == model.preset.branch.n_heads
    assert all(type(branch) is rf.Branch for branch in model.schema.branches.values())


def test_nested_bare_branches_are_independent_and_checkpoint_portable(tmp_path):
    class Basket(rf.Branch):
        transactions = Transaction

    class ExtendedBasket(Basket):
        region = rf.Category

    class Comparison(rf.Model):
        first = ExtendedBasket
        second = ExtendedBasket
        bounded = Basket(length=3, overflow="tail")
        result = rf.Boolean(mask=True)

    first = Comparison(d_model=16, n_layers=1, n_heads=2)
    second = Comparison(d_model=16, n_layers=1, n_heads=2)
    first.schema.requests["/first/transactions/merchant"].topk.append(4)

    assert first.schema.requests["/second/transactions/merchant"].topk == [2, 3]
    assert second.schema.requests["/first/transactions/merchant"].topk == [2, 3]
    assert first.schema.requests["/first/transactions/merchant"].parent is first.schema.branches["/first/transactions"]
    assert first.schema.branches["/bounded"].length == 3
    assert first.schema.shapes["/first/transactions/amount"] == (1, None, None)
    restored = rf.Model.load(first.save(tmp_path / "bare-branches.rf"))
    assert restored.schema.model_dump() == first.schema.model_dump()


@pytest.mark.parametrize(
    "construct",
    [
        lambda: Transaction(length=2, amount=rf.Number),
        lambda: Transaction(length=2, extra=rf.Number),
        lambda: Transaction(length=2, fields={"amount": rf.Number}),
        lambda: FraudDetector(d_model=16, n_layers=1, n_heads=2, account=rf.Hash),
        lambda: FraudDetector(preset="small", extra=rf.Number),
        lambda: FraudDetector(preset="small", fields={"account": rf.Hash}),
    ],
)
def test_declarative_constructors_reject_child_definitions(construct):
    with pytest.raises(TypeError, match="constructor arguments configure the node"):
        construct()


def test_multiple_inheritance_requires_explicit_resolution_of_conflicting_children():
    class Left(Transaction):
        merchant = rf.Hash

    class Right(Transaction):
        merchant = rf.Category(cased=False)

    with pytest.raises(TypeError, match="conflicting inherited children"):

        class Conflict(Left, Right):
            pass

    class Resolved(Left, Right):
        merchant = rf.Category

    class Unchanged(Transaction):
        pass

    class AlsoUnchanged(Transaction):
        pass

    class Diamond(Unchanged, AlsoUnchanged):
        pass

    assert Resolved(length=1).fields[1].type == "category"
    assert Diamond(length=1).fields[1].type == "category"


def test_invalid_class_declarations_identify_the_child():
    with pytest.raises(TypeError, match="Protocol.__call__.*Python protocol name"):

        class Protocol(rf.Model):
            __call__ = Transaction

    with pytest.raises(TypeError, match="TransactionReplacement.amount must replace an inherited child"):

        class TransactionReplacement(Transaction):
            amount = 12


@pytest.mark.parametrize("bare", [False, True])
def test_late_registered_extensions_use_the_same_declaration_path(bare):
    extension = rf.Extension(name="class_schema_custom", types=(float,))
    try:

        @extension.register(rf.Component.Request)
        class Custom(rf.RequestBase):
            type: Literal["class_schema_custom"] = "class_schema_custom"
            scale: float = 3.0

        class CustomBranch(rf.Branch):
            default = Custom
            configured = Custom(scale=5)

        schema = rf.Schema.from_tree(
            d_model=16, n_layers=1, n_heads=2, items=CustomBranch if bare else CustomBranch(length=2)
        )
        restored = rf.Schema.model_validate_json(schema.model_dump_json())

        assert isinstance(restored.requests["/items/default"], Custom)
        assert restored.requests["/items/default"].scale == 3
        assert restored.requests["/items/configured"].scale == 5
    finally:
        rf.TENSORFIELDS.pop(extension.name, None)


def test_saved_schema_is_authoritative_for_base_and_subclass_loading(tmp_path):
    model = FraudDetector(d_model=16, n_layers=1, n_heads=2)
    model.extend(rf.where("address") == "/", extra=rf.Number)
    table = pa.Table.from_pylist([{"account": "a", "transactions": [{"amount": 12, "merchant": "shop"}], "extra": 1}])
    expected = model.predict(table)
    checkpoint = model.save(tmp_path / "fraud.rf")

    for cls in (rf.Model, FraudDetector):
        restored = cls.load(checkpoint)
        assert isinstance(restored, cls)
        assert restored.schema.model_dump() == model.schema.model_dump()
        assert restored.predict(table).equals(expected)


@pytest.mark.parametrize("preset", [False, True])
def test_yaml_schema_is_authoritative_for_a_declarative_subclass(tmp_path, preset):
    path = tmp_path / "different.yaml"
    dimensions = "preset: small\n" if preset else ""
    path.write_text(
        dimensions
        + "d_model: 16\nn_layers: 1\nn_heads: 2\n"
        + "fields:\n  different:\n    type: number\n    mask: true\n"
    )
    optimizer = rf.adamw(learning_rate=1e-3)
    model = FraudDetector.from_yaml(path, batch_size=7, optimizer=optimizer)

    assert isinstance(model, FraudDetector)
    assert list(model.schema.requests) == ["/different"]
    assert model.batch_size == 7
    assert model.optimizer is optimizer
    assert model.preset == (rf.presets.SMALL if preset else None)


@pytest.mark.integration
def test_declared_model_trains_and_predicts_nested_arrow_records(tmp_path):
    model = FraudDetector(d_model=16, n_layers=1, n_heads=2)
    table = pa.Table.from_pylist(
        [
            {"account": "a", "transactions": [{"amount": 12, "merchant": "shop"}], "fraudulent": False},
            {"account": "b", "transactions": [], "fraudulent": True},
        ]
    )
    data = rf.ArrowDataModule(
        model=model,
        train=table,
        validate=table,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    model.fit(
        data,
        batch_size=2,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        rollback=False,
        early_stopping=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )

    predictions = model.predict(table)
    assert predictions.height == table.num_rows
    assert "fraudulent" in predictions["predictions"].struct.fields
