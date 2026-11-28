import pickle
import subprocess
import sys
import uuid
from decimal import Decimal
from typing import Annotated

import pyarrow as pa
import pydantic
import pytest
import torch
import yaml
from lightning.pytorch import Callback

import relflow as rf


@pytest.fixture
def inherited():
    names = []

    def make(parent=rf.Number, **options):
        name = f"test_inherit_{uuid.uuid4().hex}"
        extension = rf.Extension.inherit(parent, name=name, **options)
        names.append(name)
        return extension

    yield make

    for name in names:
        rf.TENSORFIELDS.pop(name, None)


def test_inherited_number_encodes_predicts_and_keeps_field_and_model_state_separate(inherited):
    extension = inherited()
    Measurement = extension.Request
    records = pa.table({"first": [1.0, 3.0], "second": [10.0, 30.0], "target": [2.0, 4.0]})
    model = rf.Model(preset="small", d_model=8, first=Measurement, second=Measurement, target=Measurement(predict=True))
    other = rf.Model(preset="small", d_model=8, first=Measurement)

    fields = model.encode(records, strata="train")
    other.encode(pa.table({"first": [100.0, 200.0]}), strata="train")

    assert extension.Request().type == extension.name
    assert fields["/first"].content.flatten().tolist() == [1.0, 3.0]
    assert fields["/target"].targets[rf.TensorKey.content].flatten().tolist() == [2.0, 4.0]
    assert rf.Number.normalization(model, "/first")["mean"] == 2.0
    assert rf.Number.normalization(model, "/second")["mean"] == 20.0
    assert rf.Number.normalization(other, "/first")["mean"] == 150.0
    first = model.nodes[rf.Address("first")].embedder
    second = model.nodes[rf.Address("second")].embedder
    assert first is not second and first is not other.nodes[rf.Address("first")].embedder
    assert next(first.parameters()).data_ptr() != next(second.parameters()).data_ptr()
    output = model.predict(records)
    assert len(output) == 2
    assert "target" in output["predictions"].struct.fields


def test_inherited_request_options_and_component_overrides_do_not_change_number_or_siblings(inherited, tmp_path):
    child = inherited()
    sibling = inherited()
    parent = rf.TENSORFIELDS["number"]

    @child.register(rf.Component.Request)
    class Measurement(child.Request):
        limit: Annotated[int, pydantic.Field(gt=0)] = 10

        @pydantic.field_validator("limit")
        @classmethod
        def check_limit(cls, value):
            if value % 2:
                raise ValueError("limit must be even")
            return value

    @child.register(rf.Component.loss)
    def weighted_loss(module, prediction, batch, strata):
        return parent.loss(module, prediction, batch, strata) * 2

    assert Measurement().type == child.name
    assert Measurement(n_bands=3, limit=12).limit == 12
    with pytest.raises(ValueError, match="limit must be even"):
        Measurement(limit=3)
    with pytest.raises(ValueError, match="n_bands"):
        Measurement(n_bands=0)
    with pytest.raises(ValueError, match="limit"):
        rf.Number(limit=12)
    assert child.loss is weighted_loss
    assert sibling.loss is parent.loss
    with pytest.raises(ValueError, match="already registered"):
        child.register(rf.Component.loss, parent.loss)
    with pytest.raises(ValueError, match="already registered"):
        child.register(rf.Component.Request, Measurement)

    model = rf.Model(preset="small", d_model=8, value=Measurement(limit=12, n_bands=3))
    payload = model.schema.requests[rf.Address("value")].model_dump(mode="json", exclude={"name"})
    path = tmp_path / "specialized.yaml"
    path.write_text(yaml.safe_dump({"preset": "small", "d_model": 8, "fields": {"value": payload}}))
    checkpoint = tmp_path / "specialized.ckpt"
    model.save(checkpoint)
    schema = rf.Schema.model_validate_json(model.schema.model_dump_json())
    for restored in (schema, rf.Model.from_yaml(path).schema, rf.Model.load(checkpoint).schema):
        request = restored.requests[rf.Address("value")]
        assert type(request) is Measurement
        assert (request.type, request.limit, request.n_bands) == (child.name, 12, 3)

    payload["limit"] = 3
    path.write_text(yaml.safe_dump({"preset": "small", "d_model": 8, "fields": {"value": payload}}))
    with pytest.raises(ValueError, match="limit must be even"):
        rf.Model.from_yaml(path)


def test_inherited_request_serializes_through_schema_yaml_checkpoint_and_pickle(inherited, tmp_path):
    extension = inherited()
    Measurement = extension.Request
    original = rf.Model(preset="small", d_model=8, value=Measurement(n_bands=3, alpha=0.5, query="source"))
    records = pa.table({"source": [1.0, 2.0, 3.0]})
    original.encode(records, strata="train")
    schema = rf.Schema.model_validate_json(original.schema.model_dump_json())
    assert type(schema.requests[rf.Address("value")]) is Measurement
    assert pickle.loads(pickle.dumps(Measurement(alpha=0.5))).type == extension.name

    path = tmp_path / "model.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "preset": "small",
                "d_model": 8,
                "fields": {
                    "value": original.schema.requests[rf.Address("value")].model_dump(mode="json", exclude={"name"})
                },
            }
        )
    )
    yaml_model = rf.Model.from_yaml(path)
    checkpoint = tmp_path / "model.ckpt"
    original.save(checkpoint)
    checkpoint_model = rf.Model.load(checkpoint)
    for restored in (yaml_model, checkpoint_model):
        request = restored.schema.requests[rf.Address("value")]
        assert type(request) is Measurement
        assert (request.type, request.n_bands, request.alpha, request.query) == (extension.name, 3, 0.5, ("source",))
        torch.testing.assert_close(
            original.encode(records)["/value"].content, restored.encode(records)["/value"].content
        )
    assert Measurement.normalization(checkpoint_model, "/value") == Measurement.normalization(original, "/value")


def test_inheritance_chain_uses_late_third_party_components_and_independent_overrides(inherited):
    class First(Callback):
        pass

    class Second(Callback):
        pass

    name = f"test_parent_{uuid.uuid4().hex}"
    parent = rf.Extension(
        name=name,
        types=(Decimal | int,),
        arrow={Decimal: pa.types.is_decimal},
        callbacks=(First,),
        requires={"decimal": "decimal-parent"},
        queries=(1, 2),
    )
    try:

        @parent.register(rf.Component.Request)
        class Parent(rf.RequestBase):
            type: str = name
            precision: Annotated[int, pydantic.Field(gt=0)] = 5

        child = inherited(
            Parent, types=(Decimal,), queries=slice(1, 4), callbacks=(Second,), requires={"json": "json-child"}
        )
        grandchild = inherited(child.Request)
        sibling = inherited(Parent)

        def original_loss(*args, **kwargs):
            return 1

        def child_loss(*args, **kwargs):
            return 2

        parent.register(rf.Component.loss, original_loss)
        assert child.loss is original_loss and grandchild.loss is original_loss
        child.register(rf.Component.loss, child_loss)
        assert child.loss is child_loss and grandchild.loss is child_loss
        assert parent.loss is original_loss and sibling.loss is original_loss
        assert grandchild.Request().type == grandchild.name
        assert grandchild.Request().precision == 5
        with pytest.raises(ValueError, match="precision"):
            grandchild.Request(precision=0)
        assert child.types == (Decimal,) and child.accepts(pa.decimal128(12, 2))
        assert not child.accepts(pa.int64()) and sibling.accepts(pa.int64())
        assert child.queries == slice(1, 4) and grandchild.queries == slice(1, 4)
        assert parent.queries == (1, 2)
        assert child.requires == {"decimal": "decimal-parent", "json": "json-child"}
        assert parent.requires == {"decimal": "decimal-parent"}
        assert grandchild.callback_factories == (First, Second)
        assert type(child.callbacks[0]) is First and child.callbacks[0] is not child.callbacks[0]
        assert sibling.callback_factories == (First,)
    finally:
        rf.TENSORFIELDS.pop(name, None)


def test_optional_roles_can_be_disabled_without_changing_parent_or_siblings(inherited):
    child = inherited()
    sibling = inherited()
    parent = rf.TENSORFIELDS["number"]
    child.register(rf.Component.output, None)
    child.register(rf.Component.write, None)
    assert child.components[rf.Component.output] is None
    assert child.components[rf.Component.write] is None
    assert child.output(object(), rf.Address("value")) is None
    assert child.write(object(), object(), None) is None
    assert sibling.output is parent.output and sibling.write is parent.write


def test_late_registered_third_party_components_build_a_chained_model(inherited):
    name = f"test_numeric_parent_{uuid.uuid4().hex}"
    parent = rf.Extension(name=name, types=(int | float,))
    try:

        @parent.register(rf.Component.Request)
        class Parent(rf.Number):
            type: str = name

        child = inherited(Parent)
        grandchild = inherited(child.Request)
        number = rf.TENSORFIELDS["number"]
        for role, component in number.components.items():
            if role != rf.Component.Request:
                parent.register(role, component)

        records = pa.table({"input": [1.0, 2.0], "target": [3.0, 4.0]})
        model = rf.Model(preset="small", d_model=8, input=grandchild.Request, target=grandchild.Request(predict=True))
        assert model.encode(records)["/input"].content.flatten().tolist() == [1.0, 2.0]
        assert len(model.predict(records)) == 2
        assert "target" in model.predict(records)["predictions"].struct.fields
    finally:
        rf.TENSORFIELDS.pop(name, None)


def test_inherit_rejects_unknown_parents_and_invalid_request_replacements(inherited):
    with pytest.raises(ValueError, match="registered tensorfield Request"):
        rf.Extension.inherit(rf.RequestBase, name="unregistered_parent")
    child = inherited()

    class Unrelated(rf.RequestBase):
        type: str = "number"

    with pytest.raises(TypeError, match="parent Request"):
        child.register(rf.Component.Request, Unrelated)
    with pytest.raises(ValueError, match="type discriminator"):
        child.register(rf.Component.Request, rf.Number)
    assert child.Request().type == child.name


def test_inherited_callbacks_attach_once_per_model_and_requirements_remain_enforced(inherited):
    class CustomCallback(Callback):
        pass

    parent = inherited(callbacks=(CustomCallback,))
    child = inherited(parent.Request)
    model = rf.Model(preset="small", d_model=8, a=parent.Request, b=child.Request)
    other = rf.Model(preset="small", d_model=8, a=child.Request)
    first = [callback for callback in model.configure_callbacks() if isinstance(callback, CustomCallback)]
    second = [callback for callback in other.configure_callbacks() if isinstance(callback, CustomCallback)]
    assert len(first) == len(second) == 1 and first[0] is not second[0]

    missing = f"missing_package_{uuid.uuid4().hex}"
    optional = inherited(requires={missing: "example-package"})
    dependent = inherited(optional.Request)
    with pytest.raises(ModuleNotFoundError, match="example-package") as error:
        rf.Model(preset="small", d_model=8, value=dependent.Request)
    assert error.value.name == missing


def test_loading_inherited_schema_reports_missing_registration(inherited):
    child = inherited()
    model = rf.Model(preset="small", d_model=8, value=child.Request)
    serialized = model.schema.model_dump_json()
    rf.TENSORFIELDS.pop(child.name)
    with pytest.raises(ValueError, match=f"unknown tensor field type: {child.name}"):
        rf.Schema.model_validate_json(serialized)


@pytest.mark.integration
def test_inherited_request_works_in_spawned_loader_workers(tmp_path):
    (tmp_path / "measurement_plugin.py").write_text(
        "import relflow as rf\n"
        'extension = rf.Extension.inherit(rf.Number, name="test_spawned_measurement")\n'
        "Measurement = extension.Request\n"
    )
    script = tmp_path / "worker_example.py"
    script.write_text(
        "import pyarrow as pa\n"
        "import relflow as rf\n"
        "from measurement_plugin import Measurement\n"
        "\n"
        'if __name__ == "__main__":\n'
        '    model = rf.Model(preset="small", d_model=8, batch_size=2, value=Measurement)\n'
        '    data = rf.ArrowDataModule(model=model, train=pa.table({"value": [1.0, 2.0]}),\n'
        "        num_workers=1, persistent_workers=False, pin_memory=False, shuffle=False)\n"
        "    loader = data.train_dataloader()\n"
        '    assert loader.multiprocessing_context.get_start_method() == "spawn"\n'
        "    batch = next(iter(loader))\n"
        '    assert batch.tensors["/value"].content.flatten().tolist() == [1.0, 2.0]\n'
    )
    process = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.integration
def test_inherited_numeric_target_trains_with_the_child_loss(inherited, tmp_path):
    child = inherited()
    parent = rf.TENSORFIELDS["number"]
    calls = []

    @child.register(rf.Component.loss)
    def loss(module, prediction, batch, strata):
        calls.append((str(prediction.address), strata))
        return parent.loss(module, prediction, batch, strata)

    records = pa.table({"input": [1.0, 2.0, 3.0, 4.0], "target": [2.0, 4.0, 6.0, 8.0]})
    model = rf.Model(preset="small", d_model=8, input=child.Request, target=child.Request(predict=True))
    data = rf.ArrowDataModule(
        train=records, validate=records, num_workers=0, persistent_workers=False, pin_memory=False
    )
    model.fit(
        data,
        batch_size=4,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        rollback=False,
        early_stopping=False,
        enable_progress_bar=False,
        default_root_dir=str(tmp_path),
    )
    assert ("/target", rf.Strata.train) in calls
    assert len(model.predict(records.drop(["target"]))) == 4
