import sys
import uuid
from decimal import Decimal
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from lightning.pytorch import Callback

from relflow.structs.enums import Component, Strata
from relflow.structs.tree import Address, Node
from relflow.tensorfields.base import (
    TENSORFIELDS,
    DecoderBase,
    EmbedderBase,
    Extension,
    RequestBase,
    TensorFieldBase,
)


def extension_name(prefix: str = "extension") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def build_extension() -> Extension:
    extension = Extension(name=extension_name(), types=(object,))

    class Request(RequestBase):
        type: str = extension.name

    class TensorField(TensorFieldBase):
        @classmethod
        def new(cls, input, target, present, trainable, inferred, address, schema, strata, context):
            return object()

    class Embedder(EmbedderBase):
        def __init__(self, schema, address):
            super().__init__(schema=schema, address=address)

        def forward(self, values):
            return values

    class Decoder(DecoderBase):
        pass

    extension.register(Component.Request, Request)
    extension.register(Component.TensorField, TensorField)
    extension.register(Component.Embedder, Embedder)
    extension.register(Component.Decoder, Decoder)

    def objective(module, prediction, batch, strata):
        return 3.14

    def output(module: object, address: object) -> pa.StructType:
        return pa.struct([])

    def write(module: object, prediction: object, datatype: pa.StructType | None) -> None:
        return None

    extension.register(Component.loss, objective)
    extension.register(Component.output, output)
    extension.register(Component.write, write)

    return extension


def test_extension_rejects_invalid_name():
    with pytest.raises(ValueError, match="lowercase letters"):
        Extension(name="Bad-Name", types=(str,))


def test_extension_requires_value_types_without_polluting_registry():
    name = extension_name("missingtypes")

    with pytest.raises(TypeError):
        Extension(name=name)  # ty: ignore[missing-argument]

    assert name not in TENSORFIELDS


@pytest.mark.parametrize(
    "value_types",
    [
        pytest.param([], id="list"),
        pytest.param((), id="empty"),
        pytest.param(("str",), id="non-type"),
        pytest.param((Any,), id="typing-any"),
        pytest.param((list[int],), id="parameterized-generic"),
        pytest.param((int, int | float), id="overlapping-families"),
        pytest.param((str, str), id="duplicate-type"),
        pytest.param((list,), id="list-container"),
        pytest.param((tuple,), id="tuple-container"),
        pytest.param((np.ndarray,), id="numpy-array-container"),
        pytest.param((np.int64,), id="numpy-scalar"),
    ],
)
def test_extension_rejects_invalid_value_type_families_without_polluting_registry(value_types):
    name = extension_name("badtypes")

    with pytest.raises((TypeError, ValueError)):
        Extension(name=name, types=value_types)

    assert name not in TENSORFIELDS


def test_extension_stores_separate_and_union_value_type_families():
    name = extension_name("families")
    extension = Extension(name=name, types=(str, bytes, int | float))
    try:
        assert extension.types == (str, bytes, int | float)
        with pytest.raises(AttributeError):
            extension.types = (object,)  # ty: ignore[invalid-assignment]
    finally:
        TENSORFIELDS.pop(name, None)


def test_extension_stores_immutable_runtime_requirements():
    name = extension_name("requires")
    extension = Extension(
        name=name,
        types=(str,),
        requires={"example.codec": "example-codec[fast] >= 2"},
    )
    try:
        assert extension.requires == {"example.codec": "example-codec[fast] >= 2"}
        with pytest.raises(TypeError):
            extension.requires["other"] = "other"  # type: ignore[index]
    finally:
        TENSORFIELDS.pop(name, None)


@pytest.mark.parametrize(
    ("requires", "error"),
    [
        pytest.param([], TypeError, id="non-mapping"),
        pytest.param({"bad-name": "example"}, ValueError, id="invalid-import"),
        pytest.param({1: "example"}, ValueError, id="non-string-import"),
        pytest.param({"example": ""}, ValueError, id="empty-requirement"),
        pytest.param({"example": None}, ValueError, id="non-string-requirement"),
    ],
)
def test_extension_rejects_invalid_runtime_requirements_without_polluting_registry(requires, error):
    name = extension_name("badrequires")

    with pytest.raises(error):
        Extension(name=name, types=(str,), requires=requires)

    assert name not in TENSORFIELDS


def test_extension_requires_reports_all_missing_imports_and_install_targets():
    name = extension_name("missingpackages")
    first = f"missing_{uuid.uuid4().hex}"
    second = f"missing_{uuid.uuid4().hex}"
    extension = Extension(
        name=name,
        types=(str,),
        requires={first: "example[first]", second: "example[second]"},
    )
    try:
        with pytest.raises(ModuleNotFoundError) as raised:
            extension.require(address=Address("/value"))

        message = str(raised.value)
        assert raised.value.name == first
        assert "/value" in message
        assert name in message
        assert first in message
        assert second in message
        assert "uv add 'example[first]' 'example[second]'" in message
    finally:
        TENSORFIELDS.pop(name, None)


def test_extension_requires_accepts_imported_modules_without_specs(monkeypatch: pytest.MonkeyPatch):
    name = extension_name("loadedpackage")
    module = f"loaded_{uuid.uuid4().hex}"
    marker = object()
    extension = Extension(name=name, types=(str,), requires={module: "example-loaded"})
    monkeypatch.setitem(sys.modules, module, marker)
    try:
        extension.require(address=Address("/value"))
    finally:
        TENSORFIELDS.pop(name, None)


def test_custom_value_type_requires_and_uses_an_arrow_matcher():
    missing = extension_name("decimalmissing")
    with pytest.raises(TypeError, match="custom value type.*require arrow matchers.*Decimal"):
        Extension(name=missing, types=(Decimal,))
    assert missing not in TENSORFIELDS

    name = extension_name("decimal")
    extension = Extension(name=name, types=(Decimal,), arrow={Decimal: pa.types.is_decimal})
    try:
        assert extension.accepts(pa.decimal128(12, 2))
        assert extension.accepts(pa.decimal256(50, 8))
        assert not extension.accepts(pa.float64())
    finally:
        TENSORFIELDS.pop(name, None)


def test_custom_arrow_matcher_sees_extension_type_before_storage():
    class Identifier:
        pass

    class IdentifierType(pa.ExtensionType):
        def __init__(self):
            super().__init__(pa.binary(16), f"relflow.test.identifier.{uuid.uuid4().hex}")

        def __arrow_ext_serialize__(self):
            return b""

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            return cls()

    datatype = IdentifierType()
    name = extension_name("extension")
    extension = Extension(
        name=name,
        types=(Identifier,),
        arrow={Identifier: lambda candidate: isinstance(candidate, IdentifierType)},
    )
    bytes_name = extension_name("extensionstorage")
    bytes_extension = Extension(name=bytes_name, types=(bytes,))
    try:
        assert extension.accepts(datatype)
        assert bytes_extension.accepts(datatype)

        values = pa.ExtensionArray.from_storage(
            datatype,
            pa.array([b"0123456789abcdef"], type=pa.binary(16)),
        )
        assert extension.prepare(values, address=Address("/value")) is values
        prepared = bytes_extension.prepare(values, address=Address("/value"))
        assert prepared.type == pa.binary(16)
        assert prepared.to_pylist() == [b"0123456789abcdef"]
    finally:
        TENSORFIELDS.pop(name, None)
        TENSORFIELDS.pop(bytes_name, None)


def test_extension_prepare_decodes_dictionary_wrappers_recursively():
    extension = Extension(name=extension_name("dictionary"), types=(str,))
    dictionary = pa.DictionaryArray.from_arrays(
        pa.array([0, 1, 0], type=pa.int8()),
        pa.array(["A", "B"]),
    )
    values = pa.ListArray.from_arrays(pa.array([0, 2, 3]), dictionary)
    try:
        prepared = extension.prepare(values, address=Address("/value"))

        assert prepared.type == pa.list_(pa.string())
        assert prepared.to_pylist() == [["A", "B"], ["A"]]
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_prepare_promotes_compatible_unions_recursively():
    extension = Extension(name=extension_name("union"), types=(int | float,))
    union = pa.UnionArray.from_dense(
        pa.array([0, 1, 0], type=pa.int8()),
        pa.array([0, 0, 1], type=pa.int32()),
        [pa.array([1, 3], type=pa.int64()), pa.array([2.5], type=pa.float64())],
    )
    values = pa.ListArray.from_arrays(pa.array([0, 2, 3]), union)
    try:
        prepared = extension.prepare(values, address=Address("/value"))

        assert prepared.type == pa.list_(pa.float64())
        assert prepared.to_pylist() == [[1.0, 2.5], [3.0]]
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_prepare_reports_unsafe_union_promotion_with_context():
    extension = Extension(name=extension_name("unsafeunion"), types=(int,))
    values = pa.UnionArray.from_dense(
        pa.array([0, 1], type=pa.int8()),
        pa.array([0, 0], type=pa.int32()),
        [pa.array([1], type=pa.int64()), pa.array([2**63], type=pa.uint64())],
    )
    try:
        with pytest.raises(TypeError, match=f"extension '{extension.name}'.*/value.*cannot safely normalize"):
            extension.prepare(values, address=Address("/value"))
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_custom_matchers_preserve_declared_union_family_boundaries():
    datatype = pa.dense_union(
        [
            pa.field("text", pa.string()),
            pa.field("decimal", pa.decimal128(12, 2)),
        ]
    )
    union_name = extension_name("unionfamily")
    union = Extension(
        name=union_name,
        types=(str | Decimal,),
        arrow={Decimal: pa.types.is_decimal},
    )
    separate_name = extension_name("separatefamilies")
    separate = Extension(
        name=separate_name,
        types=(str, Decimal),
        arrow={Decimal: pa.types.is_decimal},
    )
    try:
        assert union.accepts(datatype)
        assert not separate.accepts(datatype)
    finally:
        TENSORFIELDS.pop(union_name, None)
        TENSORFIELDS.pop(separate_name, None)


def test_extension_rejects_invalid_arrow_matcher_declarations():
    with pytest.raises(TypeError, match="arrow must be a mapping"):
        Extension(name=extension_name("arrowmapping"), types=(int,), arrow=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="absent from types"):
        Extension(name=extension_name("arrowkey"), types=(int,), arrow={float: pa.types.is_floating})
    with pytest.raises(TypeError, match="must be callable"):
        Extension(name=extension_name("arrowcallable"), types=(int,), arrow={int: object()})  # type: ignore[dict-item]


def test_extension_judges_python_ingress_after_arrow_canonicalization():
    values = pa.array(["text", b"bytes"])
    name = extension_name("canonical")
    extension = Extension(name=name, types=(str, bytes))
    try:
        assert pa.types.is_binary(values.type)
        assert extension.accepts(values.type)
    finally:
        TENSORFIELDS.pop(name, None)


def test_extension_warns_and_overwrites_duplicate_name():
    name = extension_name("duplicate")
    first = Extension(name=name, types=(str,))
    try:
        with pytest.warns(UserWarning, match="overriding existing tensorfield extension"):
            second = Extension(name=name, types=(str,))

        assert TENSORFIELDS[name] is second
        assert TENSORFIELDS[name] is not first
    finally:
        TENSORFIELDS.pop(name, None)


def test_extension_registers_explicit_components_without_signature_policing():
    extension = build_extension()
    try:
        assert Component.Request in extension.components
        assert Component.TensorField in extension.components
        assert Component.Embedder in extension.components
        assert Component.Decoder in extension.components
        assert Component.loss in extension.components
        assert Component.output in extension.components
        assert Component.write in extension.components

        value = extension.loss(object(), prediction=object(), batch=object(), strata=Strata.train)
        assert value == 3.14
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_write_defaults_to_no_output_when_unregistered():
    extension = Extension(name=extension_name("defaultwrite"), types=(object,))
    try:
        assert extension.write(module=object(), prediction=object(), datatype=None) is None
        assert Component.write not in extension.components
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_output_defaults_to_no_output_when_unregistered():
    extension = Extension(name=extension_name("defaultoutput"), types=(object,))
    try:
        assert extension.output(module=object(), address=object()) is None
        assert Component.output not in extension.components
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_observation_components_default_to_no_op():
    extension = Extension(name=extension_name("defaultobserve"), types=(object,))
    try:
        assert (
            extension.observe(
                field=object(),
                address=Address("/value"),
                schema=object(),
                state=None,
                learn=False,
            )
            is None
        )
        assert (
            extension.learn(
                module=object(),
                observation=object(),
                address=Address("/value"),
                strata=Strata.train,
            )
            is None
        )
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_explicit_component_role_overrides_a_misleading_function_name():
    extension = Extension(name=extension_name("explicitrole"), types=(object,))

    def output(module, prediction, datatype):
        return None

    try:
        assert extension.register("write", output) is output
        assert extension.write is output
        assert Component.output not in extension.components
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_accepts_callable_objects_without_a_name_or_annotations():
    class Writer:
        def __init__(self):
            self.arguments = None

        def __call__(self, **arguments):
            self.arguments = arguments
            return None

    extension = Extension(name=extension_name("callableobject"), types=(object,))
    writer = Writer()
    try:
        assert extension.register(Component.write, writer) is writer
        assert extension.write(module=1, prediction=2, datatype=None) is None
        assert writer.arguments == {"module": 1, "prediction": 2, "datatype": None}
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_component_decorators_preserve_callable_identity():
    extension = Extension(name=extension_name("decorator"), types=(object,))
    try:

        @extension.register("output")
        def schema_for(**arguments):
            return None

        assert extension.output is schema_for
        assert extension.output(module=object(), address=Address("value")) is None
    finally:
        TENSORFIELDS.pop(extension.name, None)


@pytest.mark.parametrize("initial", [None, lambda **arguments: None])
def test_duplicate_component_registration_preserves_the_original(initial):
    extension = Extension(name=extension_name("duplicatecomponent"), types=(object,))
    try:
        extension.register(Component.write, initial)
        before = extension.components.copy()
        with pytest.raises(ValueError, match="already registered"):
            extension.register(Component.write, lambda **arguments: None)
        assert extension.components == before
    finally:
        TENSORFIELDS.pop(extension.name, None)


@pytest.mark.parametrize(
    "component,implementation,error",
    [
        ("unknown", lambda **arguments: None, ValueError),
        (Component.loss, None, TypeError),
        (Component.observe, object(), TypeError),
        (Component.Request, object, TypeError),
        (Component.TensorField, object, TypeError),
        (Component.Embedder, object, TypeError),
        (Component.Decoder, object, TypeError),
    ],
)
def test_invalid_component_registration_does_not_change_registered_components(component, implementation, error):
    extension = Extension(name=extension_name("invalidcomponent"), types=(object,))
    try:
        extension.register(Component.write, None)
        before = extension.components.copy()
        with pytest.raises(error):
            extension.register(component, implementation)
        assert extension.components == before
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_accepts_explicit_none_write():
    extension = Extension(name=extension_name("nonehooks"), types=(object,))
    try:
        extension.register(Component.write, None)

        assert extension.write(module=object(), prediction=object(), datatype=None) is None
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_accepts_explicit_none_output():
    extension = Extension(name=extension_name("noneoutput"), types=(object,))
    try:
        extension.register(Component.output, None)

        assert extension.output(module=object(), address=object()) is None
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_write_accepts_a_return_annotation():
    extension = Extension(name=extension_name("typedwrite"), types=(object,))

    def write(module: object, prediction: object, datatype: pa.StructType | None) -> None:
        return None

    try:
        assert extension.register(Component.write, write) is write
    finally:
        TENSORFIELDS.pop(extension.name, None)


@pytest.mark.parametrize(
    "component",
    [Component.Request, Component.TensorField, Component.Embedder, Component.Decoder, Component.bind, Component.loss],
)
def test_extension_missing_required_component_reports_the_extension_and_role(component):
    extension = Extension(name=extension_name("missingcomponent"), types=(object,))
    try:
        with pytest.raises(AttributeError, match=f"{extension.name}.*{component}"):
            extension.component(component)
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_declares_callback_factories():
    class FirstCallback(Callback):
        pass

    class SecondCallback(Callback):
        pass

    extension = Extension(name=extension_name("callbacks"), types=(object,), callbacks=(FirstCallback, SecondCallback))
    try:
        assert extension.callback_factories == (FirstCallback, SecondCallback)
        assert [type(callback) for callback in extension.callbacks] == [FirstCallback, SecondCallback]
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_rejects_invalid_callback_factory_without_registering():
    class InvalidCallback:
        pass

    name = extension_name("badcallbacks")
    with pytest.raises(TypeError, match="must produce a Lightning Callback"):
        Extension(name=name, types=(object,), callbacks=(InvalidCallback,))

    assert name not in TENSORFIELDS


def test_extension_request_must_implement_leaf_contract():
    class Request(Node):
        type: str = "invalid_request"

    extension = Extension(name=extension_name("badrequest"), types=(object,))
    try:
        with pytest.raises(TypeError, match="subclass of RequestBase"):
            extension.register(Component.Request, Request)
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_tensorfield_accepts_flexible_keyword_inputs():
    class ColumnBatch(TensorFieldBase):
        @classmethod
        def new(cls, *arguments, **options):
            return object()

    extension = Extension(name=extension_name("flexibleinputs"), types=(object,))
    try:
        assert extension.register(Component.TensorField, ColumnBatch) is ColumnBatch
        assert extension.TensorField is ColumnBatch
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_extension_unknown_attributes_follow_normal_python_lookup():
    extension = Extension(name=extension_name("badattr"), types=(object,))
    try:
        with pytest.raises(AttributeError, match="not_real"):
            getattr(extension, "not_real")
        assert not hasattr(extension, "not_real")
    finally:
        TENSORFIELDS.pop(extension.name, None)
