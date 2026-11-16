import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest
import torch

from relflow.structs.enums import Component, Tokens
from relflow.tensorfields.base import TENSORFIELDS
from relflow.tensorfields.output import (
    STATE,
    array,
    embedding,
    fixed,
    inferred,
    offsets,
    shape,
    state,
    struct,
    variable,
)
from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel


def test_array_flattens_contiguous_primitive_tensor():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64).T

    result = array(values, pa.float32())

    assert result.type == pa.float32()
    assert result.to_pylist() == [1.0, 3.0, 2.0, 4.0]


def test_fixed_and_shape_preserve_singleton_axes():
    values = array(torch.arange(6), pa.int64())

    once = fixed(values, 3)
    nested = shape(values, (1, 2, 3))

    assert once.type == pa.list_(pa.field("element", pa.int64()), 3)
    assert nested.type == pa.list_(
        pa.field("element", pa.list_(pa.field("element", pa.list_(pa.field("element", pa.int64()), 3)), 2)), 1
    )
    assert len(nested) == 1


def test_fixed_rejects_incompatible_length():
    with pytest.raises(ValueError, match="cannot wrap 5 values"):
        fixed(array(torch.arange(5), pa.int64()), 2)


def test_struct_enforces_declared_order_and_types():
    dtype = pa.struct(
        [
            pa.field("value", pa.int32(), nullable=False),
            pa.field("probability", pa.float32(), nullable=False),
        ]
    )
    result = struct(
        {
            "value": array(torch.tensor([1, 2]), pa.int32()),
            "probability": array(torch.tensor([0.2, 0.8]), pa.float32()),
        },
        dtype,
    )

    assert result.type == dtype

    with pytest.raises(ValueError, match="struct fields"):
        struct(
            {
                "probability": array(torch.tensor([0.2]), pa.float32()),
                "value": array(torch.tensor([1]), pa.int32()),
            },
            dtype,
        )


def test_offsets_and_variable_build_lists_without_rows():
    counts = torch.tensor([2, 0, 1])
    values = array(torch.tensor([10, 11, 12]), pa.int64())

    assert offsets(counts).to_pylist() == [0, 2, 2, 3]
    assert variable(values, counts).to_pylist() == [[10, 11], [], [12]]


def test_variable_rejects_counts_that_do_not_consume_values():
    with pytest.raises(ValueError, match="do not consume"):
        variable(array(torch.tensor([1, 2]), pa.int64()), torch.tensor([1]))


def test_state_emits_one_probability_field_per_token():
    logits = torch.zeros(2, 1, len(Tokens))
    logits[0, 0, Tokens.valued.value] = 10.0
    logits[1, 0, Tokens.null.value] = 10.0

    result = state(logits)

    assert result.type == STATE
    assert len(result) == 2
    assert result.field(Tokens.valued.name)[0].as_py() > 0.99
    assert result.field(Tokens.null.name)[1].as_py() > 0.99


def test_inferred_flattens_boolean_mask():
    result = inferred(torch.tensor([[True, False], [False, True]]))

    assert result.type == pa.bool_()
    assert result.to_pylist() == [True, False, False, True]


def test_embedding_normalizes_and_preserves_width_one():
    result = embedding(torch.tensor([[[3.0]], [[0.0]]]))

    assert result.type == pa.list_(pa.field("element", pa.float32()), 1)
    assert result.to_pylist() == [[1.0], [0.0]]


@pytest.mark.parametrize("kind", ["fixed", "variable", "embedding"])
@pytest.mark.parametrize("nullable", [True, False])
def test_list_builders_preserve_declared_fields_across_transports(kind, nullable):
    element = pa.field("element", pa.float32(), nullable=nullable, metadata={b"unit": b"score"})
    datatype = pa.list_(element) if kind == "variable" else pa.list_(element, 2)
    if kind == "embedding":
        result = embedding(torch.tensor([[3.0, 4.0], [0.0, 0.0]]), datatype=datatype)
    else:
        values = pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float32())
        result = (
            variable(values, torch.tensor([2, 0, 2]), datatype=datatype)
            if kind == "variable"
            else fixed(values, 2, datatype=datatype)
        )
    table = pa.table({"result": result}).replace_schema_metadata({b"output-format": b"1"})
    assert result.type.value_field.equals(element, check_metadata=True)

    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    streamed = ipc.open_stream(sink.getvalue()).read_all()
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    persisted = pq.read_table(pa.BufferReader(sink.getvalue()))
    for restored in (streamed, persisted):
        assert restored.schema.equals(table.schema, check_metadata=True)
        assert restored.equals(table, check_metadata=True)


def test_fixed_nullable_vector_components_roundtrip_without_changing_width():
    datatype = pa.list_(pa.field("element", pa.float32(), nullable=True), 3)
    values = pa.array([1.0, None, 3.0, None, None, None], type=pa.float32())
    vectors = fixed(values, 3, datatype=datatype)
    table = pa.table({"vectors": vectors})
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    restored = pq.read_table(pa.BufferReader(sink.getvalue()))

    assert restored.schema.equals(table.schema, check_metadata=True)
    assert restored["vectors"].to_pylist() == [[1.0, None, 3.0], [None, None, None]]


@pytest.mark.parametrize("kind", ["fixed", "variable"])
def test_list_builders_reject_violations_of_declared_element_fields(kind):
    values = pa.array([1.0, None], type=pa.float32())
    declared = pa.field("element", pa.float32(), nullable=False)
    datatype = pa.list_(declared) if kind == "variable" else pa.list_(declared, 2)
    with pytest.raises(ValueError, match="element field is non-nullable"):
        if kind == "fixed":
            fixed(values, 2, datatype=datatype)
        else:
            variable(values, torch.tensor([2]), datatype=datatype)

    values = pa.array([1.0, 2.0], type=pa.float32())
    datatype = pa.list_(pa.float32()) if kind == "variable" else pa.list_(pa.float32(), 2)
    with pytest.raises(ValueError, match="child must be named 'element'"):
        if kind == "fixed":
            fixed(values, 2, datatype=datatype)
        else:
            variable(values, torch.tensor([2]), datatype=datatype)


@pytest.mark.parametrize("name", ["category", "enum", "set"])
def test_builtin_candidate_lists_declare_canonical_element_names(name):
    datatype = TENSORFIELDS[name].output(module=object(), address=object())
    content = datatype.field("content").type
    candidates = content if name == "set" else content.field("topk").type

    assert candidates.value_field.name == "element"


def test_vocabulary_labels_are_large_strings_cached_by_revision():
    vocabulary = OnlineVocabularyModel(size=4)
    stateful = vocabulary.state
    stateful.reserve([1, b"two"], learn=True)

    first = vocabulary.labels()
    second = vocabulary.labels()
    stateful.reserve("three", learn=True)
    third = vocabulary.labels()

    assert first is second
    assert first.type == pa.large_string()
    assert first.to_pylist() == ["1", "b'two'"]
    assert third is not first
    assert third.to_pylist() == ["1", "b'two'", "three"]


@pytest.mark.parametrize("name", ["dateparts", "hash", "text"])
def test_embedding_only_extensions_declare_no_decoded_output(name: str):
    extension = TENSORFIELDS[name]

    assert Component.output not in extension.components
    assert Component.write not in extension.components
    assert extension.output(module=object(), address=object()) is None
    assert extension.write(module=object(), prediction=object(), datatype=None) is None


DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[pytest.mark.integration, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")],
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "sliced", "empty", "nonfinite"])
def test_state_preserves_probabilities_and_storage_lifetime(device, dtype, layout):
    logits = torch.randn(2, 3, len(Tokens), dtype=dtype, device=device)
    if layout == "transposed":
        logits = logits.transpose(0, 1)
    elif layout == "sliced":
        logits = logits[:, ::2]
    elif layout == "empty":
        logits = logits[:0]
    elif layout == "nonfinite":
        logits[0, 0, 0] = torch.nan
        logits[0, 1, 1] = torch.inf
        logits[1, 0] = -torch.inf
    original = logits.clone()
    expected = logits.float().softmax(dim=-1).reshape(-1, len(Tokens)).cpu()

    written = state(logits)

    assert written.type == STATE
    assert len(written) == expected.shape[0]
    torch.testing.assert_close(logits, original, rtol=0, atol=0, equal_nan=True)
    # Later calls and source mutation must not overwrite the returned Arrow data.
    state(torch.zeros_like(logits))
    logits.zero_()
    for token in Tokens:
        values = written.field(token.name).to_numpy(zero_copy_only=False).copy()
        torch.testing.assert_close(torch.from_numpy(values), expected[:, token.value], rtol=0, atol=0, equal_nan=True)
