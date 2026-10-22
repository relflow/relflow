"""Number storage preserves Arrow precision, masks, and buffer ownership."""

import numpy as np
import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.data.ragged import coalesce
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.tensorfields.extensions.number import TensorField, number
from tests.tensorfields.helpers import tensorize


def tensorfield(values, selected):
    schema = rf.Schema.from_tree(
        d_model=8,
        n_heads=2,
        n_layers=1,
        amount=rf.Number(mask=rf.Mask(query="selected", reconstruct=True)),
    )
    selected = [selected] * len(values) if isinstance(selected, bool) else selected
    table = pa.table({"amount": values, "selected": selected})
    projection = coalesce(table, schema=schema, strata=Strata.train).fields["/amount"]
    return tensorize(TensorField, projection, number, address="/amount", schema=schema, strata=Strata.train)


def buffers(values):
    """Snapshot original Arrow storage, including independent chunk dictionaries."""
    chunks = values.chunks if isinstance(values, pa.ChunkedArray) else [values]
    return [
        tuple(None if buffer is None else buffer.to_pybytes() for buffer in chunk.buffers())
        + (
            tuple(buffer.to_pybytes() for buffer in chunk.dictionary.buffers() if buffer is not None)
            if isinstance(chunk, pa.DictionaryArray)
            else ()
        )
        for chunk in chunks
    ]


@pytest.mark.parametrize("datatype", [pa.float32(), pa.float64()])
@pytest.mark.parametrize("layout", ["plain", "sliced", "chunked", "dictionary_chunked"])
def test_number_nullable_arrow_values_preserve_input_target_masks_and_source_storage(datatype, layout):
    content = [1.25, None, -2.5, -0.0, 3.75, 4.5]
    if layout == "sliced":
        values = pa.array([99.0, *content, 100.0], type=datatype).slice(1, len(content))
    elif layout in ("chunked", "dictionary_chunked"):
        chunks = [pa.array(content[:3], type=datatype), pa.array(content[3:], type=datatype)]
        values = pa.chunked_array(
            [chunk.dictionary_encode() for chunk in chunks] if layout == "dictionary_chunked" else chunks
        )
    else:
        values = pa.array(content, type=datatype)
    original = buffers(values)
    selected = torch.tensor([False, True, False, True, False, True])
    field = tensorfield(values, selected.tolist())
    expected = torch.tensor([1.25, 0.0, -2.5, -0.0, 3.75, 4.5], dtype=torch.float64).reshape(-1, 1)
    expected_inputs = expected.masked_fill(selected.unsqueeze(-1), 0.0)
    expected_targets = expected.masked_fill(~selected.unsqueeze(-1), 0.0)
    targets = field.targets[TensorKey.content]

    torch.testing.assert_close(field.content, expected_inputs, rtol=0, atol=0)
    torch.testing.assert_close(targets, expected_targets, rtol=0, atol=0)
    assert torch.equal(torch.signbit(targets), torch.signbit(expected_targets))
    assert torch.equal(field.trainable.flatten(), selected)
    assert field.content.dtype == targets.dtype == torch.float64
    assert field.content.numpy().flags.writeable and targets.numpy().flags.writeable
    assert not np.shares_memory(field.content.numpy(), targets.numpy())
    assert buffers(values) == original

    field.content.fill_(111.0)
    torch.testing.assert_close(targets, expected_targets, rtol=0, atol=0)
    targets.fill_(222.0)
    assert buffers(values) == original


@pytest.mark.parametrize("datatype", [pa.float32(), pa.float64()])
@pytest.mark.parametrize("selected", [False, True])
def test_number_nonfinite_conversion_retains_infinities_and_signed_zero(datatype, selected):
    values = pa.array([float("nan"), float("inf"), -float("inf"), -0.0, 0.0, 1e40], type=datatype)
    original = buffers(values)
    field = tensorfield(values, selected)
    actual = field.targets[TensorKey.content] if selected else field.content
    state = field.targets[TensorKey.state] if selected else field.state
    large = 1e40 if datatype == pa.float64() else float("inf")
    expected = torch.tensor([0.0, float("inf"), -float("inf"), -0.0, 0.0, large], dtype=torch.float64).reshape(-1, 1)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))
    # NaN is not a number: it takes the ``other`` state; infinities stay valued.
    assert state.flatten().tolist() == [Tokens.other, *([Tokens.valued] * 5)]
    assert buffers(values) == original


@pytest.mark.parametrize("selected", [False, True])
def test_number_keeps_float64_values_exactly(selected):
    halfway = 1.0 + 2.0**-24
    content = [halfway, float(np.nextafter(halfway, np.inf)), -halfway, float(np.nextafter(-halfway, -np.inf))]
    field = tensorfield(pa.array(content), selected)
    actual = field.targets[TensorKey.content] if selected else field.content
    expected = torch.tensor(content, dtype=torch.float64).reshape(-1, 1)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("datatype", [pa.int64(), pa.uint64()])
@pytest.mark.parametrize("selected", [False, True])
def test_number_keeps_safe_integers_exactly(datatype, selected):
    values = pa.array([16_777_217, 16_777_219, 2**40, 2**53], type=datatype)
    field = tensorfield(values, selected)
    actual = field.targets[TensorKey.content] if selected else field.content
    expected = torch.tensor([16_777_217.0, 16_777_219.0, float(2**40), float(2**53)], dtype=torch.float64).reshape(
        -1, 1
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "datatype,value", [(pa.int64(), 2**53 + 1), (pa.int64(), -(2**53) - 1), (pa.uint64(), 2**53 + 1)]
)
@pytest.mark.parametrize("selected", [False, True])
def test_number_rejects_integers_beyond_float64_exactness(datatype, value, selected):
    with pytest.raises(ValueError, match=r"number field at '/amount' holds u?int64 values.*2\^53"):
        tensorfield(pa.array([value], type=datatype), selected)


@pytest.mark.parametrize("values", [pa.array([], type=pa.float32()), pa.array([None, None], type=pa.float64())])
def test_number_empty_and_all_null_values_still_use_float64_storage(values):
    field = tensorfield(values, True)
    assert field.content.dtype == field.targets[TensorKey.content].dtype == torch.float64
    assert field.content.shape == field.targets[TensorKey.content].shape == (len(values), 1)
    assert not field.content.any() and not field.targets[TensorKey.content].any()
