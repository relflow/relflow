from __future__ import annotations

import pyarrow as pa
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.data import arrow
from relflow.data.arrow import Encoded, mappings, variants
from relflow.structs.tree import Address


def test_batch_is_removed_from_the_public_api():
    assert not hasattr(rf, "Batch")
    assert "Encoded" not in arrow.__all__


def test_mappings_reuses_uniform_dicts_without_rebuilding_rows():
    values = [{"a": 1, "b": 2}, {"b": 3, "a": 4}]

    aligned = mappings(values)

    assert aligned == values
    assert aligned[0] is values[0]
    assert aligned[1] is values[1]


def test_mappings_aligns_sparse_rows_with_fields_introduced_later():
    values = [{"a": 1}, {"a": 2, "b": 3}]

    aligned = mappings(values)

    assert aligned == [{"a": 1, "b": None}, {"a": 2, "b": 3}]
    assert aligned[0] is not values[0]


def test_mappings_rejects_fieldless_observations():
    with pytest.raises(ValueError, match="without fields"):
        mappings([{}, {}])


def test_variants_respects_sliced_union_offsets():
    values = pa.UnionArray.from_dense(
        pa.array([0, 1, 0], type=pa.int8()),
        pa.array([0, 0, 1], type=pa.int32()),
        [pa.array([1, 3]), pa.array([2.0])],
    ).slice(1)

    codes, offsets = variants(values)

    assert codes.tolist() == [1, 0]
    assert offsets is not None
    assert offsets.tolist() == [0, 1]


def test_encoded_keeps_tensors_and_arrow_source_separate():
    source = pa.table({"value": [10, 20]})
    tensors = TensorDict({"value": torch.tensor([10, 20])}, batch_size=[2])

    encoded = Encoded(tensors=tensors, source=source, retain=("value",))

    assert encoded.tensors is tensors
    assert encoded.source is source
    assert encoded.retain == ("value",)
    assert encoded.observations == {}


def test_encoded_pins_every_tensor_payload(monkeypatch: pytest.MonkeyPatch):
    source = pa.table({"value": [10, 20]})
    tensors = TensorDict({"value": torch.tensor([10, 20])}, batch_size=[2])
    observation = TensorDict({"counts": torch.tensor([2])}, batch_size=[1])
    calls = []

    def pin(value):
        calls.append(value)
        return value.clone()

    monkeypatch.setattr(TensorDict, "pin_memory", pin)
    encoded = Encoded(
        tensors=tensors,
        source=source,
        retain=("value",),
        observations={Address("record/value"): observation},
    )

    pinned = encoded.pin_memory()

    assert calls == [tensors, observation]
    assert pinned.tensors is not tensors
    assert pinned.observations["record/value"] is not observation
    assert pinned.source is source
    assert pinned.retain == ("value",)


def test_encoded_copies_and_validates_pristine_observations():
    source = pa.table({"value": [10]})
    tensors = TensorDict({}, batch_size=[])
    observation = TensorDict({"counts": torch.ones(3, dtype=torch.int64)}, batch_size=[])
    supplied = {Address("record/value"): observation}

    encoded = Encoded(tensors=tensors, source=source, observations=supplied)
    supplied.clear()

    assert tuple(encoded.observations) == (Address("record/value"),)
    assert encoded.observations["record/value"] is observation

    with pytest.raises(TypeError, match="must be a TensorDict"):
        Encoded(tensors=tensors, source=source, observations={Address("record/value"): object()})


def test_encoded_validates_source_and_retain():
    tensors = TensorDict({}, batch_size=[])

    with pytest.raises(TypeError, match="pyarrow.Table"):
        Encoded(tensors=tensors, source=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tuple of unique"):
        Encoded(tensors=tensors, source=pa.table({"value": [10]}), retain=("value", "value"))
