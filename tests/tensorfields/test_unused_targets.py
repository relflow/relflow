"""Nonobjective target payloads stay empty without bypassing pristine validation."""

import numpy as np
import pyarrow as pa
import pytest

import relflow as rf
from relflow.data.iterables import encode
from relflow.structs.enums import TensorKey
from relflow.tensorfields.extensions.text import CachedModel
from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel


class Tokenizer:
    def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
        return {
            "input_ids": np.ones((len(texts), max_length), dtype=np.int64),
            "attention_mask": np.ones((len(texts), max_length), dtype=np.int64),
        }


@pytest.mark.parametrize(
    ("kind", "value", "options"),
    [
        ("number", 1.5, {}),
        ("boolean", True, {}),
        ("category", "known", {"p_unavailable": 0.0}),
        ("hash", "identifier", {}),
        ("text", "hello", {"model": "local-test", "max_length": 4}),
    ],
)
@pytest.mark.parametrize("objective", [False, True])
@pytest.mark.parametrize("strata", [rf.Strata.train, rf.Strata.predict])
def test_targets_follow_objectives(kind, value, options, objective, strata, monkeypatch):
    monkeypatch.setattr(CachedModel, "get_tokenizer", classmethod(lambda cls, key: Tokenizer()))
    schema = rf.Schema.model_validate(
        {
            "d_model": 8,
            "fields": {
                "name": "root",
                "type": "branch",
                "fields": [
                    {
                        "name": "value",
                        "type": kind,
                        "mask": rf.Mask(dropout=False, reconstruct=True) if objective else False,
                        **options,
                    }
                ],
            },
        }
    )
    context = {"root/value": OnlineVocabularyModel(size=8).state} if kind == "category" else {}
    batch = encode(
        pa.table({"value": [value, None]}),
        schema=schema,
        strata=strata,
        interprocess_encoding_context=context,
    )
    field = batch.tensors["root/value"]
    assert field.targets.batch_size == field.state.shape
    assert ("root/value" in schema.objectives) == objective
    assert set(field.targets.keys()) == ({TensorKey.state, TensorKey.content} if objective else set())
    assert bool(field.trainable.any()) == (objective and strata == rf.Strata.train)


@pytest.mark.parametrize(
    ("kind", "invalid"),
    [("number", "invalid"), ("boolean", "invalid"), ("hash", {"nested": 1}), ("text", 42)],
)
def test_unused_targets_preserve_masked_pristine_validation(kind, invalid):
    schema = rf.Schema.model_validate(
        {
            "d_model": 8,
            "fields": {
                "name": "root",
                "type": "branch",
                "fields": [{"name": "value", "type": kind, "mask": rf.Mask(rate=1.0, reconstruct=False)}],
            },
        }
    )
    assert "root/value" not in schema.objectives
    with pytest.raises((TypeError, ValueError), match="value"):
        encode(
            pa.table({"value": [invalid]}),
            schema=schema,
            strata=rf.Strata.train,
            interprocess_encoding_context={},
        )
