"""Nonobjective target payloads stay empty without bypassing pristine validation."""

from datetime import datetime

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
    ("field_type", "value", "options"),
    [
        (rf.Number, 1.5, {}),
        (rf.Boolean, True, {}),
        (rf.Category, "known", {"p_unavailable": 0.0}),
        (rf.Cluster, "known", {"n_clusters": 3, "p_unavailable": 0.0}),
        (rf.Set, ["known"], {"p_unavailable": 0.0}),
        (rf.DateParts, datetime(2026, 1, 1), {"dateparts": ["day_of_week"]}),
        (rf.Hash, "identifier", {}),
        (rf.Text, "hello", {"model": "local-test", "max_length": 4}),
    ],
)
@pytest.mark.parametrize("objective", [False, True])
@pytest.mark.parametrize("strata", list(rf.Strata))
def test_targets_follow_objectives(field_type, value, options, objective, strata, monkeypatch):
    monkeypatch.setattr(CachedModel, "get_tokenizer", classmethod(lambda cls, key: Tokenizer()))
    schema = rf.Schema.from_tree(
        d_model=8,
        n_layers=1,
        n_heads=4,
        value=field_type(mask=rf.Mask(dropout=False, reconstruct=True) if objective else False, **options),
    )
    context = {"/value": OnlineVocabularyModel(size=8).state} if field_type in (rf.Category, rf.Cluster, rf.Set) else {}
    batch = encode(
        pa.table({"value": [value, None]}),
        schema=schema,
        strata=strata,
        interprocess_encoding_context=context,
    )
    field = batch.tensors["/value"]
    assert field.targets.batch_size == field.state.shape
    assert ("/value" in schema.objectives) == objective
    assert set(field.targets.keys()) == ({TensorKey.state, TensorKey.content} if objective else set())
    assert bool(field.trainable.any()) == (objective and strata != rf.Strata.predict)


@pytest.mark.parametrize(
    ("field_type", "invalid"),
    [(rf.Number, "invalid"), (rf.Boolean, "invalid"), (rf.Hash, {"nested": 1}), (rf.Text, 42)],
)
def test_unused_targets_preserve_masked_pristine_validation(field_type, invalid):
    schema = rf.Schema.from_tree(
        d_model=8,
        n_layers=1,
        n_heads=4,
        value=field_type(mask=rf.Mask(rate=1.0, reconstruct=False)),
    )
    assert "/value" not in schema.objectives
    with pytest.raises((TypeError, ValueError), match="value"):
        encode(
            pa.table({"value": [invalid]}),
            schema=schema,
            strata=rf.Strata.train,
            interprocess_encoding_context={},
        )
