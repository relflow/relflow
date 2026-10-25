"""Vocabulary lookup resolves Arrow labels with Python equality and never admits NaN."""

import numpy as np
import pyarrow as pa
import pytest
import torch

from relflow.tensorfields.shared.resize import Resize
from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel, VocabularyBatch


def test_lookup_matches_admitted_labels_across_arrow_widths_and_signed_zero():
    vocabulary = OnlineVocabularyModel()
    vocabulary.load_snapshot([0.0, 2.5, 7.0])

    positions = vocabulary.lookup(pa.array([-0.0, 7.0, 9.0, 2.5], type=pa.float32()))

    assert positions.tolist() == [0, 2, -1, 1]
    assert positions.dtype == np.int64

    vocabulary.load_snapshot(["alpha", "beta"])
    assert vocabulary.lookup(pa.array(["beta", "gamma"], type=pa.large_string())).tolist() == [1, -1]
    assert vocabulary.lookup(pa.array([], type=pa.null())).tolist() == []

    vocabulary.load_snapshot([3, 1])
    assert vocabulary.lookup(pa.array([1, 3, 2], type=pa.int32())).tolist() == [1, 0, -1]


def test_lookup_cache_follows_admissions_and_snapshots():
    vocabulary = OnlineVocabularyModel()
    vocabulary.load_snapshot(["alpha"])
    assert vocabulary.lookup(pa.array(["alpha", "beta"])).tolist() == [0, -1]

    vocabulary.extend(["beta"])
    assert vocabulary.lookup(pa.array(["alpha", "beta"])).tolist() == [0, 1]

    vocabulary.load_snapshot(["beta"])
    assert vocabulary.lookup(pa.array(["alpha", "beta"])).tolist() == [-1, 0]


def test_worker_state_rejects_vocabularies_without_one_scalar_type():
    vocabulary = OnlineVocabularyModel()
    vocabulary.load_snapshot([True, 1])

    with pytest.raises(ValueError, match="one scalar type"):
        vocabulary.state.indices(pa.array([1]), learn=False)


def test_worker_state_never_admits_float_nan():
    vocabulary = OnlineVocabularyModel()
    column = pa.array([1.0, float("nan"), 2.0, float("nan")])

    state, binding = vocabulary.state.batch(column)
    indices = state.indices(column, learn=True)

    assert isinstance(binding, VocabularyBatch)
    assert binding.labels.to_pylist() == [1.0, 2.0] and binding.size == 2
    assert indices.tolist() == [0, -1, 1, -1]
    assert state.size == binding.size and len(state) == 2
    assert state.indices(column, learn=False).tolist() == [0, -1, 1, -1]


def test_resolve_admits_only_unseen_labels_in_batch_order():
    vocabulary = OnlineVocabularyModel()
    vocabulary.load_snapshot(["alpha"])
    binding = VocabularyBatch(labels=pa.array(["gamma", "alpha", "beta"]), size=3)
    resize = Resize()

    mapping, size = binding.resolve(vocabulary, learn=True, device=torch.device("cpu"), resize=resize)
    resize.commit()

    assert mapping.tolist() == [1, 0, 2, -1]
    assert size == 4
    assert list(vocabulary.master) == ["alpha", "gamma", "beta"]


def test_translate_rejects_worker_ids_the_batch_never_issued():
    binding = VocabularyBatch(labels=pa.array(["alpha", "beta"]), size=2)
    mapping = torch.tensor([4, 5, -1])

    assert binding.translate(torch.tensor([1, -1, 0]), mapping, address="/label").tolist() == [5, -1, 4]
    with pytest.raises(ValueError, match="outside"):
        binding.translate(torch.tensor([2]), mapping, address="/label")
