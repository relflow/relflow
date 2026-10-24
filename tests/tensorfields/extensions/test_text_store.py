"""The Text embedding store persists frozen-encoder vectors across runs, splits, and fields."""

import numpy as np
import pyarrow as pa
import torch

import relflow as rf
from relflow.structs.enums import Strata
from relflow.tensorfields.extensions.text import Embedder, EmbeddingStore, flush_stores, hash_rows, hash_rows_torch
from tests.tensorfields.extensions.test_text import patch_hf

MODEL = "test-text-store"  # a fake encoder from patch_hf: the tests run offline, like CI


def test_hash_rows_is_deterministic_and_row_sensitive():
    rows = np.array([[1, 2, 3, 1, 1, 0], [1, 2, 4, 1, 1, 0], [1, 2, 3, 1, 1, 0]], dtype=np.int64)
    keys = hash_rows(rows)
    assert keys.shape == (3, 2) and keys.dtype == np.int64
    assert np.array_equal(keys[0], keys[2]) and not np.array_equal(keys[0], keys[1])
    assert not np.array_equal(hash_rows(rows, salt=1)[0], keys[0])
    assert np.array_equal(hash_rows(rows), keys)
    on_device = hash_rows_torch(torch.from_numpy(rows), salt=7).numpy()
    assert np.array_equal(on_device, hash_rows(rows, salt=7)), "numpy and torch hash the same rows to the same words"


def test_store_round_trips_through_shards_and_later_openers(tmp_path):
    first = EmbeddingStore(tmp_path, "enc", 4, flush_every=1000)
    keys = hash_rows(np.arange(12, dtype=np.int64).reshape(3, 4))
    vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert first.put(keys, vectors) == 3 and first.put(keys, vectors) == 0
    found, out = first.lookup(keys[[2, 0]])
    assert found.all() and np.array_equal(out, vectors[[2, 0]])
    assert first.flush() == 3 and first.flush() == 0

    second = EmbeddingStore(tmp_path, "enc", 4)
    found, out = second.lookup(keys)
    assert found.all() and np.array_equal(out, vectors)
    more = hash_rows(np.arange(100, 108, dtype=np.int64).reshape(2, 4))
    assert second.put(more, np.ones((2, 4), dtype=np.float32)) == 2
    second.flush()

    third = EmbeddingStore(tmp_path, "enc", 4)
    found, out = third.lookup(np.concatenate((keys, more)))
    assert found.all() and len(third) == 5 and np.array_equal(out[3:], np.ones((2, 4)))
    assert len(list(tmp_path.glob("enc.*.keys.npy"))) == 2, "each flush is its own shard"
    missing, blank = third.lookup(hash_rows(np.array([[7, 7, 7, 7]], dtype=np.int64)))
    assert not missing.any() and not blank.any()
    assert len(EmbeddingStore(tmp_path, "other", 4)) == 0, "another encoder has its own files"


def test_store_flushes_itself_at_its_interval(tmp_path):
    store = EmbeddingStore(tmp_path, "enc", 2, flush_every=2)
    store.put(hash_rows(np.array([[1, 2]], dtype=np.int64)), np.zeros((1, 2), np.float32))
    assert not list(tmp_path.glob("enc.*.keys.npy"))
    store.put(hash_rows(np.array([[3, 4]], dtype=np.int64)), np.zeros((1, 2), np.float32))
    assert len(list(tmp_path.glob("enc.*.keys.npy"))) == 1 and len(store.pending_keys) == 0 and len(store) == 2


def configured(store):
    return rf.Model.xs(
        note=rf.Text(model=MODEL, store=str(store), max_length=8), label=rf.Boolean(mask=True), d_model=8, batch_size=4
    )


def test_embedder_consults_the_store_before_the_encoder(tmp_path, monkeypatch):
    patch_hf(monkeypatch)
    table = pa.table({"note": ["alpha", "beta", "alpha", None], "label": [True, False, True, False]})
    model = configured(tmp_path)
    embedder = model.nodes["/note"].embedder
    calls: list[int] = []
    original = Embedder.pool

    def counted(self, ids, mask):
        calls.append(len(ids))
        return original(self, ids, mask)

    monkeypatch.setattr(Embedder, "pool", counted)
    inputs = model.encode(table, strata=Strata.validate)
    first = embedder.encode(content=inputs["/note"].content, state=inputs["/note"].state)
    assert calls == [2], "two distinct strings reach the encoder once"
    second = embedder.encode(content=inputs["/note"].content, state=inputs["/note"].state)
    assert calls == [2] and torch.equal(first, second)
    assert flush_stores(model) == {"/note": 2}

    later = configured(tmp_path)
    calls.clear()
    third = later.nodes["/note"].embedder.encode(content=inputs["/note"].content, state=inputs["/note"].state)
    assert calls == [] and torch.allclose(first, third)

    plain = rf.Model.xs(note=rf.Text(model=MODEL, max_length=8), label=rf.Boolean(mask=True), d_model=8, batch_size=4)
    direct = plain.nodes["/note"].embedder.encode(content=inputs["/note"].content, state=inputs["/note"].state)
    assert torch.allclose(first, direct, atol=1e-5), "stored vectors are the encoder's own"


def test_warm_fills_the_store_for_every_later_split(tmp_path, monkeypatch):
    patch_hf(monkeypatch)
    model = configured(tmp_path)
    embedder = model.nodes["/note"].embedder
    # The fake tokenizer keys rows by length, so distinct strings here have distinct lengths.
    assert embedder.warm(pa.array(["a", "bbb", None, "a"])) == 2
    assert embedder.warm(["bbb", "ccccc"]) == 1
    monkeypatch.setattr(Embedder, "pool", lambda self, ids, mask: (_ for _ in ()).throw(AssertionError("encoder ran")))
    for strata in (Strata.train, Strata.validate, Strata.test):
        table = pa.table({"note": ["a", "bbb", "ccccc"], "label": [True, False, True]})
        inputs = model.encode(table, strata=strata)
        encoded = embedder.encode(content=inputs["/note"].content, state=inputs["/note"].state)
        assert encoded.shape[0] == 3 and encoded.shape[-1] == embedder.hidden_size and torch.isfinite(encoded).all()


def test_fields_with_the_same_encoder_share_one_store(tmp_path, monkeypatch):
    patch_hf(monkeypatch)
    model = rf.Model.xs(
        a=rf.Text(model=MODEL, store=str(tmp_path), max_length=8),
        b=rf.Text(model=MODEL, store=str(tmp_path), max_length=8),
        label=rf.Boolean(mask=True),
        d_model=8,
        batch_size=2,
    )
    table = pa.table({"a": ["shared"], "b": ["shared"], "label": [True]})
    inputs = model.encode(table, strata=Strata.train)
    first = model.nodes["/a"].embedder.encode(content=inputs["/a"].content, state=inputs["/a"].state)
    monkeypatch.setattr(Embedder, "pool", lambda self, ids, mask: (_ for _ in ()).throw(AssertionError("encoder ran")))
    second = model.nodes["/b"].embedder.encode(content=inputs["/b"].content, state=inputs["/b"].state)
    assert torch.equal(first, second)


def test_resident_view_answers_on_device_within_its_budget(tmp_path):
    store = EmbeddingStore(tmp_path, "enc", 4, flush_every=1000, resident=64)
    keys = hash_rows(np.arange(20, dtype=np.int64).reshape(5, 4))
    vectors = np.arange(20, dtype=np.float32).reshape(5, 4)
    store.put(keys, vectors)
    assert store.lookup_device(torch.from_numpy(keys)) is None, "pending vectors are not resident until flushed"
    store.flush()
    found, out = store.lookup_device(torch.from_numpy(keys[[4, 1]]))
    assert found.all() and torch.equal(out, torch.from_numpy(vectors[[4, 1]]))
    found, _ = store.lookup_device(torch.from_numpy(hash_rows(np.array([[9, 9, 9, 9]], dtype=np.int64))))
    assert not found.any()
    assert EmbeddingStore(tmp_path, "enc", 4, resident=0).lookup_device(torch.from_numpy(keys)) is None
    assert EmbeddingStore(tmp_path, "enc", 4).view(torch.device("cpu")) is None, "a CPU model keeps the host index"


def test_embedder_uses_the_resident_view_after_a_flush(tmp_path, monkeypatch):
    patch_hf(monkeypatch)
    model = rf.Model.xs(
        note=rf.Text(model=MODEL, store=str(tmp_path), resident=64, max_length=8),
        label=rf.Boolean(mask=True),
        d_model=8,
        batch_size=4,
    )
    embedder = model.nodes["/note"].embedder
    table = pa.table({"note": ["alpha", "beta", "alpha", None], "label": [True, False, True, False]})
    inputs = model.encode(table, strata=Strata.train)
    first = embedder.encode(content=inputs["/note"].content, state=inputs["/note"].state)
    assert flush_stores(model) == {"/note": 2}
    assert embedder.store.view(torch.device("cpu")) is not None
    monkeypatch.setattr(Embedder, "pool", lambda self, ids, mask: (_ for _ in ()).throw(AssertionError("encoder ran")))
    monkeypatch.setattr(
        embedder.store, "lookup", lambda keys: (_ for _ in ()).throw(AssertionError("host index consulted"))
    )
    second = embedder.encode(content=inputs["/note"].content, state=inputs["/note"].state)
    assert torch.equal(first, second)
