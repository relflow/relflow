"""Disk, Rust admission, bounded device state, and real owner failure recovery."""

import sys
import time

import numpy as np
import pytest
import torch

from relflow.tensorfields.extensions.text_cache.backend import Disk
from relflow.tensorfields.extensions.text_cache.device import Device
from relflow.tensorfields.extensions.text_cache.store import Client, Owner, digest

pytestmark = pytest.mark.integration


@pytest.fixture
def native_disk():
    if sys.platform == "darwin":
        pytest.skip("Moka/Arrow native binding conflict on macOS; real owner tests cover the isolated runtime")


def rows(count, length=2):
    return np.stack(
        (np.arange(count * length, dtype=np.int64).reshape(count, length), np.ones((count, length), dtype=np.int64)),
        axis=1,
    )


def populate(store, count=4):
    keys = [digest("test", str(i)) for i in range(count)]
    aliases, ready = store.identify(keys, 2)
    assert not ready.any()
    identifiers = store.tokens(aliases, rows(count), keys, "test")
    return keys, identifiers


def test_disk_roundtrip_canonical_ids_and_reopen(tmp_path, native_disk):
    path = tmp_path / "store.sqlite3"
    store = Disk(path, 1024)
    keys, ids = populate(store)
    identity = store.identity
    payload = store.get(ids, "fp32", 4, 2, 1)
    assert bytes(payload[:4]) == b"\0" * 4
    expected = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert store.put(ids, "fp32", expected, 1) == expected.tobytes()
    store.close()
    reopened = Disk(path, 0)
    try:
        assert reopened.identity == identity
        actual, ready = reopened.identify(keys, 2)
        np.testing.assert_array_equal(actual, ids)
        assert ready.all()
        payload = reopened.get(ids, "fp32", 4, 2, 2)
        assert payload[:4] == b"\1" * 4
        assert payload[4:] == expected.tobytes()
        assert reopened.stats()["hot_bytes"] == 0
    finally:
        reopened.close()


def test_aliases_share_tokens_without_duplicating_the_vector(tmp_path, native_disk):
    store = Disk(tmp_path / "store.sqlite3", 4096)
    try:
        keys = [digest("test", "alpha"), digest("test", "ALPHA")]
        aliases, _ = store.identify(keys, 2)
        same = np.repeat(rows(1), 2, axis=0)
        ids = store.tokens(aliases, same, keys, "tokenizer-v1")
        assert ids[0] == ids[1]
        assert store.stats()["aliases"] == 2
        assert store.stats()["entries"] == 1
    finally:
        store.close()


def test_conflicting_token_fill_rolls_back_without_poisoning_ram(tmp_path, native_disk):
    store = Disk(tmp_path / "store.sqlite3", 4096)
    try:
        keys, ids = populate(store, 2)
        new = keys + [digest("test", "new")]
        aliases, _ = store.identify(new, 2)
        # Known IDs are canonical; the first aliases also happen to be 1 and 2.
        wrong = rows(3)
        wrong[0, 0, 0] += 100
        with pytest.raises(ValueError, match="conflicting"):
            store.tokens(aliases, wrong, new, "test")
        _, ready = store.identify([new[-1]], 2)
        assert not ready.any()
        assert store.stats()["entries"] == 2
        assert not store.claims
    finally:
        store.close()


def test_disk_index_and_cache_stay_bounded_beyond_hot_capacity(tmp_path, native_disk):
    store = Disk(tmp_path / "store.sqlite3", 2048)
    try:
        _, ids = populate(store, 1000)
        store.get(ids, "fp32", 128, 2, 1)
        values = np.ones((1000, 128), dtype=np.float32)
        store.put(ids, "fp32", values, 1)
        for start in range(0, 1000, 100):
            result = store.get(ids[start : start + 100], "fp32", 128, 2, 2)
            assert result[:100] == b"\1" * 100
        stats = store.stats()
        assert stats["entries"] == stats["vectors"] == 1000
        assert stats["hot_bytes"] <= 2048
        assert stats["hot_entries"] < 20
        statements = []
        store.connection.set_trace_callback(statements.append)
        store.stats()
        assert not any("count(" in statement.lower() for statement in statements)
    finally:
        store.close()


def test_precision_claims_and_owner_release(tmp_path, native_disk):
    store = Disk(tmp_path / "store.sqlite3", 4096)
    try:
        _, ids = populate(store, 1)
        assert store.get(ids, "fp32", 4, 2, 1)[0] == 0
        assert store.get(ids, "fp32", 4, 2, 2)[0] == 2
        store.release(1)
        assert store.get(ids, "fp32", 4, 2, 2)[0] == 0
        store.put(ids, "fp32", np.ones((1, 4), dtype=np.float32), 2)
        assert store.get(ids, "bf16", 4, 2, 3)[0] == 0
        store.put(ids, "bf16", np.zeros((1, 4), dtype=np.float32), 3)
        assert store.stats()["vectors"] == 2
        assert store.stats()["claims"] == 0
    finally:
        store.close()


def test_rust_admission_keeps_frequent_vectors_through_a_scan(tmp_path, native_disk):
    store = Disk(tmp_path / "store.sqlite3", 64 * 1024)
    try:
        _, ids = populate(store, 3000)
        store.get(ids, "fp32", 128, 2, 1)
        store.put(ids, "fp32", np.ones((3000, 128), dtype=np.float32), 1)
        for _ in range(100):
            store.get(ids[:8], "fp32", 128, 2, 1)
        for start in range(8, len(ids), 100):
            store.get(ids[start : start + 100], "fp32", 128, 2, 1)
        before = store.stats()["vector_hits"]
        store.get(ids[:8], "fp32", 128, 2, 1)
        assert store.stats()["vector_hits"] - before >= 6
        assert store.stats()["hot_bytes"] <= 64 * 1024
    finally:
        store.close()


def test_device_cache_preserves_hot_rows_and_respects_row_and_byte_caps(tmp_path):
    view = Device.open(str(tmp_path), "fp32", 4, torch.device("cpu"), 1, 2)
    assert view is not None
    view.admit(np.array([1, 2]), torch.ones((2, 4)), np.array([1, 1]))
    view.lookup(torch.tensor([1] * 10))
    view.admit(np.array([3]), torch.full((1, 4), 3.0), np.array([2]))
    assert len(view.ids) <= 2
    assert view.lookup(torch.tensor([1]))[0].all()
    assert not view.lookup(torch.tensor([2]))[0].any()
    with torch.inference_mode():
        view.admit(np.array([4]), torch.full((1, 4), 4.0), np.array([3]))
    view.lookup(torch.tensor([1]))
    assert view.vectors.numel() * 4 <= view.limit


@pytest.mark.integration
def test_real_owner_protocol_claim_disconnect_and_durable_reopen(tmp_path):
    owner = Owner.open(tmp_path / "disk", 1, None)
    client = owner.start()
    endpoint = client.endpoint
    keys = [digest("tokenizer", "alpha")]
    aliases, _ = client.identify(keys, 2)
    ids = client.tokens(aliases, keys, rows(1), "tokenizer")
    first = Client(endpoint)
    second = Client(endpoint)
    workers = [Client(endpoint) for _ in range(40)]
    try:
        assert all(worker.stats()["identity"] == endpoint.identity for worker in workers)
        assert first.get(ids, "fp32", 4, 2)[0][0] == 0
        assert second.get(ids, "fp32", 4, 2)[0][0] == 2
        first.close()
        deadline = time.monotonic() + 5
        while second.get(ids, "fp32", 4, 2)[0][0] == 2:
            assert time.monotonic() < deadline
            second.wait()
        second.put(ids, "fp32", np.ones((1, 4), dtype=np.float32))
        with pytest.raises(RuntimeError, match="4096"):
            second.request(1, dict(n=4097, length=2), b"")
        assert client.stats()["claims"] == 0
    finally:
        for worker in workers:
            worker.close()
        first.close()
        second.close()
        process = owner.process
        owner.close()
    assert process.returncode == 0
    reopened = owner.start()
    assert reopened.endpoint.identity == endpoint.identity
    assert reopened.get(ids, "fp32", 4, 2)[0][0] == 1
    owner.close()
