"""Shared disk transactions, bounded native admission, and device row retention."""

import sys

import numpy as np
import pytest
import torch

from relflow.tensorfields.shared.cache import Payloads, digest
from relflow.tensorfields.shared.cache.backend import Disk
from relflow.tensorfields.shared.cache.device import Device

pytestmark = pytest.mark.integration


@pytest.fixture
def disk(tmp_path):
    if sys.platform == "darwin":
        pytest.skip("native Moka binding is isolated from Arrow on macOS; owner protocol tests cover this platform")
    store = Disk(tmp_path / "artifacts.sqlite3", 2048)
    yield store
    store.close()


def populate(store, count=4):
    keys = [digest("test/v1", str(i).encode()) for i in range(count)]
    aliases, objects = store.identify(keys)
    assert not objects.any()
    ids = store.bind(aliases, keys, [f"input-{i}".encode() for i in range(count)], "prepare/v1")
    return keys, ids


def test_disk_roundtrip_deduplication_and_reopen(disk):
    keys, ids = populate(disk)
    identity = disk.identity
    assert disk.fetch(ids, "output/v1", 1)[:4] == b"\0" * 4
    expected = [b"a", b"arbitrary-longer", b"", b"last"]
    assert disk.publish(ids, "output/v1", expected, 1) == Payloads.from_values(expected).pack()
    reopened = Disk(disk.path, 0)
    try:
        assert reopened.identity == identity
        _, objects = reopened.identify(keys)
        np.testing.assert_array_equal(objects, ids)
        assert (
            reopened.fetch(ids, "output/v1", 2)
            == b"\1" * 4 + Payloads.from_values(expected).pack() + Payloads.from_values([]).pack()
        )
        assert reopened.stats()["hot_bytes"] == 0

    finally:
        reopened.close()


def test_aliases_deduplicate_and_binding_conflicts_roll_back(disk):
    keys = [digest("test/v1", b"a"), digest("test/v1", b"b")]
    aliases, _ = disk.identify(keys)
    ids = disk.bind(aliases, keys, [b"same", b"same"], "prepare/v1")
    assert ids[0] == ids[1]
    assert disk.stats()["objects"] == 1
    extra = keys + [digest("test/v1", b"new")]
    aliases, _ = disk.identify(extra)
    with pytest.raises(ValueError, match="conflicting"):
        disk.bind(aliases, extra, [b"same", b"different", b"new"], "prepare/v1")
    _, ready = disk.identify(extra)
    assert ready[-1] == 0
    assert disk.stats()["objects"] == 1


def test_disk_counts_and_memory_stay_bounded(disk):
    _, ids = populate(disk, 1000)
    disk.fetch(ids, "output/v1", 1)
    disk.publish(ids, "output/v1", [b"x" * 512] * len(ids), 1)
    for start in range(0, len(ids), 100):
        assert disk.fetch(ids[start : start + 100], "output/v1", 2)[:100] == b"\1" * 100
    stats = disk.stats()
    assert stats["objects"] == stats["results"] == 1000
    assert stats["hot_bytes"] <= 2048
    assert stats["hot_entries"] < 20
    statements = []
    disk.connection.set_trace_callback(statements.append)
    disk.stats()
    assert not any("count(" in s.lower() for s in statements)


def test_claim_namespaces_and_release(disk):
    _, ids = populate(disk, 1)
    assert disk.fetch(ids, "output/f32", 1)[0] == 0
    assert disk.fetch(ids, "output/f32", 2)[0] == 2
    disk.release(1)
    assert disk.fetch(ids, "output/f32", 2)[0] == 0
    disk.publish(ids, "output/f32", [b"f32"], 2)
    assert disk.fetch(ids, "output/bf16", 3)[0] == 0
    disk.publish(ids, "output/bf16", [b"bf16"], 3)
    assert disk.stats()["results"] == 2
    assert disk.stats()["claims"] == 0


def test_device_cache_keeps_hot_rows_under_byte_and_entry_caps(tmp_path):
    view = Device.open(str(tmp_path), "f32", 4, torch.device("cpu"), 1, 2)
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
