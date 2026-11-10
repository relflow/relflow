"""The shared contract is exercised without importing any relflow extension."""

import multiprocessing as mp
import pickle
import struct
import subprocess
import sys
import time
from copy import deepcopy

import numpy as np
import pyarrow as pa
import pytest
from relflow_cache import Cache, Payloads
from relflow_cache.core import MAX_FRAME, digest

pytestmark = pytest.mark.integration


@pytest.fixture
def cache(tmp_path):
    store = Cache(tmp_path / "artifacts", memory=4096)
    yield store
    store.owner.close()


def prepare(cache, values, namespace="prepare/v1"):
    references = cache.lookup_arrow(pa.array(values, type=pa.binary()), namespace)
    ids = references.objects.copy()
    missing = np.flatnonzero(~references.ready)
    if len(missing):
        ids[missing] = cache.bind(references.take(missing), Payloads.from_values(values[i] for i in missing), namespace)
    return ids


def test_import_has_no_model_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, relflow_cache; assert not {'relflow', 'torch', 'transformers'} & sys.modules.keys()",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "kind",
    [pa.string(), pa.large_string(), pa.string_view(), pa.binary(), pa.large_binary(), pa.binary_view(), pa.binary(4)],
)
def test_arrow_hashing_matches_bytes_and_preserves_slices(cache, kind):
    if pa.types.is_fixed_size_binary(kind):
        values = [b"abcd", b"1234", b"abcd", b"0000"]
    elif pa.types.is_binary(kind) or pa.types.is_large_binary(kind) or pa.types.is_binary_view(kind):
        values = [b"", b"\x00\xff", b"hello", b"hello"]
    else:
        values = ["", "hello", "café", "你好", "😀", "hello"]
    array = pa.array([values[0], *values, values[0]], type=kind).slice(1, len(values))
    namespace = "test/arrow/v1"
    references = cache.lookup_arrow(array, namespace)
    expected = b"".join(digest(namespace, value.encode() if isinstance(value, str) else value) for value in values)
    assert references.keys.tobytes() == expected
    other = Cache(endpoint=cache.endpoint, native=False)
    np.testing.assert_array_equal(other.lookup_keys(expected).aliases, references.aliases)
    assert not references.ready.any()
    assert len(cache.lookup_arrow(pa.array([], type=kind), namespace).objects) == 0
    with pytest.raises(ValueError, match="nonnull"):
        cache.lookup_arrow(pa.array([None], type=kind), namespace)


def tensor_payload(array):
    # A field-owned stand-in codec includes dtype and dimensions. The store sees bytes.
    shape = np.asarray(array.shape, dtype="<u4")
    return (
        struct.pack("<II", array.ndim, len(array.dtype.str))
        + array.dtype.str.encode()
        + shape.tobytes()
        + array.tobytes()
    )


def test_modalities_share_one_budget_and_variable_payload_contract(cache):
    samples = [
        tensor_payload(np.arange(8, dtype="<i8").reshape(2, 4)),  # text tokens
        tensor_payload(np.arange(24, dtype=np.uint8).reshape(2, 4, 3)),  # image
        tensor_payload(np.linspace(-1, 1, 123, dtype="<f4")),  # audio
        tensor_payload(np.zeros((3, 2, 4, 3), dtype=np.uint8)),  # video clip
        b"",  # Empty artifacts are distinct from misses.
    ]
    ids = prepare(cache, samples)
    fresh = cache.fetch(ids, "frozen-encoder/weights-a/preprocess-a/output-codec-v1")
    assert fresh.status.tolist() == [0] * len(samples)
    assert [bytes(value) for value in fresh.inputs] == samples
    outputs = [b"text-embedding", b"image-patches" * 30, b"audio-frames" * 100, b"clip-tokens" * 3, b""]
    cache.publish(ids, "frozen-encoder/weights-a/preprocess-a/output-codec-v1", Payloads.from_values(outputs))
    warm = cache.fetch(ids, "frozen-encoder/weights-a/preprocess-a/output-codec-v1")
    assert warm.status.tolist() == [1] * len(samples)
    assert [bytes(value) for value in warm.hits] == outputs
    assert cache.stats()["hot_bytes"] <= 4096
    assert cache.stats()["objects"] == len(samples)
    assert cache.stats()["results"] == len(samples)
    # An image and an extracted video frame can reuse identical preparation explicitly.
    assert prepare(cache, [samples[1]])[0] == ids[1]


@pytest.mark.parametrize("change", ["weights", "codec", "sample-rate", "crop", "frame-window", "precision"])
def test_transform_identity_prevents_wrong_reuse(cache, change):
    ids = prepare(cache, [b"source-content-hash"])
    first = "encoder/weights-v1/codec-v1/sample-rate-16k/crop-full/frame-window-0-8/precision-f32"
    cache.fetch(ids, first)
    cache.publish(ids, first, Payloads.from_values([b"original"]))
    changed = first.replace(change + "-", change + "-different-")
    assert changed != first
    assert cache.fetch(ids, changed).status.tolist() == [0]
    cache.release()
    assert [bytes(value) for value in cache.fetch(ids, first).hits] == [b"original"]


def test_preparation_namespaces_and_atomic_conflict(cache):
    references = cache.lookup_arrow(pa.array([b"alias-a", b"alias-b"]), "raw/v1")
    second = references.take([1])
    cache.bind(second, Payloads.from_values([b"existing"]), "decode/v1")
    with pytest.raises(RuntimeError, match="conflicting prepared artifact"):
        cache.bind(references, Payloads.from_values([b"new", b"conflict"]), "decode/v1")
    ready = cache.lookup_arrow(pa.array([b"alias-a", b"alias-b"]), "raw/v1")
    assert ready.ready.tolist() == [False, True]
    assert cache.stats()["objects"] == 1
    with pytest.raises(RuntimeError, match="distinct preprocessing identity"):
        cache.bind(second, Payloads.from_values([b"existing"]), "decode/v2")
    aliases = cache.lookup_arrow(pa.array([b"alias-b"]), "raw/v2")
    changed = cache.bind(aliases, Payloads.from_values([b"existing"]), "decode/v2")
    assert changed[0] != ready.objects[1]


def test_claim_disconnect_handoff_and_durable_reopen(cache):
    ids = prepare(cache, [b"input"])
    # Separate transports own separate claims; all handles in one worker normally reuse one connection.
    other = Cache(endpoint=cache.endpoint, native=False)
    assert cache.fetch(ids, "result/v1").status.tolist() == [0]
    assert other.fetch(ids, "result/v1").status.tolist() == [2]
    cache.connection.close()
    deadline = time.monotonic() + 5
    while other.fetch(ids, "result/v1").status.tolist() == [2]:
        assert time.monotonic() < deadline
        other.wait()
    other.publish(ids, "result/v1", Payloads.from_values([b"durable-result"]))
    assert cache.stats()["claims"] == 0
    identity = cache.endpoint.identity
    cache.owner.close()
    reopened = Cache(cache.path, memory=0)
    try:
        assert reopened.endpoint.identity == identity
        result = reopened.fetch(ids, "result/v1")
        assert [bytes(value) for value in result.hits] == [b"durable-result"]
        assert reopened.stats()["hot_bytes"] == 0
    finally:
        reopened.owner.close()


def worker(cache, ids, sending):
    try:
        result = cache.fetch(ids, "result/v1")
        sending.send((cache.endpoint.identity, result.status.tolist(), [bytes(value) for value in result.hits]))
    finally:
        sending.close()


@pytest.mark.parametrize("method", ["spawn", "fork"])
def test_shared_worker_and_pickle_configuration(cache, method):
    if method not in mp.get_all_start_methods():
        pytest.skip(f"{method} unavailable")
    ids = prepare(cache, [b"worker-input"])
    cache.fetch(ids, "result/v1")
    cache.publish(ids, "result/v1", Payloads.from_values([b"shared-output"]))
    for copy in (deepcopy(cache), pickle.loads(pickle.dumps(cache))):
        assert copy.owner is cache.owner
        assert copy.endpoint == cache.endpoint
    cache.share()
    context = mp.get_context(method)
    receiving, sending = context.Pipe(False)
    process = context.Process(target=worker, args=(cache, ids, sending))
    process.start()
    sending.close()
    try:
        assert receiving.poll(30)
        identity, status, hits = receiving.recv()
        process.join(30)
        assert process.exitcode == 0
        assert (identity, status, hits) == (cache.endpoint.identity, [1], [b"shared-output"])
    finally:
        receiving.close()
        if process.is_alive():
            process.terminate()
            process.join(5)


def test_oversized_hot_entry_remains_on_disk_and_shared_budget(cache):
    second = Cache(cache.path, memory=4096)
    assert second.owner is cache.owner
    payload = b"x" * 8192
    ids = prepare(cache, [b"large"])
    cache.fetch(ids, "result/v1")
    cache.publish(ids, "result/v1", Payloads.from_values([payload]))
    for _ in range(3):
        result = second.fetch(ids, "result/v1")
        assert bytes(next(iter(result.hits))) == payload
    stats = cache.stats()
    assert stats["hot_bytes"] <= 4096
    assert stats["disk_reads"] == 3  # Over-budget payload never enters the hot tier.
    assert stats["results"] == 1


def test_malformed_frames_release_claims_and_reject_oversized_batches(cache):
    ids = prepare(cache, [b"input"])
    cache.fetch(ids, "result/v1")
    with pytest.raises(RuntimeError, match="4096"):
        cache.connection.request(1, dict(n=4097))
    assert cache.stats()["claims"] == 0
    with pytest.raises(ValueError, match="32 MiB"):
        Payloads.from_values([b"x" * MAX_FRAME])
    with pytest.raises(ValueError, match="offsets"):
        Payloads(memoryview(b"abc"), np.array([1, 3], dtype="<u4"))
    with pytest.raises(RuntimeError, match="offsets"):
        cache.connection.request(
            4, dict(n=1, namespace="result/v1"), ids.astype("<i8").tobytes() + struct.pack("<II", 1, 0)
        )
    with pytest.raises(RuntimeError, match="positive"):
        cache.fetch([0], "result/v1")


def test_budget_conflicts_are_explicit_and_random_views_remain_outside_cache(cache):
    with pytest.raises(ValueError, match="different RAM budget"):
        Cache(cache.path, memory=8192)
    cache.share()
    with pytest.raises(ValueError, match="different RAM budget"):
        Cache(cache.path, memory=8192)
    original = np.arange(16, dtype=np.float32)
    ids = prepare(cache, [tensor_payload(original)])
    cache.fetch(ids, "base/v1")
    cache.publish(ids, "base/v1", Payloads.from_values([original.tobytes()]))
    generator = np.random.default_rng(7302)
    pristine = np.frombuffer(next(iter(cache.fetch(ids, "base/v1").hits)), dtype=np.float32)
    augmented = pristine + generator.normal(size=len(pristine))
    assert np.any(augmented != pristine)
    np.testing.assert_array_equal(
        np.frombuffer(next(iter(cache.fetch(ids, "base/v1").hits)), dtype=np.float32), original
    )


def test_frequent_results_survive_a_stream_of_one_off_results(cache):
    ids = prepare(cache, [f"source-{i}".encode() for i in range(200)])
    cache.fetch(ids, "result/v1")
    cache.publish(ids, "result/v1", Payloads.from_values([bytes([i]) * 1024 for i in range(len(ids))]))
    for _ in range(100):
        cache.fetch(ids[:3], "result/v1")
    for start in range(3, len(ids), 20):
        cache.fetch(ids[start : start + 20], "result/v1")
        cache.fetch(ids[:3], "result/v1")
    before = cache.stats()
    warm = cache.fetch(ids[:3], "result/v1")
    after = cache.stats()
    assert [bytes(value) for value in warm.hits] == [bytes([i]) * 1024 for i in range(3)]
    assert after["hits"] - before["hits"] == 3
    assert after["disk_reads"] == before["disk_reads"]
    assert after["hot_bytes"] <= 4096
    assert after["results"] == 200


def test_variable_payload_reply_limit_can_be_retried_in_smaller_batches(cache):
    ids = prepare(cache, [b"large-a", b"large-b"])
    payload = b"a" * (17 * 2**20)
    for identifier in ids:
        cache.fetch([identifier], "result/v1")
        cache.publish([identifier], "result/v1", Payloads.from_values([payload]))
    with pytest.raises(RuntimeError, match="reduce batch size or split"):
        cache.fetch(ids, "result/v1")
    assert cache.stats()["claims"] == 0
    for identifier in ids:
        assert bytes(next(iter(cache.fetch([identifier], "result/v1").hits))) == payload
    assert cache.stats()["hot_bytes"] <= 4096
