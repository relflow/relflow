"""Opaque, local artifact storage; no encoder, tensorfield, or tensor dependencies."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import struct
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from weakref import WeakValueDictionary, finalize

import numpy as np

MAX_FRAME = 32 * 2**20
MAX_ROWS = 4096


def digest(namespace: str, value: bytes) -> bytes:
    if not namespace or "\0" in namespace or len(namespace.encode()) > 1024:
        raise ValueError("artifact namespace must be nonempty, NUL-free, and at most 1024 bytes")
    return hashlib.sha256(namespace.encode() + b"\0" + value).digest()


def directory() -> Path:
    configured = os.environ.get("RELFLOW_CACHE_DIR")
    base = os.environ.get("XDG_CACHE_HOME") or os.environ.get("LOCALAPPDATA")
    return (
        Path(configured).expanduser()
        if configured
        else (Path(base) if base else Path.home() / ".cache") / "relflow" / "artifacts-v1"
    )


@dataclass(frozen=True)
class References:
    """Alias and canonical IDs are separate domains; zero means unprepared."""

    aliases: np.ndarray
    objects: np.ndarray
    keys: np.ndarray

    @property
    def ready(self):
        return self.objects > 0

    def take(self, indices):
        return References(self.aliases[indices], self.objects[indices], self.keys[indices])


@dataclass(frozen=True)
class Payloads:
    """Contiguous bytes plus offsets, preserving empty and variable-size entries."""

    data: memoryview
    offsets: np.ndarray

    def __post_init__(self):
        offsets = np.asarray(self.offsets)
        if (
            offsets.ndim != 1
            or not len(offsets)
            or offsets[0] != 0
            or offsets[-1] != len(self.data)
            or np.any(offsets[1:] < offsets[:-1])
            or offsets.dtype != np.dtype("<u4")
        ):
            raise ValueError("artifact offsets must be uint32, start at zero, and cover the payload")
        if len(self.data) + offsets.nbytes > MAX_FRAME:
            raise ValueError("artifact batch exceeds 32 MiB; reduce batch size or split large artifacts")

    def __len__(self):
        return len(self.offsets) - 1

    def __iter__(self):
        return (
            self.data[int(start) : int(stop)] for start, stop in zip(self.offsets[:-1], self.offsets[1:], strict=True)
        )

    @classmethod
    def from_values(cls, values):
        values = list(values)
        offsets = np.zeros(len(values) + 1, dtype=np.uint64)
        offsets[1:] = np.cumsum([len(value) for value in values], dtype=np.uint64)
        if int(offsets[-1]) + len(offsets) * 4 > MAX_FRAME:
            raise ValueError("artifact batch exceeds 32 MiB; reduce batch size or split large artifacts")
        return cls(memoryview(b"".join(values)), offsets.astype("<u4"))

    @classmethod
    def fixed(cls, data, row_bytes):
        data = memoryview(data).cast("B")
        if row_bytes <= 0 or len(data) % row_bytes:
            raise ValueError("fixed artifact batch must contain whole positive-width rows")
        return cls(data, np.arange(len(data) // row_bytes + 1, dtype="<u4") * row_bytes)

    def pack(self):
        return self.offsets.tobytes() + self.data.tobytes()

    @classmethod
    def unpack(cls, data, count):
        data = memoryview(data)
        header = (count + 1) * 4
        if len(data) < header:
            raise ValueError("truncated artifact offset table")
        offsets = np.frombuffer(data[:header], dtype="<u4")
        end = header + int(offsets[-1])
        if end > len(data):
            raise ValueError("truncated artifact payload")
        return cls(data[header:end], offsets), end


@dataclass(frozen=True)
class Fetch:
    status: np.ndarray  # 0: owned miss, 1: hit, 2: another connection is computing
    hits: Payloads
    inputs: Payloads


@dataclass(frozen=True)
class Endpoint:
    port: int
    auth: bytes = field(repr=False)
    identity: str
    pid: int


def receive(stream, size):
    data = bytearray(size)
    offset = 0
    while offset < size:
        count = stream.recv_into(memoryview(data)[offset:])
        if not count:
            raise ConnectionError("artifact owner disconnected; recreate the cache handle")
        offset += count
    return data


class Connection:
    """One capability-authenticated connection per worker and transport."""

    instances = WeakValueDictionary()
    registry_lock = threading.RLock()

    def __init__(self, endpoint, native=True):
        self.endpoint, self.use_native, self.process = endpoint, native, os.getpid()
        self.lock, self.resolve_lock = threading.RLock(), threading.RLock()
        self.stream = None
        self.native = None
        if native:
            from .native import Client

            self.native = Client(endpoint.port, endpoint.auth)

    @classmethod
    def open(cls, endpoint, native=True):
        key = (endpoint, native, os.getpid())
        with cls.registry_lock:
            client = cls.instances.get(key)
            if client is None:
                client = cls.instances[key] = cls(endpoint, native)
            return client

    def request(self, op, meta=None, payload=b""):
        if self.process != os.getpid():
            self.__init__(self.endpoint, self.use_native)
        metadata = json.dumps(meta or {}, separators=(",", ":")).encode()
        size = 5 + len(metadata) + len(payload)
        if size > MAX_FRAME or len(metadata) > 65536:
            raise ValueError("artifact request exceeds 32 MiB; reduce batch size or split large artifacts")
        with self.lock:
            if self.native is not None:
                return memoryview(self.native.request(op, metadata, payload))
            if self.stream is None:
                self.stream = socket.create_connection(("127.0.0.1", self.endpoint.port), timeout=300)
                self.stream.sendall(self.endpoint.auth)
            self.stream.sendall(struct.pack("<IBI", size, op, len(metadata)) + metadata + payload)
            size = struct.unpack("<I", receive(self.stream, 4))[0]
            if not 1 <= size <= MAX_FRAME + 1:
                raise ValueError("artifact owner returned an invalid frame size")
            response = receive(self.stream, size)
            if response[0]:
                raise RuntimeError(bytes(response[1:]).decode())
            return memoryview(response)[1:]

    def close(self):
        if self.native is not None:
            self.native.close()
        if self.stream is not None:
            self.stream.close()
            self.stream = None


class Owner:
    """One disk index and managed RAM budget for all fields in a worker family."""

    instances = WeakValueDictionary()
    lock = threading.RLock()

    def __init__(self, path, memory, capacity):
        self.path, self.memory, self.capacity = path, memory, capacity
        self.pid, self.process, self.endpoint, self.finalizer = os.getpid(), None, None, None

    @classmethod
    def open(cls, path=None, memory=64 * 2**20, capacity=None):
        path = (Path(path).expanduser() if path is not None else directory()).resolve()
        if memory < 0 or (capacity is not None and capacity <= 0):
            raise ValueError("artifact RAM budget must be nonnegative and capacity must be positive")
        key = (str(path), os.getpid())
        with cls.lock:
            owner = cls.instances.get(key)
            if owner is not None and (owner.memory, owner.capacity) != (memory, capacity):
                raise ValueError(
                    "artifact store already has a different RAM budget; share its cache handle or use a separate directory"
                )
            if owner is None:
                owner = cls.instances[key] = cls(path, memory, capacity)
            return owner

    @classmethod
    def hold(cls, endpoint):
        with cls.lock:
            return next((owner for owner in cls.instances.values() if owner.endpoint == endpoint), None)

    def start(self):
        with self.lock:
            if self.process is not None:
                if self.process.poll() is not None:
                    raise RuntimeError("artifact owner exited; recreate the cache and data loader")
                return self.endpoint
            self.path.mkdir(parents=True, exist_ok=True)
            auth = os.urandom(32)
            binary = os.environ.get("RELFLOW_CACHE_OWNER") or str(Path(__file__).with_name("cache-owner"))
            log = tempfile.TemporaryFile()
            self.process = subprocess.Popen(
                [binary, str(self.path / "artifacts.sqlite3"), str(self.memory), str(self.capacity or 0)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
            )
            self.finalizer = finalize(self, stop, self.process, log, self.pid)
            self.process.stdin.write(auth)
            self.process.stdin.flush()
            ready = self.process.stdout.readline()
            if not ready:
                self.process.wait(timeout=10)
                log.seek(0)
                detail = log.read().decode(errors="replace")
                self.close()
                raise RuntimeError(f"could not start artifact owner: {detail}")
            description = json.loads(ready)
            self.endpoint = Endpoint(description["port"], auth, description["identity"], self.pid)
            return self.endpoint

    def close(self):
        if self.pid == os.getpid():
            if self.finalizer is not None:
                self.finalizer()
            self.process = self.endpoint = self.finalizer = None
            with self.lock:
                key = (str(self.path), self.pid)
                if self.instances.get(key) is self:
                    del self.instances[key]


def stop(process, log, pid):
    if pid != os.getpid():
        return
    if process.stdin is not None:
        process.stdin.close()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    if process.stdout is not None:
        process.stdout.close()
    log.close()


class Cache:
    """Batched opaque artifacts. Only configuration and a capability travel to workers.

    Call share before spawning workers. Preparation and compute happen outside the
    store; namespaces must identify their complete deterministic transformations.
    """

    def __init__(self, path=None, memory=64 * 2**20, capacity=None, endpoint=None, native=True):
        self.path, self.memory, self.capacity, self.native = path, memory, capacity, native
        self.owner = Owner.open(path, memory, capacity) if endpoint is None else Owner.hold(endpoint)
        self.saved_endpoint, self.saved_connection = endpoint, None

    def __reduce__(self):
        self.share()
        return type(self), (self.path, self.memory, self.capacity, self.endpoint, self.native)

    @property
    def endpoint(self):
        if self.saved_endpoint is None:
            self.saved_endpoint = self.owner.start()
        return self.saved_endpoint

    @property
    def connection(self):
        if self.saved_connection is None:
            self.saved_connection = Connection.open(self.endpoint, self.native)
        return self.saved_connection

    @property
    def resolve_lock(self):
        return self.connection.resolve_lock

    def share(self):
        self.connection

    def lookup_keys(self, keys):
        keys = (
            np.frombuffer(keys, dtype=np.uint8).reshape(-1, 32)
            if isinstance(keys, bytes)
            else np.asarray(keys, dtype=np.uint8)
        )
        if keys.ndim != 2 or keys.shape[1] != 32 or len(keys) > MAX_ROWS:
            raise ValueError("artifact lookup expects at most 4096 SHA256 keys")
        reply = self.connection.request(1, dict(n=len(keys)), keys.tobytes())
        return self.references(reply, keys)

    def lookup_arrow(self, values, namespace):
        digest(namespace, b"")  # Validate the namespace once, outside the native row loop.
        if values.null_count or len(values) > MAX_ROWS:
            raise ValueError("artifact lookup expects at most 4096 nonnull Arrow values")
        if self.connection.native is None:
            keys = b"".join(
                digest(namespace, value.encode() if isinstance(value, str) else value) for value in values.to_pylist()
            )
            return self.lookup_keys(keys)
        reply, keys = self.connection.native.identify_arrow(values, namespace)
        return self.references(reply, np.frombuffer(keys, dtype=np.uint8).reshape(-1, 32))

    @staticmethod
    def references(reply, keys):
        if len(reply) != len(keys) * 16:
            raise ValueError("artifact owner returned invalid reference IDs")
        ids = np.frombuffer(reply, dtype="<i8").reshape(2, len(keys)).copy()
        return References(ids[0], ids[1], keys)

    def bind(self, references, inputs, namespace):
        digest(namespace, b"")
        if len(inputs) != len(references.aliases):
            raise ValueError("prepared artifact count must match alias count")
        reply = self.connection.request(
            2,
            dict(n=len(inputs), namespace=namespace),
            np.asarray(references.aliases, dtype="<i8").tobytes() + references.keys.tobytes() + inputs.pack(),
        )
        if len(reply) != len(inputs) * 8:
            raise ValueError("artifact owner returned invalid canonical IDs")
        return np.frombuffer(reply, dtype="<i8").copy()

    def fetch(self, ids, namespace):
        digest(namespace, b"")
        ids = np.asarray(ids, dtype="<i8")
        reply = self.connection.request(3, dict(n=len(ids), namespace=namespace), ids.tobytes())
        status = np.frombuffer(reply[: len(ids)], dtype=np.uint8).copy()
        if len(status) != len(ids) or np.any(status > 2):
            raise ValueError("artifact owner returned invalid computation status")
        hits, used = Payloads.unpack(reply[len(ids) :], int((status == 1).sum()))
        inputs, consumed = Payloads.unpack(reply[len(ids) + used :], int((status == 0).sum()))
        if len(ids) + used + consumed != len(reply):
            raise ValueError("artifact owner returned trailing payload bytes")
        return Fetch(status, hits, inputs)

    def publish(self, ids, namespace, values):
        digest(namespace, b"")
        if len(ids) != len(values):
            raise ValueError("artifact result count must match ID count")
        reply = self.connection.request(
            4, dict(n=len(ids), namespace=namespace), np.asarray(ids, dtype="<i8").tobytes() + values.pack()
        )
        committed, used = Payloads.unpack(reply, len(ids))
        if used != len(reply):
            raise ValueError("artifact owner returned trailing result bytes")
        return committed

    def release(self):
        self.connection.request(7)

    def wait(self):
        self.connection.request(8)

    def stats(self):
        return json.loads(bytes(self.connection.request(5)))
