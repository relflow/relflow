"""Lazy shared cache owners and pickle-safe batched clients for Text workers."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar
from weakref import WeakValueDictionary, finalize

import numpy as np

MAX_FRAME = 32 * 2**20
MAX_ROWS = 4096


def directory() -> Path:
    """The application may move its derived text cache without changing schemas."""
    configured = os.environ.get("RELFLOW_TEXT_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or os.environ.get("LOCALAPPDATA")
    return (Path(base) if base else Path.home() / ".cache") / "relflow" / "text-v2"


def receive(stream: socket.socket, size: int) -> bytearray:
    data = bytearray(size)
    offset = 0
    while offset < size:
        count = stream.recv_into(memoryview(data)[offset:])
        if not count:
            raise ConnectionError("text cache owner exited; recreate the data loader and encode the observations again")
        offset += count
    return data


@dataclass(frozen=True)
class Endpoint:
    port: int
    auth: bytes = field(repr=False)
    identity: str
    pid: int


class Client:
    """Each worker owns its connection; copied clients reconnect to the same owner."""

    instances: ClassVar[WeakValueDictionary[tuple, Client]] = WeakValueDictionary()
    registry_lock: ClassVar[threading.RLock] = threading.RLock()

    def __init__(self, endpoint: Endpoint):
        self.endpoint = endpoint
        self.process = os.getpid()
        self.stream: socket.socket | None = None
        self.lock = threading.RLock()
        self.resolve_lock = threading.RLock()

    @classmethod
    def open(cls, endpoint: Endpoint) -> Client:
        key = (endpoint, os.getpid())
        with cls.registry_lock:
            client = cls.instances.get(key)
            if client is None:
                client = cls.instances[key] = cls(endpoint)
            return client

    def __reduce__(self):
        return type(self), (self.endpoint,)

    def request(self, op: int, meta: dict | None = None, payload: bytes = b"") -> memoryview:
        if self.process != os.getpid():
            self.__init__(self.endpoint)
        metadata = json.dumps(meta or {}, separators=(",", ":")).encode()
        size = 5 + len(metadata) + len(payload)
        if size > MAX_FRAME or len(metadata) > 65536:
            raise ValueError("text cache request exceeds 32 MiB; reduce tokenizer_batch_size or max_length")
        with self.lock:
            if self.stream is None:
                self.stream = socket.create_connection(("127.0.0.1", self.endpoint.port), timeout=300)
                self.stream.sendall(self.endpoint.auth)
            self.stream.sendall(struct.pack("<IBI", size, op, len(metadata)) + metadata + payload)
            size = struct.unpack("<I", receive(self.stream, 4))[0]
            if not 1 <= size <= MAX_FRAME + 1:
                raise ValueError("text cache owner returned an invalid frame size")
            response = receive(self.stream, size)
            if response[0]:
                raise RuntimeError(bytes(response[1:]).decode())
            return memoryview(response)[1:]

    def identify(self, keys: list[bytes], length: int) -> tuple[np.ndarray, np.ndarray]:
        output = self.request(1, dict(n=len(keys), length=length), b"".join(keys))
        if len(output) != len(keys) * 9:
            raise ValueError("text cache owner returned an invalid ID batch")
        return np.frombuffer(output[: len(keys) * 8], dtype="<i8").copy(), np.frombuffer(
            output[len(keys) * 8 :], dtype=np.uint8
        ).astype(bool)

    def tokens(self, ids: np.ndarray, keys: list[bytes], rows: np.ndarray, namespace: str) -> np.ndarray:
        output = self.request(
            2,
            dict(n=len(ids), length=rows.shape[-1], namespace=namespace),
            ids.astype("<i8", copy=False).tobytes() + b"".join(keys) + rows.astype("<i8", copy=False).tobytes(),
        )

        if len(output) != len(ids) * 8:
            raise ValueError("text cache owner returned an invalid canonical ID batch")
        return np.frombuffer(output, dtype="<i8").copy()

    def get(self, ids: np.ndarray, variant: str, width: int, length: int):
        output = self.request(
            3, dict(n=len(ids), variant=variant, width=width, length=length), ids.astype("<i8", copy=False).tobytes()
        )
        status = np.frombuffer(output[: len(ids)], dtype=np.uint8).copy()
        if np.any(status > 2):
            raise ValueError("text cache owner returned an invalid vector status")
        found, missing = int((status == 1).sum()), int((status == 0).sum())
        end = len(ids) + found * width * 4
        if len(output) != end + missing * length * 16:
            raise ValueError("text cache owner returned an invalid vector batch")
        vectors = np.frombuffer(output[len(ids) : end], dtype="<f4").reshape(found, width).copy()
        tokens = np.frombuffer(output[end:], dtype="<i8").reshape(missing, 2, length).copy()
        return status, vectors, tokens

    def put(self, ids: np.ndarray, variant: str, vectors: np.ndarray) -> np.ndarray:
        output = self.request(
            4,
            dict(n=len(ids), variant=variant, width=vectors.shape[-1]),
            ids.astype("<i8", copy=False).tobytes() + vectors.astype("<f4", copy=False).tobytes(),
        )
        if len(output) != vectors.nbytes:
            raise ValueError("text cache owner returned an invalid committed vector batch")
        return np.frombuffer(output, dtype="<f4").reshape(vectors.shape).copy()

    def release(self) -> None:
        self.request(7, dict(n=0))

    def wait(self) -> None:
        self.request(8, dict(n=0))

    def stats(self) -> dict:
        return json.loads(bytes(self.request(5)))

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None


class Owner:
    """One Rust RAM budget and disk connection shared by a trainer's loader family."""

    instances: ClassVar[WeakValueDictionary[tuple, Owner]] = WeakValueDictionary()
    lock: ClassVar[threading.RLock] = threading.RLock()

    def __init__(self, path: Path, memory: int, capacity: int | None):
        self.path, self.memory, self.capacity = path, memory, capacity
        self.pid = os.getpid()
        self.process: subprocess.Popen | None = None
        self.endpoint: Endpoint | None = None
        self.client: Client | None = None
        self.log = None
        self.finalizer = None

    @classmethod
    def open(cls, path: str | Path | None, memory: int | None, capacity: int | None) -> Owner:
        path = (Path(path).expanduser() if path is not None else directory()).resolve()
        memory = 64 if memory is None else memory
        key = (str(path), memory, capacity, os.getpid())
        with cls.lock:
            owner = cls.instances.get(key)
            if owner is None:
                owner = cls.instances[key] = cls(path, memory, capacity)
            return owner

    @classmethod
    def hold(cls, endpoint: Endpoint) -> Owner | None:
        """A same-process copy keeps its original owner alive."""
        with cls.lock:
            return next((owner for owner in cls.instances.values() if owner.endpoint == endpoint), None)

    def start(self) -> Client:
        with self.lock:
            if self.process is not None:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        "text cache owner exited; recreate the model/data loader to reopen its persistent cache"
                    )
                assert self.client is not None
                return self.client
            self.path.mkdir(parents=True, exist_ok=True)
            auth = os.urandom(32)
            log = tempfile.TemporaryFile()
            self.log = log
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("backend.py")),
                    str(self.path / "cache.sqlite3"),
                    str(self.memory * 2**20),
                    str(self.capacity or 0),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
            )
            assert self.process.stdin is not None and self.process.stdout is not None
            self.process.stdin.write(auth)
            self.process.stdin.flush()
            ready = self.process.stdout.readline()
            if not ready:
                self.process.wait(timeout=10)
                log.seek(0)
                detail = log.read().decode(errors="replace")
                self.close()
                raise RuntimeError(
                    f"could not start text cache owner: {detail}; install relflow[text] and use a writable local store directory"
                )
            description = json.loads(ready)
            self.endpoint = Endpoint(description["port"], auth, description["identity"], self.pid)
            self.client = Client.open(self.endpoint)
            self.finalizer = finalize(self, stop, self.process, self.client, log, self.pid)
            return self.client

    def close(self) -> None:
        if self.pid != os.getpid() or self.process is None:
            return
        if self.finalizer is not None:
            self.finalizer()
        else:
            stop(self.process, self.client, self.log, self.pid)
        self.process = None
        self.client = None
        self.endpoint = None


def stop(process, client, log, pid):
    """Finalization never retains its owner or terminates a fork's parent owner."""
    if pid != os.getpid():
        return
    if process.poll() is None:
        if client is not None:
            try:
                client.request(6)
            except (OSError, ConnectionError, RuntimeError):
                pass
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if client is not None:
        client.close()
    if process.stdout is not None:
        process.stdout.close()
    if log is not None:
        log.close()


def digest(namespace: str, text: str) -> bytes:
    return hashlib.sha256(namespace.encode() + b"\0" + text.encode()).digest()
