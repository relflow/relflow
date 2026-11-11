"""Portable owner for opaque artifacts; executable without importing relflow.

The optional native owner implements the same protocol and database format.
Moka stays in this isolated process, away from Arrow and encoder runtimes.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import sqlite3
import struct
import sys
import threading
from pathlib import Path
from typing import cast

import numpy as np

MAX_FRAME = 32 * 2**20
MAX_ROWS = 4096
CHARGE = 160


def receive(stream: socket.socket, size: int) -> bytearray:
    data = bytearray(size)
    offset = 0
    while offset < size:
        count = stream.recv_into(memoryview(data)[offset:])
        if not count:
            raise EOFError("artifact connection closed")
        offset += count
    return data


def pack(rows: list[bytes]) -> bytes:
    size = sum(map(len, rows))
    if size + (len(rows) + 1) * 4 > MAX_FRAME:
        raise ValueError("artifact batch exceeds 32 MiB; reduce batch size or split large artifacts")
    offsets = np.zeros(len(rows) + 1, dtype="<u4")
    offsets[1:] = np.cumsum([len(row) for row in rows], dtype=np.uint64)
    return offsets.tobytes() + b"".join(rows)


def unpack(data: memoryview, count: int) -> list[bytes]:
    header = (count + 1) * 4
    if len(data) < header:
        raise ValueError("truncated artifact offsets")
    offsets = np.frombuffer(data[:header], dtype="<u4")
    if offsets[0] or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("invalid artifact offsets")
    if header + int(offsets[-1]) != len(data):
        raise ValueError("artifact offsets must cover the complete payload")
    return [bytes(data[header + int(a) : header + int(b)]) for a, b in zip(offsets[:-1], offsets[1:], strict=True)]


class Disk:
    """Transactional aliases and artifacts with a bounded frequency-aware hot tier."""

    def __init__(self, path: str | Path, memory: int, capacity: int | None = None):
        from moka_py import Moka

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
        self.connection.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            PRAGMA cache_size=-4096;
            PRAGMA mmap_size=0;
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS aliases (id INTEGER PRIMARY KEY AUTOINCREMENT, key BLOB UNIQUE NOT NULL, object_id INTEGER);
            CREATE TABLE IF NOT EXISTS objects (id INTEGER PRIMARY KEY AUTOINCREMENT, key BLOB UNIQUE NOT NULL, payload BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS results (id INTEGER NOT NULL, namespace TEXT NOT NULL, payload BLOB NOT NULL, PRIMARY KEY(id,namespace)) WITHOUT ROWID;
            INSERT OR IGNORE INTO metadata VALUES ('format','1');
            INSERT OR IGNORE INTO metadata VALUES ('identity',lower(hex(randomblob(32))));
            INSERT OR IGNORE INTO metadata VALUES ('aliases','0');
            INSERT OR IGNORE INTO metadata VALUES ('objects','0');
            INSERT OR IGNORE INTO metadata VALUES ('results','0');
        """)
        if self.connection.execute("SELECT value FROM metadata WHERE key='format'").fetchone()[0] != "1":
            raise ValueError("unsupported artifact cache format; use a new store directory")
        self.identity = self.connection.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0]
        minimum = CHARGE if capacity is None else max(CHARGE, (memory + capacity - 1) // capacity)
        self.hot: Moka[tuple, object] = Moka(
            memory, weigher=lambda key, value: max(minimum, CHARGE + (len(value) if isinstance(value, bytes) else 16))
        )
        self.memory = memory
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.claims: dict[tuple[int, str], int] = {}
        self.owned: dict[int, set[tuple[int, str]]] = {}
        self.counters = dict(identified=0, alias_hits=0, requests=0, hits=0, disk_reads=0, compute_claims=0)

    def identify(self, keys: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
        aliases = np.empty(len(keys), dtype="<i8")
        objects = np.zeros(len(keys), dtype="<i8")
        admitted = []
        with self.lock, self.connection:
            inserted = 0
            for index, key in enumerate(keys):
                self.counters["identified"] += 1
                cached = self.hot.get(("alias", key))
                if cached is not None:
                    aliases[index], objects[index] = cast(tuple[int, int], cached)
                    self.counters["alias_hits"] += 1
                    continue
                row = self.connection.execute("SELECT id,object_id FROM aliases WHERE key=?", (key,)).fetchone()
                if row is None:
                    inserted += self.connection.execute(
                        "INSERT OR IGNORE INTO aliases(key) VALUES (?)", (key,)
                    ).rowcount
                    row = self.connection.execute("SELECT id,object_id FROM aliases WHERE key=?", (key,)).fetchone()
                aliases[index] = row[0]
                if row[1] is not None:
                    objects[index] = row[1]
                    admitted.append((key, row[0], row[1]))
            if inserted:
                self.connection.execute(
                    "UPDATE metadata SET value=CAST(value AS INTEGER)+? WHERE key='aliases'", (inserted,)
                )
        for key, alias, artifact in admitted:
            self.hot.set(("alias", key), (alias, artifact))
        self.hot.run_pending_tasks()
        return aliases, objects

    def bind(self, aliases: np.ndarray, keys: list[bytes], rows: list[bytes], namespace: str) -> np.ndarray:
        objects = np.empty(len(aliases), dtype="<i8")
        with self.lock, self.connection:
            inserted = 0
            for index, (alias, key, payload) in enumerate(zip(aliases.tolist(), keys, rows, strict=True)):
                stored = self.connection.execute("SELECT key,object_id FROM aliases WHERE id=?", (alias,)).fetchone()
                if stored is None or stored[0] != key:
                    raise ValueError(f"prepared artifact does not match alias {alias}")
                canonical = hashlib.sha256(namespace.encode() + b"\0" + payload).digest()
                if stored[1] is not None:
                    previous = self.connection.execute(
                        "SELECT key,payload FROM objects WHERE id=?", (stored[1],)
                    ).fetchone()
                    if previous != (canonical, payload):
                        raise ValueError(
                            f"conflicting prepared artifact for alias {alias}; use a distinct preprocessing identity"
                        )
                    objects[index] = stored[1]
                    continue
                inserted += self.connection.execute(
                    "INSERT OR IGNORE INTO objects(key,payload) VALUES (?,?)", (canonical, payload)
                ).rowcount
                objects[index] = self.connection.execute("SELECT id FROM objects WHERE key=?", (canonical,)).fetchone()[
                    0
                ]
                self.connection.execute("UPDATE aliases SET object_id=? WHERE id=?", (int(objects[index]), alias))
            if inserted:
                self.connection.execute(
                    "UPDATE metadata SET value=CAST(value AS INTEGER)+? WHERE key='objects'", (inserted,)
                )
        for key, alias, artifact in zip(keys, aliases.tolist(), objects.tolist(), strict=True):
            self.hot.set(("alias", key), (alias, artifact))
        self.hot.run_pending_tasks()
        return objects

    def fetch(self, ids: np.ndarray, namespace: str, owner: int) -> bytes:
        status = np.zeros(len(ids), dtype=np.uint8)
        results, inputs = [], []
        size = len(ids) + 8
        with self.lock:
            for index, identifier in enumerate(ids.tolist()):
                self.counters["requests"] += 1
                key = ("result", identifier, namespace)
                value = self.hot.get(key)
                if value is not None:
                    self.counters["hits"] += 1
                else:
                    row = self.connection.execute(
                        "SELECT payload FROM results WHERE id=? AND namespace=?", (identifier, namespace)
                    ).fetchone()
                    if row is not None:
                        value = row[0]
                        self.hot.set(key, value)
                        self.counters["disk_reads"] += 1
                if value is not None:
                    payload = cast(bytes, value)
                    size += 4 + len(payload)
                    if size > MAX_FRAME:
                        raise ValueError("artifact lookup exceeds 32 MiB; reduce batch size or split large artifacts")
                    status[index] = 1
                    results.append(payload)
                    continue
                claim = (identifier, namespace)
                if claim in self.claims and self.claims[claim] != owner:
                    status[index] = 2
                    continue
                row = self.connection.execute("SELECT payload FROM objects WHERE id=?", (identifier,)).fetchone()
                if row is None:
                    raise ValueError(f"artifact ID {identifier} has no prepared input")
                size += 4 + len(row[0])
                if size > MAX_FRAME:
                    raise ValueError("artifact lookup exceeds 32 MiB; reduce batch size or split large artifacts")
                owned = self.owned.setdefault(owner, set())
                if claim not in owned and len(owned) >= MAX_ROWS:
                    raise ValueError("connection exceeds 4096 in-flight computations")
                self.claims[claim] = owner
                owned.add(claim)
                inputs.append(row[0])
                self.counters["compute_claims"] += 1
        self.hot.run_pending_tasks()
        return status.tobytes() + pack(results) + pack(inputs)

    def publish(self, ids: np.ndarray, namespace: str, rows: list[bytes], owner: int) -> bytes:
        results = []
        with self.condition, self.connection:
            inserted = 0
            for identifier, payload in zip(ids.tolist(), rows, strict=True):
                if self.claims.get((identifier, namespace)) != owner:
                    raise ValueError(f"artifact result {identifier} has no matching compute claim")
                inserted += self.connection.execute(
                    "INSERT OR IGNORE INTO results VALUES (?,?,?)", (identifier, namespace, payload)
                ).rowcount
                results.append(
                    self.connection.execute(
                        "SELECT payload FROM results WHERE id=? AND namespace=?", (identifier, namespace)
                    ).fetchone()[0]
                )
            packed = pack(results)
            if inserted:
                self.connection.execute(
                    "UPDATE metadata SET value=CAST(value AS INTEGER)+? WHERE key='results'", (inserted,)
                )
        with self.condition:
            for identifier in ids.tolist():
                self.claims.pop((identifier, namespace), None)
                self.owned.get(owner, set()).discard((identifier, namespace))
            self.condition.notify_all()
        return packed

    def release(self, owner: int) -> None:
        with self.condition:
            for key in self.owned.pop(owner, set()):
                self.claims.pop(key, None)
            self.condition.notify_all()

    def stats(self) -> dict:
        with self.lock:
            self.hot.run_pending_tasks()
            return dict(
                self.counters,
                identity=self.identity,
                **{
                    name: int(self.connection.execute("SELECT value FROM metadata WHERE key=?", (name,)).fetchone()[0])
                    for name in ("aliases", "objects", "results")
                },
                hot_bytes=self.hot.weighted_size(),
                hot_entries=self.hot.count(),
                memory=self.memory,
                page_bytes=4 * 2**20,
                claims=len(self.claims),
            )

    def close(self):
        self.connection.close()


def dispatch(store: Disk, data: memoryview, owner: int, stop: threading.Event) -> bytes:
    if len(data) < 5:
        raise ValueError("truncated artifact request")
    op, size = data[0], struct.unpack_from("<I", data, 1)[0]
    if size > 65536 or len(data) < 5 + size:
        raise ValueError("invalid artifact metadata size")
    meta = json.loads(bytes(data[5 : 5 + size]))
    data = data[5 + size :]
    if op == 5:
        return json.dumps(store.stats()).encode()
    if op == 6:
        stop.set()
        return b""
    if op == 7:
        store.release(owner)
        return b""
    if op == 8:
        with store.condition:
            store.condition.wait(timeout=0.1)
        return b""
    n = meta["n"]
    if not isinstance(n, int) or not 0 <= n <= MAX_ROWS:
        raise ValueError("artifact request must contain at most 4096 rows")
    if op == 1:
        if len(data) != n * 32:
            raise ValueError("artifact key batch has incorrect size")
        aliases, objects = store.identify([bytes(data[i : i + 32]) for i in range(0, len(data), 32)])
        return aliases.tobytes() + objects.tobytes()
    if len(data) < n * 8:
        raise ValueError("truncated artifact ID batch")
    ids = np.frombuffer(data[: n * 8], dtype="<i8")
    if np.any(ids <= 0):
        raise ValueError("artifact IDs must be positive")
    data = data[n * 8 :]
    namespace = meta["namespace"]
    if not isinstance(namespace, str) or not namespace or "\0" in namespace or len(namespace.encode()) > 1024:
        raise ValueError("namespace must be nonempty, NUL-free, and at most 1024 bytes")
    if op == 2:
        if len(data) < n * 32:
            raise ValueError("truncated alias key batch")
        keys = [bytes(data[i : i + 32]) for i in range(0, n * 32, 32)]
        return store.bind(ids, keys, unpack(data[n * 32 :], n), namespace).tobytes()
    if op == 3:
        if data:
            raise ValueError("unexpected artifact lookup payload")
        return store.fetch(ids, namespace, owner)
    if op == 4:
        return store.publish(ids, namespace, unpack(data, n), owner)
    raise ValueError(f"unknown artifact cache operation {op}")


def serve(stream: socket.socket, store: Disk, auth: bytes, stop: threading.Event):
    owner = id(stream)
    try:
        stream.settimeout(30)
        if not hmac.compare_digest(receive(stream, 32), auth):
            return
        stream.settimeout(None)
        while not stop.is_set():
            size = struct.unpack("<I", receive(stream, 4))[0]
            if not 5 <= size <= MAX_FRAME:
                raise ValueError("artifact frame exceeds the 32 MiB limit")
            data = receive(stream, size)
            try:
                output = b"\0" + dispatch(store, memoryview(data), owner, stop)
            except (ValueError, KeyError, TypeError, sqlite3.Error) as error:
                store.release(owner)
                output = b"\1" + str(error).encode()
            stream.sendall(struct.pack("<I", len(output)) + output)
    except (EOFError, OSError):
        pass
    finally:
        store.release(owner)
        stream.close()


def main():
    path, memory, capacity = sys.argv[1:]
    auth = sys.stdin.buffer.read(32)
    if len(auth) != 32:
        raise ValueError("artifact owner requires a 32-byte local capability")
    store = Disk(path, int(memory), int(capacity) or None)
    stop = threading.Event()

    def parent():
        os.read(sys.stdin.fileno(), 1)
        stop.set()

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.settimeout(0.2)
    threading.Thread(target=parent, daemon=True).start()
    print(json.dumps(dict(port=listener.getsockname()[1], identity=store.identity)), flush=True)
    clients: list[tuple[threading.Thread, socket.socket]] = []
    try:
        while not stop.is_set():
            clients = [(thread, stream) for thread, stream in clients if thread.is_alive()]
            try:
                stream, _ = listener.accept()
            except socket.timeout:
                continue
            thread = threading.Thread(target=serve, args=(stream, store, auth, stop), daemon=True)
            clients.append((thread, stream))
            thread.start()
    finally:
        listener.close()
        for _, stream in clients:
            try:
                stream.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for thread, _ in clients:
            thread.join(2)
        store.close()


if __name__ == "__main__":
    main()
