"""Portable cache owner, also executable without importing relflow or Torch.

Moka owns frequency-aware RAM admission. SQLite owns stable IDs, token rows,
and immutable precision-specific vectors, with a bounded private page cache.
The local binary protocol never deserializes Python objects.
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
            raise EOFError("text cache connection closed")
        offset += count
    return data


class Disk:
    """One transactional index with a bounded Rust hot tier; no corpus-sized map."""

    def __init__(self, path: str | Path, memory: int, capacity: int | None = None):
        from moka_py import Moka

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA cache_size=-4096")
        self.connection.execute("PRAGMA mmap_size=0")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT, key BLOB UNIQUE NOT NULL,
                length INTEGER NOT NULL, canonical INTEGER
            );
            CREATE TABLE IF NOT EXISTS token_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT, key BLOB UNIQUE NOT NULL,
                length INTEGER NOT NULL, tokens BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER NOT NULL, variant TEXT NOT NULL, width INTEGER NOT NULL,
                value BLOB NOT NULL, PRIMARY KEY (id, variant)
            ) WITHOUT ROWID;
        """)
        self.connection.execute(
            "INSERT OR IGNORE INTO metadata VALUES ('identity', ?)",
            (hashlib.sha256(os.urandom(32)).hexdigest(),),
        )
        self.connection.execute("INSERT OR IGNORE INTO metadata VALUES ('items', '0')")
        self.connection.execute("INSERT OR IGNORE INTO metadata VALUES ('vectors', '0')")
        self.connection.execute("INSERT OR IGNORE INTO metadata VALUES ('tokens', '0')")
        self.connection.commit()
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
        self.counters = dict(
            identified=0, alias_hits=0, vector_requests=0, vector_hits=0, disk_vectors=0, encoder_claims=0
        )

    def identify(self, keys: list[bytes], length: int) -> tuple[np.ndarray, np.ndarray]:
        if length <= 0:
            raise ValueError("text token length must be positive")
        ids = np.empty(len(keys), dtype=np.int64)
        ready = np.zeros(len(keys), dtype=np.uint8)
        with self.lock, self.connection:
            for index, key in enumerate(keys):
                self.counters["identified"] += 1
                cached = self.hot.get(("key", key))
                if cached is not None:
                    ids[index], ready[index] = cast(tuple[int, int], cached)
                    self.counters["alias_hits"] += 1
                    continue
                row = self.connection.execute("SELECT id, length, canonical FROM items WHERE key=?", (key,)).fetchone()
                if row is None:
                    inserted = self.connection.execute(
                        "INSERT INTO items(key,length) VALUES (?,?) ON CONFLICT(key) DO NOTHING", (key, length)
                    )
                    if inserted.rowcount:
                        self.connection.execute("UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='items'")
                    row = self.connection.execute(
                        "SELECT id,length,canonical FROM items WHERE key=?", (key,)
                    ).fetchone()
                if row[1] != length:
                    raise ValueError("text cache key identifies a different token length")
                ids[index], ready[index] = (row[2] if row[2] is not None else row[0]), int(row[2] is not None)
                if row[2] is not None:
                    self.hot.set(("key", key), (row[2], 1))
        self.hot.run_pending_tasks()
        return ids, ready

    def tokens(self, ids: np.ndarray, rows: np.ndarray, keys: list[bytes], namespace: str) -> np.ndarray:
        canonical = np.empty(len(ids), dtype=np.int64)
        with self.lock, self.connection:
            for index, (identifier, row, key) in enumerate(zip(ids, rows, keys, strict=True)):
                payload = row.astype("<i8", copy=False).tobytes()
                stored = self.connection.execute(
                    "SELECT key,length,canonical FROM items WHERE id=?", (int(identifier),)
                ).fetchone()
                if stored is None or stored[0] != key or len(payload) != stored[1] * 16:
                    raise ValueError("text token fill does not match its alias, key, or token length")
                if stored[2] is not None:
                    previous = self.connection.execute(
                        "SELECT tokens FROM token_rows WHERE id=?", (stored[2],)
                    ).fetchone()[0]
                    if previous != payload:
                        raise ValueError(
                            "text cache received conflicting token rows; use a distinct tokenizer identity"
                        )
                    canonical[index] = stored[2]
                    continue
                token_key = hashlib.sha256(namespace.encode() + payload).digest()
                inserted = self.connection.execute(
                    "INSERT OR IGNORE INTO token_rows(key,length,tokens) VALUES (?,?,?)",
                    (token_key, stored[1], payload),
                )
                if inserted.rowcount:
                    self.connection.execute("UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='tokens'")
                canonical[index] = self.connection.execute(
                    "SELECT id FROM token_rows WHERE key=?", (token_key,)
                ).fetchone()[0]
                self.connection.execute(
                    "UPDATE items SET canonical=? WHERE id=?", (int(canonical[index]), int(identifier))
                )
        # Publish aliases only after every row in the transaction is durable.
        for key, identifier in zip(keys, canonical.tolist(), strict=True):
            self.hot.set(("key", key), (identifier, 1))
        self.hot.run_pending_tasks()
        return canonical

    def get(self, ids: np.ndarray, variant: str, width: int, length: int, owner: int):
        status = np.zeros(len(ids), dtype=np.uint8)
        vectors, tokens = [], []
        with self.lock:
            for index, identifier in enumerate(ids.tolist()):
                self.counters["vector_requests"] += 1
                cached = self.hot.get(("vector", identifier, variant))
                if cached is not None:
                    status[index] = 1
                    vectors.append(cached)
                    self.counters["vector_hits"] += 1
                    continue
                stored = self.connection.execute(
                    "SELECT width,value FROM vectors WHERE id=? AND variant=?", (identifier, variant)
                ).fetchone()
                if stored is not None:
                    if stored[0] != width or len(stored[1]) != width * 4:
                        raise ValueError("text vector width does not match its precision namespace")
                    status[index] = 1
                    vectors.append(stored[1])
                    self.counters["disk_vectors"] += 1
                    self.hot.set(("vector", identifier, variant), stored[1])
                    continue
                claim = (identifier, variant)
                if claim in self.claims and self.claims[claim] != owner:
                    status[index] = 2
                    continue
                stored = self.connection.execute(
                    "SELECT length,tokens FROM token_rows WHERE id=?", (identifier,)
                ).fetchone()
                if stored is None or stored[0] != length or stored[1] is None:
                    raise ValueError(f"text ID {identifier} has no matching token row; encode the observations again")
                owned = self.owned.setdefault(owner, set())
                if claim not in owned and len(owned) >= MAX_ROWS:
                    raise ValueError("one text cache connection cannot own more than 4096 in-flight encodings")
                self.claims[claim] = owner
                owned.add(claim)
                tokens.append(stored[1])
                self.counters["encoder_claims"] += 1
        self.hot.run_pending_tasks()
        return status.tobytes() + b"".join(vectors) + b"".join(tokens)

    def put(self, ids: np.ndarray, variant: str, rows: np.ndarray, owner: int) -> bytes:
        output = []
        with self.condition, self.connection:
            for identifier, row in zip(ids.tolist(), rows, strict=True):
                if self.claims.get((identifier, variant)) != owner:
                    raise ValueError("text vector fill has no matching encoder claim")
                payload = row.astype("<f4", copy=False).tobytes()
                inserted = self.connection.execute(
                    "INSERT OR IGNORE INTO vectors VALUES (?,?,?,?)", (identifier, variant, rows.shape[1], payload)
                )
                if inserted.rowcount:
                    self.connection.execute("UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='vectors'")
                stored = self.connection.execute(
                    "SELECT width,value FROM vectors WHERE id=? AND variant=?", (identifier, variant)
                ).fetchone()
                if stored[0] != rows.shape[1]:
                    raise ValueError("text vector width changed within one precision namespace")
                output.append(stored[1])
            # Acknowledgment follows commit; RAM eviction never deletes disk data.
        with self.condition:
            for identifier in ids.tolist():
                self.claims.pop((identifier, variant), None)
                self.owned.get(owner, set()).discard((identifier, variant))
            self.condition.notify_all()
        return b"".join(output)

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
                aliases=int(self.connection.execute("SELECT value FROM metadata WHERE key='items'").fetchone()[0]),
                entries=int(self.connection.execute("SELECT value FROM metadata WHERE key='tokens'").fetchone()[0]),
                vectors=int(self.connection.execute("SELECT value FROM metadata WHERE key='vectors'").fetchone()[0]),
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
        raise ValueError("truncated text cache request")
    op, size = data[0], struct.unpack_from("<I", data, 1)[0]
    if size > 65536 or len(data) < 5 + size:
        raise ValueError("invalid text cache metadata size")
    meta = json.loads(bytes(data[5 : 5 + size]))
    data = data[5 + size :]
    if op == 5:
        return json.dumps(store.stats()).encode()
    if op == 6:
        stop.set()
        return b""
    n = meta["n"]
    if not isinstance(n, int) or not 0 <= n <= MAX_ROWS:
        raise ValueError("text cache request must contain at most 4096 rows")
    if op == 1:
        if len(data) != n * 32:
            raise ValueError("text key batch has incorrect size")
        keys = [bytes(data[i : i + 32]) for i in range(0, len(data), 32)]
        ids, ready = store.identify(keys, meta["length"])
        return ids.astype("<i8", copy=False).tobytes() + ready.tobytes()
    if len(data) < n * 8:
        raise ValueError("truncated text ID batch")
    ids = np.frombuffer(data[: n * 8], dtype="<i8")
    data = data[n * 8 :]
    if op == 2:
        length = meta["length"]
        if length <= 0 or len(data) != n * (32 + length * 16):
            raise ValueError("text token batch has incorrect size")
        keys = [bytes(data[i : i + 32]) for i in range(0, n * 32, 32)]
        rows = np.frombuffer(data[n * 32 :], dtype="<i8").reshape(n, 2, length)
        return store.tokens(ids, rows, keys, meta["namespace"]).astype("<i8", copy=False).tobytes()
    if op == 3:
        if data or not 0 < meta["width"] <= MAX_FRAME // 4 or meta["length"] <= 0:
            raise ValueError("invalid text vector lookup dimensions")
        if n * max(meta["width"] * 4, meta["length"] * 16) + n > MAX_FRAME:
            raise ValueError("text lookup reply exceeds 32 MiB; use smaller batches")
        return store.get(ids, meta["variant"], meta["width"], meta["length"], owner)
    if op == 4:
        width = meta["width"]
        if width <= 0 or len(data) != n * width * 4:
            raise ValueError("text vector fill has incorrect size")
        return store.put(ids, meta["variant"], np.frombuffer(data, dtype="<f4").reshape(n, width), owner)
    if op == 7:
        store.release(owner)
        return b""
    if op == 8:
        with store.condition:
            store.condition.wait(timeout=0.1)
        return b""
    raise ValueError(f"unknown text cache operation {op}")


def serve(stream: socket.socket, store: Disk, auth: bytes, stop: threading.Event):
    owner = id(stream)
    try:
        stream.settimeout(30)
        if not hmac.compare_digest(receive(stream, 32), auth):
            return
        # A large CPU encoder fill may legitimately take longer than minutes.
        stream.settimeout(None)
        while not stop.is_set():
            size = struct.unpack("<I", receive(stream, 4))[0]
            if not 5 <= size <= MAX_FRAME:
                raise ValueError("text cache frame exceeds the 32 MiB limit")
            data = receive(stream, size)
            try:
                output = b"\0" + dispatch(store, memoryview(data), owner, stop)
            except (ValueError, KeyError, sqlite3.Error) as error:
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
        raise ValueError("text cache owner requires a 32-byte local capability")
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
    threads: list[tuple[threading.Thread, socket.socket]] = []
    try:
        while not stop.is_set():
            threads = [(thread, stream) for thread, stream in threads if thread.is_alive()]
            try:
                stream, _ = listener.accept()
            except socket.timeout:
                continue
            thread = threading.Thread(target=serve, args=(stream, store, auth, stop), daemon=True)
            threads.append((thread, stream))
            thread.start()
    finally:
        listener.close()
        for _, stream in threads:
            try:
                stream.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for thread, _ in threads:
            thread.join(2)
        store.close()


if __name__ == "__main__":
    main()
