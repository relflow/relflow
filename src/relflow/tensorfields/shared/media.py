"""Shared locator resolution, version snapshots, and vector caching for file-backed media."""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar, cast
from urllib.parse import unquote

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import torch
from pyarrow import fs
from tensordict import TensorDict

from relflow.data.ragged import RaggedField, Sources
from relflow.data.sources import resolve as filesystem
from relflow.structs.enums import TensorKey, Tokens
from relflow.structs.tree import Address
from relflow.tensorfields.base import Context, TensorFieldBase, masks, resolve
from relflow.tensorfields.shared.cache import MAX_FRAME, MAX_ROWS, Payloads
from relflow.tensorfields.shared.cache.vectors import VectorCache
from relflow.tensorfields.shared.frozen import Embedder as FrozenEmbedder
from relflow.tensorfields.shared.frozen import Options as FrozenOptions

if TYPE_CHECKING:
    from relflow.structs.experiment import Schema

FieldT = TypeVar("FieldT", bound=TensorFieldBase[TensorDict])


class MediaOptions(FrozenOptions, Protocol):
    base: str | None
    immutable: bool
    read_workers: int
    encoder_batch_size: int
    cache: bool | int
    store: str | None
    memory: int | None
    resident: int | None


def location(value: str, base: str | None, *, kind: str) -> str:
    if not value or "\0" in value:
        raise ValueError(f"expected a nonempty {kind} path without NUL, got {value!r}")
    if base and "://" not in value and not os.path.isabs(value):
        value = base.removesuffix("/") + "/" + value if "://" in base else os.path.join(base, value)
    if value.startswith("file://"):
        _, value = filesystem(value, None)
    return value if "://" in value else os.path.abspath(value)


@lru_cache(maxsize=32)
def remote(authority: str) -> tuple[fs.FileSystem, str]:
    """Reuse the Arrow storage client across objects in the same bucket."""
    return filesystem(authority, None)


def open_stream(
    path: str,
    resolver: Callable[[str], tuple[fs.FileSystem, str]] = remote,
) -> io.BufferedReader | io.BytesIO:
    if "://" not in path:
        return open(path, "rb")
    scheme, rest = path.split("://", 1)
    authority, _, key = rest.partition("/")
    storage, prefix = resolver(f"{scheme}://{authority}")
    name = f"{prefix.rstrip('/')}/{unquote(key)}" if prefix else unquote(key)
    # Media codecs seek through container headers and frame tables. Download
    # the compressed object sequentially once, then let the codec seek in memory.
    with storage.open_input_stream(name, compression=None) as stream:
        return io.BytesIO(stream.read())


def stamp(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def snapshot(
    path: str,
    immutable: bool,
    opener: Callable[[str], Any],
) -> tuple[bytes, bytes | None]:
    """Keep a mutable remote object's bytes for this microbatch to avoid a second read."""
    body = None
    version: tuple[int, ...] | str | None = None
    if not immutable:
        if "://" in path:
            with opener(path) as stream:
                body = stream.read()
            version = hashlib.sha256(body).hexdigest()
        elif os.name == "nt":
            # Windows can report different metadata by path and by handle after a replacement.
            with open(path, "rb") as stream:
                version = stamp(os.fstat(stream.fileno()))
        else:
            version = stamp(os.stat(path))
    return json.dumps((path, version), ensure_ascii=False, separators=(",", ":")).encode(), body


class ReferenceStore(VectorCache):
    """Only the locator and its version enter the artifact store as prepared data."""

    source_namespace: ClassVar[str]
    kind: ClassVar[str]
    input_bytes = 0  # Source descriptors have variable width.

    def identify(self, records: list[bytes]) -> np.ndarray:
        ids = np.empty(len(records), dtype=np.int64)
        longest = max(map(len, records), default=0)
        chunk = min(MAX_ROWS, (MAX_FRAME - 65536) // (longest + 64))
        if not chunk:
            raise ValueError(f"{self.kind} reference exceeds the artifact cache frame budget")
        for start in range(0, len(records), chunk):
            batch = records[start : start + chunk]
            sources = pa.array(batch, type=pa.binary())
            refs = self.client.lookup_arrow(sources, self.source_namespace)
            missing = np.flatnonzero(~refs.ready)
            if len(missing):
                refs.objects[missing] = self.client.bind(
                    refs.take(missing),
                    Payloads.from_values(batch[i] for i in missing),
                    self.source_namespace,
                )
            ids[start : start + len(batch)] = refs.objects
        return ids


def open_store(store_cls: type[ReferenceStore], request: MediaOptions) -> ReferenceStore:
    capacity = None if isinstance(request.cache, bool) else request.cache
    return store_cls(request.store, request.memory, capacity)


def tensorize(
    cls: Callable[..., FieldT],
    sources: Sources,
    *,
    schema: Schema,
    context: Context,
    kind: str,
    store_cls: type[ReferenceStore],
    opener: Callable[[str], Any],
) -> tuple[FieldT, TensorDict | None, tuple[object, ...]]:
    address = sources.address
    projection, context, observation, binding = resolve(sources, schema=schema, context=context)
    inputs, targets = projection.split()
    present, trainable, inferred = masks(projection)
    request = cast(MediaOptions, schema.requests[address])

    def encode(field: RaggedField) -> TensorDict:
        if not len(field.values):
            key, dtype = ("ids", torch.int64) if request.cache else ("paths", torch.uint8)
            return TensorDict({key: torch.zeros((*field.shape, 1), dtype=dtype)}, batch_size=field.shape)
        # Unify independent Arrow dictionaries before deduplicating across chunks.
        strings = field.values.cast(pa.large_string())
        if isinstance(strings, pa.ChunkedArray):
            strings = strings.combine_chunks()
        values = cast(pa.DictionaryArray, pc.dictionary_encode(strings))
        paths = [location(path, request.base, kind=kind) for path in cast(list[str], values.dictionary.to_pylist())]
        inverse = values.indices.to_numpy(zero_copy_only=False)
        if request.cache and (request.immutable or all("://" not in path for path in paths)):
            cache = context.state if isinstance(context.state, store_cls) else open_store(store_cls, request)
            records = [snapshot(path, request.immutable, opener)[0] for path in paths]
            ids = field.place(cache.identify(records)[inverse], fill=0)
            return TensorDict({"ids": torch.from_numpy(ids).unsqueeze(-1)}, batch_size=field.shape)
        # Uncached inputs and mutable remote URIs retain only their UTF-8 locators.
        # The trainer hashes and decodes the same downloaded bytes on a cold miss.
        raw = np.asarray([path.encode() for path in paths], dtype="S")
        packed = field.place(raw[inverse], fill=b"")
        return TensorDict(
            {"paths": torch.from_numpy(packed.view(np.uint8).reshape(*field.shape, raw.itemsize))},
            batch_size=field.shape,
        )

    try:
        return (
            cls(
                state=torch.from_numpy(inputs.dense),
                content=encode(inputs),
                present=present,
                trainable=trainable,
                inferred=inferred,
                targets=TensorDict(
                    {TensorKey.state: torch.from_numpy(targets.dense), TensorKey.content: encode(targets)}
                    if address in schema.objectives
                    else {},
                    batch_size=inputs.shape,
                ),
                batch_size=inputs.batch_size,
            ),
            observation,
            binding,
        )
    except (ValueError, OSError) as error:
        error.add_note(f"While tensorizing {kind} field {address!s}.")
        raise


class MediaEmbedder(FrozenEmbedder):
    kind: ClassVar[str]
    store_cls: ClassVar[type[ReferenceStore]]

    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)
        request = cast(MediaOptions, schema.requests[address])
        self.request: Any = request
        self.store: ReferenceStore | None = open_store(self.store_cls, request) if request.cache else None

    @property
    def context(self) -> ReferenceStore | None:
        return self.store

    def stream(self, path: str) -> Any:
        raise NotImplementedError

    def variant(self, device: torch.device) -> str:
        raise NotImplementedError

    def pool(
        self, records: Sequence[bytes] | Payloads, device: torch.device, bodies: dict[bytes, bytes]
    ) -> torch.Tensor:
        raise NotImplementedError

    def lookup(self, ids: torch.Tensor, bodies: dict[bytes, bytes]) -> torch.Tensor:
        assert self.store is not None
        return self.store.resolve(
            ids,
            variant=self.variant(ids.device),
            width=self.hidden_size,
            resident=self.request.resident,
            pool=partial(self.pool, bodies=bodies),
        )

    def paths(self, paths: Sequence[str], device: torch.device) -> torch.Tensor:
        """Resolve unique locators, keeping mutable remote bodies only for the active batch."""
        if not paths:
            return torch.empty((0, self.hidden_size), dtype=torch.float32, device=device)
        if self.store is None:
            return self.pool([snapshot(path, True, self.stream)[0] for path in paths], device, {})
        if self.request.immutable or all("://" not in path for path in paths):
            records = [snapshot(path, self.request.immutable, self.stream)[0] for path in paths]
            return self.lookup(torch.from_numpy(self.store.identify(records)).to(device), {})

        size = self.request.encoder_batch_size
        vectors = []
        with ThreadPoolExecutor(max_workers=self.request.read_workers) as executor:
            for start in range(0, len(paths), size):
                batch = paths[start : start + size]
                snapshots = list(executor.map(partial(snapshot, immutable=False, opener=self.stream), batch))
                records = [record for record, body in snapshots]
                bodies = {record: body for record, body in snapshots if body is not None}
                ids = self.store.identify(records)
                vectors.append(self.lookup(torch.from_numpy(ids).to(device), bodies))
        return torch.cat(vectors)

    @torch.compiler.disable
    def encode(self, content: TensorDict, state: torch.Tensor) -> torch.Tensor:
        valued = state.reshape(-1).eq(Tokens.valued.value)
        result = torch.zeros((state.numel(), self.hidden_size), dtype=torch.float32, device=state.device)
        if valued.any():
            if "ids" in content.keys():
                if self.store is None:
                    raise ValueError(
                        f"{self.kind} field at '{self.address}' received cache IDs with cache=False; encode its observations again"
                    )
                result[valued] = self.lookup(cast(torch.Tensor, content["ids"]).reshape(-1)[valued], {})
            else:
                paths = cast(torch.Tensor, content["paths"])
                width = paths.shape[-1]
                packed = paths.reshape(-1, width)[valued].contiguous().cpu().numpy().view(f"S{width}").ravel()
                unique, inverse = np.unique(packed, return_inverse=True)
                encoded = self.paths([row.decode() for row in unique], state.device)
                result[valued] = encoded.index_select(0, torch.from_numpy(inverse).to(state.device))
        return result.reshape(*state.shape, self.hidden_size)

    def warm(self, values: Sequence[str | None] | pa.Array | pa.ChunkedArray | pl.Series) -> int:
        """Populate the local vector cache at the current device and precision."""
        if self.store is None:
            raise ValueError(f"{self.kind} field at {self.address!s} requires cache=True to warm")
        if isinstance(values, str):
            values = (values,)
        before = self.store.stats()["results"]
        for start in range(0, len(values), MAX_ROWS):
            batch = values[start : start + MAX_ROWS]
            if isinstance(batch, (pa.Array, pa.ChunkedArray)):
                paths: Sequence[str | None] = batch.to_pylist()
            elif isinstance(batch, pl.Series):
                paths = batch.to_list()
            else:
                paths = batch
            unique = list(
                dict.fromkeys(location(path, self.request.base, kind=self.kind) for path in paths if path is not None)
            )
            if unique:
                self.paths(unique, self.anchor.device)
        return self.store.stats()["results"] - before
