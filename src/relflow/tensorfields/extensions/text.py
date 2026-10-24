from __future__ import annotations

import enum
import hashlib
import math
import os
import sys
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import InitVar, dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, TypeAlias, Unpack, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from beartype import beartype
from lightning.pytorch import Callback, LightningModule, Trainer
from tensordict import TensorDict, tensorclass

from relflow.data.ragged import RaggedField
from relflow.structs.enums import Component, Metric, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address, FieldOptions
from relflow.tensorfields.base import (
    Context,
    DecoderBase,
    EmbedderBase,
    Extension,
    RequestBase,
    TensorFieldBase,
    TensorInput,
)
from relflow.tensorfields.shared.counter import Counter, CounterUpdateCallback, tally
from relflow.tensorfields.shared.jitter import Jitter
from relflow.tensorfields.shared.reconstruction import Errors, Metrics, Scores, errors

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from relflow.architecture.node import NodeModule
    from relflow.architecture.root import Model
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch


HASH_SEED = 0x9E3779B97F4A7C15


def hash_weights(width: int) -> np.ndarray:
    """Two rows of odd 64-bit weights for token rows of one width, fixed for the life of a store."""
    generator = np.random.default_rng(HASH_SEED ^ width)
    weights = generator.integers(1, 2**63 - 1, size=(2, width), dtype=np.int64)
    return weights | np.int64(1)


def hash_rows(rows: np.ndarray, *, salt: int = 0) -> np.ndarray:
    """Two independent 64-bit multilinear hashes of every token row, as two int64 words per row.

    The first word indexes the store; the second guards against a first-word
    collision, which then reads as a miss rather than as a wrong vector. The
    arithmetic wraps modulo 2**64, so ``hash_rows_torch`` reproduces it bit for
    bit on any device.
    """
    if rows.ndim != 2:
        raise ValueError(f"token rows must be a 2-D array, got shape {rows.shape}")
    weights = hash_weights(rows.shape[1])
    values = rows.astype(np.int64, copy=False) + np.int64(1)
    with np.errstate(over="ignore"):
        words = (values[:, None, :] * weights[None, :, :]).sum(axis=2, dtype=np.int64)
        words += np.int64(np.uint64(salt).astype(np.int64)) * np.array([1, 3], dtype=np.int64)
    return words


def hash_rows_torch(rows: torch.Tensor, *, salt: int = 0) -> torch.Tensor:
    """``hash_rows`` on a device tensor of token rows; identical words, no host round trip."""
    weights = torch.from_numpy(hash_weights(rows.shape[1])).to(rows.device)
    values = rows.to(torch.int64) + 1
    words = torch.stack(((values * weights[0]).sum(dim=1), (values * weights[1]).sum(dim=1)), dim=1)
    offset = torch.tensor([1, 3], dtype=torch.int64, device=rows.device) * int(np.uint64(salt).astype(np.int64))
    return words + offset


class EmbeddingStore:
    """Persistent frozen-encoder vectors keyed by token row, shared by every run that opens the directory.

    A string's pooled vector does not depend on the task, the split, or the
    training history, so one store serves training, validation, test, and
    prediction alike, and every later run. Vectors live in memory-mapped
    shards under ``directory``, named by the encoder fingerprint; each process
    appends its own shards, so readers never block and no file is rewritten.
    Lookups hash whole token rows and binary-search a sorted index, so a batch
    resolves in a few vectorized operations. New vectors wait in memory until
    ``flush``, which the extension's callback runs at epoch and phase ends.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        fingerprint: str,
        hidden_size: int,
        *,
        flush_every: int = 65_536,
        resident: int | None = None,
    ):
        if hidden_size <= 0 or flush_every <= 0 or (resident is not None and resident < 0):
            raise ValueError("EmbeddingStore needs a positive hidden size and flush interval and a nonnegative budget")
        self.directory = Path(directory)
        self.fingerprint = fingerprint
        self.hidden_size = hidden_size
        self.flush_every = flush_every
        self.resident = resident  # MiB of accelerator memory a device view may take; None decides per device
        self.lock = RLock()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shards: list[np.ndarray] = []
        self.offsets = np.zeros(1, dtype=np.int64)
        self.k1 = np.empty(0, dtype=np.int64)
        self.k2 = np.empty(0, dtype=np.int64)
        self.locator = np.empty(0, dtype=np.int64)
        self.pending_keys = np.empty((0, 2), dtype=np.int64)
        self.pending_vectors = np.empty((0, hidden_size), dtype=np.float32)
        self.pending_order = np.empty(0, dtype=np.int64)
        self.loaded: set[str] = set()
        self.views: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = {}
        self.reload()

    instances: ClassVar[dict[tuple[str, str, int], EmbeddingStore]] = {}
    registry_lock: ClassVar[RLock] = RLock()

    @classmethod
    def open(cls, directory: str | os.PathLike[str], fingerprint: str, hidden_size: int, **options) -> EmbeddingStore:
        """One store object per directory and encoder within a process, so fields share pending vectors."""
        key = (str(Path(directory).resolve()), fingerprint, hidden_size)
        with cls.registry_lock:
            store = cls.instances.get(key)
            if store is None:
                store = cls.instances[key] = cls(directory, fingerprint, hidden_size, **options)
            elif options.get("resident") is not None:
                store.resident = options["resident"]
            return store

    def __len__(self) -> int:
        return int(len(self.k1) + len(self.pending_keys))

    def __reduce__(self):
        # Copies and spawned workers reopen the directory; pending vectors are persisted first.
        self.flush()
        return EmbeddingStore.open, (self.directory, self.fingerprint, self.hidden_size)

    def reload(self) -> int:
        """Index every shard in the directory that this instance has not loaded yet."""
        with self.lock:
            added = 0
            for keys_path in sorted(self.directory.glob(f"{self.fingerprint}.*.keys.npy")):
                vectors_path = keys_path.with_name(keys_path.name.replace(".keys.npy", ".vectors.npy"))
                if keys_path.name in self.loaded or not vectors_path.exists():
                    continue
                keys = np.load(keys_path)
                vectors = np.load(vectors_path, mmap_mode="r")
                if keys.shape != (len(vectors), 2) or vectors.shape[1:] != (self.hidden_size,):
                    raise ValueError(
                        f"embedding shard {keys_path} does not match its vectors or hidden size {self.hidden_size}"
                    )
                self.admit(keys, vectors)
                self.loaded.add(keys_path.name)
                added += len(keys)
            return added

    def admit(self, keys: np.ndarray, vectors: np.ndarray) -> None:
        """Merge one shard into the sorted index."""
        start = int(self.offsets[-1])
        self.shards.append(vectors)
        self.offsets = np.append(self.offsets, start + len(vectors))
        k1 = np.concatenate((self.k1, keys[:, 0]))
        k2 = np.concatenate((self.k2, keys[:, 1]))
        locator = np.concatenate((self.locator, np.arange(start, start + len(vectors), dtype=np.int64)))
        order = np.argsort(k1, kind="stable")
        self.k1, self.k2, self.locator = k1[order], k2[order], locator[order]
        self.views.clear()  # device views are rebuilt from the new index on their next use

    def fetch(self, positions: np.ndarray) -> np.ndarray:
        """Vectors at sorted index positions, gathered shard by shard."""
        rows = self.locator[positions]
        out = np.empty((len(rows), self.hidden_size), dtype=np.float32)
        shard = np.searchsorted(self.offsets, rows, side="right") - 1
        for index in np.unique(shard):
            selected = np.flatnonzero(shard == index)
            local = rows[selected] - self.offsets[index]
            order = np.argsort(local)
            out[selected[order]] = self.shards[index][local[order]]
        return out

    def lookup(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Which keys the store holds, and their vectors in key order (rows of misses are left zero)."""
        with self.lock:
            found = np.zeros(len(keys), dtype=bool)
            out = np.zeros((len(keys), self.hidden_size), dtype=np.float32)
            for k1, k2, gather in (
                (self.k1, self.k2, self.fetch),
                (
                    self.pending_keys[self.pending_order, 0],
                    self.pending_keys[self.pending_order, 1],
                    lambda positions: self.pending_vectors[self.pending_order[positions]],
                ),
            ):
                if not len(k1) or found.all():
                    continue
                positions = np.searchsorted(k1, keys[:, 0])
                clipped = np.minimum(positions, len(k1) - 1)
                hit = (positions < len(k1)) & (k1[clipped] == keys[:, 0]) & (k2[clipped] == keys[:, 1]) & ~found
                if hit.any():
                    out[hit] = gather(clipped[hit])
                    found |= hit
            return found, out

    def put(self, keys: np.ndarray, vectors: np.ndarray) -> int:
        """Remember vectors for keys the store does not hold yet; returns how many were new."""
        if keys.shape != (len(vectors), 2) or vectors.shape[1:] != (self.hidden_size,):
            raise ValueError(
                f"store put needs keys [n, 2] and vectors [n, {self.hidden_size}], got {keys.shape} and {vectors.shape}"
            )
        with self.lock:
            _, first = np.unique(keys[:, 0], return_index=True)
            keys, vectors = keys[first], np.asarray(vectors[first], dtype=np.float32)
            found, _ = self.lookup(keys)
            new = ~found
            if not new.any():
                return 0
            self.pending_keys = np.concatenate((self.pending_keys, keys[new]))
            self.pending_vectors = np.concatenate((self.pending_vectors, vectors[new]))
            self.pending_order = np.argsort(self.pending_keys[:, 0], kind="stable")
            if len(self.pending_keys) >= self.flush_every:
                self.flush()
            return int(new.sum())

    def flush(self) -> int:
        """Write pending vectors as one new shard and merge it into the index; returns how many were written."""
        with self.lock:
            count = len(self.pending_keys)
            if not count:
                return 0
            token = uuid.uuid4().hex[:12]
            vectors_path = self.directory / f"{self.fingerprint}.{token}.vectors.npy"
            keys_path = self.directory / f"{self.fingerprint}.{token}.keys.npy"
            # Vectors land before keys, so a reader never indexes keys whose vectors are missing.
            for path, payload in ((vectors_path, self.pending_vectors), (keys_path, self.pending_keys)):
                temporary = path.with_name(path.name + ".tmp")
                with open(temporary, "wb") as handle:
                    np.save(handle, payload)
                os.replace(temporary, path)
            self.admit(np.load(keys_path), np.load(vectors_path, mmap_mode="r"))
            self.loaded.add(keys_path.name)
            self.pending_keys = np.empty((0, 2), dtype=np.int64)
            self.pending_vectors = np.empty((0, self.hidden_size), dtype=np.float32)
            self.pending_order = np.empty(0, dtype=np.int64)
            return count

    def budget(self, device: torch.device) -> int:
        """Bytes a device view may take: the configured MiB, or a quarter of the device's memory up to 2 GiB."""
        if self.resident is not None:
            return self.resident * 2**20
        if device.type == "cuda":
            total = torch.cuda.get_device_properties(device).total_memory
            return int(min(total // 4, 2 * 2**30))
        return 0

    def view(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """The flushed index and vectors on ``device`` when they fit its budget; built once per index version."""
        name = str(device)
        if name in self.views:
            return self.views[name]
        with self.lock:
            if name in self.views:
                return self.views[name]
            size = len(self.k1) * (self.hidden_size * 4 + 16)
            # The host index already serves a CPU model; an explicit budget still allows a CPU view for tests and tools.
            if not len(self.k1) or size > self.budget(device) or (device.type == "cpu" and self.resident is None):
                self.views[name] = None
                return None
            vectors = torch.from_numpy(self.fetch(np.arange(len(self.k1)))).to(device)
            k1 = torch.from_numpy(self.k1).to(device)
            k2 = torch.from_numpy(self.k2).to(device)
            self.views[name] = (k1, k2, vectors)
            return self.views[name]

    def lookup_device(self, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Device-side lookup against the resident view: (found, vectors), or None when no view fits."""
        resident = self.view(keys.device)
        if resident is None:
            return None
        k1, k2, vectors = resident
        positions = torch.searchsorted(k1, keys[:, 0].contiguous()).clamp_max(len(k1) - 1)
        found = (k1[positions] == keys[:, 0]) & (k2[positions] == keys[:, 1])
        return found, vectors.index_select(0, positions)


def flush_stores(module: Model) -> dict[str, int]:
    """Persist every Text field's pending vectors; returns how many each field wrote."""
    written: dict[str, int] = {}
    for address, request in module.schema.requests.items():
        if getattr(request, "type", None) != "text":
            continue
        store = getattr(cast("NodeModule", module.nodes[address]).embedder, "store", None)
        if store is not None:
            written[str(address)] = store.flush()
    return written


class StoreFlush(Callback):
    """Flush Text embedding stores at the end of every epoch and phase, so no computed vector is lost."""

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        flush_stores(cast("Model", pl_module))

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        flush_stores(cast("Model", pl_module))

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        flush_stores(cast("Model", pl_module))

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        flush_stores(cast("Model", pl_module))

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        flush_stores(cast("Model", pl_module))

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        flush_stores(cast("Model", pl_module))


text: Extension = Extension(
    name="text",
    types=(str,),
    requires={"transformers": "relflow[text]"},
    callbacks=(CounterUpdateCallback, Metrics, StoreFlush),
)

INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
DEFAULT_TEXT_MODEL = "google/bert_uncased_L-2_H-128_A-2"

CacheKey: TypeAlias = tuple[str, torch.dtype | None, bytes]


class EmbeddingCache:
    """Transient CPU embeddings with least-frequency, then least-recent eviction."""

    def __init__(self, capacity: int | None):
        self.capacity = capacity
        self.values: dict[CacheKey, torch.Tensor] = {}
        self.frequencies: dict[CacheKey, int] = {}
        self.buckets: dict[int, OrderedDict[CacheKey, None]] = {}
        self.minimum = 0
        self.lock = RLock()

    def __len__(self) -> int:
        return len(self.values)

    def __reduce__(self):
        # Model copies and spawned workers start cold, without serializing
        # input-derived vectors or the process-local lock.
        return type(self), (self.capacity,)

    def get(self, key: CacheKey) -> torch.Tensor | None:
        with self.lock:
            if key not in self.values:
                return None
            frequency = self.frequencies[key]
            self.buckets[frequency].pop(key)
            if not self.buckets[frequency]:
                del self.buckets[frequency]
                if self.minimum == frequency:
                    self.minimum += 1
            self.frequencies[key] = frequency + 1
            self.buckets.setdefault(frequency + 1, OrderedDict())[key] = None
            return self.values[key]

    def put(self, key: CacheKey, value: torch.Tensor, *, count: int) -> None:
        with self.lock:
            if key in self.values:
                return
            if self.capacity is not None and len(self.values) == self.capacity:
                evicted, _ = self.buckets[self.minimum].popitem(last=False)
                del self.values[evicted], self.frequencies[evicted]
                if not self.buckets[self.minimum]:
                    del self.buckets[self.minimum]
            self.values[key] = value.detach().to(device="cpu", dtype=torch.float32).clone()
            self.frequencies[key] = 1
            self.buckets.setdefault(1, OrderedDict())[key] = None
            self.minimum = 1
            for _ in range(count - 1):
                self.get(key)


class TokenCache:
    """Transient LRU token rows owned by one field and process.

    Entry count and retained strings/array payload are bounded independently;
    Python container overhead is outside the payload limit. Copies start cold.
    """

    def __init__(self, capacity: int, byte_limit: int = 64 * 1024 * 1024):
        self.capacity = capacity
        self.byte_limit = byte_limit
        self.values: OrderedDict[str, tuple[np.ndarray, np.ndarray, int]] = OrderedDict()
        self.retained_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.process = os.getpid()
        self.signature: tuple[object, ...] | None = None
        self.tokenizer: object | None = None
        self.lock = RLock()

    def __len__(self) -> int:
        self.isolate()
        with self.lock:
            return len(self.values)

    def __reduce__(self):
        return type(self), (self.capacity, self.byte_limit)

    def isolate(self) -> None:
        if self.process != os.getpid():
            # A fork must not inherit an acquired parent lock or input corpus.
            self.__init__(self.capacity, self.byte_limit)

    def clear(self) -> None:
        self.isolate()
        with self.lock:
            self.values.clear()
            self.retained_bytes = 0
            self.hits = self.misses = self.evictions = 0

    def encode(
        self, texts: list[str], tokenizer: PreTrainedTokenizerBase, *, length: int, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize unique strings, restoring cached rows in the caller's order."""
        self.isolate()
        backend = getattr(tokenizer, "backend_tokenizer", None)
        processing = []
        for name in ("normalizer", "pre_tokenizer", "post_processor"):
            component = getattr(backend, name, None)
            processing.append(component.__getstate__() if component is not None else None)
        special = repr(getattr(tokenizer, "special_tokens_map", None))
        signature = (
            length,
            getattr(tokenizer, "padding_side", "right"),
            getattr(tokenizer, "truncation_side", "right"),
            getattr(tokenizer, "pad_token_id", None),
            special,
            getattr(tokenizer, "vocab_size", None),
            repr(getattr(tokenizer, "added_tokens_decoder", None)),
            getattr(tokenizer, "do_lower_case", None),
            getattr(tokenizer, "split_special_tokens", None),
            tuple(processing),
        )
        ids = np.empty((len(texts), length), dtype=np.int64)
        mask = np.empty_like(ids)
        with self.lock:
            if self.tokenizer is not tokenizer or self.signature != signature:
                self.clear()
                self.tokenizer = tokenizer
                self.signature = signature
            missing: list[int] = []
            for index, value in enumerate(texts):
                cached = self.values.get(value)
                if cached is None:
                    missing.append(index)
                    self.misses += 1
                else:
                    self.values.move_to_end(value)
                    ids[index], mask[index] = cached[:2]
                    self.hits += 1
            for start in range(0, len(missing), batch_size):
                selected = missing[start : start + batch_size]
                encoded = tokenizer(
                    [texts[index] for index in selected],
                    padding="max_length",
                    truncation=True,
                    max_length=length,
                    return_tensors="np",
                )
                token_ids = np.asarray(encoded[INPUT_IDS], dtype=np.int64)
                attention = np.asarray(encoded[ATTENTION_MASK], dtype=np.int64)
                expected = (len(selected), length)
                if token_ids.shape != expected or attention.shape != expected:
                    raise ValueError(
                        f"text tokenizer returned shapes {token_ids.shape}/{attention.shape}, expected {expected}"
                    )
                for offset, index in enumerate(selected):
                    ids[index], mask[index] = token_ids[offset], attention[offset]
                    text = texts[index]
                    retained = sys.getsizeof(text) + token_ids[offset].nbytes + attention[offset].nbytes
                    if retained > self.byte_limit:
                        continue
                    while self.values and (
                        len(self.values) >= self.capacity or self.retained_bytes + retained > self.byte_limit
                    ):
                        _, (_, _, removed) = self.values.popitem(last=False)
                        self.retained_bytes -= removed
                        self.evictions += 1
                    self.values[text] = (token_ids[offset].copy(), attention[offset].copy(), retained)
                    self.retained_bytes += retained
        return ids, mask


class Pooling(enum.StrEnum):
    cls = "cls"
    mean = "mean"
    pooler = "pooler"


class Objective(enum.StrEnum):
    l1 = "l1"
    l2 = "l2"

    def loss(self, diff: torch.Tensor) -> torch.Tensor:
        """Mean error over every coordinate of the selected embeddings."""
        if self == Objective.l1:
            diff = diff.absolute()
        else:
            diff = diff.square()

        return diff.mean()


@dataclass
class CachedModel:
    _models: ClassVar[dict[str, "CachedModel"]] = {}
    _tokenizers: ClassVar[dict[str, PreTrainedTokenizerBase]] = {}

    key: str
    model: PreTrainedModel
    hidden_size: int
    device: torch.device | None = None

    @classmethod
    def get_model(
        cls,
        key: str,
    ) -> "CachedModel":
        if key not in cls._models:
            from transformers import AutoModel

            model = cast("PreTrainedModel", AutoModel.from_pretrained(key))
            model.eval()
            model.requires_grad_(False)
            hidden_size = getattr(model.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError(f"text model '{key}' does not expose `config.hidden_size`")
            cls._models[key] = cls(key=key, model=model, hidden_size=int(hidden_size))

        return cls._models[key]

    @classmethod
    def get_tokenizer(cls, key: str) -> PreTrainedTokenizerBase:
        if key not in cls._tokenizers:
            from transformers import AutoTokenizer

            tokenizer = cast("PreTrainedTokenizerBase", AutoTokenizer.from_pretrained(key))

            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                elif tokenizer.sep_token is not None:
                    tokenizer.pad_token = tokenizer.sep_token
                else:
                    raise ValueError(f"text model '{key}' tokenizer does not define a pad/eos/sep token")
            cls._tokenizers[key] = tokenizer

        return cls._tokenizers[key]

    def module_for(self, device: torch.device) -> PreTrainedModel:
        target = torch.device(device)
        if self.device != target:
            self.model.to(target)
            self.device = target

        self.model.eval()
        return self.model


@text.register(Component.Request)
class Request(RequestBase):
    """String input represented by a frozen Hugging Face encoder.

    ``model`` names a pretrained model or local path. ``max_length`` limits
    tokens; ``tokenizer_batch_size`` and ``encoder_batch_size`` bound each
    tokenizer and encoder call. ``encoder_pooling`` selects ``"cls"``,
    ``"mean"``, or ``"pooler"``. ``cache=True`` retains pooled frozen encoder
    outputs in CPU memory; a positive integer bounds the cache with least-used
    eviction. Token rows use a separate LRU with 4,096 entries for ``True`` or
    at most N entries for ``cache=N``, plus a 64 MiB string/array payload limit;
    Python container overhead is additional. ``True`` leaves the frozen
    embedding count unbounded. Both caches are per field/process and excluded from
    checkpoints; token caches also start cold after copying or process changes.
    ``store`` names a directory of persistent frozen-encoder vectors keyed by
    encoder configuration and token row; every split, run, and field with the
    same encoder shares it, misses are encoded once and appended, and the
    in-process embedding cache is bypassed. ``resident`` is the accelerator
    memory in MiB a store may occupy so lookups never leave the device; ``None``
    allows a quarter of the device up to 2 GiB and ``0`` keeps the store on the
    host. ``jitter`` perturbs pooled training inputs.
    Reconstructing masks use ``"l1"`` or ``"l2"`` ``objective`` loss against
    pristine frozen embeddings. Prediction can export an embedding with
    ``embed=True``; this field does not generate text.
    """

    model_config = pydantic.ConfigDict(str_strip_whitespace=True)

    type: Literal["text"] = "text"
    model: Annotated[str, pydantic.Field(min_length=1)] = DEFAULT_TEXT_MODEL
    max_length: Annotated[int, pydantic.Field(gt=0, default=128)] = 128
    tokenizer_batch_size: Annotated[int, pydantic.Field(gt=0, default=4096)] = 4096
    encoder_batch_size: Annotated[int, pydantic.Field(gt=0, default=32)] = 32
    encoder_pooling: Pooling = Pooling.cls
    cache: pydantic.StrictBool | Annotated[pydantic.StrictInt, pydantic.Field(gt=0)] = False
    store: Annotated[str, pydantic.Field(min_length=1)] | None = None
    resident: Annotated[pydantic.StrictInt, pydantic.Field(ge=0)] | None = None
    objective: Objective = Objective.l2
    jitter: Jitter = pydantic.Field(default_factory=Jitter)

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            model: str = DEFAULT_TEXT_MODEL,
            max_length: int = 128,
            tokenizer_batch_size: int = 4096,
            encoder_batch_size: int = 32,
            encoder_pooling: Pooling | Literal["cls", "mean", "pooler"] = Pooling.cls,
            cache: bool | int = False,
            store: str | None = None,
            resident: int | None = None,
            objective: Objective | Literal["l1", "l2"] = Objective.l2,
            jitter: Jitter = ...,
            type: Literal["text"] = "text",
            **options: Unpack[FieldOptions],
        ) -> None: ...


@text.register(Component.observe)
def observe(
    field: RaggedField,
    *,
    address: Address,
    schema: Schema,
    state: object | None,
    learn: bool,
) -> TensorDict | None:
    """Count pristine text states without tokenizing hidden content."""

    if not learn:
        return None
    return TensorDict(
        {TensorKey.state: tally(torch.from_numpy(field.dense.copy()), len(Tokens))},
        batch_size=[],
    )


@text.register(Component.TensorField)
@tensorclass
class TensorField(TensorFieldBase[TensorDict]):
    content: TensorDict
    state: torch.Tensor
    present: torch.Tensor
    trainable: torch.Tensor
    inferred: torch.Tensor
    targets: TensorDict

    if TYPE_CHECKING:
        # TensorClass metadata is accepted by its generated initializer.
        batch_size: InitVar[int | torch.Size | list[int] | tuple[int, ...] | None] = None
        device: InitVar[torch.device | str | int | None] = None
        names: InitVar[list[str | None] | None] = None

    @classmethod
    def new(
        cls,
        input: RaggedField,
        target: RaggedField,
        present: torch.Tensor,
        trainable: torch.Tensor,
        inferred: torch.Tensor,
        address: Address,
        schema: Schema,
        strata: Strata,
        context: Context,
    ) -> TensorField:
        request: Request = cast(Request, schema.requests[address])

        def encode(field: RaggedField) -> TensorDict:
            length = request.max_length
            size = math.prod(field.shape)
            encoded_ids = np.zeros((size, length), dtype=np.int64)
            encoded_mask = np.zeros((size, length), dtype=np.int64)

            if len(field.values):
                tokenizer = CachedModel.get_tokenizer(request.model)
                placement = field.placement.to_numpy(zero_copy_only=False)
                values = field.values
                if pa.types.is_dictionary(values.type):
                    values = values.cast(cast(pa.DictionaryType, values.type).value_type)
                dictionary = pc.dictionary_encode(values)
                if isinstance(dictionary, pa.ChunkedArray):
                    dictionary = dictionary.combine_chunks()
                dictionary = cast(pa.DictionaryArray, dictionary)
                unique = dictionary.dictionary
                inverse = dictionary.indices.to_numpy(zero_copy_only=False)
                token_cache = context.state if request.cache and isinstance(context.state, TokenCache) else None
                if token_cache is not None:
                    cached_ids, cached_mask = token_cache.encode(
                        unique.to_numpy(zero_copy_only=False).tolist(),
                        tokenizer,
                        length=length,
                        batch_size=request.tokenizer_batch_size,
                    )
                    encoded_ids[placement] = cached_ids[inverse]
                    encoded_mask[placement] = cached_mask[inverse]
                else:
                    repeated = len(unique) != len(field.values)
                    for start in range(0, len(unique), request.tokenizer_batch_size):
                        stop = min(start + request.tokenizer_batch_size, len(unique))
                        texts = unique.slice(start, stop - start).to_numpy(zero_copy_only=False).tolist()
                        encoded = tokenizer(
                            texts,
                            padding="max_length",
                            truncation=True,
                            max_length=length,
                            return_tensors="np",
                        )
                        selected = np.flatnonzero((inverse >= start) & (inverse < stop)) if repeated else None
                        positions = placement[selected] if repeated else placement[start:stop]
                        expected = (stop - start, length)
                        for name, destination in (
                            (INPUT_IDS, encoded_ids),
                            (ATTENTION_MASK, encoded_mask),
                        ):
                            values = np.asarray(encoded[name], dtype=np.int64)
                            if values.shape != expected:
                                raise ValueError(
                                    f"text tokenizer '{request.model}' returned {name} shape {values.shape}, "
                                    f"expected {expected}"
                                )
                            destination[positions] = values[inverse[selected] - start] if repeated else values

            return TensorDict(
                {
                    INPUT_IDS: torch.from_numpy(encoded_ids.reshape(*field.shape, length)),
                    ATTENTION_MASK: torch.from_numpy(encoded_mask.reshape(*field.shape, length)),
                },
                batch_size=field.shape,
            )

        state_tensor = torch.from_numpy(input.dense)

        return cls(
            state=state_tensor,
            content=encode(input),
            present=present,
            trainable=trainable,
            inferred=inferred,
            targets=TensorDict(
                {
                    TensorKey.state: torch.from_numpy(target.dense),
                    TensorKey.content: encode(target),
                }
                if address in schema.objectives
                else {},
                batch_size=input.shape,
            ),
            batch_size=input.batch_size,
        )


@text.register(Component.Embedder)
class Embedder(EmbedderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = cast(Request, schema.requests[address])

        self.origin: Address = address
        self.destination: Address = cast("Branch", request.parent).address
        cached_model = CachedModel.get_model(request.model)
        self.__dict__["_cached_model"] = cached_model
        self.hidden_size: int = cached_model.hidden_size
        self.request = request
        self.cache = EmbeddingCache(None if request.cache is True else request.cache) if request.cache else None
        self.tokens = TokenCache(4096 if request.cache is True else request.cache) if request.cache else None
        self.store = (
            EmbeddingStore.open(request.store, self.fingerprint(request), self.hidden_size, resident=request.resident)
            if request.store
            else None
        )
        self.jitter: Jitter = request.jitter

        self.embeddings = torch.nn.Embedding(
            num_embeddings=len(Tokens),
            embedding_dim=schema.d_model,
        )
        self.counter = Counter(address=address, size=len(Tokens))
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.hidden_size, out_features=schema.d_model),
            torch.nn.GELU(approximate="tanh"),
        )

    @staticmethod
    def fingerprint(request: Request) -> str:
        """The store name for one encoder configuration; precision joins the row hash, not the name."""
        payload = f"{request.model}|{request.encoder_pooling}".encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    @property
    def context(self) -> TokenCache | None:
        return self.tokens

    def consult(self, valued_ids: torch.Tensor, valued_mask: torch.Tensor) -> torch.Tensor:
        """Pooled vectors for token rows from the store, encoding and remembering only the rows it lacks.

        Rows are hashed on their own device. A store small enough to be resident
        there answers without a host round trip; the host index answers the rest,
        and misses reach the encoder once per distinct row.
        """
        assert self.store is not None
        device = valued_ids.device
        dtype = torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type) else None
        salt = int.from_bytes(hashlib.sha256(f"{device.type}|{dtype}".encode()).digest()[:8], "big")
        keys = hash_rows_torch(torch.cat((valued_ids, valued_mask), dim=1), salt=salt)
        out = torch.empty((len(keys), self.hidden_size), device=device, dtype=torch.float32)
        found = torch.zeros(len(keys), dtype=torch.bool, device=device)
        resident = self.store.lookup_device(keys)
        if resident is not None:
            found, vectors = resident
            out[found] = vectors[found]
        remaining = (~found).nonzero(as_tuple=False).reshape(-1)
        if not len(remaining):
            return out
        host_keys = keys.index_select(0, remaining).cpu().numpy()
        host_found, host_vectors = self.store.lookup(host_keys)
        if host_found.any():
            hit = remaining[torch.from_numpy(host_found).to(device)]
            out[hit] = torch.from_numpy(host_vectors[host_found]).to(device)
        missing = np.flatnonzero(~host_found)
        if len(missing):
            unique_keys, first, inverse = np.unique(host_keys[missing], axis=0, return_index=True, return_inverse=True)
            rows = remaining[torch.from_numpy(missing).to(device)]
            select = remaining[torch.from_numpy(missing[first]).to(device)]
            fresh = self.pool(valued_ids.index_select(0, select), valued_mask.index_select(0, select))
            out[rows] = fresh.index_select(0, torch.from_numpy(inverse.reshape(-1)).to(device))
            self.store.put(unique_keys, fresh.detach().cpu().numpy())
        return out

    def warm(self, values: Sequence[str] | pa.Array | pa.ChunkedArray) -> int:
        """Encode strings the store lacks and persist them; returns how many vectors were new.

        Run it under the same device and autocast context as training, since
        precision is part of the key. Nulls are skipped; repeated strings count once.
        """
        if self.store is None:
            raise ValueError(f"text field at '{self.origin}' has no store to warm; set Text(store=...)")
        texts = values.to_pylist() if isinstance(values, (pa.Array, pa.ChunkedArray)) else list(values)
        unique = list(dict.fromkeys(text for text in texts if text is not None))
        tokenizer = CachedModel.get_tokenizer(self.request.model)
        device = next(self.linear.parameters()).device
        before = len(self.store)
        for start in range(0, len(unique), self.request.tokenizer_batch_size):
            encoded = tokenizer(
                unique[start : start + self.request.tokenizer_batch_size],
                padding="max_length",
                truncation=True,
                max_length=self.request.max_length,
                return_tensors="np",
            )
            ids = torch.from_numpy(np.asarray(encoded[INPUT_IDS], dtype=np.int64)).to(device)
            mask = torch.from_numpy(np.asarray(encoded[ATTENTION_MASK], dtype=np.int64)).to(device)
            self.consult(ids, mask)
        self.store.flush()
        return len(self.store) - before

    @property
    def padding_side(self) -> str:
        return getattr(CachedModel.get_tokenizer(self.request.model), "padding_side", "right")

    @beartype
    def encode(
        self,
        content: TensorDict,
        state: torch.Tensor,
    ) -> torch.Tensor:
        N, *dims = state.shape
        D = math.prod((N, *dims))

        flat_state = state.reshape(D)
        flat_ids = cast(torch.Tensor, content[INPUT_IDS]).reshape(D, -1)
        flat_mask = cast(torch.Tensor, content[ATTENTION_MASK]).reshape(D, -1)

        embeddings = torch.zeros((D, self.hidden_size), device=flat_ids.device, dtype=torch.float32)
        valued = flat_state.eq(Tokens.valued.value)
        if not valued.any():
            return embeddings.reshape(N, *dims, self.hidden_size)

        valued_ids = flat_ids[valued]
        valued_mask = flat_mask[valued]
        if self.store is not None:
            embeddings[valued] = self.consult(valued_ids, valued_mask)
            return embeddings.reshape(N, *dims, self.hidden_size)
        if self.cache is None:
            embeddings[valued] = self.pool(valued_ids, valued_mask)
            return embeddings.reshape(N, *dims, self.hidden_size)

        # Exact token rows include padding and the attention mask. Equal
        # truncated inputs can reuse a vector; different precision/device
        # contexts must not silently reuse a lower-precision result.
        device = valued_ids.device
        dtype = torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type) else None
        rows = (
            torch.stack((valued_ids, valued_mask), dim=1)
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .reshape(len(valued_ids), -1)
        )
        keys = cast(
            list[bytes], rows.view(np.dtype((np.void, rows.shape[1] * rows.dtype.itemsize))).reshape(-1).tolist()
        )
        hits: list[torch.Tensor] = []
        hit_positions: list[int] = []
        missing: dict[CacheKey, list[int]] = {}
        for position, row in enumerate(keys):
            key = (str(device), dtype, row)
            value = self.cache.get(key)
            if value is None:
                missing.setdefault(key, []).append(position)
            else:
                hit_positions.append(position)
                hits.append(value)

        valued_embeddings = torch.empty((len(keys), self.hidden_size), device=device, dtype=torch.float32)
        if hits:
            valued_embeddings.index_copy_(
                0,
                torch.tensor(hit_positions, device=device),
                torch.stack(hits).to(device),
            )
        if missing:
            indices = torch.tensor([positions[0] for positions in missing.values()], device=device)
            fresh = self.pool(valued_ids.index_select(0, indices), valued_mask.index_select(0, indices))
            positions = torch.tensor([position for group in missing.values() for position in group], device=device)
            inverse = torch.tensor(
                [index for index, group in enumerate(missing.values()) for _ in group], device=device
            )
            valued_embeddings.index_copy_(0, positions, fresh.index_select(0, inverse))
            host = fresh.detach().to("cpu")
            for (key, group), value in zip(missing.items(), host, strict=True):
                self.cache.put(key, value, count=len(group))
        embeddings[valued] = valued_embeddings
        return embeddings.reshape(N, *dims, self.hidden_size)

    def pool(self, valued_ids: torch.Tensor, valued_mask: torch.Tensor) -> torch.Tensor:
        """Encode token rows with the frozen model and restore their input order."""
        model = self.__dict__["_cached_model"].module_for(valued_ids.device)

        restore_order: torch.Tensor | None = None
        starts = range(0, valued_ids.size(0), self.request.encoder_batch_size)
        stops = [min(start + self.request.encoder_batch_size, valued_ids.size(0)) for start in starts]
        if self.padding_side == "right":
            lengths = valued_mask.count_nonzero(dim=-1)
            restore_order = torch.argsort(lengths, stable=True)
            valued_ids = valued_ids.index_select(0, restore_order)
            valued_mask = valued_mask.index_select(0, restore_order)
            sorted_lengths = lengths.index_select(0, restore_order)

            # Copy every microbatch boundary length to the host in one
            # synchronization, rather than calling .item() before each encoder
            # invocation. A valued row with no attended tokens is forwarded at
            # full width to preserve the previous cls/pooler behavior.
            metadata_indexes = torch.as_tensor(
                [*starts, *(stop - 1 for stop in stops)],
                device=sorted_lengths.device,
            )
            metadata = sorted_lengths.index_select(0, metadata_indexes).tolist()
            boundary = len(stops)
            chunk_minimums = metadata[:boundary]
            chunk_maximums = metadata[boundary:]
            widths = [
                valued_ids.size(1) if minimum == 0 else maximum
                for minimum, maximum in zip(chunk_minimums, chunk_maximums, strict=True)
            ]
        else:
            # Removing leading padding changes absolute position ids for many
            # models. Keep the full width and original order for left-padded
            # tokenizers.
            widths = [valued_ids.size(1)] * len(stops)

        encoded: list[torch.Tensor] = []
        for start, stop, width in zip(starts, stops, widths, strict=True):
            batch_ids = valued_ids[start:stop, :width]
            batch_mask = valued_mask[start:stop, :width]

            with torch.inference_mode():
                outputs = model(
                    input_ids=batch_ids,
                    attention_mask=batch_mask,
                )

            if self.request.encoder_pooling == Pooling.pooler:
                pooled = getattr(outputs, "pooler_output", None)
                if pooled is None:
                    raise ValueError(f"text model '{self.request.model}' does not expose pooler_output")
            else:
                hidden = getattr(outputs, "last_hidden_state", None)
                if hidden is None:
                    raise ValueError(f"text model '{self.request.model}' does not expose last_hidden_state")

                if self.request.encoder_pooling == Pooling.cls:
                    pooled = hidden[:, 0]
                else:
                    mask = batch_mask.unsqueeze(-1).to(dtype=hidden.dtype)
                    denom = mask.sum(dim=1).clamp_min(1.0)
                    pooled = (hidden * mask).sum(dim=1) / denom

            encoded.append(pooled.to(dtype=torch.float32))

        batched_embeddings = torch.cat(encoded, dim=0)
        if restore_order is None:
            valued_embeddings = batched_embeddings
        else:
            valued_embeddings = torch.empty_like(batched_embeddings)
            valued_embeddings[restore_order] = batched_embeddings
        return valued_embeddings

    @beartype
    def forward(self, inputs: TensorInput) -> Parcel:
        N, *dims = inputs.state.shape
        D = math.prod((N, *dims))

        state = inputs.state.reshape(D)
        valued = state.eq(Tokens.valued.value).unsqueeze(-1)
        encoded = self.encode(content=cast(TensorDict, inputs.content), state=inputs.state)
        encoded = encoded.reshape(D, self.hidden_size)
        if self.training:
            eligible = valued & torch.isfinite(encoded)
            encoded = self.jitter.apply(encoded, eligible)
        projected = self.linear(encoded) * valued
        embeddings = self.embeddings(state)

        return Parcel(
            payload=(embeddings + projected).reshape(N, *dims, -1),
            present=torch.ones(N, dtype=torch.bool, device=embeddings.device),
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )


@text.register(Component.learn)
def learn(
    module: Model,
    observation: TensorDict,
    *,
    address: Address,
    strata: Strata,
) -> None:
    """Apply pristine text-state counts to the model resource."""

    if strata != Strata.train:
        raise ValueError(f"text learner at '{address}' requires train strata, got {strata}")
    embedder: Embedder = cast(Embedder, module.nodes[address].embedder)
    embedder.counter.learn(cast(torch.Tensor, observation[TensorKey.state]))


@text.register(Component.Decoder)
class Decoder(DecoderBase):
    coordinatewise = True

    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = cast(Request, schema.requests[address])
        hidden_size = CachedModel.get_model(request.model).hidden_size
        self.classification = torch.nn.Linear(
            in_features=schema.d_model,
            out_features=len(Tokens),
        )
        self.linear = torch.nn.Linear(
            in_features=schema.d_model,
            out_features=hidden_size,
        )
        self.metrics = Scores(address, Errors)

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict:
        return TensorDict(
            source={
                TensorKey.state: self.classification(pooled),
                TensorKey.content: self.linear(pooled),
            }
        )


@text.register(Component.loss)
def loss(
    module: Model,
    prediction: Prediction,
    batch: TensorField,
    strata: Strata,
) -> torch.Tensor:
    address: Address = prediction.address
    request: Request = cast(Request, module.schema.requests[address])
    embedder: Embedder = cast(Embedder, module.nodes[address].embedder)
    decoder: Decoder = cast(Decoder, module.nodes[address].decoder)

    trainable = batch.trainable.reshape(-1)
    state_targets = cast(torch.Tensor, batch.targets[TensorKey.state]).reshape(-1)
    state_inputs = cast(torch.Tensor, prediction.payload[TensorKey.state]).reshape(-1, len(Tokens))

    loss: torch.Tensor = module.track(
        (address, strata, Metric.loss, TensorKey.state),
        value=(
            torch.nn.functional.cross_entropy(
                input=state_inputs,
                target=state_targets,
                weight=embedder.counter.weight,
                reduction="none",
            )
            .masked_select(trainable)
            .mean()
        ),
    )

    module.track(
        (address, strata, Metric.accuracy, TensorKey.state),
        value=state_inputs.argmax(dim=1).eq(state_targets).masked_select(trainable).float().mean(),
    )

    valued = trainable & state_targets.eq(Tokens.valued.value)
    inputs = cast(torch.Tensor, prediction.payload[TensorKey.content]).reshape(-1, embedder.hidden_size)
    predicted = inputs[valued]
    observed = inputs.new_zeros((0, embedder.hidden_size))
    if predicted.numel():
        if TensorKey.embedding not in batch.targets.keys():
            batch.targets[TensorKey.embedding] = embedder.encode(
                content=cast(TensorDict, batch.targets[TensorKey.content]),
                state=cast(torch.Tensor, batch.targets[TensorKey.state]),
            )
        targets = cast(torch.Tensor, batch.targets[TensorKey.embedding]).reshape(-1, embedder.hidden_size)
        observed = targets[valued]
        loss += module.track(
            (address, strata, Metric.loss, TensorKey.content),
            value=request.objective.loss(predicted.subtract(observed)),
        )

    decoder.metrics.record(module, strata, *errors(predicted, observed))

    return loss
