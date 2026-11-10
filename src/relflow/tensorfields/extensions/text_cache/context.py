"""Compact IDs at tensorization, with frozen-encoder fallback in the trainer."""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, cast

import numpy as np
import pyarrow as pa
import torch

from relflow.tensorfields.extensions.text_cache.device import Device
from relflow.tensorfields.extensions.text_cache.store import MAX_FRAME, MAX_ROWS, Client, Endpoint, Owner, digest


def signature(tokenizer: Any) -> tuple:
    """Mutable tokenizer settings that can change an already loaded tokenizer."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    processing = []
    for name in ("normalizer", "pre_tokenizer", "post_processor"):
        component = getattr(backend, name, None)
        processing.append(component.__getstate__() if component is not None else None)
    return (
        getattr(tokenizer, "padding_side", "right"),
        getattr(tokenizer, "truncation_side", "right"),
        getattr(tokenizer, "pad_token_id", None),
        repr(getattr(tokenizer, "special_tokens_map", None)),
        getattr(tokenizer, "vocab_size", None),
        repr(getattr(tokenizer, "added_tokens_decoder", None)),
        getattr(tokenizer, "do_lower_case", None),
        getattr(tokenizer, "split_special_tokens", None),
        tuple(processing),
    )


def tokenizer_identity(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        configuration = json.loads(backend.to_str())
        configuration.pop("padding", None)
        configuration.pop("truncation", None)
        payload = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    elif callable(getattr(tokenizer, "get_vocab", None)):
        payload = json.dumps(tokenizer.get_vocab(), sort_keys=True, separators=(",", ":"))
    else:
        code = getattr(type(tokenizer).__call__, "__code__", None)
        payload = repr((type(tokenizer).__module__, type(tokenizer).__qualname__, code.co_code if code else None))
    return hashlib.sha256(payload.encode()).hexdigest()


class Encoding:
    """Only configuration and the local capability travel to spawned workers."""

    def __init__(
        self,
        model: str,
        length: int,
        batch_size: int,
        path: str | None,
        memory: int | None,
        capacity: int | None,
        endpoint: Endpoint | None = None,
    ):
        self.model, self.length, self.batch_size = model, length, batch_size
        self.path, self.memory, self.capacity = path, memory, capacity
        self.endpoint = endpoint
        self.owner = Owner.open(path, memory, capacity) if endpoint is None else Owner.hold(endpoint)
        self.connection: Client | None = None
        self.tokenizer = None
        self.base = ""
        self.settings = None
        self.namespace = ""
        self.lock = threading.RLock()
        self.views: dict[str, Device] = {}

    def __reduce__(self):
        return type(self), (
            self.model,
            self.length,
            self.batch_size,
            self.path,
            self.memory,
            self.capacity,
            self.endpoint,
        )

    @property
    def client(self) -> Client:
        if self.connection is None:
            if self.endpoint is None:
                assert self.owner is not None
                self.endpoint = self.owner.start().endpoint
            self.connection = Client.open(self.endpoint)
        return self.connection

    def share(self) -> None:
        self.client

    def identify(self, texts: list[str] | pa.Array, tokenizer: Any) -> np.ndarray:
        if isinstance(texts, pa.Array):
            texts = cast(list[str], texts.to_pylist())
        with self.lock:
            settings = signature(tokenizer)
            if self.tokenizer is not tokenizer:
                self.base = tokenizer_identity(tokenizer)
                self.tokenizer = tokenizer
                self.namespace = ""
            if self.settings != settings or not self.namespace:
                self.namespace = hashlib.sha256(
                    repr((self.model, self.base, self.length, settings)).encode()
                ).hexdigest()
                self.settings = settings
            keys = [digest(self.namespace, text) for text in texts]
            output = np.empty(len(texts), dtype=np.int64)
            chunk = min(MAX_ROWS, self.batch_size, (MAX_FRAME - 65536) // (self.length * 16 + 40))
            if chunk <= 0:
                raise ValueError(f"text token length {self.length} exceeds the cache frame budget; reduce max_length")
            for start in range(0, len(texts), chunk):
                stop = min(start + chunk, len(texts))
                ids, ready = self.client.identify(keys[start:stop], self.length)
                missing = np.flatnonzero(~ready)
                if len(missing):
                    encoded = tokenizer(
                        [texts[start + int(i)] for i in missing],
                        padding="max_length",
                        truncation=True,
                        max_length=self.length,
                        return_tensors="np",
                    )
                    rows = np.stack(
                        (
                            np.asarray(encoded["input_ids"], dtype=np.int64),
                            np.asarray(encoded["attention_mask"], dtype=np.int64),
                        ),
                        axis=1,
                    )
                    if rows.shape != (len(missing), 2, self.length):
                        raise ValueError(
                            f"text tokenizer '{self.model}' returned {rows.shape}; expected {(len(missing), 2, self.length)}"
                        )
                    ids[missing] = self.client.tokens(
                        ids[missing], [keys[start + int(i)] for i in missing], rows, self.namespace
                    )
                output[start:stop] = ids
            return output

    @torch.compiler.disable
    def resolve(
        self, identifiers: torch.Tensor, *, variant: str, width: int, resident: int | None, pool
    ) -> torch.Tensor:
        client = self.client
        name = f"{identifiers.device}|{variant}"
        view = self.views.get(name)
        if view is None:
            view = Device.open(client.endpoint.identity, variant, width, identifiers.device, resident, self.capacity)
            if view is not None:
                self.views[name] = view
        with client.resolve_lock:
            if view is None:
                return self.fetch(identifiers, variant, width, pool)
            with view.lock:
                found, cached = view.lookup(identifiers)
                if found.all():
                    # A fresh tensor is safe for later training after an
                    # inference_mode warm-up and cannot alias the cache table.
                    return cached.clone()
                output = torch.empty((len(identifiers), width), device=identifiers.device, dtype=torch.float32)
                output[found] = cached[found]
                missing = identifiers[~found]
                host = missing.detach().cpu().numpy()
                unique, inverse, counts = np.unique(host, return_inverse=True, return_counts=True)
                values = self.fetch(torch.from_numpy(unique).to(identifiers.device), variant, width, pool)
                output[~found] = values.index_select(0, torch.from_numpy(inverse).to(identifiers.device))
                view.admit(unique, values, counts)
                return output

    def fetch(self, identifiers: torch.Tensor, variant: str, width: int, pool) -> torch.Tensor:
        host = identifiers.detach().cpu().numpy()
        unique, inverse = np.unique(host, return_inverse=True)
        output = np.empty((len(unique), width), dtype=np.float32)
        chunk = min(MAX_ROWS, (MAX_FRAME - 65536) // max(width * 4, self.length * 16))
        if chunk <= 0:
            raise ValueError("text encoder vector or token row exceeds the cache frame budget")
        try:
            for start in range(0, len(unique), chunk):
                selected = unique[start : start + chunk]
                pending = np.arange(len(selected))
                while len(pending):
                    status, vectors, tokens = self.client.get(selected[pending], variant, width, self.length)
                    output[start + pending[status == 1]] = vectors
                    owned = pending[status == 0]
                    if len(owned):
                        device = identifiers.device
                        fresh = pool(
                            torch.from_numpy(tokens[:, 0]).to(device), torch.from_numpy(tokens[:, 1]).to(device)
                        )
                        committed = self.client.put(selected[owned], variant, fresh.detach().cpu().numpy())
                        output[start + owned] = committed
                    pending = pending[status == 2]
                    if len(pending):
                        self.client.wait()
        except BaseException:
            self.client.release()
            raise
        return torch.from_numpy(output[inverse]).to(identifiers.device)

    def stats(self) -> dict:
        return self.client.stats()
