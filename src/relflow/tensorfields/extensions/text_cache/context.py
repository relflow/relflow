"""Compact IDs at tensorization, with frozen-encoder fallback in the trainer."""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

import numpy as np
import pyarrow as pa

from relflow.tensorfields.extensions.text_cache.tokens import tokens
from relflow.tensorfields.shared.cache import MAX_FRAME, MAX_ROWS, Endpoint
from relflow.tensorfields.shared.cache.vectors import VectorCache


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


class Encoding(VectorCache):
    """Only configuration and the local capability travel to spawned workers."""

    vector_namespace = "rf.text.vector.f32/v1"

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
        super().__init__(path, memory, capacity, endpoint)
        self.input_bytes = length * 16 + 4
        self.tokenizer = None
        self.base = ""
        self.settings = None
        self.namespace = ""
        self.lock = threading.RLock()

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

    def identify(self, texts: list[str] | pa.Array, tokenizer: Any) -> np.ndarray:
        if not isinstance(texts, pa.Array):
            texts = pa.array(texts, type=pa.large_string())
        with self.lock:
            settings = signature(tokenizer)
            if self.tokenizer is not tokenizer:
                self.base = tokenizer_identity(tokenizer)
                self.tokenizer = tokenizer
                self.namespace = ""
            if self.settings != settings or not self.namespace:
                self.namespace = hashlib.sha256(
                    repr(("rf.text.tokens/v2", self.model, self.base, self.length, settings)).encode()
                ).hexdigest()
                self.settings = settings
            output = np.empty(len(texts), dtype=np.int64)
            chunk = min(MAX_ROWS, self.batch_size, (MAX_FRAME - 65536) // (self.length * 16 + 48))
            if chunk <= 0:
                raise ValueError(f"text token length {self.length} exceeds the cache frame budget; reduce max_length")
            for start in range(0, len(texts), chunk):
                selected = texts.slice(start, min(chunk, len(texts) - start))
                references = self.client.lookup_arrow(selected, self.namespace)
                ids = references.objects
                missing = np.flatnonzero(~references.ready)
                if len(missing):
                    strings = selected.take(missing).to_pylist()
                    encoded = tokenizer(
                        strings, padding="max_length", truncation=True, max_length=self.length, return_tensors="np"
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
                    ids[missing] = self.client.bind(
                        references.take(missing), tokens(rows, tokenizer.pad_token_id), self.namespace
                    )
                output[start : start + len(selected)] = ids
            return output
