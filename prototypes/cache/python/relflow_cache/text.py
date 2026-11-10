"""Text-specific preparation and fixed-vector codecs over the opaque cache.

NATIVE_TEXT_MODE=owner uses the Python wire client for the transport ablation;
full (the default) hashes Arrow buffers and performs I/O through PyO3.
"""

from __future__ import annotations

import hashlib
import os
from typing import Literal

import numpy as np
import pyarrow as pa

from relflow.structs.enums import Component
from relflow.tensorfields.base import Extension
from relflow.tensorfields.extensions import text
from relflow.tensorfields.extensions.text_cache.context import Encoding, signature, tokenizer_identity

from .core import MAX_FRAME, MAX_ROWS, Cache, Owner, Payloads


class TextCache(Cache):
    """Only this adapter interprets token matrices and float32 pooled vectors."""

    def tokens(self, references, rows, namespace):
        length = rows.shape[-1]
        codec = np.dtype([("length", "<u4"), ("tokens", "<i8", (2, length))])
        records = np.empty(len(rows), dtype=codec)
        records["length"], records["tokens"] = length, rows
        return self.bind(references, Payloads.fixed(records, codec.itemsize), namespace)

    def get(self, ids, variant, width, length):
        fetched = self.fetch(ids, f"rf.text.vector.f32/v1:{variant}")
        if np.any(np.diff(fetched.hits.offsets) != width * 4):
            raise ValueError("cached text vector width does not match encoder namespace")
        vectors = np.frombuffer(fetched.hits.data, dtype="<f4").reshape(-1, width).copy()
        codec = np.dtype([("length", "<u4"), ("tokens", "<i8", (2, length))])
        if np.any(np.diff(fetched.inputs.offsets) != codec.itemsize):
            raise ValueError("cached text preparation has a different token shape")
        records = np.frombuffer(fetched.inputs.data, dtype=codec)
        if np.any(records["length"] != length):
            raise ValueError("cached text preparation has a different token length")
        return fetched.status, vectors, records["tokens"].copy()

    def put(self, ids, variant, vectors):
        rows = np.ascontiguousarray(vectors, dtype="<f4")
        committed = self.publish(ids, f"rf.text.vector.f32/v1:{variant}", Payloads.fixed(rows, rows.shape[-1] * 4))
        if np.any(np.diff(committed.offsets) != rows.shape[-1] * 4):
            raise ValueError("committed text vectors have a different width")
        return np.frombuffer(committed.data, dtype="<f4").reshape(rows.shape).copy()


class NativeEncoding(Encoding):
    def __init__(self, model, length, batch_size, path, memory, capacity, endpoint=None):
        super().__init__(model, length, batch_size, path, memory, capacity, endpoint)
        self.owner = (
            Owner.open(path, (64 if memory is None else memory) * 2**20, capacity)
            if endpoint is None
            else Owner.hold(endpoint)
        )

    @property
    def client(self):
        if self.connection is None:
            if self.endpoint is None:
                self.endpoint = self.owner.start()
            self.connection = TextCache(endpoint=self.endpoint, native=os.environ.get("NATIVE_TEXT_MODE") != "owner")
        return self.connection

    def identify(self, texts, tokenizer):
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
                    repr(("rf.text.tokens/v1", self.model, self.base, self.length, settings)).encode()
                ).hexdigest()
                self.settings = settings
            output = np.empty(len(texts), dtype=np.int64)
            chunk = min(MAX_ROWS, self.batch_size, (MAX_FRAME - 65536) // (self.length * 16 + 48))
            if chunk <= 0:
                raise ValueError("token length exceeds native frame budget; reduce max_length")
            for start in range(0, len(texts), chunk):
                selected = texts.slice(start, min(chunk, len(texts) - start))
                references = self.client.lookup_arrow(selected, self.namespace)
                ids = references.objects
                missing = np.flatnonzero(~references.ready)
                if len(missing):
                    # Python strings exist only at the tokenizer's cold library boundary.
                    strings = selected.take(pa.array(missing)).to_pylist()
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
                        raise ValueError("tokenizer returned an invalid token batch")
                    ids[missing] = self.client.tokens(references.take(missing), rows, self.namespace)
                output[start : start + len(selected)] = ids
            return output


extension = Extension(
    name="text_native",
    types=(str,),
    requires={"transformers": "relflow[text]", "relflow_cache": "native prototype"},
    callbacks=text.text.callback_factories,
)


@extension.register(Component.Request)
class Request(text.Request):
    type: Literal["text_native"] = "text_native"


@extension.register(Component.Embedder)
class Embedder(text.Embedder):
    def __init__(self, schema, address):
        super().__init__(schema, address)
        request = self.request
        capacity = None if isinstance(request.cache, bool) else request.cache
        self.store = (
            NativeEncoding(
                request.model, request.max_length, request.tokenizer_batch_size, request.store, request.memory, capacity
            )
            if request.cache
            else None
        )


for role, component in text.text.components.items():
    if role not in (Component.Request, Component.Embedder):
        extension.register(role, component)
