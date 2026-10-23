from __future__ import annotations

import enum
import math
import os
import sys
from collections import OrderedDict
from dataclasses import InitVar, dataclass
from threading import RLock
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, TypeAlias, Unpack, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from relflow.data.ragged import RaggedField
from relflow.helpers import Jitter
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

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from relflow.architecture.root import Model
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch


text: Extension = Extension(
    name="text",
    types=(str,),
    requires={"transformers": "relflow[text]"},
)
text.callback(CounterUpdateCallback)

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
        if self == Objective.l1:
            diff = diff.absolute()
        else:
            diff = diff.square()

        return diff.mean(dim=1)


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
    ``jitter`` perturbs pooled training inputs.
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

    @property
    def context(self) -> TokenCache | None:
        return self.tokens

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
    if not valued.any():
        return loss

    inputs = cast(torch.Tensor, prediction.payload[TensorKey.content]).reshape(-1, embedder.hidden_size)
    if TensorKey.embedding not in batch.targets.keys():
        batch.targets[TensorKey.embedding] = embedder.encode(
            content=cast(TensorDict, batch.targets[TensorKey.content]),
            state=cast(torch.Tensor, batch.targets[TensorKey.state]),
        )
    targets = cast(torch.Tensor, batch.targets[TensorKey.embedding]).reshape(-1, embedder.hidden_size)
    diff = inputs.subtract(targets)

    loss += module.track(
        (address, strata, Metric.loss, TensorKey.content),
        value=request.objective.loss(diff).masked_select(valued).mean(),
    )

    module.track(
        (address, strata, Metric.mae, TensorKey.content),
        value=diff.absolute().mean(dim=1).masked_select(valued).mean(),
    )

    module.track(
        (address, strata, Metric.rmse, TensorKey.content),
        value=diff.square().mean(dim=1).sqrt().masked_select(valued).mean(),
    )

    return loss
