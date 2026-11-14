from __future__ import annotations

import enum
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import InitVar, dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Unpack, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from beartype import beartype
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
from relflow.tensorfields.extensions.text_cache.context import Encoding
from relflow.tensorfields.shared.counter import Counter, CounterUpdateCallback, tally
from relflow.tensorfields.shared.jitter import Jitter
from relflow.tensorfields.shared.reconstruction import Errors, Metrics, Scores, errors

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from relflow.architecture.root import Model
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch


text: Extension = Extension(
    name="text",
    types=(str,),
    requires={"transformers": "relflow[text]", "moka_py": "relflow[text]"},
    callbacks=(CounterUpdateCallback, Metrics),
)

INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
IDENTIFIERS = "ids"
DEFAULT_TEXT_MODEL = "google/bert_uncased_L-2_H-128_A-2"


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
    revision: str | None = None
    versions: tuple | None = None

    def identity(self) -> str:
        parameters = tuple(self.model.parameters()) if callable(getattr(self.model, "parameters", None)) else ()
        buffers = tuple(self.model.buffers()) if callable(getattr(self.model, "buffers", None)) else ()
        config = self.model.config
        configuration = config.to_dict() if callable(getattr(config, "to_dict", None)) else vars(config)
        serialized = json.dumps(
            (configuration, getattr(config, "_attn_implementation", None)), sort_keys=True, default=str
        )
        versions = (serialized, *((id(p), p._version, p.dtype, tuple(p.shape)) for p in (*parameters, *buffers)))
        if self.revision is None or self.versions != versions:
            digest = hashlib.sha256()
            digest.update(serialized.encode())
            digest.update(self.key.encode())
            state = (
                (*self.model.named_parameters(), *self.model.named_buffers())
                if callable(getattr(self.model, "named_parameters", None))
                else ()
            )
            for name, value in state:
                digest.update(name.encode())
                digest.update(str((value.dtype, tuple(value.shape))).encode())
                digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
            if not state:
                digest.update(repr((type(self.model).__module__, type(self.model).__qualname__)).encode())
            self.revision = digest.hexdigest()
            self.versions = versions
        return self.revision

    @classmethod
    def get_model(
        cls,
        key: str,
    ) -> "CachedModel":
        if key not in cls._models:
            from transformers import AutoModel

            # Version counters must remain available after inference-only warm-up.
            with torch.inference_mode(False):
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
    ``"mean"``, or ``"pooler"``. Automatic ``cache=True`` sends stable IDs
    from loader workers and reuses pristine vectors through bounded device
    tables, a shared Rust frequency-aware RAM cache, and a persistent SQLite
    store. ``cache=N`` also caps hot entries; ``False`` disables all tiers.
    ``memory`` budgets shared RAM payload in MiB (default 64); ``resident``
    budgets device tables and replacement workspace (CUDA default: one eighth
    of free memory, up to 512 MiB; other devices: 0). Process overhead and OS
    file cache are additional. ``store`` selects a writable local directory,
    otherwise the platform cache directory is used. Disk entries persist
    without a size limit; only hot entries occupy managed RAM. Encoder weights,
    tokenizer configuration, pooling, device type, and precision distinguish
    entries. Checkpoints omit all runtime resources. ``jitter`` perturbs pooled
    training inputs after the pristine cache lookup.
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
    cache: pydantic.StrictBool | Annotated[pydantic.StrictInt, pydantic.Field(gt=0)] = True
    store: Annotated[str, pydantic.Field(min_length=1)] | None = None
    resident: Annotated[pydantic.StrictInt, pydantic.Field(ge=0)] | None = None
    memory: Annotated[pydantic.StrictInt, pydantic.Field(ge=0)] | None = None
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
            cache: bool | int = True,
            store: str | None = None,
            resident: int | None = None,
            memory: int | None = None,
            objective: Objective | Literal["l1", "l2"] = Objective.l2,
            jitter: Jitter = ...,
            type: Literal["text"] = "text",
            **options: Unpack[FieldOptions],
        ) -> None: ...


def encoding(request: Request) -> Encoding:
    capacity = None if isinstance(request.cache, bool) else request.cache
    return Encoding(
        request.model, request.max_length, request.tokenizer_batch_size, request.store, request.memory, capacity
    )


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
            identifiers = np.zeros((size, 1), dtype=np.int64) if request.cache else None
            encoded_ids = np.zeros((size, length), dtype=np.int64) if identifiers is None else None
            encoded_mask = np.zeros((size, length), dtype=np.int64) if identifiers is None else None
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
                if identifiers is not None:
                    storage = context.state if isinstance(context.state, Encoding) else encoding(request)
                    rows = storage.identify(unique, tokenizer)
                    identifiers[placement, 0] = rows[inverse]
                else:
                    assert encoded_ids is not None and encoded_mask is not None
                    repeated = len(unique) != len(field.values)
                    for start in range(0, len(unique), request.tokenizer_batch_size):
                        stop = min(start + request.tokenizer_batch_size, len(unique))
                        encoded = tokenizer(
                            unique.slice(start, stop - start).to_pylist(),
                            padding="max_length",
                            truncation=True,
                            max_length=length,
                            return_tensors="np",
                        )
                        selected = np.flatnonzero((inverse >= start) & (inverse < stop)) if repeated else None
                        positions = placement[selected] if repeated else placement[start:stop]
                        for name, destination in ((INPUT_IDS, encoded_ids), (ATTENTION_MASK, encoded_mask)):
                            rows = np.asarray(encoded[name], dtype=np.int64)
                            if rows.shape != (stop - start, length):
                                raise ValueError(
                                    f"text tokenizer '{request.model}' returned {name} shape {rows.shape}; expected {(stop - start, length)}"
                                )
                            destination[positions] = rows[inverse[selected] - start] if repeated else rows
            if identifiers is not None:
                content = {IDENTIFIERS: torch.from_numpy(identifiers.reshape(*field.shape, 1))}
            else:
                assert encoded_ids is not None and encoded_mask is not None
                content = {
                    INPUT_IDS: torch.from_numpy(encoded_ids.reshape(*field.shape, length)),
                    ATTENTION_MASK: torch.from_numpy(encoded_mask.reshape(*field.shape, length)),
                }
            return TensorDict(cast(Any, content), batch_size=field.shape)

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
        self.store = encoding(request) if request.cache else None
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
    def context(self) -> Encoding | None:
        return self.store

    @torch.compiler.disable
    def variant(self, device: torch.device) -> str:
        dtype = torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type) else None
        hardware = torch.cuda.get_device_capability(device) if device.type == "cuda" else None
        payload = (
            cast(CachedModel, self.__dict__["_cached_model"]).identity(),
            self.request.encoder_pooling.value,
            str(dtype),
            device.type,
            hardware,
            torch.get_float32_matmul_precision(),
            torch.__version__,
        )
        return hashlib.sha256(repr(payload).encode()).hexdigest()

    def warm(self, values: Sequence[str] | pa.Array | pa.ChunkedArray) -> int:
        """Populate the automatic cache using the current device and autocast precision."""
        if self.store is None:
            raise ValueError(f"text field at '{self.origin}' has cache=False; enable caching before warming")
        texts = values.to_pylist() if isinstance(values, (pa.Array, pa.ChunkedArray)) else list(values)
        texts = list(dict.fromkeys(text for text in texts if text is not None))
        before = self.store.stats()["results"]
        device = self.anchor.device
        tokenizer = CachedModel.get_tokenizer(self.request.model)
        for start in range(0, len(texts), self.request.tokenizer_batch_size):
            ids = self.store.identify(texts[start : start + self.request.tokenizer_batch_size], tokenizer)
            self.store.resolve(
                torch.from_numpy(ids).to(device),
                variant=self.variant(device),
                width=self.hidden_size,
                resident=self.request.resident,
                pool=self.pool,
            )
        return self.store.stats()["results"] - before

    @property
    def padding_side(self) -> str:
        return getattr(CachedModel.get_tokenizer(self.request.model), "padding_side", "right")

    @beartype
    def encode(self, content: TensorDict, state: torch.Tensor) -> torch.Tensor:
        shape = tuple(state.shape)
        size = state.numel()
        device = state.device
        embeddings = torch.zeros((size, self.hidden_size), device=device, dtype=torch.float32)
        valued = state.reshape(-1).eq(Tokens.valued.value)
        if not valued.any():
            return embeddings.reshape(*shape, self.hidden_size)
        if IDENTIFIERS in content.keys():
            if self.store is None:
                raise ValueError(
                    f"text field at '{self.origin}' received cache IDs with cache=False; encode its observations again"
                )
            ids = cast(torch.Tensor, content[IDENTIFIERS]).reshape(size)[valued]
            embeddings[valued] = self.store.resolve(
                ids,
                variant=self.variant(device),
                width=self.hidden_size,
                resident=self.request.resident,
                pool=self.pool,
            )
        else:
            ids = cast(torch.Tensor, content[INPUT_IDS])
            mask = cast(torch.Tensor, content[ATTENTION_MASK])
            embeddings[valued] = self.pool(
                ids.reshape(size, ids.shape[-1])[valued], mask.reshape(size, mask.shape[-1])[valued]
            )
        return embeddings.reshape(*shape, self.hidden_size)

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
