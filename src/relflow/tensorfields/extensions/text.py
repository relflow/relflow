from __future__ import annotations

import enum
import math
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import InitVar, dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self, Unpack, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from relflow.data.ragged import RaggedField
from relflow.structs.enums import Component, TensorKey, Tokens
from relflow.structs.tree import Address, FieldOptions
from relflow.tensorfields.base import (
    Context,
    Extension,
    RequestBase,
    TensorFieldBase,
    masks,
    resolve,
)
from relflow.tensorfields.extensions.text_cache.context import Encoding
from relflow.tensorfields.extensions.text_cache.tokens import token_rows
from relflow.tensorfields.shared.cache import Payloads
from relflow.tensorfields.shared.counter import CounterUpdateCallback
from relflow.tensorfields.shared.frozen import Backbone, Objective, learn, loss, observe
from relflow.tensorfields.shared.frozen import Decoder as FrozenDecoder
from relflow.tensorfields.shared.frozen import Embedder as FrozenEmbedder
from relflow.tensorfields.shared.jitter import Jitter
from relflow.tensorfields.shared.reconstruction import Metrics

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from relflow.data.ragged import Sources
    from relflow.structs.experiment import Schema


text: Extension = Extension(
    name="text",
    types=(str,),
    requires={"transformers": "relflow[text]", "moka_py": "relflow[text]"},
    callbacks=(CounterUpdateCallback, Metrics),
)

INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
IDENTIFIERS = "ids"
DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Pooling(enum.StrEnum):
    cls = "cls"
    mean = "mean"
    pooler = "pooler"


@dataclass
class CachedModel(Backbone):
    # Text owns its CPU execution policy even when another modality uses the same model ID.
    _models: ClassVar[dict[str, Backbone]] = {}
    cpu_model: PreTrainedModel | None = None
    cpu_revision: str | None = None

    _tokenizers: ClassVar[dict[str, PreTrainedTokenizerBase]] = {}

    def module_for(self, device: torch.device) -> PreTrainedModel:
        model = super().module_for(device)
        if device.type != "cpu":
            return model
        if all(not t.is_floating_point() or t.dtype == torch.float32 for t in (*model.parameters(), *model.buffers())):
            return model
        revision = self.identity()
        if self.cpu_revision != revision:
            # Keep the checkpoint weights available at their native precision for accelerators.
            with torch.inference_mode(False):
                self.cpu_model = deepcopy(model).float()
            self.cpu_revision = revision
        assert self.cpu_model is not None
        return self.cpu_model

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


@text.register(Component.Request)
class Request(RequestBase):
    """String input represented by a frozen Hugging Face encoder.

    ``model`` defaults to all-MiniLM-L6-v2 with masked mean pooling. Custom
    encoders retain CLS pooling unless configured explicitly. ``encoder_rms``
    scales each pooled input to unit RMS before jitter and projection.
    CPU encoder inference uses float32. ``max_length`` limits
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
    encoder_pooling: Pooling = pydantic.Field(
        default_factory=lambda data: Pooling.mean if data.get("model") == DEFAULT_TEXT_MODEL else Pooling.cls
    )
    encoder_rms: pydantic.StrictBool = True
    cache: pydantic.StrictBool | Annotated[pydantic.StrictInt, pydantic.Field(gt=0)] = True
    store: Annotated[str, pydantic.Field(min_length=1)] | None = None
    resident: Annotated[pydantic.StrictInt, pydantic.Field(ge=0)] | None = None
    memory: Annotated[pydantic.StrictInt, pydantic.Field(ge=0)] | None = None
    objective: Objective = Objective.l2
    jitter: Jitter = pydantic.Field(default_factory=Jitter)

    @pydantic.model_validator(mode="before")
    @classmethod
    def check_checkpoint_scaling(cls, data: Any, info: pydantic.ValidationInfo) -> Any:
        if isinstance(info.context, dict) and info.context.get("checkpoint") and isinstance(data, dict):
            return {"encoder_rms": False, **data}
        return data

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            model: str = DEFAULT_TEXT_MODEL,
            max_length: int = 128,
            tokenizer_batch_size: int = 4096,
            encoder_batch_size: int = 32,
            encoder_pooling: Pooling | Literal["cls", "mean", "pooler"] = ...,
            encoder_rms: bool = True,
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


text.register(Component.observe, observe)


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
        sources: Sources,
        *,
        schema: Schema,
        context: Context,
    ) -> tuple[Self, TensorDict | None, tuple[object, ...]]:
        address = sources.address
        projection, context, observation, binding = resolve(sources, schema=schema, context=context)
        input, target = projection.split()
        present, trainable, inferred = masks(projection)
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

        result = cls(
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
        return result, observation, binding


@text.register(Component.Embedder)
class Embedder(FrozenEmbedder):
    _version = 2  # PyTorch module metadata. Version 1 projected raw pooled vectors.
    backbone = CachedModel

    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)
        self.request = cast(Request, schema.requests[address])
        if self.request.encoder_rms:
            self.input_norm = torch.nn.RMSNorm(self.hidden_size, eps=1e-12, elementwise_affine=False)
        self.store = encoding(self.request) if self.request.cache else None

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Lightning can restore tensors into an already constructed, shape-compatible graph.
        if local_metadata.get("version") == 1 and self.request.encoder_rms:
            self.request.encoder_rms = False
            self.input_norm = torch.nn.Identity()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @property
    def context(self) -> Encoding | None:
        return self.store

    def variant(self, device: torch.device) -> str:
        pooling = self.request.encoder_pooling.value
        if device.type == "cpu":
            with torch.autocast("cpu", enabled=False):
                return self._cached_model.variant(device, f"{pooling}:cpu-fp32")
        return self._cached_model.variant(device, pooling)

    def cached(self, inputs: Payloads, device: torch.device) -> torch.Tensor:
        rows = token_rows(inputs, self.request.max_length)
        return self.pool(torch.from_numpy(rows[:, 0]).to(device), torch.from_numpy(rows[:, 1]).to(device))

    def warm(self, values: Sequence[str] | pa.Array | pa.ChunkedArray) -> int:
        """Populate the automatic cache using the execution precision for the current device."""
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
                pool=self.cached,
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
                pool=self.cached,
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
        model = self._cached_model.module_for(valued_ids.device)

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

            with torch.inference_mode(), torch.autocast("cpu", enabled=False):
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


text.register(Component.learn, learn)
text.register(Component.loss, loss)


@text.register(Component.Decoder)
class Decoder(FrozenDecoder):
    coordinatewise = True
    backbone = CachedModel
