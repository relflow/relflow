"""Audio references, concurrent decoding, resampling, and frozen audio-encoder embeddings."""

from __future__ import annotations

import enum
import hashlib
import importlib.util
import io
import json
import os
import platform
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import InitVar
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, TypeAlias, Unpack, cast
from urllib.parse import unquote

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from pyarrow import fs
from tensordict import TensorDict, tensorclass

from relflow.data.ragged import RaggedField, Sources
from relflow.data.sources import resolve as filesystem
from relflow.structs.enums import Component, TensorKey, Tokens
from relflow.structs.tree import Address, FieldOptions
from relflow.tensorfields.base import Context, Extension, RequestBase, TensorFieldBase, masks, resolve
from relflow.tensorfields.shared.cache import MAX_FRAME, MAX_ROWS, Payloads
from relflow.tensorfields.shared.cache.vectors import VectorCache
from relflow.tensorfields.shared.counter import CounterUpdateCallback
from relflow.tensorfields.shared.frozen import Backbone, Objective, learn, loss, observe
from relflow.tensorfields.shared.frozen import Decoder as FrozenDecoder
from relflow.tensorfields.shared.frozen import Embedder as FrozenEmbedder
from relflow.tensorfields.shared.jitter import Jitter
from relflow.tensorfields.shared.reconstruction import Metrics

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from relflow.structs.experiment import Schema

DEFAULT_MODEL = "openai/whisper-base"
DEFAULT_WINDOW = 10.0

audio = Extension(
    name="audio",
    types=(str,),
    requires={
        "soundfile": "relflow[audio]",
        "soxr": "relflow[audio]",
        "transformers": "relflow[audio]",
        "moka_py": "relflow[audio]",
    },
    callbacks=(CounterUpdateCallback, Metrics),
)


Clip: TypeAlias = Literal["chunks", "head", "center", "tail"]


class Pooling(enum.StrEnum):
    auto = "auto"
    mean = "mean"
    cls = "cls"
    pooler = "pooler"


class CachedModel(Backbone):
    @classmethod
    def get_model(cls, key: str) -> Backbone:
        if key not in cls._models:
            from transformers import AutoConfig, AutoModel

            # Version counters must remain available after inference-only warm-up.
            with torch.inference_mode(False):
                config = AutoConfig.from_pretrained(key)
                kind = getattr(config, "model_type", None)
                if kind in {"clap", "clap_audio_model"}:
                    from transformers import ClapAudioModelWithProjection

                    loaded: Any = ClapAudioModelWithProjection.from_pretrained(key).audio_model
                else:
                    loaded = AutoModel.from_pretrained(key)
                    if getattr(config, "is_encoder_decoder", False) and callable(getattr(loaded, "get_encoder", None)):
                        loaded = loaded.get_encoder()
                model = cast("PreTrainedModel", loaded)
            model.eval()
            model.requires_grad_(False)
            hidden_size = getattr(model.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError(f"encoder '{key}' does not expose `config.hidden_size`")
            cls._models[key] = cls(key=key, model=model, hidden_size=int(hidden_size))
        return cls._models[key]


@audio.register(Component.Request)
class Request(RequestBase):
    """Use a local audio path or an object-store URI as context for prediction.

    The frozen audio encoder supplies one vector per clip. Decoding, channel
    selection, and high-quality resampling run in a bounded thread pool;
    encoder_batch_size limits each model call. Recordings longer than window use
    clip="chunks" by default, averaging up to max_windows overlapping windows
    into one deterministic vector. clip="head", "center", and "tail" read a
    single window. A reconstructing mask targets the pristine vector.
    embed=True exports the field's learned representation.

    Relative paths resolve against base (the working directory by default).
    Set immutable=True only for paths that always identify the same bytes,
    such as content-addressed object names. Warm hits then skip source storage.
    Mutable local paths use file identity and timestamps; mutable remote paths
    use a hash of the object bytes. store, memory, resident, and cache follow
    the Text and Image cache contract. Encoder assets and caches live outside
    checkpoints.
    """

    model_config = pydantic.ConfigDict(str_strip_whitespace=True, allow_inf_nan=False)
    type: Literal["audio"] = "audio"
    model: Annotated[str, pydantic.Field(min_length=1)] = DEFAULT_MODEL
    base: Annotated[str, pydantic.Field(min_length=1)] | None = None
    immutable: pydantic.StrictBool = False
    offset: Annotated[float, pydantic.Field(ge=0.0)] = 0.0
    duration: Annotated[float, pydantic.Field(gt=0.0)] | None = None
    window: Annotated[float, pydantic.Field(gt=0.0)] | None = None
    stride: Annotated[float, pydantic.Field(gt=0.0)] | None = None
    max_windows: Annotated[pydantic.StrictInt, pydantic.Field(gt=0)] = 4
    clip: Clip = "chunks"
    channels: Literal["mean"] | Annotated[pydantic.StrictInt, pydantic.Field(ge=0)] = "mean"
    encoder_pooling: Pooling = Pooling.auto
    read_workers: Annotated[pydantic.StrictInt, pydantic.Field(gt=0)] = 4
    encoder_batch_size: Annotated[pydantic.StrictInt, pydantic.Field(gt=0)] = 16
    cache: pydantic.StrictBool | Annotated[pydantic.StrictInt, pydantic.Field(gt=0)] = True
    store: Annotated[str, pydantic.Field(min_length=1)] | None = None
    memory: Annotated[pydantic.StrictInt, pydantic.Field(ge=0)] | None = None
    resident: Annotated[pydantic.StrictInt, pydantic.Field(ge=0)] | None = None
    objective: Objective = Objective.l2
    jitter: Jitter = pydantic.Field(default_factory=Jitter)

    @pydantic.field_validator("offset", "duration", "window", "stride", mode="before")
    @classmethod
    def check_seconds(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("audio time bounds must be numbers, not booleans")
        return value

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            model: str = DEFAULT_MODEL,
            base: str | None = None,
            immutable: bool = False,
            offset: float = 0.0,
            duration: float | None = None,
            window: float | None = None,
            stride: float | None = None,
            max_windows: int = 4,
            clip: Clip = "chunks",
            channels: Literal["mean"] | int = "mean",
            encoder_pooling: Pooling | Literal["auto", "mean", "cls", "pooler"] = Pooling.auto,
            read_workers: int = 4,
            encoder_batch_size: int = 16,
            cache: bool | int = True,
            store: str | None = None,
            memory: int | None = None,
            resident: int | None = None,
            objective: Objective | Literal["l1", "l2"] = Objective.l2,
            jitter: Jitter = ...,
            type: Literal["audio"] = "audio",
            **options: Unpack[FieldOptions],
        ) -> None: ...


def location(value: str, base: str | None) -> str:
    if not value or "\0" in value:
        raise ValueError(f"expected a nonempty audio path without NUL, got {value!r}")
    if base and "://" not in value and not os.path.isabs(value):
        value = base.removesuffix("/") + "/" + value if "://" in base else os.path.join(base, value)
    if value.startswith("file://"):
        _, value = filesystem(value, None)
    return value if "://" in value else os.path.abspath(value)


@lru_cache(maxsize=32)
def remote(authority: str) -> tuple[fs.FileSystem, str]:
    """Reuse the Arrow storage client across objects in the same bucket."""
    return filesystem(authority, None)


def open_audio(path: str):
    if "://" not in path:
        return open(path, "rb")
    scheme, rest = path.split("://", 1)
    authority, _, key = rest.partition("/")
    storage, prefix = remote(f"{scheme}://{authority}")
    name = f"{prefix.rstrip('/')}/{unquote(key)}" if prefix else unquote(key)
    # Audio decoders seek through container headers and frame tables. Download
    # the object sequentially once, then let the codec seek in memory.
    with storage.open_input_stream(name, compression=None) as stream:
        return io.BytesIO(stream.read())


def stamp(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def snapshot(path: str, immutable: bool) -> tuple[bytes, bytes | None]:
    """Keep a mutable remote object's bytes for this microbatch to avoid a second read."""
    body = None
    version: tuple[int, ...] | str | None = None
    if not immutable:
        if "://" in path:
            with open_audio(path) as stream:
                body = stream.read()
            version = hashlib.sha256(body).hexdigest()
        elif os.name == "nt":
            # Windows can report different metadata by path and by handle after a replacement.
            with open(path, "rb") as stream:
                version = stamp(os.fstat(stream.fileno()))
        else:
            version = stamp(os.stat(path))
    return json.dumps((path, version), ensure_ascii=False, separators=(",", ":")).encode(), body


def native_window(extractor: Any) -> float:
    chunk = getattr(extractor, "chunk_length", None)
    if isinstance(chunk, (int, float)) and chunk > 0:
        return float(chunk)
    maximum = getattr(extractor, "max_length_s", None)
    if isinstance(maximum, (int, float)) and maximum > 0:
        return float(maximum)
    frames = getattr(extractor, "max_length", None)
    if type(extractor).__name__ == "ASTFeatureExtractor" and isinstance(frames, int) and frames > 0:
        return float(frames) * 0.01
    return DEFAULT_WINDOW


def spans(
    total: int,
    rate: int,
    *,
    offset: float,
    duration: float | None,
    window: float,
    stride: float | None,
    max_windows: int,
    clip: Clip,
) -> list[tuple[int, int]]:
    """Select deterministic sample intervals before reading full audio streams."""
    if rate <= 0 or total <= 0:
        raise ValueError("audio source contains no readable frames")
    start = round(offset * rate)
    stop = total if duration is None else min(total, start + max(1, round(duration * rate)))
    if start >= stop:
        raise ValueError(f"audio clip has no samples after offset={offset:g}s (source duration {total / rate:.3f}s)")
    length = stop - start
    width = max(1, round(window * rate))
    if length <= width:
        return [(start, stop)]
    if clip == "head":
        return [(start, start + width)]
    if clip == "center" or max_windows == 1:
        left = start + (length - width) // 2
        return [(left, left + width)]
    if clip == "tail":
        return [(stop - width, stop)]
    hop = max(1, round((window / 2.0 if stride is None else stride) * rate))
    last = stop - width
    starts = list(range(start, last + 1, hop))
    if starts[-1] != last:
        starts.append(last)
    if len(starts) > max_windows:
        starts = np.rint(np.linspace(start, last, max_windows)).astype(np.int64).tolist()
    return [(int(left), int(left) + width) for left in starts]


def waveforms(
    record: bytes,
    bodies: dict[bytes, bytes],
    *,
    target_rate: int,
    offset: float,
    duration: float | None,
    window: float,
    stride: float | None,
    max_windows: int,
    clip: Clip,
    channels: Literal["mean"] | int,
) -> list[np.ndarray]:
    import soundfile as sf
    import soxr

    path, version = json.loads(record)
    try:
        stream = io.BytesIO(bodies[record]) if isinstance(version, str) else open_audio(path)
        with stream:
            if isinstance(version, list) and list(stamp(os.fstat(stream.fileno()))) != version:
                raise ValueError(f"audio {path!r} changed after tensorization; encode the observations again")
            with sf.SoundFile(stream) as source:
                rate = int(source.samplerate)
                count = int(source.channels)
                if isinstance(channels, int) and channels >= count:
                    raise ValueError(f"audio {path!r} has {count} channel(s); cannot select channel index {channels}")
                if source.seekable() and 0 < source.frames < 2**60:
                    intervals = spans(
                        int(source.frames),
                        rate,
                        offset=offset,
                        duration=duration,
                        window=window,
                        stride=stride,
                        max_windows=max_windows,
                        clip=clip,
                    )
                    pieces = []
                    for left, right in intervals:
                        source.seek(left)
                        block = source.read(right - left, dtype="float32", always_2d=True)
                        if len(block):
                            pieces.append(block)
                else:
                    full = source.read(dtype="float32", always_2d=True)
                    intervals = spans(
                        len(full),
                        rate,
                        offset=offset,
                        duration=duration,
                        window=window,
                        stride=stride,
                        max_windows=max_windows,
                        clip=clip,
                    )
                    pieces = [full[left:right] for left, right in intervals if right > left]
            if isinstance(version, list) and list(stamp(os.fstat(stream.fileno()))) != version:
                raise ValueError(f"audio {path!r} changed during decoding; encode the observations again")
        if not pieces:
            raise ValueError(f"audio {path!r} yielded no decoded samples")
        target_width = max(1, round(window * target_rate))
        prepared: list[np.ndarray] = []
        for block in pieces:
            if channels == "mean":
                mono = (
                    np.ascontiguousarray(block[:, 0], dtype=np.float32)
                    if block.shape[1] == 1
                    else block.mean(axis=1, dtype=np.float32)
                )
            else:
                mono = np.ascontiguousarray(block[:, channels], dtype=np.float32)
            if not np.isfinite(mono).all():
                raise ValueError(f"audio {path!r} contains non-finite samples")
            if rate != target_rate:
                mono = np.asarray(soxr.resample(mono, rate, target_rate, quality="HQ"), dtype=np.float32)
                if len(mono) > target_width:
                    mono = mono[:target_width]
            if not len(mono):
                raise ValueError(f"audio {path!r} resampled to zero samples at {target_rate} Hz")
            prepared.append(mono)
        return prepared
    except (OSError, RuntimeError, ValueError) as error:
        error.add_note(f"While reading or preparing audio {path!r}.")
        raise


class Store(VectorCache):
    """Only the locator and its version enter the artifact store as prepared data."""

    vector_namespace = "rf.audio.vec.f32/v1"
    input_bytes = 0  # Source descriptors have variable width.

    def identify(self, records: list[bytes]) -> np.ndarray:
        namespace = "rf.audio.source/v1"
        ids = np.empty(len(records), dtype=np.int64)
        longest = max(map(len, records), default=0)
        chunk = min(MAX_ROWS, (MAX_FRAME - 65536) // (longest + 64))
        if not chunk:
            raise ValueError("audio reference exceeds the artifact cache frame budget")
        for start in range(0, len(records), chunk):
            batch = records[start : start + chunk]
            sources = pa.array(batch, type=pa.binary())
            refs = self.client.lookup_arrow(sources, namespace)
            missing = np.flatnonzero(~refs.ready)
            if len(missing):
                refs.objects[missing] = self.client.bind(
                    refs.take(missing), Payloads.from_values(batch[i] for i in missing), namespace
                )
            ids[start : start + len(batch)] = refs.objects
        return ids


def store(request: Request) -> Store:
    capacity = None if isinstance(request.cache, bool) else request.cache
    return Store(request.store, request.memory, capacity)


@audio.register(Component.TensorField)
@tensorclass
class TensorField(TensorFieldBase[TensorDict]):
    content: TensorDict
    state: torch.Tensor
    present: torch.Tensor
    trainable: torch.Tensor
    inferred: torch.Tensor
    targets: TensorDict

    if TYPE_CHECKING:
        batch_size: InitVar[int | torch.Size | list[int] | tuple[int, ...] | None] = None
        device: InitVar[torch.device | str | int | None] = None
        names: InitVar[list[str | None] | None] = None

    @classmethod
    def new(
        cls, sources: Sources, *, schema: Schema, context: Context
    ) -> tuple[Self, TensorDict | None, tuple[object, ...]]:
        address = sources.address
        projection, context, observation, binding = resolve(sources, schema=schema, context=context)
        inputs, targets = projection.split()
        present, trainable, inferred = masks(projection)
        request = cast(Request, schema.requests[address])

        def encode(field: RaggedField) -> TensorDict:
            if not len(field.values):
                key, dtype = ("ids", torch.int64) if request.cache else ("paths", torch.uint8)
                return TensorDict({key: torch.zeros((*field.shape, 1), dtype=dtype)}, batch_size=field.shape)
            # Unify independent Arrow dictionaries before deduplicating across chunks.
            strings = field.values.cast(pa.large_string())
            if isinstance(strings, pa.ChunkedArray):
                strings = strings.combine_chunks()
            values = cast(pa.DictionaryArray, pc.dictionary_encode(strings))
            paths = [location(path, request.base) for path in cast(list[str], values.dictionary.to_pylist())]
            inverse = values.indices.to_numpy(zero_copy_only=False)
            if request.cache and (request.immutable or all("://" not in path for path in paths)):
                cache = context.state if isinstance(context.state, Store) else store(request)
                records = [snapshot(path, request.immutable)[0] for path in paths]
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
            error.add_note(f"While tensorizing audio field {address!s}.")
            raise


def pool_outputs(
    outputs: Any,
    model: Any,
    attention_mask: torch.Tensor | None,
    pooling: Pooling,
    hidden_size: int,
    model_name: str,
) -> torch.Tensor:
    kind = getattr(getattr(model, "config", None), "model_type", None)
    if pooling == Pooling.pooler or (
        pooling == Pooling.auto
        and getattr(outputs, "pooler_output", None) is not None
        and kind in {"audio-spectrogram-transformer", "clap", "clap_audio_model"}
    ):
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None or pooled.ndim != 2 or pooled.shape[1] != hidden_size:
            raise ValueError(f"audio encoder {model_name!r} must expose pooler_output of width {hidden_size}")
        return pooled.float().contiguous()

    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is not None and hidden.ndim == 4 and pooling in {Pooling.auto, Pooling.mean}:
        if hidden.shape[1] != hidden_size or hidden.shape[2] == 0 or hidden.shape[3] == 0:
            raise ValueError(f"audio encoder {model_name!r} spatial output must have {hidden_size} channels")
        return hidden.mean(dim=(-2, -1)).float().contiguous()
    if hidden is None or hidden.ndim != 3 or hidden.shape[1] == 0 or hidden.shape[2] != hidden_size:
        raise ValueError(f"audio encoder {model_name!r} must return [batch, frames, {hidden_size}] hidden states")
    if pooling == Pooling.cls:
        return hidden[:, 0].float().contiguous()
    if attention_mask is None:
        return hidden.mean(dim=1).float().contiguous()

    frames = hidden.shape[1]
    reduce_mask = getattr(model, "_get_feature_vector_attention_mask", None)
    if callable(reduce_mask) and attention_mask.shape[1] != frames:
        valid = cast(torch.Tensor, reduce_mask(frames, attention_mask)).bool()
    elif attention_mask.shape[1] == frames:
        valid = attention_mask.bool()
    else:
        scale = attention_mask.shape[1] / frames
        counts = (attention_mask.sum(dim=-1).float() / scale).ceil().long().clamp(min=1, max=frames)
        valid = torch.arange(frames, device=hidden.device).unsqueeze(0) < counts.unsqueeze(1)
    empty = ~valid.any(dim=1)
    if empty.any():
        valid = valid.clone()
        valid[empty, 0] = True
    weights = valid.unsqueeze(-1).to(dtype=hidden.dtype)
    return ((hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)).float().contiguous()


def minimum_samples(config: Any, rate: int) -> int:
    receptive = 1
    kernels = getattr(config, "conv_kernel", ()) or ()
    strides = getattr(config, "conv_stride", ()) or ()
    for kernel, step in reversed(list(zip(kernels, strides, strict=False))):
        receptive = (receptive - 1) * int(step) + int(kernel)
    return max(receptive, round(0.025 * rate), 1)


def encode_clips(
    model: Any,
    extractor: Any,
    clips: list[np.ndarray],
    device: torch.device,
    *,
    rate: int,
    pooling: Pooling,
    hidden_size: int,
    model_name: str,
) -> torch.Tensor:
    if not clips:
        return torch.empty((0, hidden_size), dtype=torch.float32, device=device)
    names = tuple(getattr(extractor, "model_input_names", ("input_values",)))
    config = getattr(model, "config", None)
    floor = minimum_samples(config, rate)
    clips = [clip if len(clip) >= floor else np.pad(clip, (0, floor - len(clip))) for clip in clips]
    kind = getattr(config, "model_type", None)
    group_norm = getattr(config, "feat_extract_norm", None) == "group" and "input_values" in names

    # Group-normalized waveform stems are sensitive to zero-padding across unequal clip lengths.
    if group_norm and len({len(clip) for clip in clips}) > 1:
        buckets: dict[int, list[int]] = {}
        for index, clip in enumerate(clips):
            buckets.setdefault(len(clip), []).append(index)
        combined = torch.empty((len(clips), hidden_size), dtype=torch.float32, device=device)
        for indices in buckets.values():
            combined[indices] = encode_clips(
                model,
                extractor,
                [clips[i] for i in indices],
                device,
                rate=rate,
                pooling=pooling,
                hidden_size=hidden_size,
                model_name=model_name,
            )
        return combined

    kwargs: dict[str, Any] = {"sampling_rate": rate, "return_tensors": "np"}
    if "is_longer" in names:
        if not getattr(config, "enable_fusion", False):
            limit = getattr(extractor, "nb_max_samples", None)
            if isinstance(limit, int) and limit > 0:
                clips = [clip[:limit] for clip in clips]
    elif "input_features" in names:
        kwargs["return_attention_mask"] = True
    elif "attention_mask" in names:
        kwargs["padding"] = True
        kwargs["return_attention_mask"] = True

    with torch.autocast(device_type="cpu", enabled=False), torch.autocast(device_type=device.type, enabled=False):
        prepared = extractor(clips, **kwargs)
    mask_array = prepared.get("attention_mask")
    attention_mask = torch.as_tensor(np.asarray(mask_array), device=device) if mask_array is not None else None
    forward_inputs: dict[str, torch.Tensor] = {}
    for key in names:
        if key not in prepared or key == "attention_mask":
            continue
        forward_inputs[key] = torch.as_tensor(np.asarray(prepared[key]), device=device)
    if attention_mask is not None and "attention_mask" in names and not group_norm and kind != "whisper":
        forward_inputs["attention_mask"] = attention_mask
    if not forward_inputs:
        raise ValueError(f"audio feature extractor for {model_name!r} produced none of {names!r}")

    with torch.no_grad():
        outputs = model(**forward_inputs)
    pooled = pool_outputs(outputs, model, attention_mask, pooling, hidden_size, model_name)
    if pooled.shape != (len(clips), hidden_size):
        raise ValueError(
            f"audio encoder {model_name!r} returned shape {tuple(pooled.shape)}; expected {(len(clips), hidden_size)}"
        )
    return pooled


@audio.register(Component.Embedder)
class Embedder(FrozenEmbedder):
    backbone = CachedModel

    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)
        from transformers import AutoFeatureExtractor

        self.request = cast(Request, schema.requests[address])
        self.extractor = AutoFeatureExtractor.from_pretrained(self.request.model)
        self.store = store(self.request) if self.request.cache else None

    @property
    def context(self) -> Store | None:
        return self.store

    @property
    def sampling_rate(self) -> int:
        rate = getattr(self.extractor, "sampling_rate", 16000)
        if not isinstance(rate, int) or rate <= 0:
            raise ValueError(f"audio feature extractor for {self.request.model!r} must expose a positive sampling_rate")
        return rate

    @property
    def window_seconds(self) -> float:
        return native_window(self.extractor) if self.request.window is None else self.request.window

    def variant(self, device: torch.device) -> str:
        import soundfile as sf
        import soxr
        from transformers import __version__ as transformers

        serialize = getattr(self.extractor, "to_json_string", None)
        extractor_state = serialize() if callable(serialize) else repr(vars(self.extractor))
        settings = (
            type(self.extractor).__qualname__,
            extractor_state,
            self.sampling_rate,
            self.request.offset,
            self.request.duration,
            self.window_seconds,
            self.request.stride,
            self.request.max_windows,
            self.request.clip,
            self.request.channels,
            self.request.encoder_pooling.value,
            sf.__version__,
            getattr(sf, "__libsndfile_version__", None),
            soxr.__version__,
            getattr(soxr, "__libsoxr_version__", None),
            importlib.util.find_spec("torchaudio") is not None,
            sys.platform,
            platform.machine(),
            transformers,
        )
        return self._cached_model.variant(device, repr(settings))

    def pool(
        self, records: Sequence[bytes] | Payloads, device: torch.device, bodies: dict[bytes, bytes]
    ) -> torch.Tensor:
        """Decode and resample the next bounded batch on CPU while encoding the current batch."""
        if isinstance(records, Payloads):
            records = [bytes(row) for row in records]
        if not records:
            return torch.empty((0, self.hidden_size), dtype=torch.float32, device=device)
        model = self._cached_model.module_for(device)
        size = self.request.encoder_batch_size
        rate = self.sampling_rate
        starts = iter(range(0, len(records), size))
        encoded: list[torch.Tensor] = []
        prepare = partial(
            waveforms,
            bodies=bodies,
            target_rate=rate,
            offset=self.request.offset,
            duration=self.request.duration,
            window=self.window_seconds,
            stride=self.request.stride,
            max_windows=self.request.max_windows,
            clip=self.request.clip,
            channels=self.request.channels,
        )
        try:
            with ThreadPoolExecutor(max_workers=self.request.read_workers) as executor:

                def submit(start: int):
                    return [executor.submit(prepare, row) for row in records[start : start + size]]

                pending = submit(next(starts))
                while pending:
                    decoded = [job.result() for job in pending]
                    following = next(starts, None)
                    pending = submit(following) if following is not None else []

                    flat = [window for windows in decoded for window in windows]
                    counts = [len(windows) for windows in decoded]
                    window_vectors = torch.cat(
                        [
                            encode_clips(
                                model,
                                self.extractor,
                                flat[offset : offset + size],
                                device,
                                rate=rate,
                                pooling=self.request.encoder_pooling,
                                hidden_size=self.hidden_size,
                                model_name=self.request.model,
                            )
                            for offset in range(0, len(flat), size)
                        ],
                        dim=0,
                    )
                    if all(count == 1 for count in counts):
                        encoded.append(window_vectors)
                    else:
                        encoded.append(
                            torch.stack([chunk.mean(dim=0) for chunk in window_vectors.split(counts, dim=0)], dim=0)
                        )
        except (OSError, RuntimeError, ValueError) as error:
            error.add_note(f"While encoding audio field {self.address!s}.")
            raise
        return torch.cat(encoded, dim=0)

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
        if self.store is None:
            return self.pool([snapshot(path, True)[0] for path in paths], device, {})
        if self.request.immutable or all("://" not in path for path in paths):
            records = [snapshot(path, self.request.immutable)[0] for path in paths]
            return self.lookup(torch.from_numpy(self.store.identify(records)).to(device), {})

        size = self.request.encoder_batch_size
        vectors = []
        with ThreadPoolExecutor(max_workers=self.request.read_workers) as executor:
            for start in range(0, len(paths), size):
                batch = paths[start : start + size]
                snapshots = list(executor.map(partial(snapshot, immutable=False), batch))
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
                        f"audio field at '{self.address}' received cache IDs with cache=False; encode its observations again"
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

    def warm(self, values: Sequence[str | None] | pa.Array | pa.ChunkedArray) -> int:
        """Populate the local vector cache at the current device and precision."""
        if self.store is None:
            raise ValueError(f"audio field at {self.address!s} requires cache=True to warm")
        if isinstance(values, str):
            values = (values,)
        before = self.store.stats()["results"]
        for start in range(0, len(values), MAX_ROWS):
            batch = values[start : start + MAX_ROWS]
            paths = batch.to_pylist() if isinstance(batch, (pa.Array, pa.ChunkedArray)) else batch
            unique = list(dict.fromkeys(location(path, self.request.base) for path in paths if path is not None))
            if unique:
                self.paths(unique, self.anchor.device)
        return self.store.stats()["results"] - before


@audio.register(Component.Decoder)
class Decoder(FrozenDecoder):
    coordinatewise = True
    backbone = CachedModel


audio.register(Component.observe, observe)
audio.register(Component.learn, learn)
audio.register(Component.loss, loss)
