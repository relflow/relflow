"""Image references, concurrent decoding, and frozen vision-transformer embeddings."""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import InitVar, dataclass
from functools import lru_cache, partial
from typing import IO, TYPE_CHECKING, Annotated, Any, Literal, Self, Unpack, cast
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
from relflow.tensorfields.shared.frozen import Decoder, Objective, learn, loss, observe
from relflow.tensorfields.shared.frozen import Embedder as FrozenEmbedder
from relflow.tensorfields.shared.jitter import Jitter
from relflow.tensorfields.shared.reconstruction import Metrics

if TYPE_CHECKING:
    from PIL.Image import Image as Raster

    from relflow.structs.experiment import Schema

DEFAULT_MODEL = "facebook/dinov2-small"
image = Extension(
    name="image",
    types=(str,),
    requires={"PIL": "relflow[image]", "transformers": "relflow[image]", "moka_py": "relflow[image]"},
    callbacks=(CounterUpdateCallback, Metrics),
)


@image.register(Component.Request)
class Request(RequestBase):
    """Use a local image path or an object-store URI as context for prediction.

    The frozen vision transformer supplies one CLS vector for each image. Its
    processor controls resizing and normalization. Reads and RGB preparation
    run in a bounded thread pool; encoder_batch_size limits each model call.
    jpeg_short_edge opts into decoder-side JPEG reduction before preprocessing.
    A reconstructing mask targets the pristine vector. embed=True exports the
    field's learned representation.

    Relative paths resolve against base (the working directory by default).
    Set immutable=True only for paths that always identify the same bytes,
    such as content-addressed object names. Warm hits then skip source storage.
    Mutable local paths use file identity and timestamps; mutable remote paths
    use a hash of the object bytes. store, memory, resident, and cache follow
    the Text cache contract. Encoder assets and caches live outside checkpoints.
    """

    model_config = pydantic.ConfigDict(str_strip_whitespace=True)
    type: Literal["image"] = "image"
    model: Annotated[str, pydantic.Field(min_length=1)] = DEFAULT_MODEL
    base: Annotated[str, pydantic.Field(min_length=1)] | None = None
    immutable: pydantic.StrictBool = False
    jpeg_short_edge: Annotated[pydantic.StrictInt, pydantic.Field(gt=0)] | None = None
    read_workers: Annotated[pydantic.StrictInt, pydantic.Field(gt=0)] = 4
    encoder_batch_size: Annotated[pydantic.StrictInt, pydantic.Field(gt=0)] = 16
    cache: pydantic.StrictBool | Annotated[pydantic.StrictInt, pydantic.Field(gt=0)] = True
    store: Annotated[str, pydantic.Field(min_length=1)] | None = None
    memory: Annotated[pydantic.StrictInt, pydantic.Field(ge=0)] | None = None
    resident: Annotated[pydantic.StrictInt, pydantic.Field(ge=0)] | None = None
    objective: Objective = Objective.l2
    jitter: Jitter = pydantic.Field(default_factory=Jitter)

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            model: str = DEFAULT_MODEL,
            base: str | None = None,
            immutable: bool = False,
            jpeg_short_edge: int | None = None,
            read_workers: int = 4,
            encoder_batch_size: int = 16,
            cache: bool | int = True,
            store: str | None = None,
            memory: int | None = None,
            resident: int | None = None,
            objective: Objective | Literal["l1", "l2"] = Objective.l2,
            jitter: Jitter = ...,
            type: Literal["image"] = "image",
            **options: Unpack[FieldOptions],
        ) -> None: ...


def location(value: str, base: str | None) -> str:
    if not value or "\0" in value:
        raise ValueError(f"expected a nonempty image path without NUL, got {value!r}")
    if base and "://" not in value and not os.path.isabs(value):
        value = base.removesuffix("/") + "/" + value if "://" in base else os.path.join(base, value)
    if value.startswith("file://"):
        _, value = filesystem(value, None)
    return value if "://" in value else os.path.abspath(value)


@lru_cache(maxsize=32)
def remote(authority: str) -> tuple[fs.FileSystem, str]:
    """Reuse the Arrow storage client across objects in the same bucket."""
    return filesystem(authority, None)


def open_image(path: str):
    if "://" not in path:
        return open(path, "rb")
    scheme, rest = path.split("://", 1)
    authority, _, key = rest.partition("/")
    storage, prefix = remote(f"{scheme}://{authority}")
    name = f"{prefix.rstrip('/')}/{unquote(key)}" if prefix else unquote(key)
    # Pillow seeks through headers and frames. Download the compressed object
    # sequentially once, then let the codec seek in memory.
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
            with open_image(path) as stream:
                body = stream.read()
            version = hashlib.sha256(body).hexdigest()
        elif os.name == "nt":
            # Windows can report different metadata by path and by handle after a replacement.
            with open(path, "rb") as stream:
                version = stamp(os.fstat(stream.fileno()))
        else:
            version = stamp(os.stat(path))
    return json.dumps((path, version), ensure_ascii=False, separators=(",", ":")).encode(), body


@dataclass(frozen=True, slots=True)
class Crop:
    """Apply the standard BiT shortest-edge resize and center crop directly to RGB pixels."""

    edge: int
    width: int
    height: int

    @classmethod
    def from_processor(cls, processor: Any) -> Self | None:
        from transformers.models.bit.image_processing_pil_bit import BitImageProcessorPil

        # Other processors may own different ordering, interpolation or padding.
        if (
            type(processor) is not BitImageProcessorPil
            or not processor.do_resize
            or not processor.do_center_crop
            or processor.resample is None
            or dict(processor.size).keys() != {"shortest_edge"}
            or dict(processor.crop_size).keys() != {"width", "height"}
        ):
            return None
        edge = processor.size["shortest_edge"]
        width, height = processor.crop_size["width"], processor.crop_size["height"]
        if min(width, height) <= 0 or max(width, height) > edge:
            return None
        return cls(edge, width, height)

    def apply(self, rgb: Raster, resample: Any) -> Raster:
        width, height = rgb.size
        if width <= height:
            resized = (self.edge, self.edge * height // width)
        else:
            resized = (self.edge * width // height, self.edge)
        left, top = (resized[0] - self.width) // 2, (resized[1] - self.height) // 2
        return rgb.resize(resized, resample).crop((left, top, left + self.width, top + self.height))


def pixels(
    record: bytes,
    processor: Any,
    bodies: dict[bytes, bytes],
    *,
    crop: Crop | None = None,
    jpeg_short_edge: int | None = None,
) -> np.ndarray:
    from PIL import Image, ImageOps

    path, version = json.loads(record)
    try:
        stream = io.BytesIO(bodies[record]) if isinstance(version, str) else open_image(path)
        with stream:
            if isinstance(version, list) and list(stamp(os.fstat(stream.fileno()))) != version:
                raise ValueError(f"image {path!r} changed after tensorization; encode the observations again")
            with Image.open(cast(IO[bytes], stream)) as source:
                if jpeg_short_edge is not None and source.format == "JPEG":
                    source.draft(source.mode, (jpeg_short_edge, jpeg_short_edge))
                ImageOps.exif_transpose(source, in_place=True)
                rgb = source if source.mode == "RGB" else source.convert("RGB")
                if crop is None:
                    prepared = processor(images=rgb, return_tensors="np")
                else:
                    prepared = processor(
                        images=crop.apply(rgb, processor.resample),
                        do_resize=False,
                        do_center_crop=False,
                        return_tensors="np",
                    )
                values = prepared["pixel_values"][0]
            if isinstance(version, list) and list(stamp(os.fstat(stream.fileno()))) != version:
                raise ValueError(f"image {path!r} changed during decoding; encode the observations again")
        return values
    except (OSError, ValueError) as error:
        error.add_note(f"While reading or preparing image {path!r}.")
        raise


class Store(VectorCache):
    """Only the locator and its version enter the artifact store as prepared data."""

    vector_namespace = "rf.image.cls.f32/v1"
    input_bytes = 0  # Source descriptors have variable width.

    def identify(self, records: list[bytes]) -> np.ndarray:
        namespace = "rf.image.source/v1"
        ids = np.empty(len(records), dtype=np.int64)
        longest = max(map(len, records), default=0)
        chunk = min(MAX_ROWS, (MAX_FRAME - 65536) // (longest + 64))
        if not chunk:
            raise ValueError("image reference exceeds the artifact cache frame budget")
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


@image.register(Component.TensorField)
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
            error.add_note(f"While tensorizing image field {address!s}.")
            raise


@image.register(Component.Embedder)
class Embedder(FrozenEmbedder):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)
        # The top-level 5.9 export also checks torchvision; the PIL backend only needs Pillow.
        from transformers.models.auto.image_processing_auto import AutoImageProcessor

        self.request = cast(Request, schema.requests[address])
        self.processor = AutoImageProcessor.from_pretrained(self.request.model, backend="pil")
        self.store = store(self.request) if self.request.cache else None

    @property
    def context(self) -> Store | None:
        return self.store

    def variant(self, device: torch.device) -> str:
        from PIL import __version__ as pillow
        from PIL import features
        from transformers import __version__ as transformers

        settings = (
            type(self.processor).__qualname__,
            self.processor.to_json_string(),
            self.request.jpeg_short_edge,
            pillow,
            features.version_feature("libjpeg_turbo"),
            transformers,
        )
        return self._cached_model.variant(device, repr(settings))

    def pool(
        self, records: Sequence[bytes] | Payloads, device: torch.device, bodies: dict[bytes, bytes]
    ) -> torch.Tensor:
        """Prepare the next bounded batch on CPU while the current batch is encoded."""
        if isinstance(records, Payloads):
            records = [bytes(row) for row in records]
        model = self._cached_model.module_for(device)
        size = self.request.encoder_batch_size
        starts = iter(range(0, len(records), size))
        encoded = []
        prepare = partial(
            pixels,
            processor=self.processor,
            bodies=bodies,
            crop=Crop.from_processor(self.processor),
            jpeg_short_edge=self.request.jpeg_short_edge,
        )
        try:
            with ThreadPoolExecutor(max_workers=self.request.read_workers) as executor:

                def submit(start: int):
                    return [executor.submit(prepare, row) for row in records[start : start + size]]

                pending = submit(next(starts))
                while pending:
                    batch = torch.from_numpy(np.stack([job.result() for job in pending])).to(device)
                    following = next(starts, None)
                    pending = submit(following) if following is not None else []
                    with torch.no_grad():
                        hidden = getattr(model(pixel_values=batch), "last_hidden_state", None)
                        if (
                            hidden is None
                            or hidden.ndim != 3
                            or hidden.shape[0] != len(batch)
                            or hidden.shape[1] == 0
                            or hidden.shape[2] != self.hidden_size
                        ):
                            raise ValueError(
                                f"image encoder {self.request.model!r} must return [batch, tokens, {self.hidden_size}] with a CLS token"
                            )
                        encoded.append(hidden[:, 0].float().contiguous())
        except (ValueError, OSError) as error:
            error.add_note(f"While encoding image field {self.address!s}.")
            raise
        return torch.cat(encoded)

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
                        f"image field at '{self.address}' received cache IDs with cache=False; encode its observations again"
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
            raise ValueError(f"image field at {self.address!s} requires cache=True to warm")
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


image.register(Component.observe, observe)
image.register(Component.learn, learn)
image.register(Component.Decoder, Decoder)
image.register(Component.loss, loss)
