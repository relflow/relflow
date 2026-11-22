"""Image references, concurrent decoding, and frozen vision-transformer embeddings."""

from __future__ import annotations

import io
import json
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import InitVar, dataclass
from functools import partial
from typing import IO, TYPE_CHECKING, Annotated, Any, Literal, Self, Unpack, cast

import numpy as np
import pydantic
import torch
from tensordict import TensorDict, tensorclass

from relflow.data.ragged import Sources
from relflow.structs.enums import Component
from relflow.structs.tree import Address, FieldOptions
from relflow.tensorfields.base import Context, Extension, RequestBase, TensorFieldBase
from relflow.tensorfields.shared import media
from relflow.tensorfields.shared.cache import Payloads
from relflow.tensorfields.shared.counter import CounterUpdateCallback
from relflow.tensorfields.shared.frozen import Decoder, Objective, learn, loss, observe
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


remote = media.remote
stamp = media.stamp


def location(value: str, base: str | None) -> str:
    return media.location(value, base, kind="image")


def open_image(path: str):
    return media.open_stream(path, remote)


def snapshot(path: str, immutable: bool) -> tuple[bytes, bytes | None]:
    return media.snapshot(path, immutable, open_image)


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


class Store(media.ReferenceStore):
    vector_namespace = "rf.image.cls.f32/v1"
    source_namespace = "rf.image.source/v1"
    kind = "image"


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
        return media.tensorize(
            cls,
            sources,
            schema=schema,
            context=context,
            kind="image",
            store_cls=Store,
            opener=lambda path: open_image(path),
        )


@image.register(Component.Embedder)
class Embedder(media.MediaEmbedder):
    kind = "image"
    store_cls = Store

    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)
        # The top-level 5.9 export also checks torchvision; the PIL backend only needs Pillow.
        from transformers.models.auto.image_processing_auto import AutoImageProcessor

        self.request = cast(Request, schema.requests[address])
        self.processor = AutoImageProcessor.from_pretrained(self.request.model, backend="pil")

    def stream(self, path: str):
        return open_image(path)

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
        if not records:
            return torch.empty((0, self.hidden_size), dtype=torch.float32, device=device)
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


image.register(Component.observe, observe)
image.register(Component.learn, learn)
image.register(Component.Decoder, Decoder)
image.register(Component.loss, loss)
