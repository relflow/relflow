"""Video references, concurrent decoding, temporal sampling, and frozen video-encoder embeddings."""

from __future__ import annotations

import enum
import inspect
import io
import json
import math
import os
import platform
import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import InitVar
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, TypeAlias, Unpack, cast

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
from relflow.tensorfields.shared.frozen import Backbone, Objective, learn, loss, observe
from relflow.tensorfields.shared.frozen import Decoder as FrozenDecoder
from relflow.tensorfields.shared.jitter import Jitter
from relflow.tensorfields.shared.reconstruction import Metrics

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from relflow.structs.experiment import Schema

DEFAULT_MODEL = "microsoft/xclip-base-patch32"
DEFAULT_FRAMES = 8
SEEK_GAP_SECONDS = 4.0

video = Extension(
    name="video",
    types=(str,),
    requires={
        "av": "relflow[video]",
        "PIL": "relflow[video]",
        "transformers": "relflow[video]",
        "moka_py": "relflow[video]",
    },
    callbacks=(CounterUpdateCallback, Metrics),
)


Clip: TypeAlias = Literal["chunks", "head", "center", "tail"]


class Pooling(enum.StrEnum):
    auto = "auto"
    mean = "mean"
    cls = "cls"
    pooler = "pooler"


class XCLIPTower(torch.nn.Module):
    """Video-only tower unwrapped from an X-CLIP checkpoint without its text encoder."""

    def __init__(self, model: Any):
        super().__init__()
        self.config = model.config
        self.vision_model = model.vision_model
        self.visual_projection = model.visual_projection
        self.mit = model.mit

    def forward(self, *, pixel_values: torch.Tensor) -> Any:
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        vision = self.vision_model(pixel_values=pixel_values.reshape(-1, num_channels, height, width))
        projected = self.visual_projection(vision.pooler_output).view(batch_size, num_frames, -1)
        return self.mit(projected)


class CachedModel(Backbone):
    @classmethod
    def get_model(cls, key: str) -> Backbone:
        if key not in cls._models:
            from transformers import AutoConfig, AutoModel

            # Version counters must remain available after inference-only warm-up.
            with torch.inference_mode(False):
                config = AutoConfig.from_pretrained(key)
                kind = getattr(config, "model_type", None)
                if kind == "xclip":
                    from transformers import XCLIPModel

                    loaded: Any = XCLIPTower(XCLIPModel.from_pretrained(key))
                elif kind == "vjepa2":
                    from transformers import VJEPA2Model

                    loaded = VJEPA2Model.from_pretrained(key).encoder
                else:
                    loaded = AutoModel.from_pretrained(key)
                model = cast("PreTrainedModel", loaded)
            model.eval()
            model.requires_grad_(False)
            hidden_size = getattr(model.config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(model.config, "projection_dim", None)
            if hidden_size is None:
                raise ValueError(
                    f"video encoder '{key}' does not expose `config.hidden_size` or `config.projection_dim`"
                )
            cls._models[key] = cls(key=key, model=model, hidden_size=int(hidden_size))
        return cls._models[key]


@video.register(Component.Request)
class Request(RequestBase):
    """Use a local video path or an object-store URI as context for prediction.

    The frozen video encoder supplies one vector per video. Keyframe seeking,
    frame decoding, display-rotation normalization, and spatial preprocessing
    run in a bounded thread pool; encoder_batch_size limits each model call.
    With window=None (the default), num_frames frames are sampled uniformly
    across the selected span. Setting window enables temporal chunking: videos
    longer than window use clip="chunks" by default, averaging up to
    max_windows windows into one deterministic vector, while clip="head",
    "center", and "tail" sample a single window. A reconstructing mask targets
    the pristine vector. embed=True exports the field's learned representation.

    Relative paths resolve against base (the working directory by default).
    Set immutable=True only for paths that always identify the same bytes,
    such as content-addressed object names. Warm hits then skip source storage.
    Mutable local paths use file identity and timestamps; mutable remote paths
    use a hash of the object bytes. store, memory, resident, and cache follow
    the Text, Image, and Audio cache contract. Encoder assets and caches live
    outside checkpoints.
    """

    model_config = pydantic.ConfigDict(str_strip_whitespace=True, allow_inf_nan=False)
    type: Literal["video"] = "video"
    model: Annotated[str, pydantic.Field(min_length=1)] = DEFAULT_MODEL
    base: Annotated[str, pydantic.Field(min_length=1)] | None = None
    immutable: pydantic.StrictBool = False
    offset: Annotated[float, pydantic.Field(ge=0.0)] = 0.0
    duration: Annotated[float, pydantic.Field(gt=0.0)] | None = None
    window: Annotated[float, pydantic.Field(gt=0.0)] | None = None
    stride: Annotated[float, pydantic.Field(gt=0.0)] | None = None
    max_windows: Annotated[pydantic.StrictInt, pydantic.Field(gt=0)] = 4
    clip: Clip = "chunks"
    num_frames: Annotated[pydantic.StrictInt, pydantic.Field(gt=0)] | None = None
    encoder_pooling: Pooling = Pooling.auto
    read_workers: Annotated[pydantic.StrictInt, pydantic.Field(gt=0)] = 4
    encoder_batch_size: Annotated[pydantic.StrictInt, pydantic.Field(gt=0)] = 8
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
            raise ValueError("video time bounds must be numbers, not booleans")
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
            num_frames: int | None = None,
            encoder_pooling: Pooling | Literal["auto", "mean", "cls", "pooler"] = Pooling.auto,
            read_workers: int = 4,
            encoder_batch_size: int = 8,
            cache: bool | int = True,
            store: str | None = None,
            memory: int | None = None,
            resident: int | None = None,
            objective: Objective | Literal["l1", "l2"] = Objective.l2,
            jitter: Jitter = ...,
            type: Literal["video"] = "video",
            **options: Unpack[FieldOptions],
        ) -> None: ...


remote = media.remote
stamp = media.stamp


def location(value: str, base: str | None) -> str:
    return media.location(value, base, kind="video")


def open_video(path: str):
    return media.open_stream(path, remote, seekable=True)


def snapshot(path: str, immutable: bool) -> tuple[bytes, bytes | None]:
    return media.snapshot(path, immutable, open_video)


def native_frames(config: Any) -> int | None:
    for owner in (config, getattr(config, "vision_config", None)):
        if owner is None:
            continue
        for attribute in ("num_frames", "frames_per_clip"):
            value = getattr(owner, attribute, None)
            if isinstance(value, int) and value > 0:
                return value
    return None


def tubelet_size(config: Any) -> int | None:
    value = getattr(config, "tubelet_size", None)
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], int) and value[0] > 0:
        return int(value[0])
    if isinstance(value, int) and value > 0:
        return value
    return None


def load_processor(model_name: str, config: Any = None) -> Any:
    """Load a PIL-backed frame processor without requiring torchvision."""
    from transformers.models.auto.image_processing_auto import AutoImageProcessor

    kind = getattr(config, "model_type", None)
    if kind == "vivit":
        from transformers import VivitImageProcessor

        return VivitImageProcessor.from_pretrained(model_name)

    try:
        return AutoImageProcessor.from_pretrained(model_name, backend="pil")
    except (OSError, ValueError, KeyError, ImportError) as image_error:
        # Upstream ViViT checkpoints spell image_processor_type as "ViViTImageProcessor".
        pre_path = resolve_asset(model_name, "preprocessor_config.json")
        if pre_path is not None:
            raw_pre = json.loads(Path(pre_path).read_text(encoding="utf-8"))
            if str(raw_pre.get("image_processor_type", "")).lower() == "vivitimageprocessor":
                from transformers import VivitImageProcessor

                return VivitImageProcessor.from_pretrained(model_name)

        # Checkpoints such as V-JEPA 2 ship video_preprocessor_config.json only.
        config_path = resolve_asset(model_name, "video_preprocessor_config.json")
        if config_path is None:
            raise ValueError(
                f"video encoder {model_name!r} has neither a supported image processor nor video_preprocessor_config.json"
            ) from image_error

        from transformers.models.videomae.image_processing_pil_videomae import VideoMAEImageProcessorPil

        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        allowed = (
            "do_resize",
            "size",
            "resample",
            "do_center_crop",
            "crop_size",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "default_to_square",
        )
        options = {key: raw[key] for key in allowed if key in raw and raw[key] is not None}
        if "crop_size" in options and "size" not in options:
            crop = options["crop_size"]
            edge = crop["height"] if isinstance(crop, dict) else int(crop)
            options["size"] = {"shortest_edge": int(edge * 256 / 224)}
        return VideoMAEImageProcessorPil(**options)


def resolve_asset(model_name: str, filename: str) -> str | None:
    root = Path(model_name)
    if root.exists():
        local = root / filename
        return str(local) if local.is_file() else None
    from transformers.utils.hub import cached_file

    try:
        return cached_file(model_name, filename, _raise_exceptions_for_missing_entries=False)
    except (OSError, ValueError):
        return None


def spans(
    total: float,
    *,
    offset: float,
    duration: float | None,
    window: float | None,
    stride: float | None,
    max_windows: int,
    clip: Clip,
) -> list[tuple[float, float]]:
    """Select deterministic temporal windows in seconds before decoding video frames."""
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("video source contains no readable duration")
    start = float(offset)
    stop = float(total) if duration is None else min(float(total), start + float(duration))
    if start >= stop:
        raise ValueError(f"video clip has no frames after offset={offset:g}s (source duration {total:.3f}s)")
    length = stop - start
    if window is None or length <= window:
        return [(start, stop)]
    if clip == "head":
        return [(start, start + window)]
    if clip == "tail":
        return [(stop - window, stop)]
    if clip == "center" or max_windows == 1:
        left = start + (length - window) / 2.0
        return [(left, left + window)]
    hop = window / 2.0 if stride is None else float(stride)
    last = stop - window
    steps = max(0, math.ceil((last - start) / hop - 1e-9))
    # Include the final window even when it falls between hops.
    if steps + 1 > max_windows:
        starts = np.linspace(start, last, max_windows).tolist()
    else:
        starts = [start + i * hop for i in range(steps)] + [last]
    return [(float(left), float(left) + window) for left in starts]


def sample_times(start: float, stop: float, num_frames: int) -> list[float]:
    """Choose deterministic midpoint timestamps strictly inside each equal bin."""
    width = stop - start
    return [start + (index + 0.5) * width / num_frames for index in range(num_frames)]


def orient(frame: Any, rgb: np.ndarray) -> np.ndarray:
    """Apply container DISPLAYMATRIX rotation and mirror flips so frames are upright."""
    side_data = getattr(frame, "side_data", None)
    matrix_data = side_data.get("DISPLAYMATRIX") if side_data is not None else None
    if matrix_data is not None:
        raw = bytes(matrix_data)
        if len(raw) >= 36:
            mat = np.frombuffer(raw[:36], dtype=np.int32).reshape(3, 3)
            a, b = round(float(mat[0, 0]) / 65536.0), round(float(mat[0, 1]) / 65536.0)
            c, d = round(float(mat[1, 0]) / 65536.0), round(float(mat[1, 1]) / 65536.0)
            if abs(b) == 1 and abs(c) == 1 and a == 0 and d == 0:
                out = np.transpose(rgb, (1, 0, 2))
                if b < 0:
                    out = out[::-1, :]
                if c < 0:
                    out = out[:, ::-1]
                return np.ascontiguousarray(out)
            if abs(a) == 1 and abs(d) == 1 and b == 0 and c == 0:
                out = rgb
                if d < 0:
                    out = out[::-1, :]
                if a < 0:
                    out = out[:, ::-1]
                return np.ascontiguousarray(out)
    quarter = round(int(getattr(frame, "rotation", 0) or 0) / 90) % 4
    if quarter == 0:
        return rgb
    return np.ascontiguousarray(np.rot90(rgb, k=quarter))


def stream_bounds(container: Any, video_stream: Any) -> tuple[float, float, float]:
    """Resolve (origin_seconds, duration_seconds, frame_step_seconds) without decoding pixels."""
    import av

    time_base = float(video_stream.time_base) if video_stream.time_base is not None else 1.0 / av.time_base
    rate = float(video_stream.average_rate) if video_stream.average_rate else 0.0
    if rate <= 0.0 and getattr(video_stream, "guessed_rate", None):
        rate = float(video_stream.guessed_rate)
    step = 1.0 / rate if rate > 0.0 else 1.0 / 30.0

    origin = float(video_stream.start_time * time_base) if video_stream.start_time is not None else 0.0
    if video_stream.duration is not None and video_stream.duration > 0:
        return origin, float(video_stream.duration * time_base), step
    if container.duration is not None and container.duration > 0:
        if video_stream.start_time is None and container.start_time is not None:
            origin = float(container.start_time / av.time_base)
        return origin, float(container.duration / av.time_base), step
    if video_stream.frames > 0 and rate > 0.0:
        return origin, float(video_stream.frames / rate), step

    # Container omitted duration metadata (common in WebM/MKV/streamed files):
    # scan packet timestamps only, without invoking the video decoder.
    first_time: float | None = None
    last_time: float | None = None
    last_duration: float = 0.0
    packets = 0
    for packet in container.demux(video_stream):
        if packet.dts is None and packet.pts is None and packet.size == 0:
            continue
        packets += 1
        stamp_pts = packet.pts if packet.pts is not None else packet.dts
        if stamp_pts is not None:
            seconds = float(stamp_pts * time_base)
            if first_time is None or seconds < first_time:
                first_time = seconds
            if last_time is None or seconds >= last_time:
                last_time = seconds
                last_duration = float(packet.duration * time_base) if packet.duration else 0.0
    container.seek(0)
    if packets == 0:
        raise ValueError("video source contains no readable frames")
    if first_time is not None and last_time is not None:
        total = (last_time - first_time) + (last_duration if last_duration > 0.0 else step)
        return first_time, max(total, step), step
    return origin, max(packets * step, step), step


def frame_time(frame: Any, index: int, origin: float, step: float, time_base: float) -> float:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is not None:
        return float(frame.pts * time_base)
    if frame.dts is not None:
        return float(frame.dts * time_base)
    return origin + index * step


def extract_frames(
    container: Any,
    video_stream: Any,
    targets: Sequence[float],
    *,
    origin: float,
    step: float,
    time_base: float,
) -> dict[float, np.ndarray]:
    """Decode only the frames nearest to sorted target timestamps using keyframe seeks."""
    import av

    if not targets:
        return {}
    selected: dict[float, np.ndarray] = {}
    last_frame: Any | None = None
    last_rgb: np.ndarray | None = None

    def as_rgb(frame: Any) -> np.ndarray:
        nonlocal last_frame, last_rgb
        if frame is last_frame and last_rgb is not None:
            return last_rgb
        last_frame = frame
        last_rgb = orient(frame, frame.to_ndarray(format="rgb24"))
        return last_rgb

    def seek_to(seconds: float) -> Iterator[Any]:
        pts = max(0, int(seconds / time_base)) if origin >= 0.0 else int(seconds / time_base)
        container.seek(pts, backward=True, stream=video_stream)
        return iter(container.decode(video_stream))

    def pull(decoder: Iterator[Any], index: int) -> tuple[float | None, Any | None, int]:
        frame = next(decoder, None)
        if frame is None:
            return None, None, index
        return frame_time(frame, index, origin, step, time_base), frame, index + 1

    def restart() -> tuple[Iterator[Any], float | None, Any | None, float | None, Any | None, int]:
        container.seek(0)
        stream_iter = iter(container.decode(video_stream))
        first_t, first_frame, seen = pull(stream_iter, 0)
        return stream_iter, None, None, first_t, first_frame, seen

    seekable = True
    absolute = [origin + timestamp for timestamp in targets]
    if absolute[0] > origin + SEEK_GAP_SECONDS:
        try:
            decoder = seek_to(absolute[0])
            prev_t, prev_frame = None, None
            curr_t, curr_frame, count = pull(decoder, 0)
        except (av.FFmpegError, OSError, ValueError):
            seekable = False
            decoder, prev_t, prev_frame, curr_t, curr_frame, count = restart()
    else:
        decoder = iter(container.decode(video_stream))
        prev_t, prev_frame = None, None
        curr_t, curr_frame, count = pull(decoder, 0)

    # Guard against containers whose keyframe index overshoots the first requested timestamp.
    if (
        curr_frame is not None
        and curr_t is not None
        and curr_t > absolute[0] + 1e-6
        and absolute[0] > origin + SEEK_GAP_SECONDS
    ):
        seekable = False
        decoder, prev_t, prev_frame, curr_t, curr_frame, count = restart()

    for rel, target in zip(targets, absolute, strict=True):
        if seekable and curr_t is not None and target - curr_t > SEEK_GAP_SECONDS:
            previous_pos = curr_t
            try:
                candidate = seek_to(target)
                next_t, next_frame, next_count = pull(candidate, count)
                if next_frame is not None and next_t is not None and next_t <= target + 1e-6:
                    if next_t <= previous_pos:
                        # No closer keyframe exists between previous_pos and target; avoid repeated regressive seeks.
                        seekable = False
                    decoder = candidate
                    prev_t, prev_frame = None, None
                    curr_t, curr_frame, count = next_t, next_frame, next_count
                else:
                    # Overshot seek: fall back cleanly to sequential decoding from start.
                    seekable = False
                    decoder, prev_t, prev_frame, curr_t, curr_frame, count = restart()
            except (av.FFmpegError, OSError, ValueError):
                seekable = False

        while curr_frame is not None and curr_t is not None and curr_t < target:
            prev_t, prev_frame = curr_t, curr_frame
            curr_t, curr_frame, count = pull(decoder, count)

        if prev_frame is not None and curr_frame is not None and prev_t is not None and curr_t is not None:
            chosen = prev_frame if abs(target - prev_t) <= abs(curr_t - target) else curr_frame
        elif curr_frame is not None:
            chosen = curr_frame
        elif prev_frame is not None:
            chosen = prev_frame
        else:
            raise ValueError("video source yielded no decoded frames")
        selected[rel] = as_rgb(chosen)
    return selected


def clips(
    record: bytes,
    processor: Any,
    bodies: dict[bytes, bytes],
    *,
    offset: float,
    duration: float | None,
    window: float | None,
    stride: float | None,
    max_windows: int,
    clip: Clip,
    num_frames: int,
) -> list[np.ndarray]:
    """Decode sampled windows for one video and run spatial preprocessing per window."""
    import av

    path, version = json.loads(record)
    try:
        stream = io.BytesIO(bodies[record]) if isinstance(version, str) else open_video(path)
        with stream:
            if isinstance(version, list) and list(stamp(os.fstat(stream.fileno()))) != version:
                raise ValueError(f"video {path!r} changed after tensorization; encode the observations again")
            with av.open(stream) as container:
                if not container.streams.video:
                    raise ValueError(f"video {path!r} contains no video stream")
                video_stream = container.streams.video[0]
                video_stream.thread_count = 1
                time_base = float(video_stream.time_base) if video_stream.time_base is not None else 1.0 / av.time_base
                origin, total, step = stream_bounds(container, video_stream)
                selected_spans = spans(
                    total,
                    offset=offset,
                    duration=duration,
                    window=window,
                    stride=stride,
                    max_windows=max_windows,
                    clip=clip,
                )
                per_window = [sample_times(left, right, num_frames) for left, right in selected_spans]
                unique_targets = sorted({timestamp for times in per_window for timestamp in times})
                decoded = extract_frames(
                    container,
                    video_stream,
                    unique_targets,
                    origin=origin,
                    step=step,
                    time_base=time_base,
                )
            if isinstance(version, list) and list(stamp(os.fstat(stream.fileno()))) != version:
                raise ValueError(f"video {path!r} changed during decoding; encode the observations again")

        prepared: list[np.ndarray] = []
        with torch.autocast(device_type="cpu", enabled=False):
            for times in per_window:
                frames = [decoded[timestamp] for timestamp in times]
                batch = processor(frames, return_tensors="np")
                values = batch.get("pixel_values")
                if values is None:
                    values = batch.get("pixel_values_videos")
                if values is None:
                    raise ValueError(
                        f"video processor for {path!r} produced neither 'pixel_values' nor 'pixel_values_videos'"
                    )
                array = np.asarray(values, dtype=np.float32)
                if array.ndim != 5 or array.shape[0] != 1 or array.shape[1] != num_frames:
                    raise ValueError(
                        f"video processor returned shape {tuple(array.shape)}; expected (1, {num_frames}, C, H, W)"
                    )
                prepared.append(np.ascontiguousarray(array[0]))
        return prepared
    except (av.FFmpegError, OSError, RuntimeError, ValueError) as error:
        error.add_note(f"While reading or preparing video {path!r}.")
        raise


class Store(media.ReferenceStore):
    vector_namespace = "rf.video.vec.f32/v1"
    source_namespace = "rf.video.source/v1"
    kind = "video"


@video.register(Component.TensorField)
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
            kind="video",
            store_cls=Store,
            opener=lambda path: open_video(path),
        )


def pool_outputs(
    outputs: Any,
    model: Any,
    pooling: Pooling,
    hidden_size: int,
    model_name: str,
) -> torch.Tensor:
    kind = getattr(getattr(model, "config", None), "model_type", None)
    if pooling == Pooling.pooler or (
        pooling == Pooling.auto and getattr(outputs, "pooler_output", None) is not None and kind in {"xclip", "vivit"}
    ):
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None or pooled.ndim != 2 or pooled.shape[1] != hidden_size:
            raise ValueError(f"video encoder {model_name!r} must expose pooler_output of width {hidden_size}")
        return pooled.float().contiguous()

    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None or hidden.ndim != 3 or hidden.shape[1] == 0 or hidden.shape[2] != hidden_size:
        raise ValueError(f"video encoder {model_name!r} must return [batch, tokens, {hidden_size}] hidden states")
    if pooling == Pooling.cls or (pooling == Pooling.auto and kind == "timesformer"):
        return hidden[:, 0].float().contiguous()
    return hidden.mean(dim=1).float().contiguous()


def input_key(model: Any) -> str:
    kind = getattr(getattr(model, "config", None), "model_type", None)
    if kind == "vjepa2":
        return "pixel_values_videos"
    parameters = inspect.signature(model.forward).parameters
    if "pixel_values_videos" in parameters and "pixel_values" not in parameters:
        return "pixel_values_videos"
    return "pixel_values"


def encode_clips(
    model: Any,
    windows: Sequence[np.ndarray],
    device: torch.device,
    *,
    pooling: Pooling,
    hidden_size: int,
    model_name: str,
) -> torch.Tensor:
    if not windows:
        return torch.empty((0, hidden_size), dtype=torch.float32, device=device)
    batch = torch.from_numpy(np.stack(windows)).to(device)
    key = input_key(model)
    with torch.no_grad():
        outputs = model(**{key: batch})
    pooled = pool_outputs(outputs, model, pooling, hidden_size, model_name)
    if pooled.shape != (len(windows), hidden_size):
        raise ValueError(
            f"video encoder {model_name!r} returned shape {tuple(pooled.shape)}; expected {(len(windows), hidden_size)}"
        )
    return pooled


@video.register(Component.Embedder)
class Embedder(media.MediaEmbedder):
    backbone = CachedModel
    kind = "video"
    store_cls = Store

    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)
        self.request = cast(Request, schema.requests[address])
        config = self._cached_model.model.config
        self.processor = load_processor(self.request.model, config=config)
        kind = getattr(config, "model_type", None)
        native = native_frames(config)
        requested = self.request.num_frames
        if (
            kind in {"videomae", "vivit", "xclip"}
            and native is not None
            and requested is not None
            and requested != native
        ):
            raise ValueError(
                f"video field {address} requests num_frames={requested} for {self.request.model!r}, "
                f"which requires exactly {native} frames. Omit num_frames or set num_frames={native}, "
                "and use window with clip='chunks' for longer videos."
            )
        chosen = requested or native or DEFAULT_FRAMES
        if kind == "vjepa2":
            tubelet = tubelet_size(config) or 2
            if chosen % tubelet != 0:
                raise ValueError(
                    f"video field {address} requests num_frames={chosen} for {self.request.model!r}, "
                    f"which must be a multiple of tubelet_size={tubelet}"
                )
        self.clip_frames = chosen

    def stream(self, path: str):
        return open_video(path)

    def variant(self, device: torch.device) -> str:
        import av
        from PIL import __version__ as pillow
        from transformers import __version__ as transformers

        serialize = getattr(self.processor, "to_json_string", None)
        processor_state = serialize() if callable(serialize) else repr(vars(self.processor))
        settings = (
            type(self.processor).__qualname__,
            processor_state,
            self.request.offset,
            self.request.duration,
            self.request.window,
            self.request.stride,
            self.request.max_windows,
            self.request.clip,
            self.clip_frames,
            self.request.encoder_pooling.value,
            av.__version__,
            tuple(sorted(av.library_versions.items())),
            pillow,
            sys.platform,
            platform.machine(),
            transformers,
        )
        return self._cached_model.variant(device, repr(settings))

    def pool(
        self, records: Sequence[bytes] | Payloads, device: torch.device, bodies: dict[bytes, bytes]
    ) -> torch.Tensor:
        """Decode and preprocess the next bounded batch on CPU while encoding the current batch."""
        import av

        if isinstance(records, Payloads):
            records = [bytes(row) for row in records]
        if not records:
            return torch.empty((0, self.hidden_size), dtype=torch.float32, device=device)
        model = self._cached_model.module_for(device)
        size = self.request.encoder_batch_size
        starts = iter(range(0, len(records), size))
        encoded: list[torch.Tensor] = []
        prepare = partial(
            clips,
            processor=self.processor,
            bodies=bodies,
            offset=self.request.offset,
            duration=self.request.duration,
            window=self.request.window,
            stride=self.request.stride,
            max_windows=self.request.max_windows,
            clip=self.request.clip,
            num_frames=self.clip_frames,
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
                                flat[offset : offset + size],
                                device,
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
        except (av.FFmpegError, OSError, RuntimeError, ValueError) as error:
            error.add_note(f"While encoding video field {self.address!s}.")
            raise
        return torch.cat(encoded, dim=0)


@video.register(Component.Decoder)
class Decoder(FrozenDecoder):
    coordinatewise = True
    backbone = CachedModel


video.register(Component.observe, observe)
video.register(Component.learn, learn)
video.register(Component.loss, loss)
