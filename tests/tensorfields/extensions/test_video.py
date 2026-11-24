import io
import json
import os
import threading
from collections import Counter
from types import SimpleNamespace

import av
import numpy as np
import polars as pl
import pyarrow as pa
import pytest
import torch
from pyarrow import fs

import relflow as rf
from relflow.tensorfields.extensions import video
from relflow.tensorfields.shared.frozen import Backbone


class FrameProcessor:
    def __init__(self):
        self.scale = 1.0

    def to_json_string(self):
        return str(self.scale)

    def __call__(self, frames, *, return_tensors="np", **kwargs):
        assert return_tensors == "np"
        stacked = np.stack([np.asarray(frame, dtype=np.float32) / 255.0 * self.scale for frame in frames], axis=0)
        # Convert (T, H, W, C) -> (1, T, C, H, W)
        return {"pixel_values": np.transpose(stacked, (0, 3, 1, 2))[None]}


class Visual(torch.nn.Module):
    def __init__(self, num_frames: int = 4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=3, model_type="custom", num_frames=num_frames)
        self.batches: list[int] = []
        self.weight = torch.nn.Parameter(torch.ones(3), requires_grad=False)

    def forward(self, *, pixel_values: torch.Tensor):
        self.batches.append(len(pixel_values))
        # (B, T, C, H, W) -> per-channel spatial+temporal mean (B, 3)
        features = pixel_values.mean(dim=(1, 3, 4)) * self.weight
        return SimpleNamespace(last_hidden_state=features.unsqueeze(1))


def write_video(
    path,
    frames: list[np.ndarray],
    *,
    fps: int = 10,
    codec: str = "mpeg4",
    container_format: str | None = None,
    rotation: int = 0,
    gop_size: int | None = None,
    pts_offset: int = 0,
):
    height, width = frames[0].shape[:2]
    open_kwargs = {"format": container_format} if container_format is not None else {}
    with av.open(str(path), "w", **open_kwargs) as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "bgr0" if codec == "ffv1" else "yuvj420p" if codec == "mjpeg" else "yuv420p"
        if gop_size is not None:
            stream.codec_context.gop_size = gop_size
        if rotation:
            stream.set_display_rotation(rotation)
        for index, array in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(np.asarray(array, dtype=np.uint8), format="rgb24")
            if pts_offset:
                frame.pts = pts_offset + index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def solid_frames(count: int, rgb: tuple[int, int, int], size: tuple[int, int] = (16, 16)) -> list[np.ndarray]:
    frame = np.empty((size[0], size[1], 3), dtype=np.uint8)
    frame[..., 0], frame[..., 1], frame[..., 2] = rgb
    return [frame for _ in range(count)]


@pytest.fixture
def visual(monkeypatch, tmp_path):
    processor = FrameProcessor()
    model = Visual(num_frames=4)
    monkeypatch.setattr(Backbone, "_models", {"tiny": Backbone("tiny", model, 3)})
    monkeypatch.setattr(video, "load_processor", lambda *args, **kwargs: processor)
    monkeypatch.setenv("RELFLOW_CACHE_DIR", str(tmp_path / "cache"))
    write_video(tmp_path / "bright.mp4", solid_frames(8, (200, 120, 40)), fps=8)
    write_video(tmp_path / "glöcke.mp4", solid_frames(8, (40, 160, 220)), fps=8)
    return model


def build(base, **options):
    return rf.Model(d_model=8, n_heads=2, n_layers=1, clip=rf.Video(model="tiny", base=str(base), **options))


def vectors(model, paths):
    field = model.encode(pl.DataFrame({"clip": paths}, schema={"clip": pl.String}))["/clip"]
    return model.nodes["/clip"].embedder.encode(field.content, field.state)


@pytest.mark.parametrize("resident", [0, 1])
def test_concurrent_reads_compact_ids_and_warm_parity(visual, tmp_path, monkeypatch, resident):
    model = build(tmp_path, immutable=True, read_workers=2, encoder_batch_size=2, resident=resident)
    paths = ["glöcke.mp4", None, "bright.mp4", "glöcke.mp4"]
    data = pl.DataFrame({"clip": paths})
    assert model.encode(data)["/clip"].content["ids"].numel() == len(paths)
    reads = Counter()
    barrier = threading.Barrier(2)
    original = video.open_video

    def concurrent(path):
        reads[path] += 1
        barrier.wait(timeout=10)
        return original(path)

    monkeypatch.setattr(video, "open_video", concurrent)
    first = vectors(model, paths)
    assert sorted(reads.values()) == [1, 1] and visual.batches == [2]
    assert first[1, 0].count_nonzero() == 0
    assert torch.equal(first[0, 0], first[3, 0])
    # bright has higher R than B; glöcke has higher B than R.
    assert first[2, 0, 0] > first[2, 0, 2]
    assert first[0, 0, 2] > first[0, 0, 0]

    monkeypatch.setattr(video, "open_video", lambda *args: pytest.fail("warm hit opened a video file"))
    assert torch.equal(vectors(model, paths), first)
    assert model.nodes["/clip"].embedder.warm(pa.array(paths)) == 0
    assert model.nodes["/clip"].embedder.warm("bright.mp4") == 0
    assert visual.batches == [2]

    monkeypatch.setattr(video, "open_video", original)
    direct = build(tmp_path, cache=False, encoder_batch_size=1)
    torch.testing.assert_close(vectors(direct, paths), first)
    assert visual.batches == [2, 1, 1]
    with pytest.raises(ValueError, match="cache=True"):
        direct.nodes["/clip"].embedder.warm(paths)


def test_mutable_reference_changes_and_failed_reads_can_retry(visual, tmp_path):
    model = build(tmp_path)
    path = tmp_path / "bright.mp4"
    assert video.location(path.name, tmp_path.as_uri()) == str(path)
    root = tmp_path.parents[-1]
    assert video.location("clip.mp4", root.as_uri()) == str(root / "clip.mp4")
    first = vectors(model, [path.as_uri()])
    write_video(tmp_path / "replacement.mp4", solid_frames(8, (20, 30, 240)), fps=8)
    os.replace(tmp_path / "replacement.mp4", path)
    assert not torch.equal(first, vectors(model, [str(path)]))

    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not-a-video")
    with pytest.raises((av.FFmpegError, OSError, RuntimeError, ValueError)) as error:
        vectors(model, [str(broken)])
    assert any("/clip" in note for note in error.value.__notes__)
    assert model.nodes["/clip"].embedder.store.stats()["claims"] == 0
    write_video(broken, solid_frames(4, (100, 100, 100)), fps=8)
    assert vectors(model, [str(broken)]).count_nonzero() == 3

    # Audio-only container has no video stream.
    audio_only = tmp_path / "audio_only.mkv"
    with av.open(str(audio_only), "w") as out:
        ast = out.add_stream("pcm_s16le", rate=8000)
        af = av.AudioFrame.from_ndarray(np.zeros((1, 800), dtype=np.int16), format="s16", layout="mono")
        af.sample_rate = 8000
        for pkt in ast.encode(af):
            out.mux(pkt)
        for pkt in ast.encode():
            out.mux(pkt)
    with pytest.raises(ValueError, match="no video stream") as error:
        vectors(model, [str(audio_only)])
    assert any("/clip" in note for note in error.value.__notes__)

    with pytest.raises(FileNotFoundError, match="missing.mp4"):
        vectors(model, ["missing.mp4"])
    with pytest.raises(ValueError, match="nonempty video path"):
        vectors(model, [""])
    with pytest.raises(ValueError, match="numbers, not booleans"):
        rf.Video(offset=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="numbers, not booleans"):
        rf.Video(window=False)  # type: ignore[arg-type]


def test_snapshot_cannot_publish_replaced_file_under_old_id(visual, tmp_path):
    model = build(tmp_path)
    field = model.encode(pl.DataFrame({"clip": ["bright.mp4"]}))["/clip"]
    write_video(tmp_path / "other.mp4", solid_frames(8, (10, 20, 30)), fps=8)
    os.replace(tmp_path / "other.mp4", tmp_path / "bright.mp4")
    encoder = model.nodes["/clip"].embedder
    with pytest.raises(ValueError, match="changed after tensorization"):
        encoder.encode(field.content, field.state)
    assert encoder.store.stats()["results"] == 0


@pytest.mark.parametrize("immutable", [False, True])
def test_remote_snapshot_reads_body_once_and_skips_decode_on_hits(visual, tmp_path, monkeypatch, immutable):
    storage = fs.SubTreeFileSystem(str(tmp_path), fs.LocalFileSystem())
    monkeypatch.setattr(video, "remote", lambda authority: (storage, ""))
    opens = []
    decodes = []
    open_video, clips = video.open_video, video.clips

    def opened(path):
        opens.append(path)
        return open_video(path)

    def decoded(*args, **kwargs):
        decodes.append(1)
        return clips(*args, **kwargs)

    monkeypatch.setattr(video, "open_video", opened)
    monkeypatch.setattr(video, "clips", decoded)
    model = build("s3://fixtures", immutable=immutable)
    first = vectors(model, ["bright.mp4", "bright.mp4"])
    assert len(opens) == len(decodes) == 1
    second = vectors(model, ["bright.mp4"])
    assert torch.equal(first[0], second[0])
    assert len(opens) == (1 if immutable else 2)
    assert len(decodes) == 1 and visual.batches == [1]
    if not immutable:
        write_video(tmp_path / "next.mp4", solid_frames(8, (10, 240, 10)), fps=8)
        os.replace(tmp_path / "next.mp4", tmp_path / "bright.mp4")
        assert not torch.equal(first[0], vectors(model, ["bright.mp4"])[0])
        assert len(opens) == 3 and len(decodes) == 2


@pytest.mark.parametrize("access", ["seekable", "stream", "unsupported"])
def test_remote_window_reads(visual, tmp_path, monkeypatch, access):
    # 12-second multi-keyframe video (~1 MiB) so seekable tail reads fetch well under half the object.
    rng = np.random.default_rng(7)
    frames = [rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8) for _ in range(120)]
    path = tmp_path / "long.mp4"
    write_video(path, frames, fps=10, codec="mjpeg", gop_size=1)
    read = Counter()

    class Reader(io.FileIO):
        def seekable(self):
            return access == "seekable"

        def readinto(self, buf):
            count = super().readinto(buf)
            read["bytes"] += count
            return count

    class Storage:
        def open_input_file(self, name):
            assert name == path.name
            if access == "unsupported":
                raise NotImplementedError
            return Reader(path, "rb")

        def open_input_stream(self, name, *, compression):
            read["full"] += 1
            return path.open("rb")

    monkeypatch.setattr(video, "remote", lambda authority: (Storage(), ""))
    processor = FrameProcessor()
    opts = dict(
        processor=processor,
        bodies={},
        offset=0.0,
        duration=None,
        window=1.0,
        stride=None,
        max_windows=1,
        clip="tail",
        num_frames=4,
    )
    remote_out = video.clips(json.dumps(("s3://bucket/long.mp4", None)).encode(), **opts)
    local_out = video.clips(video.snapshot(str(path), True)[0], **opts)
    np.testing.assert_array_equal(remote_out, local_out)
    if access == "seekable":
        assert 0 < read["bytes"] < path.stat().st_size // 2 and read["full"] == 0
    else:
        assert read["full"] == 1


def test_temporal_windowing_sampling_and_cache_invalidation(visual, tmp_path):
    # 3-second video at 10 fps: 0..1s intensity ~50, 1..2s intensity ~125, 2..3s intensity ~225.
    frames = solid_frames(10, (50, 50, 50)) + solid_frames(10, (125, 125, 125)) + solid_frames(10, (225, 225, 225))
    path = tmp_path / "segmented.mkv"
    write_video(path, frames, fps=10, codec="ffv1", container_format="matroska")

    head = vectors(build(tmp_path, window=1.0, clip="head"), [path.name])[0, 0, 0]
    center = vectors(build(tmp_path, window=1.0, clip="center"), [path.name])[0, 0, 0]
    tail = vectors(build(tmp_path, window=1.0, clip="tail", max_windows=1), [path.name])[0, 0, 0]
    chunks = vectors(build(tmp_path, window=1.0, stride=1.0, max_windows=3, clip="chunks"), [path.name])[0, 0, 0]
    sliced = vectors(build(tmp_path, offset=2.0, duration=1.0, window=1.0), [path.name])[0, 0, 0]
    whole = vectors(build(tmp_path, window=None, num_frames=6), [path.name])[0, 0, 0]

    torch.testing.assert_close(head, torch.tensor(50.0 / 255.0), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(center, torch.tensor(125.0 / 255.0), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(tail, torch.tensor(225.0 / 255.0), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(sliced, tail, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(chunks, (head + center + tail) / 3.0, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(whole, (head + center + tail) / 3.0, atol=1e-3, rtol=1e-3)

    with pytest.raises(ValueError, match="no frames after offset"):
        vectors(build(tmp_path, offset=5.0), [path.name])

    model = build(tmp_path, immutable=True)
    initial = vectors(model, ["bright.mp4"])
    encoder = model.nodes["/clip"].embedder
    encoder.processor.scale = 0.5
    torch.testing.assert_close(vectors(model, ["bright.mp4"]), 0.5 * initial)
    with torch.no_grad():
        visual.weight.mul_(0.5)
    torch.testing.assert_close(vectors(model, ["bright.mp4"]), 0.25 * initial)


def test_single_frame_single_keyframe_nonzero_pts_and_unknown_duration(visual, tmp_path):
    # 1-frame clip sampled with num_frames=4 repeats the single frame cleanly.
    single = tmp_path / "single.mkv"
    write_video(single, solid_frames(1, (180, 90, 45)), fps=10, codec="ffv1", container_format="matroska")
    vec = vectors(build(tmp_path, num_frames=4), [single.name])[0, 0]
    torch.testing.assert_close(vec, torch.tensor([180.0 / 255.0, 90.0 / 255.0, 45.0 / 255.0]), atol=1e-3, rtol=1e-3)

    # Non-zero start PTS matches zero-based clip.
    seq = solid_frames(10, (60, 120, 180)) + solid_frames(10, (200, 100, 50))
    zero_pts = tmp_path / "zero_pts.mkv"
    shifted_pts = tmp_path / "shifted_pts.mkv"
    write_video(zero_pts, seq, fps=10, codec="ffv1", container_format="matroska", pts_offset=0)
    write_video(shifted_pts, seq, fps=10, codec="ffv1", container_format="matroska", pts_offset=5000)
    v_zero = vectors(build(tmp_path, window=1.0, clip="tail"), [zero_pts.name])
    v_shift = vectors(build(tmp_path, window=1.0, clip="tail"), [shifted_pts.name])
    torch.testing.assert_close(v_zero, v_shift, atol=1e-4, rtol=1e-4)

    # Long single-keyframe video (> SEEK_GAP_SECONDS) avoids regressive seek loops.
    long_frames = solid_frames(60, (40, 40, 40)) + solid_frames(60, (200, 200, 200))
    single_gop = tmp_path / "single_gop.mp4"
    write_video(single_gop, long_frames, fps=10, gop_size=500)
    head = vectors(build(tmp_path, window=1.0, clip="head"), [single_gop.name])[0, 0, 0]
    tail = vectors(build(tmp_path, window=1.0, clip="tail"), [single_gop.name])[0, 0, 0]
    assert head < 0.25 and tail > 0.70

    # Raw MPEG-4 elementary stream has no duration metadata -> exercises packet-level scan fallback.
    raw = tmp_path / "no_duration.m4v"
    with av.open(str(raw), "w", format="m4v") as out:
        st = out.add_stream("mpeg4", rate=10)
        st.width, st.height, st.pix_fmt = 16, 16, "yuv420p"
        for frame_arr in solid_frames(10, (40, 40, 40)) + solid_frames(10, (210, 210, 210)):
            for pkt in st.encode(av.VideoFrame.from_ndarray(frame_arr, format="rgb24")):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    with av.open(str(raw)) as probe:
        assert probe.duration is None and probe.streams.video[0].duration is None
    raw_head = vectors(build(tmp_path, window=0.8, clip="head"), [raw.name])[0, 0, 0]
    raw_tail = vectors(build(tmp_path, window=0.8, clip="tail"), [raw.name])[0, 0, 0]
    assert raw_head < 0.25 and raw_tail > 0.70


def test_display_rotation_metadata_normalization(tmp_path):
    # Asymmetric 16x32 frame with a bright red square in top-left (y=0..4, x=0..4).
    base = np.zeros((16, 32, 3), dtype=np.uint8)
    base[:4, :4, 0] = 255
    processor = FrameProcessor()

    for angle, expected_shape, corner in [
        (0, (16, 32), (slice(0, 4), slice(0, 4))),
        (90, (32, 16), (slice(28, 32), slice(0, 4))),
        (180, (16, 32), (slice(12, 16), slice(28, 32))),
        (270, (32, 16), (slice(0, 4), slice(12, 16))),
    ]:
        path = tmp_path / f"rot_{angle}.mp4"
        write_video(path, [base, base], fps=10, rotation=angle)
        prepared = video.clips(
            video.snapshot(str(path), True)[0],
            processor=processor,
            bodies={},
            offset=0.0,
            duration=None,
            window=None,
            stride=None,
            max_windows=1,
            clip="head",
            num_frames=2,
        )
        # Shape is (T, C, H, W)
        assert prepared[0].shape == (2, 3, expected_shape[0], expected_shape[1])
        red = prepared[0][0, 0]
        assert red[corner].mean() > 0.6
        # Opposite corner must stay dark.
        assert red.mean() < 0.15


def test_nested_fields_share_pristine_vectors_and_preserve_holes(visual, tmp_path):
    def recording(**kw):
        return rf.Video(model="tiny", base=str(tmp_path), **kw)

    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        reels=rf.Branch(clip=recording(embed=True)),
        answer=recording(mask=True),
    )
    data = pl.DataFrame(
        {
            "reels": [[{"clip": "bright.mp4"}, None, {"clip": "glöcke.mp4"}], [], None],
            "answer": ["bright.mp4", "glöcke.mp4", None],
        }
    )
    batch = model.encode(data, strata="train")
    hidden = batch["/answer"]
    assert not hidden.content["ids"].any()
    left = model.nodes["/reels/clip"].embedder
    assert left.warm(["bright.mp4", "glöcke.mp4"]) == 2
    target = model.nodes["/answer"].embedder
    encoded = target.encode(hidden.targets["content"], hidden.targets["state"])
    assert encoded[:, 0].count_nonzero() == 6
    reports = model.predict(data)
    assert reports.columns == ["representations"]
    assert [row["parents"] for row in reports["representations"][0]] == [[0], [2]]
    assert left.linear is not target.linear and visual.batches == [2]
    copy = rf.Schema.model_validate_json(model.schema.model_dump_json())
    assert copy.requests["/reels/clip"].model == "tiny"


def test_oversize_descriptor_batch_splits_and_chunked_dictionaries(visual, tmp_path, monkeypatch):
    model = build(tmp_path, resident=0, embed=True)
    store = model.nodes["/clip"].embedder.store
    fetch = store.client.fetch

    def limited(ids, variant):
        if len(ids) > 1:
            raise ValueError("artifact lookup exceeds 32 MiB; reduce batch size or split large artifacts")
        return fetch(ids, variant)

    monkeypatch.setattr(store.client, "fetch", limited)
    chunks = [
        pa.array(["bright.mp4", None]).dictionary_encode(),
        pa.array(["glöcke.mp4", "bright.mp4"]).dictionary_encode(),
    ]
    data = pa.table({"clip": pa.chunked_array(chunks)})
    field = model.encode(data)["/clip"]
    result = model.nodes["/clip"].embedder.encode(field.content, field.state)
    assert result[1, 0].count_nonzero() == 0
    assert torch.equal(result[0, 0], result[3, 0])
    assert visual.batches == [1, 1] and store.stats()["claims"] == 0
    assert model.predict(data.slice(0, 0)).height == 0


def test_deterministic_span_selection_pooling_and_invalid_outputs(visual, tmp_path):
    assert video.spans(2.0, offset=0.0, duration=None, window=None, stride=None, max_windows=4, clip="chunks") == [
        (0.0, 2.0)
    ]
    assert video.spans(5.0, offset=0.0, duration=None, window=1.0, stride=None, max_windows=3, clip="chunks") == [
        (0.0, 1.0),
        (2.0, 3.0),
        (4.0, 5.0),
    ]
    assert video.spans(5.0, offset=1.0, duration=2.0, window=1.0, stride=None, max_windows=1, clip="chunks") == [
        (1.5, 2.5)
    ]
    exact = video.spans(0.6, offset=0.0, duration=None, window=0.2, stride=0.2, max_windows=4, clip="chunks")
    assert len(exact) == 3 and exact[0] == pytest.approx((0.0, 0.2)) and exact[-1] == pytest.approx((0.4, 0.6))
    assert video.sample_times(1.0, 3.0, 4) == pytest.approx([1.25, 1.75, 2.25, 2.75])
    with pytest.raises(ValueError, match="no readable duration"):
        video.spans(0.0, offset=0.0, duration=None, window=1.0, stride=None, max_windows=4, clip="chunks")

    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    outputs = SimpleNamespace(last_hidden_state=hidden, pooler_output=torch.tensor([[7.0, 8.0]]))
    for kind, expected in [
        ("xclip", [[7.0, 8.0]]),
        ("vivit", [[7.0, 8.0]]),
        ("timesformer", [[1.0, 2.0]]),
        ("videomae", [[3.0, 4.0]]),
        ("vjepa2", [[3.0, 4.0]]),
    ]:
        model = SimpleNamespace(config=SimpleNamespace(model_type=kind))
        torch.testing.assert_close(
            video.pool_outputs(outputs, model, video.Pooling.auto, 2, "m"), torch.tensor(expected)
        )
    generic = SimpleNamespace(config=SimpleNamespace(model_type="videomae"))
    torch.testing.assert_close(
        video.pool_outputs(outputs, generic, video.Pooling.cls, 2, "m"), torch.tensor([[1.0, 2.0]])
    )
    torch.testing.assert_close(
        video.pool_outputs(outputs, generic, video.Pooling.mean, 2, "m"), torch.tensor([[3.0, 4.0]])
    )
    torch.testing.assert_close(
        video.pool_outputs(outputs, generic, video.Pooling.pooler, 2, "m"), torch.tensor([[7.0, 8.0]])
    )
    with pytest.raises(ValueError, match="pooler_output"):
        video.pool_outputs(SimpleNamespace(last_hidden_state=hidden), generic, video.Pooling.pooler, 2, "m")
    with pytest.raises(ValueError, match=r"\[batch, tokens, 2\]"):
        video.pool_outputs(SimpleNamespace(last_hidden_state=torch.zeros(1, 2)), generic, video.Pooling.mean, 2, "m")

    embedder = build(tmp_path).nodes["/clip"].embedder
    assert embedder.pool([], torch.device("cpu"), {}).shape == (0, 3)

    # Multi-window chunking strictly respects encoder_batch_size per forward call.
    path = tmp_path / "chunked.mkv"
    write_video(
        path,
        solid_frames(10, (50, 50, 50)) + solid_frames(10, (150, 150, 150)) + solid_frames(10, (230, 230, 230)),
        fps=10,
        codec="ffv1",
        container_format="matroska",
    )
    visual.batches.clear()
    chunked_model = build(
        tmp_path, window=1.0, stride=1.0, max_windows=3, clip="chunks", encoder_batch_size=2, cache=False
    )
    out = vectors(chunked_model, [path.name])
    assert out.shape == (1, 1, 3) and max(visual.batches) <= 2 and sum(visual.batches) == 3

    # Reject a 2D image-only processor returning 4D tensors instead of 5D video tensors.
    def bad_processor(frames, *, return_tensors="np"):
        return {"pixel_values": np.zeros((4, 3, 16, 16), dtype=np.float32)}

    with pytest.raises(ValueError, match=r"expected \(1, 4, C, H, W\)"):
        video.clips(
            video.snapshot(str(path), True)[0],
            processor=bad_processor,
            bodies={},
            offset=0.0,
            duration=None,
            window=None,
            stride=None,
            max_windows=1,
            clip="head",
            num_frames=4,
        )


def test_missing_processor_config_is_rejected(tmp_path):
    empty_dir = tmp_path / "empty_model"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="video_preprocessor_config.json"):
        video.load_processor(str(empty_dir))


@pytest.mark.integration
def test_xclip_videomae_vjepa2_vivit_and_timesformer_encoder_contracts(tmp_path):
    from transformers import (
        TimesformerConfig,
        TimesformerModel,
        VideoMAEConfig,
        VideoMAEModel,
        VivitConfig,
        VivitImageProcessor,
        VivitModel,
        VJEPA2Config,
        VJEPA2Model,
        XCLIPConfig,
        XCLIPModel,
        XCLIPTextConfig,
        XCLIPVisionConfig,
    )
    from transformers.models.videomae.image_processing_pil_videomae import VideoMAEImageProcessorPil

    device = torch.device("cpu")
    clip_4 = [np.full((32, 32, 3), 50 + 40 * i, dtype=np.uint8) for i in range(4)]
    clip_6 = [np.full((32, 32, 3), 30 + 30 * i, dtype=np.uint8) for i in range(6)]
    clip_path = tmp_path / "sample.mkv"
    write_video(clip_path, clip_4, fps=4, codec="ffv1", container_format="matroska")

    # 1. X-CLIP unwraps its video tower (drops text_model) and matches get_video_features().pooler_output.
    xclip_dir = tmp_path / "xclip"
    VideoMAEImageProcessorPil(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}).save_pretrained(
        xclip_dir
    )
    raw_xclip = XCLIPModel(
        XCLIPConfig(
            text_config=XCLIPTextConfig(
                hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, projection_dim=16
            ),
            vision_config=XCLIPVisionConfig(
                image_size=32,
                patch_size=16,
                num_frames=4,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                projection_dim=16,
                mit_hidden_size=16,
                mit_intermediate_size=32,
                mit_num_hidden_layers=1,
                mit_num_attention_heads=2,
            ),
            projection_dim=16,
        )
    ).eval()
    raw_xclip.save_pretrained(xclip_dir)
    xclip_backbone = video.CachedModel.get_model(str(xclip_dir))
    assert xclip_backbone.hidden_size == 16 and not hasattr(xclip_backbone.model, "text_model")
    xclip_proc = video.load_processor(str(xclip_dir))
    prepared = xclip_proc(clip_4, return_tensors="np")["pixel_values"][0]
    with torch.no_grad():
        reference = raw_xclip.get_video_features(pixel_values=torch.from_numpy(prepared[None])).pooler_output
    encoded = video.encode_clips(
        xclip_backbone.model, [prepared], device, pooling=video.Pooling.auto, hidden_size=16, model_name="xclip"
    )
    torch.testing.assert_close(encoded, reference, atol=1e-6, rtol=1e-6)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        bf16 = video.encode_clips(
            xclip_backbone.model, [prepared], device, pooling=video.Pooling.auto, hidden_size=16, model_name="xclip"
        )
    assert bf16.dtype == torch.float32 and torch.isfinite(bf16).all()
    with pytest.raises(ValueError, match=r"/clip.*num_frames=6.*exactly 4"):
        rf.Model(d_model=8, n_layers=1, n_heads=2, clip=rf.Video(model=str(xclip_dir), num_frames=6, cache=False))

    # 2. VideoMAE uses mean pooling in auto mode and rejects wrong num_frames.
    vmae_dir = tmp_path / "videomae"
    VideoMAEImageProcessorPil(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}).save_pretrained(
        vmae_dir
    )
    VideoMAEModel(
        VideoMAEConfig(
            image_size=32,
            patch_size=16,
            num_frames=4,
            tubelet_size=2,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        )
    ).save_pretrained(vmae_dir)
    vmae_model = rf.Model(d_model=8, n_layers=1, n_heads=2, clip=rf.Video(model=str(vmae_dir), cache=False))
    vmae_field = vmae_model.encode(pl.DataFrame({"clip": [str(clip_path)]}))["/clip"]
    vmae_vec = vmae_model.nodes["/clip"].embedder.encode(vmae_field.content, vmae_field.state)
    assert vmae_vec.shape == (1, 1, 16) and torch.isfinite(vmae_vec).all()
    with pytest.raises(ValueError, match=r"num_frames=6.*exactly 4"):
        rf.Model(d_model=8, n_layers=1, n_heads=2, clip=rf.Video(model=str(vmae_dir), num_frames=6, cache=False))

    # 3. V-JEPA 2 unwraps .encoder, loads video_preprocessor_config.json without torchvision, and validates tubelet_size.
    vjepa_dir = tmp_path / "vjepa2"
    vjepa_dir.mkdir()
    (vjepa_dir / "video_preprocessor_config.json").write_text(
        json.dumps(
            {
                "video_processor_type": "VJEPA2VideoProcessor",
                "crop_size": {"height": 32, "width": 32},
                "size": {"shortest_edge": 36},
                "do_resize": True,
                "do_center_crop": True,
                "do_rescale": True,
                "rescale_factor": 1.0 / 255.0,
                "do_normalize": True,
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
                "resample": 2,
            }
        )
    )
    VJEPA2Model(
        VJEPA2Config(
            crop_size=32,
            patch_size=16,
            frames_per_clip=4,
            tubelet_size=2,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            mlp_ratio=2.0,
            pred_hidden_size=16,
            pred_num_hidden_layers=1,
            pred_num_attention_heads=2,
        )
    ).save_pretrained(vjepa_dir)
    vjepa_backbone = video.CachedModel.get_model(str(vjepa_dir))
    assert vjepa_backbone.hidden_size == 16 and not hasattr(vjepa_backbone.model, "predictor")
    # Variable frame count divisible by tubelet_size=2 succeeds.
    vjepa_model = rf.Model(
        d_model=8, n_layers=1, n_heads=2, clip=rf.Video(model=str(vjepa_dir), num_frames=6, cache=False)
    )
    vjepa_field = vjepa_model.encode(pl.DataFrame({"clip": [str(clip_path)]}))["/clip"]
    vjepa_vec = vjepa_model.nodes["/clip"].embedder.encode(vjepa_field.content, vjepa_field.state)
    assert vjepa_vec.shape == (1, 1, 16) and torch.isfinite(vjepa_vec).all()
    with pytest.raises(ValueError, match="multiple of tubelet_size=2"):
        rf.Model(d_model=8, n_layers=1, n_heads=2, clip=rf.Video(model=str(vjepa_dir), num_frames=5, cache=False))

    # 4. ViViT supports auto/pooler/cls/mean pooling and rejects wrong num_frames.
    vivit_dir = tmp_path / "vivit"
    VivitImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}).save_pretrained(vivit_dir)
    VivitModel(
        VivitConfig(
            image_size=32,
            num_frames=4,
            tubelet_size=[2, 16, 16],
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        )
    ).save_pretrained(vivit_dir)
    vivit_backbone = video.CachedModel.get_model(str(vivit_dir))
    vivit_proc = video.load_processor(str(vivit_dir))
    vivit_in = vivit_proc(clip_4, return_tensors="np")["pixel_values"][0]
    for mode in (video.Pooling.auto, video.Pooling.pooler, video.Pooling.cls, video.Pooling.mean):
        out = video.encode_clips(
            vivit_backbone.model, [vivit_in], device, pooling=mode, hidden_size=16, model_name="vivit"
        )
        assert out.shape == (1, 16) and torch.isfinite(out).all()
    with pytest.raises(ValueError, match=r"num_frames=6.*exactly 4"):
        rf.Model(d_model=8, n_layers=1, n_heads=2, clip=rf.Video(model=str(vivit_dir), num_frames=6, cache=False))

    # 5. TimeSformer uses CLS pooling in auto mode and supports variable frame counts.
    ts_dir = tmp_path / "timesformer"
    VideoMAEImageProcessorPil(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}).save_pretrained(ts_dir)
    TimesformerModel(
        TimesformerConfig(
            image_size=32,
            patch_size=16,
            num_frames=4,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        )
    ).save_pretrained(ts_dir)
    ts_backbone = video.CachedModel.get_model(str(ts_dir))
    ts_proc = video.load_processor(str(ts_dir))
    ts_in = ts_proc(clip_6, return_tensors="np")["pixel_values"][0]
    auto_out = video.encode_clips(
        ts_backbone.model, [ts_in], device, pooling=video.Pooling.auto, hidden_size=16, model_name="ts"
    )
    cls_out = video.encode_clips(
        ts_backbone.model, [ts_in], device, pooling=video.Pooling.cls, hidden_size=16, model_name="ts"
    )
    assert torch.equal(auto_out, cls_out)


@pytest.mark.integration
def test_offline_xclip_training_prediction_and_checkpoint_with_loader_worker(tmp_path):
    from transformers import XCLIPConfig, XCLIPModel, XCLIPTextConfig, XCLIPVisionConfig
    from transformers.models.videomae.image_processing_pil_videomae import VideoMAEImageProcessorPil

    assets = tmp_path / "encoder"
    VideoMAEImageProcessorPil(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}).save_pretrained(assets)
    XCLIPModel(
        XCLIPConfig(
            text_config=XCLIPTextConfig(
                hidden_size=12, intermediate_size=24, num_hidden_layers=1, num_attention_heads=3, projection_dim=12
            ),
            vision_config=XCLIPVisionConfig(
                image_size=32,
                patch_size=16,
                num_frames=4,
                hidden_size=12,
                intermediate_size=24,
                num_hidden_layers=1,
                num_attention_heads=3,
                projection_dim=12,
                mit_hidden_size=12,
                mit_intermediate_size=24,
                mit_num_hidden_layers=1,
                mit_num_attention_heads=3,
            ),
            projection_dim=12,
        )
    ).save_pretrained(assets)

    paths = []
    for name, color in (("red.mp4", (220, 40, 40)), ("blue.mkv", (40, 40, 220))):
        path = tmp_path / name
        write_video(
            path,
            solid_frames(8, color, size=(32, 32)),
            fps=8,
            codec="ffv1" if name.endswith(".mkv") else "mpeg4",
            container_format="matroska" if name.endswith(".mkv") else None,
        )
        paths.append(str(path))

    def recording(**options):
        return rf.Video(model=str(assets), store=str(tmp_path / "store"), **options)

    model = rf.Model(d_model=8, n_layers=1, n_heads=2, clip=recording(embed=True), target=recording(mask=True))
    data = pl.DataFrame({"clip": paths, "target": list(reversed(paths))})
    embedder = model.nodes["/clip"].embedder
    before = embedder.linear[0].weight.detach().clone()
    dm = rf.PolarsDataModule(train=data, validate=data, num_workers=1, persistent_workers=False, pin_memory=False)
    model.fit(
        dm,
        batch_size=2,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        logger=False,
        rollback=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    assert not torch.equal(before, embedder.linear[0].weight)
    assert all(p.grad is None for p in video.CachedModel.get_model(str(assets)).model.parameters())
    assert embedder.store.stats()["results"] == 2
    predicted = model.predict(data)
    assert predicted.columns == ["representations"]
    checkpoint = tmp_path / "video.ckpt"
    model.save(checkpoint)
    restored = rf.Model.load(checkpoint)
    assert restored.predict(data).equals(predicted)


@rf.preprocess
def prepare_camera(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.rename({"raw_footage": "footage"})


@rf.postprocess
def attach_alert(frame: pl.DataFrame) -> pl.DataFrame:
    prob = frame["predictions"].struct.field("alert").struct.field("content").struct.field("probability")
    return frame.with_columns((prob >= 0.5).alias("alert_flag"))


@pytest.mark.integration
def test_end_to_end_multiformat_relational_video_pipeline(tmp_path, monkeypatch):
    import lightning.pytorch as lit
    import pyarrow.ipc as ipc
    import yaml
    from fastapi.testclient import TestClient
    from transformers import VideoMAEConfig, VideoMAEModel
    from transformers.models.videomae.image_processing_pil_videomae import VideoMAEImageProcessorPil

    monkeypatch.setenv("RELFLOW_CACHE_DIR", str(tmp_path / "default_cache"))
    lit.seed_everything(505, workers=True)

    encoder_dir = tmp_path / "videomae_encoder"
    VideoMAEImageProcessorPil(
        size={"shortest_edge": 16},
        crop_size={"height": 16, "width": 16},
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ).save_pretrained(encoder_dir)
    # Deterministic VideoMAE backbone whose patch projection separates red vs blue energy
    # and temporal color shifts cleanly across 4-frame clips.
    vmae_init = VideoMAEModel(
        VideoMAEConfig(
            image_size=16,
            patch_size=16,
            num_frames=4,
            tubelet_size=2,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        )
    )
    with torch.no_grad():
        for param in vmae_init.parameters():
            param.zero_()
        proj = vmae_init.embeddings.patch_embeddings.projection
        # proj.weight has shape (16, 3, 2, 16, 16)
        proj.weight[:8, 0] = 0.05  # red channel
        proj.weight[:8, 2] = -0.05  # blue channel
        proj.weight[8:, 2] = 0.05
        proj.weight[8:, 0] = -0.05
        vmae_init.encoder.layer[0].layernorm_before.weight.fill_(1.0)
        vmae_init.encoder.layer[0].layernorm_after.weight.fill_(1.0)
    vmae_init.save_pretrained(encoder_dir)

    media_dir = tmp_path / "videos"
    media_dir.mkdir()
    formats = (
        ("mp4", "mpeg4", None),
        ("h264.mp4", "libx264", None),
        ("webm", "libvpx", "webm"),
        ("mkv", "ffv1", "matroska"),
        ("mov", "mjpeg", "mov"),
    )

    def write_clip(index: int, alert: bool, split: str) -> str:
        ext, codec, fmt = formats[index % len(formats)]
        rng = np.random.default_rng(7000 + index + (500 if split == "test" else 0))
        # Place the distinguishing visual cue in the final third (0.4s..0.6s) at 20 fps (12 frames)
        # so multi-window clip="chunks" (window=0.2s, stride=0.2s, max_windows=3) is required to see it.
        frames: list[np.ndarray] = []
        for idx in range(12):
            noise = rng.integers(-8, 9, size=(16, 16, 3), dtype=np.int16)
            if idx < 8:
                base = np.array([128, 128, 128], dtype=np.int16)
            elif alert:
                base = np.array([235, 60, 30], dtype=np.int16)
            else:
                base = np.array([30, 60, 235], dtype=np.int16)
            frames.append(np.clip(base + noise, 0, 255).astype(np.uint8))
        name = f"{split}_{index:03d}.{ext}"
        write_video(media_dir / name, frames, fps=20, codec=codec, container_format=fmt)
        return name

    def make_split(split: str, count: int) -> pl.DataFrame:
        rows = []
        for idx in range(count):
            alert = bool(idx % 2)
            pattern = (idx // 2) % 3
            primary = write_clip(idx * 2, alert, split)
            followup = write_clip(idx * 2 + 1, alert, split)
            cameras = (
                []
                if pattern == 0
                else [{"clip": None, "angle": "front"}, {"clip": followup, "angle": "side"}]
                if pattern == 1
                else [{"clip": primary, "angle": "front"}, {"clip": followup, "angle": "rear"}]
            )
            rows.append(
                {
                    "incident_id": f"{split}-{idx}",
                    "raw_footage": primary,
                    "cameras": cameras,
                    "alert": alert,
                    "severity": "high" if alert else "normal",
                }
            )
        return pl.DataFrame(rows)

    train_df = make_split("train", 24)
    val_df = make_split("val", 8)
    test_df = make_split("test", 12)

    train_parquet = tmp_path / "train.parquet"
    val_parquet = tmp_path / "val.parquet"
    test_parquet = tmp_path / "test.parquet"
    train_df.write_parquet(train_parquet)
    val_df.write_parquet(val_parquet)
    test_df.write_parquet(test_parquet)

    store_dir = tmp_path / "video_store"

    class Camera(rf.Branch):
        clip = rf.Video(
            model=str(encoder_dir),
            base=str(media_dir),
            store=str(store_dir),
            window=0.2,
            stride=0.2,
            max_windows=3,
            clip="chunks",
        )
        angle = rf.Category

    class IncidentModel(rf.Model):
        footage = rf.Video(
            model=str(encoder_dir),
            base=str(media_dir),
            store=str(store_dir),
            window=0.2,
            stride=0.2,
            max_windows=3,
            clip="chunks",
            embed=True,
        )
        cameras = Camera(length=3, overflow="tail")
        alert = rf.Boolean(mask=True)
        severity = rf.Enum("normal", "high", mask=True)

    model = IncidentModel(d_model=32, n_layers=2, n_heads=4)
    backbone = video.CachedModel.get_model(str(encoder_dir)).model
    initial_backbone = {name: param.detach().clone() for name, param in backbone.named_parameters()}

    # Pre-warm from a Polars Series and verify duplicate calls add zero new cache entries.
    warmed = model.nodes["/footage"].embedder.warm(train_df["raw_footage"])
    assert warmed == len(train_df)
    assert model.nodes["/footage"].embedder.warm(train_df["raw_footage"]) == 0

    dm = rf.ArrowDataModule(
        model=model,
        train=str(train_parquet),
        validate=str(val_parquet),
        test=str(test_parquet),
        preprocessor=prepare_camera,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    model.fit(
        dm,
        batch_size=8,
        max_epochs=32,
        learning_rate=4e-3,
        early_stopping=False,
        rollback=False,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )

    for name, param in backbone.named_parameters():
        assert param.grad is None
        assert torch.equal(param, initial_backbone[name])

    val_metrics = model.validate(dm, accelerator="cpu", devices=1, logger=False, enable_progress_bar=False)
    test_metrics = model.test(dm, accelerator="cpu", devices=1, logger=False, enable_progress_bar=False)
    assert val_metrics and test_metrics

    unseen_inputs = test_df.select("incident_id", "raw_footage", "cameras")
    whole = model.predict(unseen_inputs, preprocess=prepare_camera, postprocess=attach_alert, retain=("incident_id",))
    chunked = model.predict(
        unseen_inputs,
        batch_size=3,
        preprocess=prepare_camera,
        postprocess=attach_alert,
        retain=("incident_id",),
    )
    assert set(whole.columns) == {"inputs", "predictions", "representations", "alert_flag"}
    assert whole["alert_flag"].to_list() == chunked["alert_flag"].to_list()

    truth = test_df["alert"].to_numpy()
    alert_pred = whole["alert_flag"].to_numpy()
    sev_pred = whole["predictions"].struct.field("severity").struct.field("content").struct.field("value").to_numpy()
    assert np.mean(alert_pred == truth) >= 0.85
    assert np.mean((sev_pred == "high") == truth) >= 0.85

    # Streaming Parquet batch Writer parity.
    written_dir = tmp_path / "batch_output"
    pred_dm = rf.PolarsDataModule(
        model=model,
        predict=unseen_inputs,
        preprocessor=prepare_camera,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    lit.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        callbacks=[rf.Writer(str(written_dir))],
    ).predict(model, pred_dm, return_predictions=False)
    written_table = pl.read_parquet(written_dir / "*.parquet")
    assert written_table.height == len(test_df)

    # Checkpoint save, inspect, preflight check, and reload into a cold store.
    ckpt_path = tmp_path / "incident.rf"
    model.save(ckpt_path)
    inspected = rf.inspect(ckpt_path)
    assert inspected["command"] == "inspect"
    checked = rf.check(
        ckpt_path,
        data={"train": str(train_parquet)},
        preprocessor=prepare_camera,
        execute=True,
        sample=8,
    )
    assert all(phase["status"] == "passed" for phase in checked["phases"])

    restored = rf.Model.load(ckpt_path)
    restored.update(rf.where("type") == "video", store=str(tmp_path / "cold_store"))
    reloaded = restored.predict(
        unseen_inputs, preprocess=prepare_camera, postprocess=attach_alert, retain=("incident_id",)
    )
    assert reloaded["alert_flag"].to_list() == whole["alert_flag"].to_list()

    # YAML schema declaration loading.
    yaml_path = tmp_path / "schema.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "d_model": 16,
                "n_layers": 1,
                "n_heads": 2,
                "fields": {
                    "footage": {
                        "type": "video",
                        "model": str(encoder_dir),
                        "base": str(media_dir),
                        "store": str(tmp_path / "yaml_store"),
                        "window": 0.2,
                        "clip": "tail",
                    },
                    "alert": {"type": "boolean", "mask": True},
                },
            }
        )
    )
    yaml_model = rf.Model.from_yaml(yaml_path)
    assert yaml_model.predict(unseen_inputs.select("raw_footage"), preprocess=prepare_camera).height == len(test_df)

    # Live FastAPI ASGI serving over both JSON and Arrow IPC streams.
    deployment = (
        rf.Deployment(model=restored, accelerator="cpu", retain=("incident_id",))
        .preprocess(prepare_camera)
        .postprocess(attach_alert)
    )
    with TestClient(deployment.app()) as client:
        assert client.get("/health").status_code == 200
        json_rows = unseen_inputs.head(2).to_dicts()
        r_json = client.post("/predict", json=json_rows)
        assert r_json.status_code == 200, r_json.text
        served_flags = [row["alert_flag"] for row in r_json.json()]
        assert served_flags == whole["alert_flag"].head(2).to_list()

        ipc_frame = unseen_inputs.slice(2, 3).to_arrow()
        sink = io.BytesIO()
        with ipc.new_stream(sink, ipc_frame.schema) as writer:
            writer.write_table(ipc_frame)
        r_ipc = client.post(
            "/predict",
            content=sink.getvalue(),
            headers={
                "content-type": "application/vnd.apache.arrow.stream",
                "accept": "application/vnd.apache.arrow.stream",
            },
        )
        assert r_ipc.status_code == 200
        with ipc.open_stream(io.BytesIO(r_ipc.content)) as reader:
            ipc_out = pl.from_arrow(reader.read_all())
        assert ipc_out["alert_flag"].to_list() == whole["alert_flag"].slice(2, 3).to_list()
