import ast
import io
import os
import threading
from collections import Counter
from functools import partial
from types import SimpleNamespace

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
import soundfile as sf
import torch
from pyarrow import fs

import relflow as rf
from relflow.tensorfields.extensions import audio
from relflow.tensorfields.shared.frozen import Backbone


class Extractor:
    sampling_rate = 16000
    model_input_names = ["input_values", "attention_mask"]

    def __init__(self):
        self.scale = 1.0

    def to_json_string(self):
        return str(self.scale)

    def __call__(self, waveforms, *, sampling_rate, return_tensors="np", **kwargs):
        assert sampling_rate == self.sampling_rate and return_tensors == "np"
        width = max(len(row) for row in waveforms)
        values = np.zeros((len(waveforms), width), dtype=np.float32)
        mask = np.zeros((len(waveforms), width), dtype=np.int64)
        for index, row in enumerate(waveforms):
            values[index, : len(row)] = row * self.scale
            mask[index, : len(row)] = 1
        return {"input_values": values, "attention_mask": mask}


class Acoustic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=3, model_type="custom")
        self.batches = []
        self.weight = torch.nn.Parameter(torch.ones(3), requires_grad=False)

    def forward(self, *, input_values, attention_mask=None):
        self.batches.append(len(input_values))
        if attention_mask is None:
            mean = input_values.mean(dim=-1)
            energy = input_values.square().mean(dim=-1)
            peak = input_values.abs().amax(dim=-1)
        else:
            valid = attention_mask.to(dtype=input_values.dtype)
            denom = valid.sum(dim=-1).clamp_min(1.0)
            mean = (input_values * valid).sum(dim=-1) / denom
            energy = (input_values.square() * valid).sum(dim=-1) / denom
            peak = (input_values.abs() * valid).amax(dim=-1)
        features = torch.stack([mean, energy, peak], dim=-1) * self.weight
        return SimpleNamespace(last_hidden_state=features.unsqueeze(1))


def write_wave(path, samples, rate=16000):
    sf.write(path, np.asarray(samples, dtype=np.float32), rate)


@pytest.fixture
def acoustic(monkeypatch, tmp_path):
    extractor = Extractor()
    model = Acoustic()
    monkeypatch.setattr(Backbone, "_models", {"tiny": Backbone("tiny", model, 3)})
    from transformers import AutoFeatureExtractor

    monkeypatch.setattr(AutoFeatureExtractor, "from_pretrained", lambda *args, **kwargs: extractor)
    monkeypatch.setenv("RELFLOW_CACHE_DIR", str(tmp_path / "cache"))
    write_wave(tmp_path / "positive.wav", np.full(1600, 0.5, dtype=np.float32))
    write_wave(tmp_path / "glöcke.flac", np.full(1600, -0.25, dtype=np.float32))
    return model


def build(base, **options):
    return rf.Model(d_model=8, n_heads=2, n_layers=1, clip=rf.Audio(model="tiny", base=str(base), **options))


def vectors(model, paths):
    field = model.encode(pl.DataFrame({"clip": paths}, schema={"clip": pl.String}))["/clip"]
    return model.nodes["/clip"].embedder.encode(field.content, field.state)


@pytest.mark.parametrize("resident", [0, 1])
def test_concurrent_reads_compact_ids_and_warm_parity(acoustic, tmp_path, monkeypatch, resident):
    model = build(tmp_path, immutable=True, read_workers=2, encoder_batch_size=2, resident=resident)
    paths = ["glöcke.flac", None, "positive.wav", "glöcke.flac"]
    data = pl.DataFrame({"clip": paths})
    assert model.encode(data)["/clip"].content["ids"].numel() == len(paths)
    reads = Counter()
    barrier = threading.Barrier(2)
    original = audio.open_audio

    def concurrent(path):
        reads[path] += 1
        barrier.wait(timeout=10)
        return original(path)

    monkeypatch.setattr(audio, "open_audio", concurrent)
    first = vectors(model, paths)
    assert sorted(reads.values()) == [1, 1] and acoustic.batches == [2]
    expected_pos = [0.5, 0.25, 0.5]
    expected_neg = [-0.25, 0.0625, 0.25]
    torch.testing.assert_close(
        first[:, 0],
        torch.tensor([expected_neg, [0.0, 0.0, 0.0], expected_pos, expected_neg]),
        atol=1e-4,
        rtol=1e-4,
    )
    monkeypatch.setattr(audio, "open_audio", lambda *args: pytest.fail("warm hit opened an audio file"))
    assert torch.equal(vectors(model, paths), first)
    assert model.nodes["/clip"].embedder.warm(pa.array(paths)) == 0
    assert model.nodes["/clip"].embedder.warm("positive.wav") == 0
    assert acoustic.batches == [2]
    monkeypatch.setattr(audio, "open_audio", original)
    direct = build(tmp_path, cache=False, encoder_batch_size=1)
    torch.testing.assert_close(vectors(direct, paths), first)
    assert acoustic.batches == [2, 1, 1]
    with pytest.raises(ValueError, match="cache=True"):
        direct.nodes["/clip"].embedder.warm(paths)


def test_mutable_reference_changes_and_failed_reads_can_retry(acoustic, tmp_path):
    model = build(tmp_path)
    path = tmp_path / "positive.wav"
    assert audio.location(path.name, tmp_path.as_uri()) == str(path)
    root = tmp_path.parents[-1]
    assert audio.location("call.wav", root.as_uri()) == str(root / "call.wav")
    first = vectors(model, [path.as_uri()])
    write_wave(tmp_path / "replacement.wav", np.full(1600, -0.5, dtype=np.float32))
    os.replace(tmp_path / "replacement.wav", path)
    assert not torch.equal(first, vectors(model, [str(path)]))

    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not-audio")
    with pytest.raises((OSError, RuntimeError)) as error:
        vectors(model, [str(broken)])
    assert any("/clip" in note for note in error.value.__notes__)
    assert model.nodes["/clip"].embedder.store.stats()["claims"] == 0
    write_wave(broken, np.full(800, 0.2, dtype=np.float32))
    assert vectors(model, [str(broken)]).count_nonzero() == 3

    with pytest.raises(FileNotFoundError, match="missing.wav"):
        vectors(model, ["missing.wav"])
    with pytest.raises(ValueError, match="nonempty audio path"):
        vectors(model, [""])
    with pytest.raises(ValueError, match="numbers, not booleans"):
        rf.Audio(offset=True)  # type: ignore[arg-type]


def test_snapshot_cannot_publish_replaced_file_under_old_id(acoustic, tmp_path):
    model = build(tmp_path)
    field = model.encode(pl.DataFrame({"clip": ["positive.wav"]}))["/clip"]
    write_wave(tmp_path / "other.wav", np.full(1600, -0.1, dtype=np.float32))
    os.replace(tmp_path / "other.wav", tmp_path / "positive.wav")
    encoder = model.nodes["/clip"].embedder
    with pytest.raises(ValueError, match="changed after tensorization"):
        encoder.encode(field.content, field.state)
    assert encoder.store.stats()["results"] == 0


@pytest.mark.parametrize("immutable", [False, True])
def test_remote_snapshot_reads_body_once_and_skips_decode_on_hits(acoustic, tmp_path, monkeypatch, immutable):
    storage = fs.SubTreeFileSystem(str(tmp_path), fs.LocalFileSystem())
    monkeypatch.setattr(audio, "remote", lambda authority: (storage, ""))
    opens = []
    decodes = []
    open_audio, waveforms = audio.open_audio, audio.waveforms

    def opened(path):
        opens.append(path)
        return open_audio(path)

    def decoded(*args, **kwargs):
        decodes.append(1)
        return waveforms(*args, **kwargs)

    monkeypatch.setattr(audio, "open_audio", opened)
    monkeypatch.setattr(audio, "waveforms", decoded)
    model = build("s3://fixtures", immutable=immutable)
    first = vectors(model, ["positive.wav", "positive.wav"])
    assert len(opens) == len(decodes) == 1
    second = vectors(model, ["positive.wav"])
    assert torch.equal(first[0], second[0])
    assert len(opens) == (1 if immutable else 2)
    assert len(decodes) == 1 and acoustic.batches == [1]
    if not immutable:
        write_wave(tmp_path / "next.wav", np.full(1600, 0.125, dtype=np.float32))
        os.replace(tmp_path / "next.wav", tmp_path / "positive.wav")
        assert not torch.equal(first[0], vectors(model, ["positive.wav"])[0])
        assert len(opens) == 3 and len(decodes) == 2


def test_channels_resampling_windowing_and_cache_invalidation(acoustic, tmp_path):
    stereo = np.column_stack(
        [
            np.concatenate([np.full(4800, 0.2), np.full(4800, 0.4), np.full(4800, 0.8)]),
            np.concatenate([np.full(4800, 0.6), np.full(4800, 0.4), np.full(4800, 0.0)]),
        ]
    ).astype(np.float32)
    path = tmp_path / "segmented_48k.flac"
    sf.write(path, stereo, 48000, subtype="PCM_24")

    left = vectors(build(tmp_path, channels=0, window=0.1, clip="head"), [path.name])[0, 0]
    right = vectors(build(tmp_path, channels=1, window=0.1, clip="head"), [path.name])[0, 0]
    mixed = vectors(build(tmp_path, channels="mean", window=0.1, clip="head"), [path.name])[0, 0]
    assert left[0] < mixed[0] < right[0]
    with pytest.raises(ValueError, match="cannot select channel index 2"):
        vectors(build(tmp_path, channels=2), [path.name])

    # Segment means in channel 0 over 0.1s windows are 0.2, 0.4, and 0.8.
    head = vectors(build(tmp_path, channels=0, window=0.1, clip="head"), [path.name])[0, 0, 0]
    center = vectors(build(tmp_path, channels=0, window=0.1, clip="center"), [path.name])[0, 0, 0]
    tail = vectors(build(tmp_path, channels=0, window=0.1, clip="tail", max_windows=1), [path.name])[0, 0, 0]
    chunks = vectors(build(tmp_path, channels=0, window=0.1, stride=0.1, max_windows=3, clip="chunks"), [path.name])[
        0, 0, 0
    ]
    sliced = vectors(build(tmp_path, channels=0, offset=0.2, duration=0.1, window=0.1), [path.name])[0, 0, 0]
    torch.testing.assert_close(head, torch.tensor(0.2), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(center, torch.tensor(0.4), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(tail, torch.tensor(0.8), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(sliced, tail, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(chunks, (head + center + tail) / 3.0, atol=1e-2, rtol=1e-2)

    with pytest.raises(ValueError, match="no samples after offset"):
        vectors(build(tmp_path, offset=1.0), [path.name])

    model = build(tmp_path, immutable=True)
    initial = vectors(model, ["positive.wav"])
    encoder = model.nodes["/clip"].embedder
    encoder.extractor.scale = 0.5
    torch.testing.assert_close(vectors(model, ["positive.wav"])[:, :, 0], 0.5 * initial[:, :, 0])
    with torch.no_grad():
        acoustic.weight.mul_(0.5)
    torch.testing.assert_close(vectors(model, ["positive.wav"])[:, :, 0], 0.25 * initial[:, :, 0])


def test_nested_fields_share_pristine_vectors_and_preserve_holes(acoustic, tmp_path):
    def recording(**kw):
        return rf.Audio(model="tiny", base=str(tmp_path), **kw)

    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        calls=rf.Branch(clip=recording(embed=True)),
        answer=recording(predict=True),
    )
    data = pl.DataFrame(
        {
            "calls": [[{"clip": "positive.wav"}, None, {"clip": "glöcke.flac"}], [], None],
            "answer": ["positive.wav", "glöcke.flac", None],
        }
    )
    batch = model.encode(data, strata="train")
    hidden = batch["/answer"]
    assert not hidden.content["ids"].any()
    left = model.nodes["/calls/clip"].embedder
    assert left.warm(["positive.wav", "glöcke.flac"]) == 2
    target = model.nodes["/answer"].embedder
    encoded = target.encode(hidden.targets["content"], hidden.targets["state"])
    assert encoded[:, 0].count_nonzero() == 6
    reports = model.predict(data)
    assert reports.columns == ["representations"]
    assert [row["parents"] for row in reports["representations"][0]] == [[0], [2]]
    assert left.linear is not target.linear and acoustic.batches == [2]
    copy = rf.Schema.model_validate_json(model.schema.model_dump_json())
    assert copy.requests["/calls/clip"].model == "tiny"


def test_oversize_descriptor_batch_splits_and_chunked_dictionaries(acoustic, tmp_path, monkeypatch):
    model = build(tmp_path, resident=0, embed=True)
    store = model.nodes["/clip"].embedder.store
    fetch = store.client.fetch

    def limited(ids, variant):
        if len(ids) > 1:
            raise ValueError("artifact lookup exceeds 32 MiB; reduce batch size or split large artifacts")
        return fetch(ids, variant)

    monkeypatch.setattr(store.client, "fetch", limited)
    chunks = [
        pa.array(["positive.wav", None]).dictionary_encode(),
        pa.array(["glöcke.flac", "positive.wav"]).dictionary_encode(),
    ]
    data = pa.table({"clip": pa.chunked_array(chunks)})
    field = model.encode(data)["/clip"]
    result = model.nodes["/clip"].embedder.encode(field.content, field.state)
    assert result[1, 0].count_nonzero() == 0
    assert torch.equal(result[0, 0], result[3, 0])
    assert acoustic.batches == [1, 1] and store.stats()["claims"] == 0
    assert model.predict(data.slice(0, 0)).height == 0


def test_capped_span_planning(monkeypatch):
    def bounded_range(*args):
        hops = range(*args)
        assert len(hops) <= 4
        return hops

    monkeypatch.setattr(audio, "range", bounded_range, raising=False)
    plan = partial(audio.spans, 3600 * 16000, 16000, offset=0, window=1, clip="chunks", max_windows=4)
    assert plan(duration=None, stride=0.0001) == [
        (0, 16000),
        (19194667, 19210667),
        (38389333, 38405333),
        (57584000, 57600000),
    ]
    assert plan(duration=2, stride=0.4) == [(0, 16000), (6400, 22400), (12800, 28800), (16000, 32000)]


def test_legacy_tail_vectors_are_recomputed(acoustic, tmp_path, monkeypatch):
    model = build(tmp_path, window=0.025, clip="tail", max_windows=1, resident=0)
    embedder = model.nodes["/clip"].embedder
    backbone = embedder._cached_model
    variant = backbone.variant

    def legacy(device, processing):
        settings = ast.literal_eval(processing)
        assert settings[-1] == "tail/v2"
        return variant(device, repr(settings[:-1]))

    with monkeypatch.context() as previous:
        previous.setattr(backbone, "variant", legacy)
        vectors(model, ["positive.wav"])
    vectors(model, ["positive.wav"])
    vectors(model, ["positive.wav"])
    assert acoustic.batches == [1, 1]


@pytest.mark.parametrize("access", ["seekable", "stream", "unsupported"])
def test_remote_window_reads(acoustic, tmp_path, monkeypatch, access):
    import json

    path = tmp_path / "long.wav"
    write_wave(path, np.linspace(-0.5, 0.5, 1_600_000, dtype=np.float32))
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

    monkeypatch.setattr(audio, "remote", lambda authority: (Storage(), ""))
    opts = dict(
        target_rate=16000, offset=0, duration=None, window=1, stride=None, max_windows=1, clip="tail", channels="mean"
    )
    remote = audio.waveforms(json.dumps(("s3://bucket/long.wav", None)).encode(), {}, **opts)
    local = audio.waveforms(audio.snapshot(str(path), True)[0], {}, **opts)
    np.testing.assert_array_equal(remote, local)
    if access == "seekable":
        assert 0 < read["bytes"] < path.stat().st_size // 2 and read["full"] == 0
    else:
        assert read["full"] == 1


def test_deterministic_span_selection_and_pooling_modes():
    assert audio.spans(
        16000, 16000, offset=0.0, duration=None, window=2.0, stride=None, max_windows=4, clip="chunks"
    ) == [(0, 16000)]
    chunks = audio.spans(80000, 16000, offset=0.0, duration=None, window=1.0, stride=None, max_windows=3, clip="chunks")
    assert chunks == [(0, 16000), (32000, 48000), (64000, 80000)]
    assert audio.spans(
        80000, 16000, offset=1.0, duration=2.0, window=1.0, stride=None, max_windows=1, clip="chunks"
    ) == [(24000, 40000)]
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    outputs = SimpleNamespace(last_hidden_state=hidden, pooler_output=torch.tensor([[7.0, 8.0]]))
    model = SimpleNamespace(config=SimpleNamespace(model_type="whisper"))
    torch.testing.assert_close(
        audio.pool_outputs(outputs, model, mask, audio.Pooling.auto, 2, "m"), torch.tensor([[2.0, 3.0]])
    )
    torch.testing.assert_close(
        audio.pool_outputs(outputs, model, mask, audio.Pooling.cls, 2, "m"), torch.tensor([[1.0, 2.0]])
    )
    torch.testing.assert_close(
        audio.pool_outputs(outputs, model, mask, audio.Pooling.pooler, 2, "m"), torch.tensor([[7.0, 8.0]])
    )
    zero_mask = torch.zeros((1, 3), dtype=torch.int64)
    torch.testing.assert_close(
        audio.pool_outputs(outputs, model, zero_mask, audio.Pooling.mean, 2, "m"), torch.tensor([[1.0, 2.0]])
    )
    assert (
        audio.minimum_samples(
            SimpleNamespace(conv_kernel=(10, 3, 3, 3, 3, 2, 2), conv_stride=(5, 2, 2, 2, 2, 2, 2)), 16000
        )
        == 400
    )


def test_empty_audio_pool(acoustic, tmp_path):
    embedder = build(tmp_path).nodes["/clip"].embedder
    assert embedder.pool([], torch.device("cpu"), {}).shape == (0, 3)


@pytest.fixture
def streaming_flac(tmp_path):
    rate = 8000
    t = np.arange(150_001) / rate
    samples = np.column_stack((np.sin(140 * t), np.cos(310 * t)))
    known = tmp_path / "known.flac"
    sf.write(known, samples, rate, subtype="PCM_16")
    data = bytearray(known.read_bytes())
    # In FLAC STREAMINFO a zero 36-bit total sample count means the length is unknown.
    assert data[:4] == b"fLaC" and int.from_bytes(data[18:26], "big") % (1 << 36) == len(t)
    data[21] &= 0xF0
    data[22:26] = bytes(4)
    unknown = tmp_path / "streaming.flac"
    unknown.write_bytes(data)
    return known, unknown


@pytest.mark.parametrize("clip", ["head", "center", "tail", "chunks"])
@pytest.mark.parametrize("offset,duration,channels", [(0, None, "mean"), (0.125125, 3.25, 1), (18.74, 2.0, 0)])
def test_streaming_flac_windows_match_known_length(streaming_flac, clip, offset, duration, channels):
    decode = partial(
        audio.waveforms,
        bodies={},
        target_rate=16000,
        offset=offset,
        duration=duration,
        window=0.2,
        stride=None,
        max_windows=3,
        clip=clip,
        channels=channels,
    )
    known, unknown = (decode(audio.snapshot(str(path), True)[0]) for path in streaming_flac)
    np.testing.assert_array_equal(unknown, known)


@pytest.mark.parametrize("clip,duration,stop", [("head", None, 8800), ("head", 0.05, 8400), ("tail", 0.4, 11200)])
def test_streaming_flac_stops_at_requested_frame(streaming_flac, monkeypatch, clip, duration, stop):
    read = sf.SoundFile.read
    consumed = 0

    def bounded(source, frames, **kwargs):
        nonlocal consumed
        assert 0 < frames <= stop - consumed
        block = read(source, frames, **kwargs)
        consumed += len(block)
        return block

    monkeypatch.setattr(sf.SoundFile, "read", bounded)
    audio.waveforms(
        audio.snapshot(str(streaming_flac[1]), True)[0],
        {},
        target_rate=8000,
        offset=1,
        duration=duration,
        window=0.1,
        stride=None,
        max_windows=2,
        clip=clip,
        channels="mean",
    )
    assert consumed == stop


def test_streaming_flac_offset_past_end(streaming_flac):
    record = audio.snapshot(str(streaming_flac[1]), True)[0]
    with pytest.raises(ValueError, match=r"no samples after offset=20s .*18.750s"):
        audio.waveforms(
            record,
            {},
            target_rate=8000,
            offset=20,
            duration=None,
            window=1,
            stride=None,
            max_windows=2,
            clip="chunks",
            channels="mean",
        )


@pytest.mark.integration
def test_whisper_hubert_ast_and_clap_encoder_contracts(tmp_path):
    from transformers import (
        ASTConfig,
        ASTFeatureExtractor,
        ASTModel,
        ClapAudioConfig,
        ClapAudioModelWithProjection,
        ClapFeatureExtractor,
        HubertConfig,
        HubertModel,
        Wav2Vec2FeatureExtractor,
        WhisperConfig,
        WhisperFeatureExtractor,
        WhisperModel,
    )

    device = torch.device("cpu")
    impulse = np.full(50, 0.25, dtype=np.float32)
    short = np.sin(np.linspace(0.0, 20.0, 8000, dtype=np.float32))
    longer = np.cos(np.linspace(0.0, 60.0, 24000, dtype=np.float32))

    # Whisper unwraps the encoder and pools over valid frames rather than 30s padding.
    whisper_dir = tmp_path / "whisper"
    WhisperFeatureExtractor(chunk_length=4, n_samples=64000, nb_max_frames=400, feature_size=32).save_pretrained(
        whisper_dir
    )
    WhisperModel(
        WhisperConfig(
            d_model=16,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_layers=1,
            decoder_attention_heads=2,
            decoder_ffn_dim=32,
            num_mel_bins=32,
            max_source_positions=200,
        )
    ).save_pretrained(whisper_dir)
    whisper_backbone = audio.CachedModel.get_model(str(whisper_dir))
    assert whisper_backbone.hidden_size == 16 and not hasattr(whisper_backbone.model, "decoder")
    whisper_fe = WhisperFeatureExtractor.from_pretrained(whisper_dir)
    assert audio.native_window(whisper_fe) == 4.0
    for window in (None, 4):
        rf.Model(d_model=8, n_layers=1, n_heads=2, clip=rf.Audio(model=str(whisper_dir), window=window, cache=False))
    with pytest.raises(ValueError, match=r"/clip.*4.1s.*up to 4s.*chunks"):
        rf.Model(d_model=8, n_layers=1, n_heads=2, clip=rf.Audio(model=str(whisper_dir), window=4.1, cache=False))
    whisper_out = audio.encode_clips(
        whisper_backbone.model,
        whisper_fe,
        [short, longer],
        device,
        rate=16000,
        pooling=audio.Pooling.auto,
        hidden_size=16,
        model_name="whisper",
    )
    solo_short = audio.encode_clips(
        whisper_backbone.model,
        whisper_fe,
        [short],
        device,
        rate=16000,
        pooling=audio.Pooling.auto,
        hidden_size=16,
        model_name="whisper",
    )
    torch.testing.assert_close(whisper_out[0], solo_short[0], atol=1e-6, rtol=1e-6)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        bf16_out = audio.encode_clips(
            whisper_backbone.model,
            whisper_fe,
            [short],
            device,
            rate=16000,
            pooling=audio.Pooling.auto,
            hidden_size=16,
            model_name="whisper",
        )
    assert bf16_out.dtype == torch.float32 and torch.isfinite(bf16_out).all()

    # Group-normalized waveform models bucket unequal lengths so padding cannot alter embeddings.
    hubert = HubertModel(
        HubertConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            conv_dim=(16, 16, 16, 16, 16, 16, 16),
            feat_extract_norm="group",
        )
    ).eval()
    wav_fe = Wav2Vec2FeatureExtractor(return_attention_mask=True)
    batched = audio.encode_clips(
        hubert, wav_fe, [short, longer], device, rate=16000, pooling=audio.Pooling.auto, hidden_size=16, model_name="h"
    )
    solo = audio.encode_clips(
        hubert, wav_fe, [short], device, rate=16000, pooling=audio.Pooling.auto, hidden_size=16, model_name="h"
    )
    torch.testing.assert_close(batched[0], solo[0], atol=1e-6, rtol=1e-6)
    impulse_hubert = audio.encode_clips(
        hubert, wav_fe, [impulse], device, rate=16000, pooling=audio.Pooling.auto, hidden_size=16, model_name="h"
    )
    assert impulse_hubert.shape == (1, 16) and impulse_hubert.abs().sum() > 0

    # AST supports pooler_output in auto/pooler modes and token pooling in cls/mean modes.
    ast = ASTModel(
        ASTConfig(
            max_length=64,
            num_mel_bins=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        )
    ).eval()
    ast_fe = ASTFeatureExtractor(max_length=64, num_mel_bins=32)
    assert audio.native_window(ast_fe) == pytest.approx(0.64)
    for mode in (audio.Pooling.auto, audio.Pooling.pooler, audio.Pooling.cls, audio.Pooling.mean):
        out = audio.encode_clips(
            ast, ast_fe, [impulse], device, rate=16000, pooling=mode, hidden_size=16, model_name="a"
        )
        assert out.shape == (1, 16) and torch.isfinite(out).all()

    # CLAP unwraps audio_model and avoids random truncation on overlong clips.
    clap_dir = tmp_path / "clap"
    ClapFeatureExtractor(
        sampling_rate=48000, max_length_s=1, nb_max_samples=48000, truncation="rand_trunc"
    ).save_pretrained(clap_dir)
    ClapAudioModelWithProjection(
        ClapAudioConfig(
            hidden_size=32,
            depths=[1, 1, 1, 1],
            num_attention_heads=[2, 2, 2, 2],
            patch_embeds_hidden_size=4,
            enable_fusion=False,
        )
    ).save_pretrained(clap_dir)
    clap_backbone = audio.CachedModel.get_model(str(clap_dir))
    clap_fe = ClapFeatureExtractor.from_pretrained(clap_dir)
    overlong = np.linspace(-1.0, 1.0, 72000, dtype=np.float32)
    first_clap = audio.encode_clips(
        clap_backbone.model,
        clap_fe,
        [overlong],
        device,
        rate=48000,
        pooling=audio.Pooling.auto,
        hidden_size=32,
        model_name="clap",
    )
    second_clap = audio.encode_clips(
        clap_backbone.model,
        clap_fe,
        [overlong],
        device,
        rate=48000,
        pooling=audio.Pooling.auto,
        hidden_size=32,
        model_name="clap",
    )
    assert torch.equal(first_clap, second_clap)


@pytest.mark.integration
def test_fused_clap_bounded_windows_ignore_batch_order_and_rng(monkeypatch):
    from transformers import ClapAudioConfig, ClapAudioModelWithProjection, ClapFeatureExtractor

    model = ClapAudioModelWithProjection(
        ClapAudioConfig(
            hidden_size=32, depths=[1] * 4, num_attention_heads=[2] * 4, patch_embeds_hidden_size=4, enable_fusion=True
        )
    ).audio_model.eval()
    fe = ClapFeatureExtractor(sampling_rate=48000, max_length_s=1, truncation="fusion")
    encode = partial(
        audio.encode_clips,
        model,
        fe,
        device=torch.device("cpu"),
        rate=48000,
        pooling=audio.Pooling.auto,
        hidden_size=32,
        model_name="fused",
    )
    short = np.sin(np.arange(24000) * 0.1).astype(np.float32)
    full = np.cos(np.arange(48000) * 0.7).astype(np.float32)
    monkeypatch.setattr(np.random, "randint", lambda *args: pytest.fail("random selection for bounded CLAP windows"))
    first = encode([short, short, full])
    torch.testing.assert_close(first[0], first[1])
    torch.testing.assert_close(first[0], encode([short])[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(first, encode([full, short, short]).flip([0]), atol=1e-5, rtol=1e-5)
    assert not torch.allclose(first[0], first[2])
    with pytest.raises(ValueError, match="48000 samples"):
        encode([np.concatenate([short, full])])


@pytest.mark.integration
def test_offline_whisper_training_prediction_and_checkpoint_with_loader_worker(tmp_path):
    from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperModel

    assets = tmp_path / "encoder"
    WhisperModel(
        WhisperConfig(
            d_model=12,
            encoder_layers=1,
            encoder_attention_heads=3,
            encoder_ffn_dim=24,
            decoder_layers=1,
            decoder_attention_heads=3,
            decoder_ffn_dim=24,
            num_mel_bins=32,
            max_source_positions=100,
        )
    ).save_pretrained(assets)
    WhisperFeatureExtractor(chunk_length=2, n_samples=32000, nb_max_frames=200, feature_size=32).save_pretrained(assets)

    paths = []
    for name, freq, rate in (("low.wav", 220.0, 16000), ("high.flac", 880.0, 24000)):
        path = tmp_path / name
        grid = np.linspace(0.0, 0.5, int(rate * 0.5), endpoint=False, dtype=np.float32)
        stereo = np.column_stack([np.sin(2 * np.pi * freq * grid), np.cos(2 * np.pi * freq * grid)])
        sf.write(path, stereo, rate)
        paths.append(str(path))

    def recording(**options):
        return rf.Audio(model=str(assets), store=str(tmp_path / "store"), **options)

    model = rf.Model(d_model=8, n_layers=1, n_heads=2, clip=recording(embed=True), target=recording(predict=True))
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
    assert all(p.grad is None for p in audio.CachedModel.get_model(str(assets)).model.parameters())
    assert embedder.store.stats()["results"] == 2
    predicted = model.predict(data)
    assert predicted.columns == ["representations"]
    checkpoint = tmp_path / "audio.ckpt"
    model.save(checkpoint)
    restored = rf.Model.load(checkpoint)
    assert restored.predict(data).equals(predicted)


@rf.preprocess
def prepare_voicemail(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.rename({"raw_voicemail": "voicemail"})


@rf.postprocess
def attach_flag(frame: pl.DataFrame) -> pl.DataFrame:
    prob = frame["predictions"].struct.field("escalated").struct.field("content").struct.field("probability")
    return frame.with_columns((prob >= 0.5).alias("escalated_flag"))


@pytest.mark.integration
def test_end_to_end_multiformat_relational_audio_pipeline(tmp_path, monkeypatch):
    import io

    import lightning.pytorch as lit
    import pyarrow.ipc as ipc
    import yaml
    from fastapi.testclient import TestClient
    from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperModel

    monkeypatch.setenv("RELFLOW_CACHE_DIR", str(tmp_path / "default_cache"))
    lit.seed_everything(404, workers=True)

    encoder_dir = tmp_path / "whisper_encoder"
    WhisperFeatureExtractor(chunk_length=1, n_samples=16000, nb_max_frames=100, feature_size=32).save_pretrained(
        encoder_dir
    )
    whisper_init = WhisperModel(
        WhisperConfig(
            d_model=16,
            encoder_layers=0,
            encoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_layers=1,
            decoder_attention_heads=2,
            decoder_ffn_dim=32,
            num_mel_bins=32,
            max_source_positions=50,
        )
    )
    with torch.no_grad():
        whisper_init.encoder.conv1.weight.zero_()
        whisper_init.encoder.conv1.bias.zero_()
        whisper_init.encoder.conv1.weight[:8, :10, 1] = 1.0
        whisper_init.encoder.conv1.weight[:8, 20:, 1] = -1.0
        whisper_init.encoder.conv1.weight[8:, 20:, 1] = 1.0
        whisper_init.encoder.conv1.weight[8:, :10, 1] = -1.0
        whisper_init.encoder.conv2.weight.zero_()
        whisper_init.encoder.conv2.bias.zero_()
        for ch in range(16):
            whisper_init.encoder.conv2.weight[ch, ch, 1] = 1.0
        whisper_init.encoder.embed_positions.weight.zero_()
    whisper_init.save_pretrained(encoder_dir)

    media_dir = tmp_path / "recordings"
    media_dir.mkdir()
    formats = (
        ("wav", "WAV", "PCM_16", 48000, 2),
        ("flac", "FLAC", "PCM_16", 24000, 2),
        ("ogg", "OGG", "VORBIS", 16000, 1),
        ("opus.ogg", "OGG", "OPUS", 48000, 1),
        ("mp3", "MP3", None, 16000, 1),
        ("aiff", "AIFF", "PCM_16", 22050, 2),
    )

    def write_recording(index: int, urgent: bool, split: str) -> str:
        ext, fmt, subtype, rate, ch = formats[index % len(formats)]
        rng = np.random.default_rng(9000 + index + (500 if split == "test" else 0))
        # Place the distinguishing acoustic cue in the final third (0.4s..0.6s) so
        # multi-window clip="chunks" (window=0.2s, max_windows=3) is required to observe it.
        lead = np.linspace(0.0, 0.4, int(0.4 * rate), endpoint=False, dtype=np.float64)
        tail = np.linspace(0.0, 0.2, int(0.2 * rate), endpoint=False, dtype=np.float64)
        neutral = 0.25 * np.sin(2.0 * np.pi * rng.uniform(850.0, 950.0) * lead)
        cue_freq = rng.uniform(3100.0, 3800.0) if urgent else rng.uniform(160.0, 230.0)
        cue = 0.75 * np.sin(2.0 * np.pi * cue_freq * tail + rng.uniform(0.0, 2.0 * np.pi))
        mono = np.concatenate([neutral, cue]).astype(np.float32)
        samples = np.column_stack([mono, 0.8 * mono]) if ch == 2 else mono
        name = f"{split}_{index:03d}.{ext}"
        sf.write(media_dir / name, samples, rate, format=fmt, subtype=subtype)
        return name

    def make_split(split: str, count: int) -> pl.DataFrame:
        rows = []
        for idx in range(count):
            urgent = bool(idx % 2)
            pattern = (idx // 2) % 3
            primary = write_recording(idx * 2, urgent, split)
            followup = write_recording(idx * 2 + 1, urgent, split)
            calls = (
                []
                if pattern == 0
                else [{"clip": None, "channel": "phone"}, {"clip": followup, "channel": "app"}]
                if pattern == 1
                else [{"clip": primary, "channel": "phone"}, {"clip": followup, "channel": "web"}]
            )
            rows.append(
                {
                    "case_id": f"{split}-{idx}",
                    "raw_voicemail": primary,
                    "calls": calls,
                    "escalated": urgent,
                    "priority": "urgent" if urgent else "routine",
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

    store_dir = tmp_path / "audio_store"

    class Call(rf.Branch):
        clip = rf.Audio(
            model=str(encoder_dir),
            base=str(media_dir),
            store=str(store_dir),
            window=0.2,
            stride=0.2,
            max_windows=3,
            clip="chunks",
        )
        channel = rf.Category

    class EscalationModel(rf.Model):
        voicemail = rf.Audio(
            model=str(encoder_dir),
            base=str(media_dir),
            store=str(store_dir),
            window=0.2,
            stride=0.2,
            max_windows=3,
            clip="chunks",
            embed=True,
        )
        calls = Call(length=3, overflow="tail")
        escalated = rf.Boolean(predict=True)
        priority = rf.Enum("routine", "urgent", predict=True)

    model = EscalationModel(d_model=32, n_layers=2, n_heads=4)
    backbone = audio.CachedModel.get_model(str(encoder_dir)).model
    initial_backbone = {name: param.detach().clone() for name, param in backbone.named_parameters()}

    # Pre-warm from a Polars Series and verify duplicate calls add zero new cache entries.
    warmed = model.nodes["/voicemail"].embedder.warm(train_df["raw_voicemail"])
    assert warmed == len(train_df)
    assert model.nodes["/voicemail"].embedder.warm(train_df["raw_voicemail"]) == 0

    dm = rf.ArrowDataModule(
        model=model,
        train=str(train_parquet),
        validate=str(val_parquet),
        test=str(test_parquet),
        preprocessor=prepare_voicemail,
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

    unseen_inputs = test_df.select("case_id", "raw_voicemail", "calls")
    whole = model.predict(unseen_inputs, preprocess=prepare_voicemail, postprocess=attach_flag, retain=("case_id",))
    chunked = model.predict(
        unseen_inputs,
        batch_size=3,
        preprocess=prepare_voicemail,
        postprocess=attach_flag,
        retain=("case_id",),
    )
    assert set(whole.columns) == {"inputs", "predictions", "representations", "escalated_flag"}
    assert whole["escalated_flag"].to_list() == chunked["escalated_flag"].to_list()

    truth = test_df["escalated"].to_numpy()
    esc_pred = whole["escalated_flag"].to_numpy()
    prio_pred = whole["predictions"].struct.field("priority").struct.field("content").struct.field("value").to_numpy()
    assert np.mean(esc_pred == truth) >= 0.85
    assert np.mean((prio_pred == "urgent") == truth) >= 0.85

    # Streaming Parquet batch Writer parity.
    written_dir = tmp_path / "batch_output"
    pred_dm = rf.PolarsDataModule(
        model=model,
        predict=unseen_inputs,
        preprocessor=prepare_voicemail,
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
    ckpt_path = tmp_path / "escalation.rf"
    model.save(ckpt_path)
    inspected = rf.inspect(ckpt_path)
    assert inspected["command"] == "inspect"
    checked = rf.check(
        ckpt_path,
        data={"train": str(train_parquet)},
        preprocessor=prepare_voicemail,
        execute=True,
        sample=8,
    )
    assert all(phase["status"] == "passed" for phase in checked["phases"])

    restored = rf.Model.load(ckpt_path)
    restored.update(rf.where("type") == "audio", store=str(tmp_path / "cold_store"))
    reloaded = restored.predict(
        unseen_inputs, preprocess=prepare_voicemail, postprocess=attach_flag, retain=("case_id",)
    )
    assert reloaded["escalated_flag"].to_list() == whole["escalated_flag"].to_list()

    # YAML schema declaration loading.
    yaml_path = tmp_path / "schema.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "d_model": 16,
                "n_layers": 1,
                "n_heads": 2,
                "fields": {
                    "voicemail": {
                        "type": "audio",
                        "model": str(encoder_dir),
                        "base": str(media_dir),
                        "store": str(tmp_path / "yaml_store"),
                        "window": 0.2,
                        "clip": "tail",
                    },
                    "escalated": {"type": "boolean", "predict": True},
                },
            }
        )
    )
    yaml_model = rf.Model.from_yaml(yaml_path)
    assert yaml_model.predict(unseen_inputs.select("raw_voicemail"), preprocess=prepare_voicemail).height == len(
        test_df
    )

    # Live FastAPI ASGI serving over both JSON and Arrow IPC streams.
    deployment = (
        rf.Deployment(model=restored, accelerator="cpu", retain=("case_id",))
        .preprocess(prepare_voicemail)
        .postprocess(attach_flag)
    )
    with TestClient(deployment.app()) as client:
        assert client.get("/health").status_code == 200
        json_rows = unseen_inputs.head(2).to_dicts()
        r_json = client.post("/predict", json=json_rows)
        assert r_json.status_code == 200, r_json.text
        served_flags = [row["escalated_flag"] for row in r_json.json()]
        assert served_flags == whole["escalated_flag"].head(2).to_list()

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
        assert ipc_out["escalated_flag"].to_list() == whole["escalated_flag"].slice(2, 3).to_list()
