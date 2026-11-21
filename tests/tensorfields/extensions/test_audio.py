import os
import threading
from collections import Counter
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
    tail = vectors(build(tmp_path, channels=0, window=0.1, clip="tail"), [path.name])[0, 0, 0]
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
        answer=recording(mask=True),
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
        out = audio.encode_clips(ast, ast_fe, [short], device, rate=16000, pooling=mode, hidden_size=16, model_name="a")
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
    assert all(p.grad is None for p in audio.CachedModel.get_model(str(assets)).model.parameters())
    assert embedder.store.stats()["results"] == 2
    predicted = model.predict(data)
    assert predicted.columns == ["representations"]
    checkpoint = tmp_path / "audio.ckpt"
    model.save(checkpoint)
    restored = rf.Model.load(checkpoint)
    assert restored.predict(data).equals(predicted)
