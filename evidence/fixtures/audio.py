"""Rebuild the Audio Evidence fixtures on CPU.

Run with the relflow audio extra installed:
    python evidence/fixtures/audio.py /path/to/new/P124.npz
    python evidence/fixtures/audio.py /path/to/new/P125.npz --teacher /path/to/whisper-base --device cuda
The Hugging Face cache is used when populated. Only this authoring command needs weights.
"""

import argparse
import hashlib
import io
import json
from itertools import product
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from media import AUDIO, embed

RATE = 16000


def synthesize(seed, high, pulsed):
    """Matched harmonic balance, level jitter and noise floor across all four states of a seed."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 2.0, 2 * RATE, endpoint=False, dtype=np.float64)
    fundamental = rng.uniform(780.0, 1080.0) if high else rng.uniform(155.0, 235.0)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=3)
    overtone_2, overtone_3 = rng.uniform(0.25, 0.55), rng.uniform(0.10, 0.30)
    carrier = (
        np.sin(2.0 * np.pi * fundamental * grid + phases[0])
        + overtone_2 * np.sin(2.0 * np.pi * 2.0 * fundamental * grid + phases[1])
        + overtone_3 * np.sin(2.0 * np.pi * 3.0 * fundamental * grid + phases[2])
    )
    if pulsed:
        pulse_hz = rng.uniform(4.0, 6.5)
        pulse_phase = rng.uniform(0.0, 1.0)
        cycle = (grid * pulse_hz + pulse_phase) % 1.0
        gate = np.clip((0.38 - cycle) * 25.0, 0.0, 1.0) * np.clip(cycle * 25.0, 0.0, 1.0)
        envelope = 0.04 + 0.96 * gate
    else:
        drift_hz = rng.uniform(0.4, 1.1)
        envelope = 0.82 + 0.12 * np.sin(2.0 * np.pi * drift_hz * grid + rng.uniform(0.0, 2.0 * np.pi))
    waveform = carrier * envelope
    waveform /= np.max(np.abs(waveform)) + 1e-8
    waveform = rng.uniform(0.45, 0.75) * waveform + rng.normal(0.0, 0.012, size=len(grid))
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(waveform, -0.99, 0.99).astype(np.float32), RATE, format="FLAC")
    return buffer.getvalue()


def freeze(output, audio_cache=None):
    split = np.array(["train"] * 12 + ["validation"] * 4 + ["test"] * 8)
    seeds = np.arange(8400, 8400 + len(split))
    blobs = [synthesize(int(seed), high, pulsed) for seed, high, pulsed in product(seeds, (False, True), (False, True))]
    waveforms = []
    for raw in blobs:
        with sf.SoundFile(io.BytesIO(raw)) as source:
            waveforms.append(source.read(dtype="float32"))
    keys = np.array([hashlib.sha256(raw).hexdigest()[:16] + ".flac" for raw in blobs])
    vectors, metadata = embed(AUDIO, waveforms, np.repeat(split == "train", 4), audio_cache)
    with output.open("xb") as stream:
        np.savez_compressed(
            stream,
            split=split,
            seed=seeds,
            clip=keys.reshape(-1, 2, 2),
            clip_vectors=vectors.reshape(-1, 2, 2, vectors.shape[-1]),
            metadata=json.dumps({"clip": metadata}, sort_keys=True),
        )


def distill(output, teacher_dir, device):
    """Distill Whisper-base on unlabeled harmonic mixtures independent of the P125 splits."""
    from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperModel

    for name, digest in {
        "model.safetensors": "07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b",
        "config.json": "a153c53883a6799b6f056b4a8d1a515c9926d03994682ba88a7616618d7da0c1",
        "preprocessor_config.json": "9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666",
    }.items():
        with (teacher_dir / name).open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != digest:
                raise ValueError(f"Expected the pinned {AUDIO} {name} at {teacher_dir}")

    seed = 85291
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    grid = np.arange(RATE // 2) / RATE
    clips = []
    transcodes = []
    for _ in range(512):
        freqs = np.exp(rng.uniform(np.log(65), np.log(6000), rng.integers(1, 5)))
        amplitudes = rng.uniform(0.1, 1.0, len(freqs))
        phases = rng.uniform(0, 2 * np.pi, len(freqs))
        wave = (amplitudes[:, None] * np.sin(2 * np.pi * freqs[:, None] * grid + phases[:, None])).sum(0)
        wave *= rng.uniform(0.2, 0.8) / np.max(np.abs(wave))
        pcm = (wave + rng.normal(0, rng.uniform(0, 0.02), len(grid))).astype(np.float32)
        clips.append(pcm)
        buffer = io.BytesIO()
        sf.write(buffer, pcm, RATE, format="OGG", subtype="VORBIS")
        buffer.seek(0)
        transcodes.append(sf.read(buffer, dtype="float32")[0])

    teacher = WhisperModel.from_pretrained(teacher_dir, local_files_only=True).encoder.to(device).eval()
    processor = WhisperFeatureExtractor.from_pretrained(teacher_dir, local_files_only=True)
    # Each half-second recording has 25 valid encoder frames at Whisper's 20 ms hop.
    targets = []
    with torch.no_grad():
        for start in range(0, len(clips), 16):
            features = processor(clips[start : start + 16], sampling_rate=RATE, return_tensors="pt").input_features
            targets.append(teacher(features.to(device)).last_hidden_state[:, :25].mean(1).cpu())
    teacher_vectors = torch.cat(targets).double()
    centered = teacher_vectors - teacher_vectors.mean(0)
    _, _, axes = torch.linalg.svd(centered, full_matrices=False)
    basis = axes[:32].clone()
    basis *= basis[torch.arange(len(basis)), basis.abs().argmax(1)].sign().unsqueeze(1)
    scores = (centered @ basis.T).float()
    targets = torch.nn.functional.normalize(scores)

    config = dict(
        d_model=32,
        encoder_layers=0,
        decoder_layers=0,
        encoder_attention_heads=1,
        decoder_attention_heads=1,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        num_mel_bins=80,
        max_source_positions=50,
        vocab_size=4,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    student = WhisperModel(WhisperConfig(**config)).encoder
    with torch.no_grad():
        student.embed_positions.weight.zero_()
    compact = WhisperFeatureExtractor(chunk_length=1, feature_size=80)
    # Both acoustic views have the same teacher target. No pitch labels enter distillation.
    inputs = compact(clips + transcodes, sampling_rate=RATE, return_tensors="pt").input_features
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.002)
    for _ in range(600):
        batch = torch.randint(len(inputs), (32,))
        predicted = student(inputs[batch]).last_hidden_state[:, :25].mean(1)
        loss = 1 - torch.nn.functional.cosine_similarity(predicted, targets[batch % len(clips)]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    metadata = dict(teacher=AUDIO, seed=seed, clips=len(clips), views=["pcm", "vorbis"], steps=600, config=config)
    with output.open("xb") as target:
        np.savez_compressed(
            target,
            metadata=json.dumps(metadata),
            **{name: param.numpy() for name, param in student.state_dict().items()},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--audio-cache", help="Optional separate Hugging Face cache for the audio checkpoint")
    parser.add_argument(
        "--teacher", type=Path, help="Distill the offline P125 encoder from a pinned Whisper-base directory"
    )
    parser.add_argument("--device", default="cpu", help="Device for the frozen teacher during distillation")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.teacher:
        distill(args.output, args.teacher, args.device)
    else:
        freeze(args.output, args.audio_cache)
