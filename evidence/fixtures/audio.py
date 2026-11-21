"""Rebuild P124's frozen Whisper-base audio vectors on CPU.

Run with the relflow audio extra installed:
    python evidence/fixtures/audio.py /path/to/new/P124.npz
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
from huggingface_hub import snapshot_download
from media import project
from transformers import AutoFeatureExtractor, AutoModel

AUDIO = ("openai/whisper-base", "e37978b90ca9030d5170a5c07aadb050351a65bb")
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


@torch.inference_mode()
def embed_audio(blobs, train, cache_dir=None):
    """Encode FLAC bytes with the pinned Whisper-base encoder and fit train-only PCA."""
    repo, revision = AUDIO
    path = snapshot_download(
        repo,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json"],
    )
    extractor = AutoFeatureExtractor.from_pretrained(path, local_files_only=True)
    encoder = AutoModel.from_pretrained(path, local_files_only=True).get_encoder().eval()
    waveforms = []
    for raw in blobs:
        with sf.SoundFile(io.BytesIO(raw)) as source:
            waveforms.append(source.read(dtype="float32"))
    pooled = []
    for start in range(0, len(waveforms), 8):
        batch = waveforms[start : start + 8]
        inputs = extractor(batch, sampling_rate=RATE, return_attention_mask=True, return_tensors="pt")
        hidden = encoder(input_features=inputs.input_features).last_hidden_state
        scale = inputs.attention_mask.shape[1] / hidden.shape[1]
        counts = (inputs.attention_mask.sum(dim=-1).float() / scale).ceil().long().clamp(min=1, max=hidden.shape[1])
        valid = (torch.arange(hidden.shape[1]).unsqueeze(0) < counts.unsqueeze(1)).unsqueeze(-1).float()
        pooled.append((hidden * valid).sum(dim=1) / valid.sum(dim=1))
    vectors = torch.cat(pooled, dim=0)
    scores, variance = project(vectors, train)
    return scores, dict(
        model=repo,
        revision=revision,
        pooling="valid-frame mean",
        width=int(vectors.shape[1]),
        train_variance_retained=variance,
    )


def freeze(output, audio_cache=None):
    split = np.array(["train"] * 12 + ["validation"] * 4 + ["test"] * 8)
    seeds = np.arange(8400, 8400 + len(split))
    blobs = [synthesize(int(seed), high, pulsed) for seed, high, pulsed in product(seeds, (False, True), (False, True))]
    keys = np.array([hashlib.sha256(raw).hexdigest()[:16] + ".flac" for raw in blobs])
    vectors, metadata = embed_audio(blobs, np.repeat(split == "train", 4), audio_cache)
    with output.open("xb") as stream:
        np.savez_compressed(
            stream,
            split=split,
            seed=seeds,
            clip=keys.reshape(-1, 2, 2),
            clip_vectors=vectors.reshape(-1, 2, 2, vectors.shape[-1]),
            metadata=json.dumps({"clip": metadata}, sort_keys=True),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--audio-cache", help="Optional separate Hugging Face cache for the audio checkpoint")
    args = parser.parse_args()
    torch.set_num_threads(1)
    freeze(args.output, args.audio_cache)
