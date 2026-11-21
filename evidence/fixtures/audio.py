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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--audio-cache", help="Optional separate Hugging Face cache for the audio checkpoint")
    args = parser.parse_args()
    torch.set_num_threads(1)
    freeze(args.output, args.audio_cache)
