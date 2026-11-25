"""Rebuild the Video Evidence fixture on CPU.

Run with the relflow video extra installed:
    python evidence/fixtures/video.py /path/to/new/P126.npz
The Hugging Face cache is used when populated. Only this authoring command needs weights.
"""

import argparse
import hashlib
import io
import json
from itertools import product
from pathlib import Path

import av
import numpy as np
import torch
from media import VIDEO, embed

FPS = 8
FRAMES = 8


def synthesize(seed, striped, pulsing):
    """Matched palette, orientation, spacing, and foreground balance across all four states of a seed."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:144, :144]
    angle = rng.uniform(0.0, np.pi)
    period = rng.uniform(26.0, 38.0)
    phase = rng.uniform(0.0, period)
    u = (x * np.cos(angle) + y * np.sin(angle) + phase) % period - period / 2.0
    v = (y * np.cos(angle) - x * np.sin(angle) + phase) % period - period / 2.0
    ink = (np.abs(u) < period * 0.19) if striped else (u**2 + v**2 < (period * 0.35) ** 2)
    dark = rng.integers(15, 85, size=3)
    light = rng.integers(175, 245, size=3)
    base = np.where(ink[:, :, None], dark, light).astype(np.float32)

    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=FPS, options={"crf": "18", "preset": "fast"})
        stream.width = 144
        stream.height = 144
        stream.pix_fmt = "yuv420p"
        for step in range(FRAMES):
            if pulsing:
                scale = 0.30 + 0.70 * (step % 2)
                radius = 24 + step * 8
                ring = (x - 72) ** 2 + (y - 72) ** 2 < radius**2
                pixels = np.where(ring[:, :, None], base * scale, base * (1.0 - 0.45 * (step % 2)))
            else:
                pixels = base * 0.70
            frame = av.VideoFrame.from_ndarray(np.clip(pixels, 0, 255).astype(np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def freeze(output, video_cache=None):
    split = np.array(["train"] * 12 + ["validation"] * 4 + ["test"] * 8)
    seeds = np.arange(9200, 9200 + len(split))
    blobs = [
        synthesize(int(seed), striped, pulsing)
        for seed, striped, pulsing in product(seeds, (False, True), (False, True))
    ]
    decoded = []
    for raw in blobs:
        with av.open(io.BytesIO(raw)) as container:
            decoded.append([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    keys = np.array([hashlib.sha256(raw).hexdigest()[:16] + ".mp4" for raw in blobs])
    vectors, metadata = embed(VIDEO, decoded, np.repeat(split == "train", 4), video_cache)
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
    parser.add_argument("--video-cache", help="Optional separate Hugging Face cache for the video checkpoint")
    args = parser.parse_args()
    torch.set_num_threads(1)
    freeze(args.output, args.video_cache)
