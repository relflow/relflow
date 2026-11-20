"""Rebuild P123's small frozen question and tile fixture on CPU.

python evidence/fixtures/questions.py /path/to/new/P123.npz
"""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch
from media import IMAGE, TEXT, embed
from PIL import Image

# Each row pairs two wordings for the question-switch comparison. Rows 0–11 train,
# 12–15 validate and 16–23 test. Every question is crossed with every tile in its split.
QUESTIONS = [
    ("Is the background blue?", "Does the pattern have stripes?"),
    ("Is the paper colored blue?", "Do you see parallel stripes?"),
    ("Is there a blue background behind the marks?", "Are there lines across the tile?"),
    ("Would you call the base color blue?", "Is the design made of stripes?"),
    ("Is blue the color under the pattern?", "Are the dark marks long bands?"),
    ("Does the sheet have a blue tint?", "Do the markings form parallel lines?"),
    ("Is the surface behind the ink blue?", "Is the paper covered with stripes?"),
    ("Has the background been painted blue?", "Are the marks arranged as straight bands?"),
    ("Does the tile have a blue base?", "Do stripes run across the background?"),
    ("Is the blank part of the paper blue?", "Is the print a pattern of parallel lines?"),
    ("Does the ink sit on blue paper?", "Is this tile striped?"),
    ("Is the fill color a shade of blue?", "Do the markings look like long stripes?"),
    ("Is the backdrop a blue shade?", "Does this print consist of straight stripes?"),
    ("Is blue the main background color?", "Do parallel bands cross the sheet?"),
    ("Is the color beneath the ink blue?", "Are the dark shapes continuous lines?"),
    ("Does the background look blue to you?", "Is a striped design visible here?"),
    ("Is the paper behind the shapes blue?", "Do long stripes span the tile?"),
    ("Would you describe the backdrop as blue?", "Does the ink form parallel bands?"),
    ("Do you see a blue ground beneath the design?", "Is this a pattern of straight lines?"),
    ("Is the sheet itself a shade of blue?", "Are stripes printed on the sheet?"),
    ("Are the shapes printed on a blue surface?", "Do the dark marks look like stripes?"),
    ("Is blue visible between the marks?", "Are there long parallel lines in the design?"),
    ("Does the underlying paper look blue?", "Would you call the pattern striped?"),
    ("Is blue the background hue of this tile?", "Do you see bands of ink across the paper?"),
]


def tile(seed, blue, striped):
    """Share geometry and shade jitter across all four variants of a design seed."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:144, :144]
    angle = rng.uniform(0, np.pi)
    period = rng.uniform(20, 34)
    u = (x * np.cos(angle) + y * np.sin(angle) + rng.uniform(0, period)) % period - period / 2
    v = (y * np.cos(angle) - x * np.sin(angle) + rng.uniform(0, period)) % period - period / 2
    ink = abs(u) < period * 0.15 if striped else u**2 + v**2 < (period * 0.31) ** 2
    paper = np.array([66, 150, 232] if blue else [242, 149, 62]) + rng.integers(-12, 13, size=3)
    pigment = rng.integers(16, 43, size=3)
    return Image.fromarray(np.where(ink[:, :, None], pigment, paper).astype(np.uint8))


def freeze(output, image_cache=None):
    split = np.array(["train"] * 12 + ["validation"] * 4 + ["test"] * 8)
    questions = np.array(QUESTIONS)
    seeds = np.arange(6700, 6700 + len(split))
    pictures = [tile(seed, blue, striped) for seed, blue, striped in product(seeds, (False, True), (False, True))]
    keys = np.array([hashlib.sha256(picture.tobytes()).hexdigest()[:16] + ".png" for picture in pictures])
    q, question_metadata = embed(TEXT, questions.ravel().tolist(), np.repeat(split == "train", 2))
    t, tile_metadata = embed(IMAGE, pictures, np.repeat(split == "train", 4), image_cache)
    with output.open("xb") as stream:
        np.savez_compressed(
            stream,
            split=split,
            seed=seeds,
            question=questions,
            question_vectors=q.reshape(*questions.shape, -1),
            tile=keys.reshape(-1, 2, 2),
            tile_vectors=t.reshape(-1, 2, 2, t.shape[-1]),
            metadata=json.dumps({"question": question_metadata, "tile": tile_metadata}, sort_keys=True),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--image-cache", help="Optional separate Hugging Face cache for the image checkpoint")
    args = parser.parse_args()
    torch.set_num_threads(1)
    freeze(args.output, args.image_cache)
