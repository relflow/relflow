"""Rebuild P122's frozen vectors from the two published encoder snapshots on CPU.

Run with the relflow image and text extras installed:
    python evidence/fixtures/media.py /path/to/new/P122.npz
The Hugging Face cache is used when populated. Only this authoring command needs weights.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer
from transformers.models.auto.image_processing_auto import AutoImageProcessor

TEXT = ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
IMAGE = ("facebook/dinov2-small", "ed25f3a31f01632728cabb09d1542f84ab7b0056")
AUDIO = ("openai/whisper-base", "e37978b90ca9030d5170a5c07aadb050351a65bb")
WIDTH = 8
# Within each topic the first twelve sentences are training, the next four
# validation, and the final eight test. Each sentence is an original input.
COOKING = """Fold the beaten eggs into the flour.
Chop the carrots for the soup.
Knead the dough until smooth.
Melt the butter in a frying pan.
Roast the potatoes until golden.
Whisk the cream into soft peaks.
Season the broth with salt.
Slice the tomatoes for the salad.
Grill the onions over hot coals.
Stir the rice as it simmers.
Peel the garlic before frying it.
Roll the pastry on a floured board.
Steam the broccoli until tender.
Beat the sugar into the batter.
Grate the cheese over the pasta.
Bring the lentil stew to a boil.
Sear the steak on both sides.
Bake the loaf until the crust browns.
Dress the greens with lemon juice.
Poach the pears in spiced syrup.
Toast the breadcrumbs in olive oil.
Mash the beans with a little cumin.
Marinate the tofu before dinner.
Drain the noodles in a colander.""".splitlines()
MUSIC = """Tune the strings on the guitar.
Play a scale on the piano.
Tap a steady rhythm on the drum.
Sing the chorus in harmony.
Rehearse the melody with the choir.
Strum a chord on the ukulele.
Practice the solo on the trumpet.
Pluck the bass strings in time.
Conduct the orchestra through the finale.
Keep the beat with the metronome.
Bow the violin through the long note.
Finger the next chord on the banjo.
Repeat the arpeggio on the harp.
Hum the tune before the recital.
Strike the cymbal at the downbeat.
Transpose the ballad to a higher key.
Blow across the mouth of the flute.
Accompany the singer on the keyboard.
Clap along with the percussion section.
Improvise a riff on the saxophone.
Hold the final note of the anthem.
Read the score before the concert.
Adjust the reed on the clarinet.
Listen for the cellos in the quartet.""".splitlines()


def tile(seed, striped):
    """Matched colors, spacing, rotation and phase for both patterns at each seed."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:144, :144]
    angle = rng.uniform(0, np.pi)
    period = rng.uniform(20, 35)
    u = (x * np.cos(angle) + y * np.sin(angle) + rng.uniform(0, period)) % period - period / 2
    v = (y * np.cos(angle) - x * np.sin(angle) + rng.uniform(0, period)) % period - period / 2
    # Approximately equal foreground area in both classes.
    ink = abs(u) < period * 0.16 if striped else u**2 + v**2 < (period * 0.32) ** 2
    dark, light = rng.integers(10, 95, size=3), rng.integers(165, 250, size=3)
    return Image.fromarray(np.where(ink[:, :, None], dark, light).astype(np.uint8))


def project(vectors, train):
    """Fit centering, PCA directions and a common scale exclusively on training inputs."""
    values = vectors.double().numpy()
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    centered = values - values[train].mean(axis=0)
    _, singular, basis = np.linalg.svd(centered[train], full_matrices=False)
    axes = basis[:WIDTH].copy()
    axes *= np.sign(axes[np.arange(WIDTH), abs(axes).argmax(axis=1)])[:, None]
    scores = centered @ axes.T
    scores /= np.sqrt(np.mean(scores[train] ** 2))
    return scores.astype("<f2"), float(np.sum(singular[:WIDTH] ** 2) / np.sum(singular**2))


@torch.inference_mode()
def embed(checkpoint, inputs, train, cache_dir=None):
    """Freeze one pinned encoder's outputs with a projection fitted on training identities."""
    repo, revision = checkpoint
    path = snapshot_download(
        repo,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer*.json",
            "special_tokens_map.json",
            "vocab.txt",
        ],
    )
    model = AutoModel.from_pretrained(path, local_files_only=True).eval()
    if checkpoint == TEXT:
        tokens = AutoTokenizer.from_pretrained(path, local_files_only=True)(inputs, padding=True, return_tensors="pt")
        mask = tokens.attention_mask.unsqueeze(-1)
        vectors = (model(**tokens).last_hidden_state * mask).sum(1) / mask.sum(1)
        pooling = "attention-mask mean"
    elif checkpoint == IMAGE:
        processor = AutoImageProcessor.from_pretrained(path, backend="pil", local_files_only=True)
        vectors = torch.cat(
            [
                model(**processor(images=inputs[i : i + 4], return_tensors="pt")).last_hidden_state[:, 0]
                for i in range(0, len(inputs), 4)
            ]
        )
        pooling = "CLS"
    elif checkpoint == AUDIO:
        extractor = AutoFeatureExtractor.from_pretrained(path, local_files_only=True)
        encoder = model.get_encoder()
        pooled = []
        for i in range(0, len(inputs), 8):
            batch = extractor(
                inputs[i : i + 8],
                sampling_rate=extractor.sampling_rate,
                return_attention_mask=True,
                return_tensors="pt",
            )
            hidden = encoder(input_features=batch.input_features).last_hidden_state
            scale = batch.attention_mask.shape[1] / hidden.shape[1]
            counts = (batch.attention_mask.sum(dim=-1).float() / scale).ceil().long().clamp(min=1, max=hidden.shape[1])
            valid = (torch.arange(hidden.shape[1]).unsqueeze(0) < counts.unsqueeze(1)).unsqueeze(-1).float()
            pooled.append((hidden * valid).sum(dim=1) / valid.sum(dim=1))
        vectors = torch.cat(pooled, dim=0)
        pooling = "valid-frame mean"
    else:
        raise ValueError(f"No fixture pooling rule for {repo} at {revision}")
    scores, variance = project(vectors, train)
    return scores, dict(
        model=repo, revision=revision, pooling=pooling, width=vectors.shape[1], train_variance_retained=variance
    )


def freeze(output, image_cache=None):
    split = np.tile(["train"] * 12 + ["validation"] * 4 + ["test"] * 8, 2)
    labels = np.repeat([False, True], 24)
    notes = COOKING + MUSIC
    pictures = [tile(4000 + i % 24, bool(label)) for i, label in enumerate(labels)]
    keys = [hashlib.sha256(picture.tobytes()).hexdigest()[:16] + ".png" for picture in pictures]
    data = dict(split=split, label=labels, note=notes, tile=keys)
    metadata = {}
    data["note_vectors"], metadata["note"] = embed(TEXT, notes, split == "train")
    data["tile_vectors"], metadata["tile"] = embed(IMAGE, pictures, split == "train", image_cache)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **data, metadata=json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--image-cache", help="Optional separate Hugging Face cache for the image checkpoint")
    args = parser.parse_args()
    torch.set_num_threads(1)
    freeze(args.output, args.image_cache)
