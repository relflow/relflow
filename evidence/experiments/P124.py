# %% [markdown]
# ---
# title: Can a model track directional pitch and rhythm shifts across audio recordings?
# description: Compare ordered pairs of unseen recordings in a branch using cached Whisper-base embeddings.
# evidence-group: Representation
# evidence-readout:
# - {label: Pitch shift accuracy, metric: [conditions, both, pitch, accuracy], format: percent}
# - {label: Rhythm shift accuracy, metric: [conditions, both, rhythm, accuracy], format: percent}
# - {label: Converse accuracy when reversed, metric: [reversed_converse, both_targets], format: percent}
# evidence-review:
#   question: Can relflow track ordered pitch and rhythm transitions across unseen audio recordings in a sequence?
#   modeling: Train one model on a two-step recording branch to classify whether pitch moved lower, stayed the same,
#     or moved higher and whether rhythm became calmer, stayed the same, or became pulsing. Pair distinct synthesis
#     seeds within each split, then score both clips, single-clip controls, and reversed clip order.
#   comparison:
#     kind: Controlled comparison
#     metric: Accuracy
#     conditions:
#     - {label: Pitch shift with both clips, path: [conditions, both, pitch, accuracy], format: percent}
#     - {label: Pitch shift with first clip only, path: [conditions, first_only, pitch, accuracy], format: percent}
#     - {label: Rhythm shift with both clips, path: [conditions, both, rhythm, accuracy], format: percent}
#     - {label: Rhythm shift with first clip only, path: [conditions, first_only, rhythm, accuracy], format: percent}
#     definition: Fraction of held-out distinct-seed recording pairs assigned the correct three-way transition label.
#     direction: Higher accuracy is better. Any single recording allows at most 50 percent accuracy.
#   note: Every test recording comes from an unseen synthesis seed. Paired clips never share a seed, so unchanged transitions still compare distinct waveforms and vectors.
# evidence-plot:
#   renderer: plot
#   focus: Do predicted transition labels follow ordered pitch and rhythm changes across unseen recording pairs?
#   comparison: Compare the four acoustic states of the first recording against the four states of the second recording.
#   limit: Each cell averages 56 ordered pairs of distinct held-out synthesis seeds.
# ---
#
# Across two ordered recordings, pitch register can fall, stay steady, or rise while
# rhythmic pulsing independently calms, stays steady, or starts pulsing.
#
# ## Example
#
# ```yaml
# recordings:
#   - clip: cdbfd8d0d4ab3de9.flac
#   - clip: 84e71a90c21b54f0.flac
# pitch: higher
# rhythm: same
# ```
#
# The first clip is a steady low-register tone and the second is a steady high-register
# tone from a different synthesis seed. Pitch moved higher while rhythm stayed the same.
# Reversing the two clips changes the true pitch transition from `higher` to `lower`.
#
# ## Comparison
#
# Score the trained model on held-out pairs with both recordings, with only the first
# recording, with only the second recording, and with the two recordings swapped in
# order. Under swapped order, compare predictions against the converse transition
# (`lower` <-> `higher` and `calmer` <-> `pulsing`, with `same` preserved).
#
# ## Result
#
# Knowing only one recording limits deterministic accuracy to 50 percent because half
# of its pairs stay in the same state and half move in the remaining direction. High
# accuracy with both clips, single-clip accuracy at or below that ceiling, and high
# converse accuracy when the two clips trade places establish ordered comparison of
# both cached audio attributes.
#
# ## Method
#
# The training, validation and test splits use 12, four and eight independent synthesis
# seeds. Each seed generates four two-second FLAC clips combining low or high fundamental
# register with steady or pulsed amplitude envelopes under shared harmonic weights,
# level jitter and background noise. Within each split all ordered pairs of distinct
# seeds meet all 16 state transitions, producing 2,112 training, 192 validation and
# 896 test pairs. `evidence/fixtures/audio.py` encodes the 96 clips with pinned
# Whisper-base valid-frame mean pooling and fits an eight-coordinate PCA projection on
# training clips alone. `P124.npz` stores those float16 coordinates, which enter the
# native `rf.Audio` cache before each split runs. Any backbone forward raises an error.
#
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["lightning", "matplotlib", "numpy", "polars", "pyyaml>=6", "relflow[audio]>=0.5.3", "torch", "transformers>=5.9"]
#
# [tool.relflow.evidence]
# capability = "representation"
# types = ["Audio"]
# conditions = ["unseen-inputs", "missing-inputs", "varying-collections"]
# ///

"""P124 tests ordered pitch and rhythm shift detection across cached audio recordings."""

import hashlib
import json
import sys
from contextlib import ExitStack
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
import torch
from policies import CONTROL
from relflow.tensorfields.shared.cache import Payloads
from reporting import experiment, report
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperModel

FIXTURE = Path(__file__).with_suffix(".npz")
SHA256 = "be958abd3b8f8572f4073e8e993c30a0cdea1d0e10c8986d0b74b858a3b0c5a6"
BUDGET = 240
LABELS = {
    "pitch": ("lower", "same", "higher"),
    "rhythm": ("calmer", "same", "pulsing"),
}


def cache_model(directory, width):
    """Supply a local Whisper encoder shape for cache replay; forbid backbone execution."""
    WhisperFeatureExtractor(chunk_length=2, n_samples=32000, nb_max_frames=200, feature_size=32).save_pretrained(
        directory
    )
    config = WhisperConfig(
        d_model=width,
        encoder_layers=1,
        encoder_attention_heads=1,
        encoder_ffn_dim=width,
        decoder_layers=1,
        decoder_attention_heads=1,
        decoder_ffn_dim=width,
        num_mel_bins=32,
        max_source_positions=8,
    )
    WhisperModel(config).save_pretrained(directory)

    class Recording(rf.Branch):
        clip = rf.Audio(model=str(directory), immutable=True, base=str(directory), store=str(directory / "clips"))

    class Inspection(rf.Model):
        recordings = Recording(length=2)
        pitch = rf.Enum(*LABELS["pitch"], predict=True)
        rhythm = rf.Enum(*LABELS["rhythm"], predict=True)

    model = Inspection(preset=CONTROL.name)

    def reject_forward(module, args):
        raise RuntimeError("P124 cache miss reached a backbone")

    model.nodes["/recordings/clip"].embedder._cached_model.model.register_forward_pre_hook(reject_forward)
    return model


def prime(model, data, split, device):
    """Publish and verify each split's audio identities through normal branch tensorization."""
    selected = data["split"] == split
    keys = data["clip"][selected].ravel()
    inputs = pl.DataFrame({"recordings": [[{"clip": str(key)}] for key in keys]})
    field = model.encode(inputs)["/recordings/clip"]
    embedder = model.nodes["/recordings/clip"].embedder
    store = embedder.store
    namespace = f"{store.vector_namespace}:{embedder.variant(device)}"
    ids = field.content["ids"].flatten().cpu().numpy()
    if len(np.unique(ids)) != len(keys) or not (store.client.fetch(ids, namespace).status == 0).all():
        raise ValueError(f"P124 {split} must introduce distinct, uncached audio identities")
    expected = data["clip_vectors"][selected].reshape(len(keys), -1).astype("<f4")
    store.client.publish(ids, namespace, Payloads.fixed(expected, expected.strides[0]))
    restored = embedder.encode(field.content.to(device), field.state.to(device)).reshape(expected.shape)
    if not np.array_equal(restored.cpu().numpy(), expected):
        raise ValueError(f"P124 {split} audio cache lookup changed a frozen vector")


def pairs(indices):
    return [(int(left), int(right)) for left in indices for right in indices if left != right]


def records(data, split):
    indices = np.flatnonzero(data["split"] == split)
    for (left, right), high_1, pulsed_1, high_2, pulsed_2 in product(
        pairs(indices), range(2), range(2), range(2), range(2)
    ):
        yield dict(
            first=str(data["clip"][left, high_1, pulsed_1]),
            second=str(data["clip"][right, high_2, pulsed_2]),
            recordings=[
                {"clip": str(data["clip"][left, high_1, pulsed_1])},
                {"clip": str(data["clip"][right, high_2, pulsed_2])},
            ],
            pitch=LABELS["pitch"][(high_2 - high_1) + 1],
            rhythm=LABELS["rhythm"][(pulsed_2 - pulsed_1) + 1],
        )


def ceiling(frame, column, target):
    """Best deterministic accuracy from one clip position's identity alone."""
    counts = frame.group_by([column, target]).len().group_by(column).agg(pl.col("len").max().alias("best"))
    return counts["best"].sum() / len(frame)


def converse(values, labels):
    low, same, high = labels
    return np.where(values == low, high, np.where(values == high, low, same))


def condition_inputs(frame, condition):
    first = frame["first"].to_list()
    second = frame["second"].to_list()
    if condition == "both":
        rows = [[{"clip": a}, {"clip": b}] for a, b in zip(first, second, strict=True)]
    elif condition == "first_only":
        rows = [[{"clip": a}, {"clip": None}] for a in first]
    elif condition == "second_only":
        rows = [[{"clip": None}, {"clip": b}] for b in second]
    elif condition == "reversed":
        rows = [[{"clip": b}, {"clip": a}] for a, b in zip(first, second, strict=True)]
    else:
        raise ValueError(f"Unknown P124 condition: {condition}")
    return pl.DataFrame({"recordings": rows})


def evaluate(model, data, frame):
    held_out = np.flatnonzero(data["split"] == "test")
    shape = (len(pairs(held_out)), 2, 2, 2, 2)
    truth = {target: frame[target].to_numpy().reshape(shape) for target in LABELS}
    conditions, predicted, confidences = {}, {}, {}
    for condition in ("both", "first_only", "second_only", "reversed"):
        output = model.predict(condition_inputs(frame, condition), batch_size=64)["predictions"]
        conditions[condition], predicted[condition], confidences[condition] = {}, {}, {}
        for target in LABELS:
            content = output.struct.field(target).struct.field("content")
            values = content.struct.field("value").to_numpy().reshape(shape)
            probs = content.struct.field("probability").to_numpy().reshape(shape)
            if not np.isfinite(probs).all():
                raise ValueError(f"P124 {condition} produced nonfinite {target} probabilities")
            conditions[condition][target] = {"accuracy": float(np.mean(values == truth[target]))}
            predicted[condition][target] = values
            confidences[condition][target] = probs

    pitch_converse = predicted["reversed"]["pitch"] == converse(truth["pitch"], LABELS["pitch"])
    rhythm_converse = predicted["reversed"]["rhythm"] == converse(truth["rhythm"], LABELS["rhythm"])
    both_intact = (predicted["both"]["pitch"] == truth["pitch"]) & (predicted["both"]["rhythm"] == truth["rhythm"])
    reversed_converse = {
        "pitch": float(pitch_converse.mean()),
        "rhythm": float(rhythm_converse.mean()),
        "both_targets": float((pitch_converse & rhythm_converse).mean()),
    }
    matched = {
        "flip_second_pitch": float(both_intact.all(axis=3).mean()),
        "flip_second_rhythm": float(both_intact.all(axis=4).mean()),
        "order_reversal_pair": float((both_intact & pitch_converse & rhythm_converse).mean()),
    }
    ceilings = {
        position: {target: ceiling(frame, position, target) for target in LABELS} for position in ("first", "second")
    }
    return dict(
        conditions=conditions,
        reversed_converse=reversed_converse,
        matched=matched,
        ceilings=ceilings,
        observations=dict(
            axes=["seed_pair", "first_high", "first_pulsed", "second_high", "second_pulsed"],
            states=["Low steady", "Low pulsed", "High steady", "High pulsed"],
            seeds=data["seed"][held_out],
            clips=data["clip"][held_out],
            expected_pitch=truth["pitch"][0],
            expected_rhythm=truth["rhythm"][0],
            intact_correct={
                target: (predicted["both"][target] == truth[target]).astype(np.float32) for target in LABELS
            },
            confidence=confidences["both"],
        ),
    )


@experiment("P124", seed=12401)
def run(seed, steps, accelerator):
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"P124 fixture differs from its pinned SHA-256: {digest}")
    with np.load(FIXTURE, allow_pickle=False) as archive:
        data = dict(archive)
    device = torch.device("cpu")
    if accelerator in ("auto", "gpu") and torch.cuda.is_available():
        device = torch.device("cuda")
    elif accelerator in ("auto", "gpu", "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    lit.seed_everything(seed, workers=True)
    with TemporaryDirectory() as work, ExitStack() as caches:
        model = cache_model(Path(work), int(data["clip_vectors"].shape[-1])).to(device)
        caches.callback(model.nodes["/recordings/clip"].embedder.store.owner.close)
        for split in ("train", "validation"):
            prime(model, data, split, device)
        train_full = pl.DataFrame(records(data, "train")).sample(fraction=1, shuffle=True, seed=seed)
        val_full = pl.DataFrame(records(data, "validation"))
        module = rf.PolarsDataModule(
            train=train_full.select("recordings", *LABELS),
            validate=val_full.select("recordings", *LABELS),
            seed=seed,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
        )
        model.fit(
            module,
            batch_size=64,
            max_steps=BUDGET if steps is None else min(steps, BUDGET),
            max_epochs=-1,
            learning_rate=0.003,
            early_stopping=False,
            rollback=False,
            precision="32-true",
            accelerator=accelerator,
            strategy="auto",
            devices=1,
            deterministic=True,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )
        model.to(device).eval()
        prime(model, data, "test", device)
        test_full = pl.DataFrame(records(data, "test"))
        result = evaluate(model, data, test_full)
        cached = model.nodes["/recordings/clip"].embedder.store.stats()["results"]

    checks = {
        "full training budget": model.trainer.global_step == BUDGET,
        "one cached vector per recording": cached == data["clip"].size,
        "balanced single-clip ceilings": all(
            result["ceilings"][pos][target] == 0.5 for pos in ("first", "second") for target in LABELS
        ),
    }
    for target in LABELS:
        both = result["conditions"]["both"][target]["accuracy"]
        first_only = result["conditions"]["first_only"][target]["accuracy"]
        second_only = result["conditions"]["second_only"][target]["accuracy"]
        checks[f"{target} accuracy with both clips at least 0.85"] = both >= 0.85
        checks[f"{target} first-clip-only at most 0.55"] = first_only <= 0.55
        checks[f"{target} second-clip-only at most 0.55"] = second_only <= 0.55
        checks[f"{target} converse accuracy under reversed order at least 0.85"] = (
            result["reversed_converse"][target] >= 0.85
        )
    checks["both targets flip cleanly across second-clip pitch changes on at least 75 percent of pairs"] = (
        result["matched"]["flip_second_pitch"] >= 0.75
    )
    checks["both targets flip cleanly across second-clip rhythm changes on at least 75 percent of pairs"] = (
        result["matched"]["flip_second_rhythm"] >= 0.75
    )
    checks["intact and reversed transitions jointly correct on at least 75 percent of pairs"] = (
        result["matched"]["order_reversal_pair"] >= 0.75
    )

    result.update(
        steps=model.trainer.global_step,
        pairs={"train": len(train_full), "validation": len(val_full), "test": len(test_full)},
        fixture=dict(
            sha256=digest,
            bytes=FIXTURE.stat().st_size,
            width=int(data["clip_vectors"].shape[-1]),
            encoders=json.loads(data["metadata"].item()),
            cached={"clip": cached},
        ),
    )
    return result, checks


def plot(bundle, spec):
    import matplotlib.pyplot as plt
    import numpy as np

    observations = bundle.record["metrics"]["observations"]
    labels = list(observations["states"])
    panels = (
        (
            "Pitch transition accuracy",
            np.array(observations["intact_correct"]["pitch"]),
            np.array(observations["expected_pitch"]),
        ),
        (
            "Rhythm transition accuracy",
            np.array(observations["intact_correct"]["rhythm"]),
            np.array(observations["expected_rhythm"]),
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), layout="constrained", sharey=True)
    for ax, (title, correct, expected) in zip(axes, panels, strict=True):
        means = correct.mean(axis=0).reshape(4, 4)
        truth = expected.reshape(4, 4)
        ax.imshow(means, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(4), labels, rotation=25, ha="right")
        ax.set_yticks(range(4), labels)
        ax.set_xlabel("Second recording")
        ax.set_title(title, pad=10)
        for (row, col), value in np.ndenumerate(means):
            ax.text(
                col,
                row,
                f"{value:.0%}\n{truth[row, col]}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if value > 0.60 else "#152331",
            )
    axes[0].set_ylabel("First recording")
    n = panels[0][1].shape[0]
    return (
        fig,
        f"Cell accuracy across {n} distinct held-out synthesis seed pairs per transition, labeled with the true shift.",
        {
            "metric_paths": [
                ["conditions", "both", "pitch", "accuracy"],
                ["conditions", "both", "rhythm", "accuracy"],
            ]
        },
    )


if __name__ == "__main__":
    report(run)
