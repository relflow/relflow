# %% [markdown]
# ---
# title: Can a model track directional pattern and motion shifts across video clips?
# description: Compare ordered pairs of unseen video clips in a branch using cached X-CLIP embeddings.
# evidence-group: Representation
# evidence-readout:
# - {label: Pattern shift accuracy, metric: [conditions, both, pattern, accuracy], format: percent}
# - {label: Motion shift accuracy, metric: [conditions, both, motion, accuracy], format: percent}
# - {label: Converse accuracy when reversed, metric: [reversed_converse, both_targets], format: percent}
# evidence-review:
#   question: Can relflow track ordered visual pattern and temporal motion transitions across unseen video clips in a sequence?
#   modeling: Train one model on a two-step video branch to classify whether pattern shifted toward dots, stayed the same,
#     or shifted toward stripes and whether temporal motion became steadier, stayed the same, or became pulsing. Pair
#     distinct synthesis seeds within each split, then score both clips, single-clip controls, and reversed clip order.
#   comparison:
#     kind: Controlled comparison
#     metric: Accuracy
#     conditions:
#     - {label: Pattern shift with both clips, path: [conditions, both, pattern, accuracy], format: percent}
#     - {label: Pattern shift with first clip only, path: [conditions, first_only, pattern, accuracy], format: percent}
#     - {label: Motion shift with both clips, path: [conditions, both, motion, accuracy], format: percent}
#     - {label: Motion shift with first clip only, path: [conditions, first_only, motion, accuracy], format: percent}
#     definition: Fraction of held-out distinct-seed video pairs assigned the correct three-way transition label.
#     direction: Higher accuracy is better. Any single clip allows at most 50 percent accuracy.
#   note: Every test clip comes from an unseen synthesis seed. Paired clips never share a seed, so unchanged transitions still compare distinct videos and vectors.
# evidence-plot:
#   renderer: plot
#   focus: Do predicted transition labels follow ordered pattern and motion changes across unseen video pairs?
#   comparison: Compare the four visual-temporal states of the first clip against the four states of the second clip.
#   limit: Each cell averages 56 ordered pairs of distinct held-out synthesis seeds.
# ---
#
# Across two ordered video clips, visual texture can shift between dotted and striped
# patterns while temporal dynamics independently calm to steady exposure or begin
# pulsing across frames.
#
# ## Example
#
# ```yaml
# clips:
#   - video: bd397d1e9fae35a5.mp4
#   - video: 9e65c301199dee72.mp4
# pattern: to_stripes
# motion: same
# ```
#
# The first clip is a steady dotted scene and the second is a steady striped scene from
# a different synthesis seed. Pattern shifted toward stripes while motion stayed the
# same. Reversing the two clips changes the true pattern transition from `to_stripes`
# to `to_dots`.
#
# ## Comparison
#
# Score the trained model on held-out pairs with both clips, with only the first clip,
# with only the second clip, and with the two clips swapped in order. Under swapped
# order, compare predictions against the converse transition (`to_dots` <-> `to_stripes`
# and `steadier` <-> `pulsing`, with `same` preserved).
#
# ## Result
#
# Knowing only one clip limits deterministic accuracy to 50 percent because half of its
# pairs stay in the same state and half move in the remaining direction. High accuracy
# with both clips, single-clip accuracy at or below that ceiling, and high converse
# accuracy when the two clips trade places establish ordered comparison of both cached
# video attributes.
#
# ## Method
#
# The training, validation and test splits use 12, four and eight independent synthesis
# seeds. Each seed generates four one-second 8-fps MP4 clips combining dotted or striped
# textures with steady or expanding pulsing-ring exposure under shared palette,
# orientation, spacing and foreground balance. Within each split all ordered pairs of
# distinct seeds meet all 16 state transitions, producing 2,112 training, 192 validation
# and 896 test pairs. `evidence/fixtures/video.py` encodes the 96 decoded MP4 clips with
# pinned X-CLIP-base-patch32 MIT pooler outputs and fits an eight-coordinate PCA
# projection on training clips alone. `P126.npz` stores those float16 coordinates, which
# enter the native `rf.Video` cache before each split runs. Any backbone forward raises
# an error.
#
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["lightning", "matplotlib", "numpy", "polars", "pyyaml>=6", "relflow[video]>=0.5.4", "torch", "transformers>=5.9"]
#
# [tool.relflow.evidence]
# capability = "representation"
# types = ["Video"]
# conditions = ["unseen-inputs", "missing-inputs", "varying-collections"]
# ///

"""P126 tests ordered pattern and motion shift detection across cached video clips."""

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
from transformers import XCLIPConfig, XCLIPModel, XCLIPTextConfig, XCLIPVisionConfig
from transformers.models.videomae.image_processing_pil_videomae import VideoMAEImageProcessorPil

FIXTURE = Path(__file__).with_suffix(".npz")
SHA256 = "6c5f4fd7cd99a57679422f70298c19e7b66bd49ca71f2c9b4c303767e728bb3d"
BUDGET = 240
LABELS = {
    "pattern": ("to_dots", "same", "to_stripes"),
    "motion": ("steadier", "same", "pulsing"),
}


def cache_model(directory, width):
    """Supply a local X-CLIP encoder shape for cache replay; forbid backbone execution."""
    VideoMAEImageProcessorPil(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}).save_pretrained(
        directory
    )
    config = XCLIPConfig(
        text_config=XCLIPTextConfig(
            hidden_size=width,
            intermediate_size=width * 2,
            num_hidden_layers=1,
            num_attention_heads=1,
            projection_dim=width,
        ),
        vision_config=XCLIPVisionConfig(
            image_size=32,
            patch_size=16,
            num_frames=8,
            hidden_size=width,
            intermediate_size=width * 2,
            num_hidden_layers=1,
            num_attention_heads=1,
            projection_dim=width,
            mit_hidden_size=width,
            mit_intermediate_size=width * 2,
            mit_num_hidden_layers=1,
            mit_num_attention_heads=1,
        ),
        projection_dim=width,
    )
    XCLIPModel(config).save_pretrained(directory)

    class Clip(rf.Branch):
        video = rf.Video(model=str(directory), immutable=True, base=str(directory), store=str(directory / "clips"))

    class Inspection(rf.Model):
        clips = Clip(length=2)
        pattern = rf.Enum(*LABELS["pattern"], mask=True)
        motion = rf.Enum(*LABELS["motion"], mask=True)

    model = Inspection(preset=CONTROL.name)

    def reject_forward(module, args):
        raise RuntimeError("P126 cache miss reached a backbone")

    model.nodes["/clips/video"].embedder._cached_model.model.register_forward_pre_hook(reject_forward)
    return model


def prime(model, data, split, device):
    """Publish and verify each split's video identities through normal branch tensorization."""
    selected = data["split"] == split
    keys = data["clip"][selected].ravel()
    inputs = pl.DataFrame({"clips": [[{"video": str(key)}] for key in keys]})
    field = model.encode(inputs)["/clips/video"]
    embedder = model.nodes["/clips/video"].embedder
    store = embedder.store
    namespace = f"{store.vector_namespace}:{embedder.variant(device)}"
    ids = field.content["ids"].flatten().cpu().numpy()
    if len(np.unique(ids)) != len(keys) or not (store.client.fetch(ids, namespace).status == 0).all():
        raise ValueError(f"P126 {split} must introduce distinct, uncached video identities")
    expected = data["clip_vectors"][selected].reshape(len(keys), -1).astype("<f4")
    store.client.publish(ids, namespace, Payloads.fixed(expected, expected.strides[0]))
    restored = embedder.encode(field.content.to(device), field.state.to(device)).reshape(expected.shape)
    if not np.array_equal(restored.cpu().numpy(), expected):
        raise ValueError(f"P126 {split} video cache lookup changed a frozen vector")


def pairs(indices):
    return [(int(left), int(right)) for left in indices for right in indices if left != right]


def records(data, split):
    indices = np.flatnonzero(data["split"] == split)
    for (left, right), striped_1, pulsing_1, striped_2, pulsing_2 in product(
        pairs(indices), range(2), range(2), range(2), range(2)
    ):
        yield dict(
            first=str(data["clip"][left, striped_1, pulsing_1]),
            second=str(data["clip"][right, striped_2, pulsing_2]),
            clips=[
                {"video": str(data["clip"][left, striped_1, pulsing_1])},
                {"video": str(data["clip"][right, striped_2, pulsing_2])},
            ],
            pattern=LABELS["pattern"][(striped_2 - striped_1) + 1],
            motion=LABELS["motion"][(pulsing_2 - pulsing_1) + 1],
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
        rows = [[{"video": a}, {"video": b}] for a, b in zip(first, second, strict=True)]
    elif condition == "first_only":
        rows = [[{"video": a}, {"video": None}] for a in first]
    elif condition == "second_only":
        rows = [[{"video": None}, {"video": b}] for b in second]
    elif condition == "reversed":
        rows = [[{"video": b}, {"video": a}] for a, b in zip(first, second, strict=True)]
    else:
        raise ValueError(f"Unknown P126 condition: {condition}")
    return pl.DataFrame({"clips": rows})


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
                raise ValueError(f"P126 {condition} produced nonfinite {target} probabilities")
            conditions[condition][target] = {"accuracy": float(np.mean(values == truth[target]))}
            predicted[condition][target] = values
            confidences[condition][target] = probs

    pattern_converse = predicted["reversed"]["pattern"] == converse(truth["pattern"], LABELS["pattern"])
    motion_converse = predicted["reversed"]["motion"] == converse(truth["motion"], LABELS["motion"])
    both_intact = (predicted["both"]["pattern"] == truth["pattern"]) & (predicted["both"]["motion"] == truth["motion"])
    reversed_converse = {
        "pattern": float(pattern_converse.mean()),
        "motion": float(motion_converse.mean()),
        "both_targets": float((pattern_converse & motion_converse).mean()),
    }
    matched = {
        "flip_second_pattern": float(both_intact.all(axis=3).mean()),
        "flip_second_motion": float(both_intact.all(axis=4).mean()),
        "order_reversal_pair": float((both_intact & pattern_converse & motion_converse).mean()),
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
            axes=["seed_pair", "first_striped", "first_pulsing", "second_striped", "second_pulsing"],
            states=["Dotted steady", "Dotted pulsing", "Striped steady", "Striped pulsing"],
            seeds=data["seed"][held_out],
            clips=data["clip"][held_out],
            expected_pattern=truth["pattern"][0],
            expected_motion=truth["motion"][0],
            intact_correct={
                target: (predicted["both"][target] == truth[target]).astype(np.float32) for target in LABELS
            },
            confidence=confidences["both"],
        ),
    )


@experiment("P126", seed=12601)
def run(seed, steps, accelerator):
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"P126 fixture differs from its pinned SHA-256: {digest}")
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
        caches.callback(model.nodes["/clips/video"].embedder.store.owner.close)
        for split in ("train", "validation"):
            prime(model, data, split, device)
        train_full = pl.DataFrame(records(data, "train")).sample(fraction=1, shuffle=True, seed=seed)
        val_full = pl.DataFrame(records(data, "validation"))
        module = rf.PolarsDataModule(
            train=train_full.select("clips", *LABELS),
            validate=val_full.select("clips", *LABELS),
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
        cached = model.nodes["/clips/video"].embedder.store.stats()["results"]

    checks = {
        "full training budget": model.trainer.global_step == BUDGET,
        "one cached vector per clip": cached == data["clip"].size,
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
    checks["both targets flip cleanly across second-clip pattern changes on at least 75 percent of pairs"] = (
        result["matched"]["flip_second_pattern"] >= 0.75
    )
    checks["both targets flip cleanly across second-clip motion changes on at least 75 percent of pairs"] = (
        result["matched"]["flip_second_motion"] >= 0.75
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
            "Pattern transition accuracy",
            np.array(observations["intact_correct"]["pattern"]),
            np.array(observations["expected_pattern"]),
        ),
        (
            "Motion transition accuracy",
            np.array(observations["intact_correct"]["motion"]),
            np.array(observations["expected_motion"]),
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), layout="constrained", sharey=True)
    for ax, (title, correct, expected) in zip(axes, panels, strict=True):
        means = correct.mean(axis=0).reshape(4, 4)
        truth = expected.reshape(4, 4)
        ax.imshow(means, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(4), labels, rotation=25, ha="right")
        ax.set_yticks(range(4), labels)
        ax.set_xlabel("Second clip")
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
    axes[0].set_ylabel("First clip")
    n = panels[0][1].shape[0]
    return (
        fig,
        f"Cell accuracy across {n} distinct held-out synthesis seed pairs per transition, labeled with the true shift.",
        {
            "metric_paths": [
                ["conditions", "both", "pattern", "accuracy"],
                ["conditions", "both", "motion", "accuracy"],
            ]
        },
    )


if __name__ == "__main__":
    report(run)
