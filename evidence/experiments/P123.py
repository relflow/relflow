# %% [markdown]
# ---
# title: Can a question select which image feature determines the answer?
# description: Ask about the color or pattern of the same tile using frozen question and image embeddings.
# evidence-group: Representation
# evidence-readout:
# - {label: Both inputs, metric: [conditions, both, accuracy], format: percent}
# - {label: Image only, metric: [conditions, image_only, accuracy], format: percent}
# - {label: Both questions correct on the same tile, metric: [matched, switch_question], format: percent}
# evidence-review:
#   question: Does changing a written question change which property of an image determines the prediction?
#   modeling: Train one model to answer a yes-or-no question about either the background color or the printed
#     pattern of a tile. Cross unseen wordings with unseen designs. Ask both questions about each tile, then
#     change one visual property while holding the wording and the other property fixed.
#   comparison:
#     kind: Controlled comparison
#     metric: Answer accuracy
#     conditions:
#     - {label: Both inputs, path: [conditions, both, accuracy], format: percent}
#     - {label: Image only, path: [conditions, image_only, accuracy], format: percent}
#     - {label: Question only, path: [conditions, question_only, accuracy], format: percent}
#     definition: Fraction of correct yes-or-no answers across the same held-out question and tile pairs.
#     direction: Higher accuracy is better. The image alone allows at most 75 percent and the question alone 50 percent.
#   note: Every test wording and render seed is held out. The two question meanings and all four tile classes occur in training.
# evidence-plot:
#   renderer: plot
#   focus: Does the predicted probability of yes follow the property named by each question?
#   comparison: Read down a column to compare the two questions on the same set of tile designs.
#   limit: Each cell summarizes eight held-out wordings of that question and eight new render seeds.
# ---
#
# A question and a tile determine one hidden answer. The question asks either
# whether the paper is blue or whether the marks form stripes. Color and pattern
# vary independently, so a single tile can have two different correct answers.
#
# ## Example
#
# ```yaml
# question: Is the paper behind the shapes blue?
# tile: 8160e2bd8321211a.png
# answer: true
# ```
#
# This tile shows dark dots on blue paper. The color question has the answer yes.
# Asking “Do long stripes span the tile?” about the very same file changes the answer
# to no. Only the text and the opaque image reference are supplied at prediction.
#
# ## Comparison
#
# Score the same trained model with both inputs, with the question missing, and
# with the image missing. On blue dots and orange stripes, switch the question
# and count a pair only when both opposite answers are correct. Also hold the
# question fixed and change either the color or the pattern of a matched design.
#
# ## Result
#
# A question alone gives 50 percent accuracy because every wording occurs equally
# often with each answer. An image alone allows at most 75 percent because on half
# the tiles the two questions disagree. Accuracy above both ceilings, together
# with correct answer changes on the matched pairs, measures use of both fields.
#
# ## Method
#
# The training, validation and test splits contain 12, four and eight wording pairs
# and the same numbers of independent render seeds. Each seed supplies four tiles
# with matched orientation, spacing, phase and shade variation. Within each split
# all wordings meet all tiles. The resulting splits contain 1,152, 128 and 512 pairs.
# The fixture generator in `evidence/fixtures/questions.py` uses pinned MiniLM
# mean-pooled token vectors and DINOv2-Small CLS vectors. For each encoder, PCA
# and scaling fit only its training identities. The companion `P123.npz` holds
# eight float16 coordinates per input. Each split enters the native embedding
# caches before use. A backbone forward raises an error. The full run fits 240
# updates on complete pairs. Each missing-input condition uses the same weights.
#
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["lightning", "matplotlib", "numpy", "polars", "pyyaml>=6", "relflow[image,text]>=0.5.3", "tokenizers", "torch", "transformers>=5.9"]
#
# [tool.relflow.evidence]
# capability = "representation"
# types = ["Text", "Image"]
# conditions = ["unseen-inputs", "missing-inputs", "transformations"]
# ///

"""P123 tests whether language selects the relevant feature of a cached image."""

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
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
from visibility import TARGET

FIXTURE = Path(__file__).with_suffix(".npz")
SHA256 = "4c2debf0d099dc08d7633c9eed97af384bc8dc9275fdcf09b2192f3e4548755e"
BUDGET = 240


def cache_model(directory, questions, width):
    """Supply a local checkpoint shape and distinct token identities for cache replay."""
    vocab = {"[PAD]": 0, "[UNK]": 1, **{text: i + 2 for i, text in enumerate(questions)}}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]")), pad_token="[PAD]", unk_token="[UNK]"
    )
    tokenizer.save_pretrained(directory)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=width,
        num_hidden_layers=0,
        num_attention_heads=1,
        intermediate_size=width,
        max_position_embeddings=4,
    )
    BertModel(config).save_pretrained(directory)
    (directory / "preprocessor_config.json").write_text(json.dumps({"image_processor_type": "ViTImageProcessor"}))

    class Model(rf.Model):
        question = rf.Text(
            model=str(directory), max_length=2, encoder_pooling="mean", store=str(directory / "questions")
        )
        tile = rf.Image(model=str(directory), immutable=True, base=str(directory), store=str(directory / "tiles"))
        answer = rf.Boolean(**TARGET)

    model = Model(preset=CONTROL.name)

    def reject_forward(module, args):
        raise RuntimeError("P123 cache miss reached a backbone")

    for backbone in {model.nodes[path].embedder._cached_model.model for path in ("/question", "/tile")}:
        backbone.register_forward_pre_hook(reject_forward)
    return model


def prime(model, data, split, device):
    """Publish and read back each new identity using ordinary field tensorization."""
    selected = data["split"] == split
    for name, other in (("question", "tile"), ("tile", "question")):
        keys = data[name][selected].ravel()
        inputs = pl.DataFrame({name: keys}).with_columns(pl.lit(None, dtype=pl.String).alias(other))
        field = model.encode(inputs)["/" + name]
        embedder = model.nodes["/" + name].embedder
        store = embedder.store
        namespace = f"{store.vector_namespace}:{embedder.variant(device)}"
        ids = field.content["ids"].flatten().cpu().numpy()
        if len(np.unique(ids)) != len(keys) or not (store.client.fetch(ids, namespace).status == 0).all():
            raise ValueError(f"P123 {split} must introduce distinct, uncached {name} identities")
        expected = data[name + "_vectors"][selected].reshape(len(keys), -1).astype("<f4")
        store.client.publish(ids, namespace, Payloads.fixed(expected, expected.strides[0]))
        restored = embedder.encode(field.content.to(device), field.state.to(device)).reshape(expected.shape)
        if not np.array_equal(restored.cpu().numpy(), expected):
            raise ValueError(f"P123 {split} {name} cache lookup changed a frozen vector")


def records(data, split):
    indices = np.flatnonzero(data["split"] == split)
    for wording, kind, design, blue, stripes in product(indices, range(2), indices, range(2), range(2)):
        yield dict(
            question=data["question"][wording, kind],
            tile=data["tile"][design, blue, stripes],
            answer=bool(blue if kind == 0 else stripes),
        )


def ceiling(frame, field):
    """Best possible deterministic accuracy given just this field's identity."""
    counts = frame.group_by(field).agg(pl.col("answer").sum().alias("yes"), pl.len().alias("total"))
    return counts.select(pl.max_horizontal("yes", pl.col("total") - pl.col("yes")).sum()).item() / len(frame)


def evaluate(model, data, frame):
    held_out = data["split"] == "test"
    questions, tiles = data["question"][held_out], data["tile"][held_out]
    shape = (*questions.shape, *tiles.shape)
    truth = frame["answer"].to_numpy().reshape(shape)
    conditions, probabilities = {}, {}
    for condition, missing in (("both", None), ("image_only", "question"), ("question_only", "tile")):
        inputs = frame.select("question", "tile")
        if missing:
            inputs = inputs.with_columns(pl.lit(None, dtype=pl.String).alias(missing))
        output = model.predict(inputs, batch_size=64)["predictions"]
        scores = (
            output.struct.field("answer").struct.field("content").struct.field("probability").to_numpy().reshape(shape)
        )
        if not np.isfinite(scores).all():
            raise ValueError(f"P123 {condition} produced nonfinite answer probabilities")
        conditions[condition] = {"accuracy": np.mean((scores >= 0.5) == truth)}
        probabilities[condition] = scores
    correct = (probabilities["both"] >= 0.5) == truth
    # Axes are wording, question kind, render seed, blue and stripes. A match
    # succeeds only if the model gives the correct answer on BOTH members.
    matched = {
        "switch_question": correct[..., [0, 1], [1, 0]].all(axis=1).mean(),
        "switch_color_when_asked": correct[:, 0].all(axis=-2).mean(),
        "switch_pattern_when_asked": correct[:, 1].all(axis=-1).mean(),
        "change_pattern_while_asking_color": correct[:, 0].all(axis=-1).mean(),
        "change_color_while_asking_pattern": correct[:, 1].all(axis=-2).mean(),
    }
    return dict(
        conditions=conditions,
        matched=matched,
        ceilings={"image_only": ceiling(frame, "tile"), "question_only": ceiling(frame, "question")},
        observations=dict(
            axes=["wording", "question_kind", "render_seed", "blue", "stripes"],
            question_kinds=["color", "pattern"],
            questions=questions,
            render_seeds=data["seed"][held_out],
            tiles=tiles,
            expected_by_kind_and_tile=truth[0, :, 0],
            probability=probabilities,
        ),
    )


@experiment("P123", seed=12301)
def run(seed, steps, accelerator):
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"P123 fixture differs from its pinned SHA-256: {digest}")
    with np.load(FIXTURE, allow_pickle=False) as archive:
        data = dict(archive)
    device = torch.device("cpu")
    if accelerator in ("auto", "gpu") and torch.cuda.is_available():
        device = torch.device("cuda")
    elif accelerator in ("auto", "gpu", "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    lit.seed_everything(seed, workers=True)
    with TemporaryDirectory() as work, ExitStack() as caches:
        model = cache_model(Path(work), data["question"].ravel().tolist(), data["question_vectors"].shape[-1]).to(
            device
        )
        for name in ("question", "tile"):
            caches.callback(model.nodes["/" + name].embedder.store.owner.close)
        for split in ("train", "validation"):
            prime(model, data, split, device)
        train = pl.DataFrame(records(data, "train")).sample(fraction=1, shuffle=True, seed=seed)
        validation = pl.DataFrame(records(data, "validation"))
        module = rf.PolarsDataModule(
            train=train, validate=validation, seed=seed, num_workers=0, persistent_workers=False, pin_memory=False
        )
        model.fit(
            module,
            batch_size=64,
            max_steps=BUDGET if steps is None else min(steps, BUDGET),
            max_epochs=-1,
            learning_rate=0.002,
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
        test = pl.DataFrame(records(data, "test"))
        result = evaluate(model, data, test)
        counts = {name: model.nodes["/" + name].embedder.store.stats()["results"] for name in ("question", "tile")}
    checks = {
        "full training budget": model.trainer.global_step == BUDGET,
        "one cached vector per question and tile": all(counts[k] == data[k].size for k in counts),
        "balanced single-input ceilings": result["ceilings"] == {"image_only": 0.75, "question_only": 0.5},
        "both-input accuracy at least 0.85": result["conditions"]["both"]["accuracy"] >= 0.85,
        "image-only accuracy at most 0.75": result["conditions"]["image_only"]["accuracy"] <= 0.75,
        "question-only accuracy at chance": result["conditions"]["question_only"]["accuracy"] == 0.5,
        "both opposite answers correct on at least 75 percent of question switches": result["matched"][
            "switch_question"
        ]
        >= 0.75,
    }
    for key in (
        "switch_color_when_asked",
        "switch_pattern_when_asked",
        "change_pattern_while_asking_color",
        "change_color_while_asking_pattern",
    ):
        checks[f"{key.replace('_', ' ')} gives both correct answers on at least 75 percent of pairs"] = (
            result["matched"][key] >= 0.75
        )
    result.update(
        steps=model.trainer.global_step,
        pairs={"train": len(train), "validation": len(validation), "test": len(test)},
        fixture=dict(
            sha256=digest,
            bytes=FIXTURE.stat().st_size,
            width=data["question_vectors"].shape[-1],
            encoders=json.loads(data["metadata"].item()),
            cached=counts,
        ),
    )
    return result, checks


def plot(bundle, spec):
    import matplotlib.pyplot as plt
    import numpy as np

    observations = bundle.record["metrics"]["observations"]
    probabilities = np.array(observations["probability"]["both"])
    means = probabilities.mean(axis=(0, 2)).reshape(2, 4)
    targets = np.array(observations["expected_by_kind_and_tile"]).reshape(2, 4)
    fig, ax = plt.subplots(figsize=(8.2, 3.5), layout="constrained")
    ax.imshow(means, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), ["Orange dots", "Orange stripes", "Blue dots", "Blue stripes"])
    ax.set_yticks(range(2), ["Is it blue?", "Is it striped?"])
    ax.set_title("Mean predicted probability of yes", pad=12)
    for (row, col), value in np.ndenumerate(means):
        expected = "yes" if targets[row, col] else "no"
        ax.text(
            col,
            row,
            f"{value:.1%}\nExpected {expected}",
            ha="center",
            va="center",
            color="white" if value > 0.60 else "#152331",
        )
    n = probabilities.shape[0] * probabilities.shape[2]
    return (
        fig,
        f"Each cell averages {n} held-out wording and design pairs. The second line gives the correct answer.",
        {"metric_paths": [["conditions", "both", "accuracy"]]},
    )


if __name__ == "__main__":
    report(run)
