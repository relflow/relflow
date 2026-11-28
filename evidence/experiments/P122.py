# %% [markdown]
# ---
# title: Can cached text and image embeddings carry two independent labels?
# description: Recover the topic of a sentence and the pattern in an image from eight frozen coordinates per input.
# evidence-group: Representation
# evidence-readout:
# - {label: Topic accuracy, metric: [conditions, intact, topic, accuracy], format: percent}
# - {label: Pattern accuracy, metric: [conditions, intact, pattern, accuracy], format: percent}
# evidence-review:
#   question: Can relflow use compact frozen representations of previously unseen sentences and images?
#   modeling: Train one model to predict the topic of a sentence and the pattern on a tile from their cached
#     embeddings. Cross every held-out sentence with every held-out tile, then reassign each input separately
#     and score both targets again.
#   comparison:
#     kind: Controlled comparison
#     metric: Accuracy
#     conditions:
#     - {label: Topic, path: [conditions, intact, topic, accuracy], format: percent}
#     - {label: Topic with reassigned sentences, path: [conditions, shuffle_note, topic, accuracy], format: percent}
#     - {label: Pattern, path: [conditions, intact, pattern, accuracy], format: percent}
#     - {label: Pattern with reassigned images, path: [conditions, shuffle_tile, pattern, accuracy], format: percent}
#     definition: Fraction of held-out pairs classified correctly for each target. Each label occurs equally often.
#     direction: Higher accuracy is better. Chance is 50 percent for each target.
#   note: The test population contains sixteen sentences and sixteen tile designs, crossed into 256 pairs.
# evidence-plot:
#   mode: summary
#   focus: How much of each label survives when its input is reassigned?
#   comparison: Compare sentence topic and tile pattern accuracy on the same held-out pairs.
#   limit: This small authored corpus measures learning from frozen coordinates on two specific tasks.
# ---
#
# A sentence describes cooking or music. An independent image shows dots or stripes.
# Both inputs use pretrained representations compressed to eight coordinates.
#
# ## Example
#
# ```yaml
# note: Fold the beaten eggs into the flour.
# tile: 1f6c0c8da161cc6a.png
# topic: false
# pattern: true
# ```
#
# The instruction describes cooking, so the music target is false. The referenced
# tile has stripes, so the stripe target is true. Both targets are hidden at prediction.
#
# ## Comparison
#
# Score every held-out sentence with every held-out tile. Then reassign sentences
# while keeping the tiles fixed, and repeat with reassigned tiles and the original
# sentences. Each reassignment preserves the inputs and balances both donor labels
# within each original label. The other input should still predict its own target.
#
# ## Result
#
# Each target has a 50 percent chance baseline. High intact accuracy and a selective
# drop after reassignment show which field carries each label. Generalization here
# covers new instructions and new designs from these two topics and two patterns.
#
# ## Method
#
# The fixture contains 48 original sentences and 48 procedural tiles. Each modality
# contributes 24 training, eight validation and sixteen test identities. Colors,
# orientation, phase and spacing vary by design and are matched across pattern classes.
# A pinned MiniLM mean-pools sentence tokens. DINOv2-Small supplies one CLS vector
# per image. Each 384-dimensional vector is L2-normalized. Separate PCA bases and
# scales are fitted to training inputs alone. The eight retained coordinates are
# stored as float16 in the companion `P122.npz` and restored as float32. The authoring command in
# `evidence/fixtures/media.py` regenerates the asset from the pinned checkpoints.
# The experiment primes the native Text and Image caches from this asset before each
# split is used. Any attempt to run the placeholder encoder fails the experiment.
# One model trains for 100 updates. Predictions and expected outcomes are retained
# for each condition, with the identity of each held-out input.
#
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["lightning", "numpy", "polars", "pyyaml>=6", "relflow[image,text]>=0.5.3", "tokenizers", "torch", "transformers>=5.9"]
#
# [tool.relflow.evidence]
# capability = "representation"
# types = ["Text", "Image"]
# conditions = ["unseen-inputs", "changed-associations", "representations"]
# ///

"""P122 measures learning through the native Text and Image embedding caches."""

import hashlib
import json
import sys
from contextlib import ExitStack
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

BUDGET = 100
FIXTURE = Path(__file__).with_suffix(".npz")
SHA256 = "83131bc974a45c7c9e525889bc1fe0857c6ed2c56883c81e7fcd92dd910c5e93"


def cache_model(directory, notes, width):
    """Provide the encoder width and tokenizer locally. No backbone forward is permitted."""
    vocab = {"[PAD]": 0, "[UNK]": 1, **{text: i + 2 for i, text in enumerate(notes)}}
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
        note = rf.Text(model=str(directory), max_length=2, store=str(directory / "notes"))
        tile = rf.Image(model=str(directory), immutable=True, base=str(directory), store=str(directory / "tiles"))
        topic = rf.Boolean(predict=True)
        pattern = rf.Boolean(predict=True)

    model = Model(preset=CONTROL.name)

    def reject_forward(module, args):
        raise RuntimeError("P122 cache miss reached the frozen encoder")

    for backbone in {model.nodes[path].embedder._cached_model.model for path in ("/note", "/tile")}:
        backbone.register_forward_pre_hook(reject_forward)
    return model


def prime(model, data, split, device):
    """Bind vectors to the IDs produced by normal tensorization and verify readback."""
    selected = data["split"] == split
    fields = model.encode(pl.DataFrame({name: data[name][selected] for name in ("note", "tile")}))
    for name in ("note", "tile"):
        expected = data[name + "_vectors"][selected].astype("<f4")
        field = fields["/" + name]
        embedder = model.nodes["/" + name].embedder
        store = embedder.store
        namespace = f"{store.vector_namespace}:{embedder.variant(device)}"
        ids = field.content["ids"].flatten().cpu().numpy()
        if len(np.unique(ids)) != len(ids) or not (store.client.fetch(ids, namespace).status == 0).all():
            raise ValueError(f"{name} {split} must introduce distinct, uncached identities")
        store.client.publish(ids, namespace, Payloads.fixed(expected, expected.shape[1] * 4))
        restored = embedder.encode(field.content.to(device), field.state.to(device)).reshape(expected.shape)
        if not np.array_equal(restored.cpu().numpy(), expected):
            raise ValueError(f"{name} {split} cache lookup changed a projected vector")


def records(data, indices, note_donors, tile_donors):
    for i, note in zip(indices, note_donors, strict=True):
        for j, tile in zip(indices, tile_donors, strict=True):
            yield dict(
                note=data["note"][note],
                tile=data["tile"][tile],
                topic=bool(data["label"][i]),
                pattern=bool(data["label"][j]),
            )


def reassign(indices, labels, rng):
    """Permute the same keys, balancing both donor classes within every recipient class."""
    groups = [rng.permutation(np.flatnonzero(labels[indices] == label)) for label in (False, True)]
    halves = [np.split(indices[positions], 2) for positions in groups]
    donors = indices.copy()
    for group, positions in enumerate(groups):
        donors[positions] = rng.permutation(np.concatenate([parts[group] for parts in halves]))
    return donors


@experiment("P122", seed=12201)
def run(seed, steps, accelerator):
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"P122 embedding fixture differs from its pinned SHA-256: {digest}")
    with np.load(FIXTURE, allow_pickle=False) as archive:
        data = dict(archive)
    indices = {s: np.flatnonzero(data["split"] == s) for s in ("train", "validation", "test")}
    train, val, test = indices.values()
    rng = np.random.default_rng(seed)
    device = torch.device("cpu")
    if accelerator in ("auto", "gpu") and torch.cuda.is_available():
        device = torch.device("cuda")
    elif accelerator in ("auto", "gpu", "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    lit.seed_everything(seed, workers=True)
    with TemporaryDirectory() as work, ExitStack() as caches:
        model = cache_model(Path(work), data["note"].tolist(), data["note_vectors"].shape[1]).to(device)
        for name in ("note", "tile"):
            caches.callback(model.nodes["/" + name].embedder.store.owner.close)
        for split in ("train", "validation"):
            prime(model, data, split, device)
        training = pl.DataFrame(list(records(data, train, train, train))).sample(fraction=1, shuffle=True, seed=seed)
        validation = pl.DataFrame(list(records(data, val, val, val)))
        module = rf.PolarsDataModule(
            train=training, validate=validation, seed=seed, num_workers=0, persistent_workers=False, pin_memory=False
        )
        model.fit(
            module,
            batch_size=32,
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
        conditions, observations = {}, {}
        for condition, note, tile in (
            ("intact", test, test),
            ("shuffle_note", reassign(test, data["label"], rng), test),
            ("shuffle_tile", test, reassign(test, data["label"], rng)),
        ):
            frame = pl.DataFrame(list(records(data, test, note, tile)))
            output = model.predict(frame.select("note", "tile"), batch_size=64)["predictions"]
            conditions[condition], observations[condition] = {}, {}
            for target in ("topic", "pattern"):
                probability = output.struct.field(target).struct.field("content").struct.field("probability").to_numpy()
                if not np.isfinite(probability).all():
                    raise ValueError(f"{condition} produced nonfinite {target} probabilities")
                conditions[condition][target] = {
                    "accuracy": float(np.mean((probability >= 0.5) == frame[target].to_numpy()))
                }
                observations[condition][target] = probability
            observations[condition]["note_indices"] = note
            observations[condition]["tile_indices"] = tile
        counts = {name: model.nodes["/" + name].embedder.store.stats()["results"] for name in ("note", "tile")}
    checks = {
        "full training budget": model.trainer.global_step == BUDGET,
        "one cached vector per identity": all(n == len(data["label"]) for n in counts.values()),
    }
    for target, own, other in (("topic", "note", "tile"), ("pattern", "tile", "note")):
        intact = conditions["intact"][target]["accuracy"]
        shuffled = conditions["shuffle_" + own][target]["accuracy"]
        checks[f"{target} intact accuracy at least 0.80"] = intact >= 0.80
        checks[f"{target} reassigned input within 0.10 of chance"] = abs(shuffled - 0.5) <= 0.10
        checks[f"{target} accuracy gain over reassignment at least 0.25"] = intact - shuffled >= 0.25
        checks[f"{target} preserved when the other input changes"] = (
            abs(intact - conditions["shuffle_" + other][target]["accuracy"]) <= 0.05
        )
    return dict(
        conditions=conditions,
        observations=observations,
        expected=frame.select("topic", "pattern").to_dict(as_series=False),
        held_out=test,
        steps=model.trainer.global_step,
        pairs={"train": len(training), "validation": len(validation), "test": len(frame)},
        fixture=dict(
            sha256=digest,
            bytes=FIXTURE.stat().st_size,
            width=data["note_vectors"].shape[1],
            encoders=json.loads(data["metadata"].item()),
            cached=counts,
        ),
    ), checks


if __name__ == "__main__":
    report(run)
