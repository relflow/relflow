# %% [markdown]
# ---
# title: Do reconstruction embeddings help a new small-label task?
# description: Learn to reconstruct three noisy views of a hidden class, then freeze the encoder and classify using
#   eight labeled examples per class.
# evidence-readout:
# - label: Reconstruction embedding
#   metric:
#   - arms
#   - reconstruction
#   - accuracy
#   format: percent
# - label: Untrained encoder
#   metric:
#   - arms
#   - untrained
#   - accuracy
#   format: percent
# - label: Raw features
#   metric:
#   - raw_accuracy
#   format: percent
# evidence-group: Representation
# evidence-review:
#   question: Do reconstruction embeddings help a new small-label task?
#   modeling: 'Generate three noisy views of a hidden class: two categorical views and a vector. Pretrain an encoder
#     by hiding and reconstructing those views, then freeze it. Give the same nearest-centroid classifier eight labeled
#     examples per class and compare with untrained and shuffled-view encoders.'
#   comparison:
#     metric: Accuracy
#     kind: Controlled comparison
#     conditions:
#     - label: Untrained encoder
#       path:
#       - arms
#       - untrained
#       - accuracy
#       format: percent
#     - label: Shuffled-view pretraining
#       path:
#       - arms
#       - permuted_views
#       - accuracy
#       format: percent
#     - label: Reconstruction pretraining
#       path:
#       - arms
#       - reconstruction
#       - accuracy
#       format: percent
#     definition: Fraction of predictions matching the expected answer, over the population named in each condition.
#     direction: Higher is better
#     domain:
#     - 0
#     - 1
#   note: The hidden class is never a reconstruction input or target. Every encoder uses the same small labeled probe;
#     all required control margins still count toward the verdict.
#   criteria:
#     prefixes:
#       untrained: Untrained encoder
#       reconstruction: Reconstruction pretraining
#       permuted_views: Shuffled-view pretraining
# evidence-plot:
#   target: class
#   focus: Do reconstruction embeddings improve a separate held-out probe?
#   comparison: Compare trained, untrained, permuted-view, and raw representations under the same probe labels.
#   limit: A reconstruction objective need not produce the geometry required by the downstream class task.
#   geometry:
#   - reconstruction
#   - untrained
#   supplements:
#     geometry:
#       mode: representation-geometry
#   renderer: plot
# evidence-cost:
#   baseline: 46
#   candidate: 47
# ---
#
# The hidden class influences three imperfect views of a record, but the class label is absent during
# reconstruction training. Later, only a small separate labeled reference set is available. The
# experiment asks whether reconstructing shared structure makes a frozen embedding more useful for
# that later classification task than simpler representations.
#
# ## Example
#
# ```yaml
# first: arch
# second: bay
# geometry: [-0.8, -1.2, 0.4, -1.0, 0.3, 1.5, -0.7, 0.9, 0.2, -0.8, 0.6, 1.4, -0.2, 0.5, -1.1, 0.7]
# ```
#
# The hidden class is used only for the later classifier. During reconstruction, parts of the three visible views are sampled for hiding.
#
#
# The illustrated `arch` and `bay` are two possible synonyms associated with the same generating
# class. The first two geometry coordinates carry a noisy class-specific square corner, while fourteen
# additional coordinates carry independent noise. Any one view can be misleading, so a useful
# representation should retain their shared signal rather than merely reproduce the strongest-looking
# individual value.
#
# ## Comparison
#
# Use the same simple classifier on raw features, an untrained encoder, and an encoder trained with the views shuffled independently.
#
# All representations are judged by the same nearest-centroid classifier with eight labeled reference
# examples per class. Controls use the untrained encoder, reconstruction training with independently
# shuffled views, and standardized raw features. View shuffling preserves each field's distribution
# but removes cross-view agreement. Shuffling the reference labels separately checks whether the small
# probe itself supplies a spurious association.
#
# ## Result
#
# The selected run improves on the untrained encoder but misses the required 0.05 accuracy margin. Raw features score higher here; exporting an embedding alone does not establish useful transfer.
#
# Accuracy is measured on independent records after freezing the encoder; no downstream gradient
# update improves it. The selected run misses the required margin over the untrained encoder, and
# raw-feature performance remains an important comparison. Finite, unit-length embeddings and
# successful export are mechanical properties, not evidence that reconstruction necessarily gives the
# best representation for a later task.
#
# ## Method
#
# ### Process and observability
#
# Independent records have balanced hidden classes `c` in `{0,1,2,3}`. Each
# Enum view chooses one of two synonyms for `c`, with a 25% chance of replacing
# it by an independently drawn class. The Vector view has the class's corner
# of a square in its first two coordinates, plus Gaussian noise of standard
# deviation 0.8. Its other 14 coordinates are independent noise of standard
# deviation 2. Each view is imperfect; their common predictive component is c.
#
# ### Training and predeclared measurements
#
# Two identically initialized frozen control models receive 600 AdamW updates at learning
# rate 0.002, batch size 64: ordinary or independently permuted views. Each
# field is sampled for reconstruction with probability 0.35. The untrained
# control has the same initialization. Declared Enum vocabularies and unscaled
# Vectors make all three usable without training-dependent preprocessing.
#
# Training/validation/test streams contain 4,096/512/2,048 rows. A fourth
# independent stream supplies exactly eight labeled examples per class. Each
# encoder is frozen before exporting representations. The same Euclidean
# nearest-centroid rule probes each encoder. Raw features concatenate two
# one-hot codes and the full Vector; only this raw baseline is standardized,
# using unlabeled training means and scales. Nothing is fit on test records.
#
# Provisional gates require reconstruction-trained accuracy at least 0.70 and
# advantages of at least 0.05 over both encoder controls. Beating raw features
# is diagnostic. Mean accuracy across 32 shuffled probe-label assignments
# should score below 0.35; one tiny probe can accidentally align with a class.
# Frozen exports
# must be finite, unit-normalized, and leave all parameters unchanged. These
# gates can fail: failure limits the transfer claim without changing the task.
#

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.10",
#     "lightning",
#     "numpy",
#     "polars",
#     "pyarrow",
#     "pyyaml>=6",
#     "relflow>=0.5.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "training"
# types = []
# conditions = ["changed-associations", "transfer", "objectives"]
# ///

"""P073: frozen, reconstruction-only embeddings with small-label controls."""

import sys
from collections.abc import Iterator
from functools import partial
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
import torch
from policies import CONTROL
from reporting import experiment, report

BUDGET = 600
TRAIN_ROWS = 4096
TEST_ROWS = 2048
LABELS_PER_CLASS = 8
CODES = ("arch", "bay", "cove", "dune", "elm", "ford", "glen", "hill")
CENTERS = np.asarray([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])


def sample(*, rows: int, seed: int, permuted: bool = False) -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.resize(np.arange(4), rows)
    rng.shuffle(labels)
    codes = []
    for _ in range(2):
        noisy = np.where(rng.random(rows) < 0.25, rng.integers(4, size=rows), labels)
        codes.append(2 * noisy + rng.integers(2, size=rows))
    geometry = rng.normal(0.0, 2.0, (rows, 16))
    geometry[:, :2] = CENTERS[labels] + rng.normal(0.0, 0.8, (rows, 2))
    first, second = codes
    if permuted:
        first = first[rng.permutation(rows)]
        second = second[rng.permutation(rows)]
        geometry = geometry[rng.permutation(rows)]
    records = [
        {"first": CODES[a], "second": CODES[b], "geometry": vector.tolist()}
        for a, b, vector in zip(first, second, geometry, strict=True)
    ]
    return records, labels


def records(*, rows: int, seed: int, permuted: bool = False) -> Iterator[dict]:
    # Labels stay outside the training stream and the model's schema.
    yield from sample(rows=rows, seed=seed, permuted=permuted)[0]


def build() -> rf.Model:
    mask = rf.Mask(rate=0.35, reconstruct=True)

    class Model(rf.Model):
        first = rf.Enum(values=CODES, p_unavailable=0.0, mask=mask)
        second = rf.Enum(values=CODES, p_unavailable=0.0, mask=mask)
        geometry = rf.Vector(n_dim=16, mask=mask)

    model = Model(preset=CONTROL.name, embed=True)
    return model


def embedding(model: rf.Model, rows: list[dict]) -> np.ndarray:
    output = pl.DataFrame(model.predict(pa.Table.from_pylist(rows)))["representations"].to_list()
    roots = [
        sorted(
            (record for record in row if record["path"] == [] and record["parents"] == []),
            key=lambda record: record["output_index"],
        )
        for row in output
    ]
    return np.asarray([[record["embedding"] for record in root] for root in roots], dtype=np.float64).reshape(
        len(rows), -1
    )


def raw_features(rows: list[dict]) -> np.ndarray:
    identity = np.eye(len(CODES))
    return np.asarray(
        [
            np.concatenate((identity[CODES.index(row["first"])], identity[CODES.index(row["second"])], row["geometry"]))
            for row in rows
        ]
    )


def classify(reference: np.ndarray, labels: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Fit four centroids using only the supplied small labeled reference set."""
    centroids = np.stack([reference[labels == label].mean(axis=0) for label in range(4)])
    return np.square(queries[:, None, :] - centroids[None, :, :]).sum(axis=-1).argmin(axis=1)


@experiment("P073", seed=7301)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train, _ = sample(rows=TRAIN_ROWS, seed=seed * 100 + 1)
    reference, reference_labels = sample(rows=4 * LABELS_PER_CLASS, seed=seed * 100 + 3)
    test, test_labels = sample(rows=TEST_ROWS, seed=seed * 100 + 4)
    probe_rng = np.random.default_rng(seed * 100 + 5)
    shuffled_labels = [probe_rng.permutation(reference_labels) for _ in range(32)]
    metrics = {
        "train_rows": TRAIN_ROWS,
        "validation_rows": 512,
        "test_rows": TEST_ROWS,
        "probe_labels_per_class": LABELS_PER_CLASS,
        "chance_accuracy": 0.25,
        "arms": {},
    }
    checks = {}
    for name in ("untrained", "reconstruction", "permuted_views"):
        lit.seed_everything(seed, workers=True)
        model = build()
        trained_steps = 0
        if name != "untrained":
            data = rf.SyntheticDataModule(
                train=partial(records, rows=TRAIN_ROWS, seed=seed * 100 + 1, permuted=name == "permuted_views"),
                validate=partial(records, rows=512, seed=seed * 100 + 2, permuted=name == "permuted_views"),
                seed=seed,
            )
            model.fit(
                data,
                batch_size=64,
                learning_rate=0.002,
                early_stopping=False,
                rollback=False,
                precision="32-true",
                strategy="auto",
                accelerator=accelerator,
                devices=1,
                max_epochs=-1,
                max_steps=BUDGET if steps is None else min(steps, BUDGET),
                deterministic=True,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                num_sanity_val_steps=0,
            )
            trainer = model.trainer
            trained_steps = trainer.global_step
        model.eval().requires_grad_(False)
        parameters = {key: value.detach().cpu().clone() for key, value in model.named_parameters()}
        reference_vectors = embedding(model, reference)
        test_vectors = embedding(model, test)
        guessed = classify(reference_vectors, reference_labels, test_vectors)
        shuffled_accuracies = [
            float(np.mean(classify(reference_vectors, labels, test_vectors) == test_labels))
            for labels in shuffled_labels
        ]
        accuracy = float(np.mean(guessed == test_labels))
        metrics["arms"][name] = {
            "steps": trained_steps,
            "accuracy": accuracy,
            "shuffled_probe_accuracy": float(np.mean(shuffled_accuracies)),
            "shuffled_probe_repetitions": len(shuffled_labels),
            "embedding_dimensions": test_vectors.shape[1],
            "norm_max_error": float(np.max(np.abs(np.linalg.norm(test_vectors, axis=1) - 1.0))),
        }
        checks[f"{name}: export is finite and unit-normalized"] = (
            bool(np.isfinite(test_vectors).all()) and metrics["arms"][name]["norm_max_error"] < 1e-5
        )
        checks[f"{name}: export preserves every frozen parameter"] = all(
            torch.equal(parameters[key], value.detach().cpu()) for key, value in model.named_parameters()
        )
    raw_train = raw_features(train)
    mean, scale = raw_train.mean(axis=0), np.maximum(raw_train.std(axis=0), 1e-8)
    raw_prediction = classify(
        (raw_features(reference) - mean) / scale, reference_labels, (raw_features(test) - mean) / scale
    )
    metrics["raw_accuracy"] = float(np.mean(raw_prediction == test_labels))
    trained = metrics["arms"]["reconstruction"]["accuracy"]
    metrics["gain_over_raw"] = trained - metrics["raw_accuracy"]
    checks["Reconstruction embeddings reach 0.70 accuracy with eight labels per class"] = trained >= 0.70
    for control in ("untrained", "permuted_views"):
        gain = trained - metrics["arms"][control]["accuracy"]
        metrics[f"gain_over_{control}"] = gain
        checks[f"Reconstruction exceeds {control} accuracy by 0.05"] = gain >= 0.05
    checks["Shuffled probe labels remove class accuracy"] = (
        metrics["arms"]["reconstruction"]["shuffled_probe_accuracy"] < 0.35
    )
    return metrics, checks


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    from evidence.diagnostics.data import human
    from evidence.diagnostics.plots import axes_pair, dots

    m = bundle.record["metrics"]
    fig, axes = axes_pair(width=10, height=4.8)
    arms = list(m["arms"])
    dots(
        axes[0],
        [human(arm) for arm in arms] + ["Prepared raw features"],
        {"Probe accuracy": [m["arms"][arm]["accuracy"] for arm in arms] + [m["raw_accuracy"]]},
        xlabel="Accuracy with eight labeled references per class",
        reference=(m["chance_accuracy"], "Four-class chance"),
        percent=True,
    )
    dots(
        axes[1],
        [human(arm) for arm in arms],
        {"Shuffled-probe accuracy": [m["arms"][arm]["shuffled_probe_accuracy"] for arm in arms]},
        xlabel="Accuracy after shuffling reference labels",
        reference=(m["chance_accuracy"], "Four-class chance"),
        percent=True,
    )
    for ax in axes:
        ax.set_xlim(0, 1.02)
    return (
        fig,
        "All representation arms use the same held-out labels and the same small reference-label budget. The raw-feature baseline and shuffled reference labels expose gains that reconstruction loss alone cannot establish.",
        {},
    )


if __name__ == "__main__":
    report(run)
