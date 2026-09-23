# %% [markdown]
# ---
# title: Transferring reconstruction embeddings to a small-label task
# categories: [Representation]
# proof-id: P073
# description: Test whether masked reconstruction makes exported root embeddings useful for a new classifier with few labels.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P073 status >}}
#
# ## Insights
#
# Reconstruction can reward a shared signal across independently noisy views.
# The downstream class never enters the schema or reconstruction loss. A frozen
# encoder is evaluated by the same small nearest-centroid classifier used for
# raw features and encoder controls. This tests a bounded transfer hypothesis;
# setting `embed=True` alone does not establish useful representation learning.
# Raw features can already expose this synthetic signal, so outperforming that
# baseline is measured rather than assumed.
#
# Two of three CPU runs met every gate. Reconstruction embeddings scored
# 59.38–82.76% accuracy, versus 65.04–66.99% for the untrained encoder and
# 34.57–53.22% after training with independently permuted views. Reconstruction
# beat the permuted control in every seed, but seed 7301 fell below both the
# 70% accuracy gate and its untrained encoder. Raw features scored 82.18–85.16%:
# reconstruction exceeded them in only one seed, by 0.24 percentage points.
# These runs show that shared reconstruction signal can help, while the
# exported geometry and small-label transfer remain sensitive to the seed.

# %%
"""P073: frozen, reconstruction-only embeddings with small-label controls."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P073"
BUDGET = 600
TRAIN_ROWS = 4096
TEST_ROWS = 2048
LABELS_PER_CLASS = 8
CODES = ("arch", "bay", "cove", "dune", "elm", "ford", "glen", "hill")
CENTERS = np.asarray([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])

# %% [markdown]
# ## Process and observability
#
# Independent records have balanced hidden classes `c` in `{0,1,2,3}`. Each
# Enum view chooses one of two synonyms for `c`, with a 25% chance of replacing
# it by an independently drawn class. The Vector view has the class's corner
# of a square in its first two coordinates, plus Gaussian noise of standard
# deviation 0.8. Its other 14 coordinates are independent noise of standard
# deviation 2. Each view is imperfect; their common predictive component is c.
#
# ```yaml
# first: arch
# second: bay
# geometry: [-0.8, -1.2, 0.4, -1.0, 0.3, 1.5, -0.7, 0.9, 0.2, -0.8, 0.6, 1.4, -0.2, 0.5, -1.1, 0.7]
# ```
#
# Both codes support class zero. A noisy replacement can disagree:
#
# ```yaml
# first: glen
# second: bay
# geometry: [-0.8, -1.2, 0.4, -1.0, 0.3, 1.5, -0.7, 0.9, 0.2, -0.8, 0.6, 1.4, -0.2, 0.5, -1.1, 0.7]
# ```
#
# The negative training control independently permutes each complete view
# between records. All marginals survive, but their common cause is removed:
#
# ```yaml
# first: glen
# second: cove
# geometry: [-0.8, -1.2, 0.4, -1.0, 0.3, 1.5, -0.7, 0.9, 0.2, -0.8, 0.6, 1.4, -0.2, 0.5, -1.1, 0.7]
# ```
#
# ```{typst}
# //| label: fig-proof-reconstruction-transfer
# //| fig-cap: "Three noisy views reconstruct one another; the exported root supports a separate frozen probe."
# //| fig-alt: "Record exports a root embedding and contains two Enum fields and a sixteen-dimensional Vector, each with a sampled reconstruction mask. The downstream class is absent from the schema."
# #tree(node("record", kind: "root", body: [Export frozen embedding], children: (
#   node("first", type: "Enum", body: [Noisy synonym; sampled reconstruction]),
#   node("second", type: "Enum", body: [Independent noisy synonym]),
#   node("geometry", type: "Vector", body: [Two signal and fourteen noise coordinates]),
# )))
# ```


# %%
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
    return rf.Model.xs(
        batch_size=64,
        embed=True,
        first=rf.Enum(values=CODES, p_unavailable=0.0, mask=mask),
        second=rf.Enum(values=CODES, p_unavailable=0.0, mask=mask),
        geometry=rf.Vector(n_dim=16, mask=mask),
    )


def embedding(model: rf.Model, rows: list[dict]) -> np.ndarray:
    output = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    return np.asarray([row["/"]["embedding"] for row in output], dtype=np.float64).reshape(len(rows), -1)


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


# %% [markdown]
# ## Training and predeclared measurements
#
# Two identically initialized xs models receive 600 AdamW updates at learning
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


# %%
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
            model.optimizer = rf.adamw(learning_rate=0.002)
            data = rf.SyntheticDataModule(
                model=model,
                train=partial(records, rows=TRAIN_ROWS, seed=seed * 100 + 1, permuted=name == "permuted_views"),
                validate=partial(records, rows=512, seed=seed * 100 + 2, permuted=name == "permuted_views"),
                seed=seed,
            )
            trainer = lit.Trainer(
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
            trainer.fit(model, datamodule=data)
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


# %% [markdown]
# ## Evidence and limits
#
# {{< proof P073 evidence >}}
#
# The labels describe the same latent process that generated pretraining
# records, and all splits share that process. This does not establish transfer
# to new domains, general semantic embeddings, or a universal advantage over
# supervised learning. A nearest-centroid failure can reflect representation
# geometry as well as information loss. Gates are fixed before initial runs.
# The three-seed panel does not justify a reliable advantage over either raw
# features or the untrained encoder. Gates remain provisional pending ten
# seeds; failed runs are retained with the original budget and thresholds.
#
# ## Reproduce
#
# {{< proof P073 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=7301)
