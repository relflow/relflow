# %% [markdown]
# ---
# title: Do reconstruction embeddings help a new small-label task?
# categories:
# - Representation
# evidence-id: P073
# description: Learn to reconstruct three noisy views of a hidden class, then freeze the encoder and classify using
#   eight labeled examples per class.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
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
# evidence-result:
#   accelerator: cpu
#   checks:
#     Reconstruction embeddings reach 0.70 accuracy with eight labels per class: true
#     Reconstruction exceeds permuted_views accuracy by 0.05: true
#     Reconstruction exceeds untrained accuracy by 0.05: false
#     Shuffled probe labels remove class accuracy: true
#     'permuted_views: export is finite and unit-normalized': true
#     'permuted_views: export preserves every frozen parameter': true
#     'reconstruction: export is finite and unit-normalized': true
#     'reconstruction: export preserves every frozen parameter': true
#     'untrained: export is finite and unit-normalized': true
#     'untrained: export preserves every frozen parameter': true
#   duration_seconds: 49.34
#   metrics:
#     arms:
#       permuted_views:
#         accuracy: 0.3984375
#         embedding_dimensions: 64
#         norm_max_error: 1.332780972251868e-07
#         shuffled_probe_accuracy: 0.2696990966796875
#         shuffled_probe_repetitions: 32
#         steps: 600
#       reconstruction:
#         accuracy: 0.79541015625
#         embedding_dimensions: 64
#         norm_max_error: 1.1550320799802449e-07
#         shuffled_probe_accuracy: 0.224578857421875
#         shuffled_probe_repetitions: 32
#         steps: 600
#       untrained:
#         accuracy: 0.76708984375
#         embedding_dimensions: 64
#         norm_max_error: 9.627640307208196e-08
#         shuffled_probe_accuracy: 0.2564849853515625
#         shuffled_probe_repetitions: 32
#         steps: 0
#     chance_accuracy: 0.25
#     gain_over_permuted_views: 0.39697265625
#     gain_over_raw: -0.0263671875
#     gain_over_untrained: 0.0283203125
#     probe_labels_per_class: 8
#     raw_accuracy: 0.82177734375
#     test_rows: 2048
#     train_rows: 4096
#     validation_rows: 512
#   mode: full
#   outcome: not_met
#   provenance:
#     code_sha256: ec459ce426ada228a606edf3b69c04c23f912aa769250d1e1dfe9bcba6160f07
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: af95bca1b40f
#     packages:
#       lightning: 2.6.4
#       numpy: 2.4.6
#       polars: 2.0.0
#       pyarrow: 24.0.0
#       pydantic: 2.13.4
#       relflow: 0.4.1
#       tensordict: 0.12.4
#       torch: 2.12.0+cpu
#       torchmetrics: 1.9.0
#     platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.41
#     python: 3.12.16
#     relflow_path: /tmp/runtime/lib/python3.12/site-packages/relflow
#     relflow_revision: 41d6f5fb74c0ba9f0be104c896913a55adb125f5
#     relflow_source_sha256: 00fdf397aeefb3a4e705d431850e0a6294eef142989513fddcb903bb1aefa873
#     script_sha256: 5ff9b2147d4e5cdd483537c672049730cff993a2194a1e35d698622cf213bb70
#     source_sha256: e2220cca6326a17db1b7008f3df8c9b2a60f34928919ec15a25e7314005f93e9
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 7301
#   started_at: '2026-11-01T01:49:13.741717+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41261482839-1
#   candidate:
#     revision: 41d6f5fb74c0ba9f0be104c896913a55adb125f5
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   baseline:
#     revision: 1add6d8919e686dc63b7fa6c4cdbdc437be2b570
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#   manifest_sha256: f4b3bfe706e43e38785ab5bed94c3f51eb44fe7314d0a1b2f0dd3c6f3f7bef92
#   passed: true
#   github_run:
#     event: pull_request
#     head_branch: dev/evidence-in-docs
#     head_sha: 8bb7764d8186c743eb554cb4a91dedb205ad4cb9
#     html_url: https://github.com/relflow/relflow/actions/runs/41261482839
#     pull_requests:
#     - base_sha: 1add6d8919e686dc63b7fa6c4cdbdc437be2b570
#       head_sha: 8bb7764d8186c743eb554cb4a91dedb205ad4cb9
#       number: 64
#     run_attempt: 1
#     run_id: 41261482839
#   release: https://github.com/relflow/docs/releases/tag/ci-41261482839-attempt-1
#   reports_sha256: bc67a1cb3725cbf9c5a8deb65844297d70f6d2122cd3b7de5c1d15f56603864f
# ---
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
# ```{typst}
# //| label: fig-evidence-reconstruction-transfer
# //| fig-cap: "Selector notes identify hidden values. Each view is sometimes hidden; the downstream class is absent."
# //| fig-alt: "Model tree with record, first, second, geometry. Selector notes identify hidden values. Each view is sometimes hidden; the downstream class is absent."
# #tree(node("record", kind: "root", body: [Export frozen embedding], children: (
#   node("first", type: "Enum", body: [Noisy synonym; sampled reconstruction]),
#   node("second", type: "Enum", body: [Independent noisy synonym]),
#   node("geometry", type: "Vector", body: [Two signal and fourteen noise coordinates]),
# )))
# ```
#
# ## Comparison
#
# Use the same simple classifier on raw features, an untrained encoder, and an encoder trained with the views shuffled independently.
#
# ## Result
#
# {{< evidence P073 status >}}
#
# Only two of three seeds pass every check. Reconstruction can help, but raw features are stronger in two seeds; exporting an embedding alone does not make it useful.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P073 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P073 script >}}
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
#
# ### Complete experiment code
#

# %%
"""P073: frozen, reconstruction-only embeddings with small-label controls."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P073"
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
    return rf.Model.xs(
        embed=True,
        first=rf.Enum(values=CODES, p_unavailable=0.0, mask=mask),
        second=rf.Enum(values=CODES, p_unavailable=0.0, mask=mask),
        geometry=rf.Vector(n_dim=16, mask=mask),
    )


def embedding(model: rf.Model, rows: list[dict]) -> np.ndarray:
    output = model.predict(pa.Table.from_pylist(rows))["representations"].to_pylist()
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


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7301)

# %% [markdown]
# </details>
