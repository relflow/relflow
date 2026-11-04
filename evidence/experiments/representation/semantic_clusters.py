# %% [markdown]
# ---
# title: Do reconstruction embeddings separate unseen transaction styles?
# categories:
# - Representation
# evidence-id: P080
# description: Fit four clusters without labels, then measure their agreement with hidden transaction styles
#   on independent records and compare trained, untrained, shuffled-view, and raw representations.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Trained embedding ARI
#   metric:
#   - mean
#   - aligned
#   - ari
#   format: number
# - label: Untrained embedding ARI
#   metric:
#   - mean
#   - untrained
#   - ari
#   format: number
# - label: Raw feature ARI
#   metric:
#   - mean
#   - raw
#   - ari
#   format: number
# evidence-group: Representation
# ---
#
# ## Example
#
# ```yaml
# merchant_pattern: household
# payment_rhythm: frequent
# cashflow: [-0.8, -1.2, 0.4, -1.0, 0.3, 1.5, -0.7, 0.9, 0.2, -0.8, 0.6, 1.4, -0.2, 0.5, -1.1, 0.7]
# ```
#
# An account has three imperfect views of its spending style. The hidden style
# is never part of the record or model; only the evaluator receives it.
#
# ```{typst}
# //| label: fig-evidence-semantic-clusters
# //| fig-cap: "Train by reconstructing sampled fields, freeze the root embedding, then cluster without labels."
# //| fig-alt: "Account model with merchant pattern and payment rhythm Enum fields and a sixteen-coordinate cashflow Vector. The root exports one frozen embedding; hidden style labels stay outside the model."
# #tree(node("account", kind: "root", body: [Export one frozen embedding], children: (
#   node("merchant_pattern", type: "Enum", body: [Noisy spending view; sampled reconstruction]),
#   node("payment_rhythm", type: "Enum", body: [Independent noisy payment view]),
#   node("cashflow", type: "Vector(16)", body: [Two signal and fourteen nuisance coordinates]),
# )))
# ```
#
# ## Comparison
#
# Compare the same fixed clustering and neighborhood rules on reconstruction
# embeddings, identically initialized untrained embeddings, embeddings trained
# with independently shuffled views, and all raw input coordinates.
#
# ## Result
#
# {{< evidence P080 status >}}
#
# This experiment tests four synthetic styles, rather than real financial
# categories. An exported vector alone does not establish useful semantic
# structure; the trained representation need not outperform raw features.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P080 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P080 script >}}
#
# ### What the model can observe
#
# Each independent stream has four balanced hidden styles. Each Enum independently
# chooses one of two style-specific names, with a 25% chance of replacing the
# style by a uniform draw. The first two Vector coordinates locate a style at
# a corner of a square, with Gaussian noise of standard deviation 0.8; fourteen
# nuisance coordinates have standard deviation 2. Labels are never schema
# fields, reconstruction targets, or inputs to clustering.
#
# For each of three seeds, training, validation, and test contain 4,096, 512,
# and 2,048 independently generated rows. Both trained arms receive 600 AdamW
# updates at batch size 64, learning rate 0.002, and weight decay 0.01. Each
# visible field is sampled for reconstruction at rate 0.35. The shuffled-view
# control independently permutes whole views across training rows, preserving
# each marginal distribution while removing their shared signal. All encoder
# arms start from the same initialization and are calibrated on training records
# only. No validation or test score changes training duration or hyperparameters.
#
# ### Frozen export and label-free fitting
#
# Public Arrow prediction exports exactly one unit-normalized root vector per
# input record, in input order, at batch size 128. Training and test exports must
# preserve all parameters, buffers, and extension-owned extra state. Each encoder
# is evaluated on the same complete, aligned records, including the encoder
# trained on shuffled views.
#
# NumPy KMeans fits the unlabeled training exports with four clusters, four
# seeded restarts, and at most 40 Lloyd iterations. Restart selection uses
# training inertia alone. Held-out adjusted Rand index (ARI) measures cluster
# agreement without naming or relabeling clusters. A second measurement is the
# fraction of the five nearest training neighbors sharing a test record's
# hidden style. Neither measurement trains a classifier. Thirty-two fixed
# shuffled test-label assignments provide a chance control after clustering.
#
# Raw features retain both eight-way one-hot codes and all sixteen Vector
# coordinates. Their means and scales fit training rows only; standardized raw
# rows are then normalized to unit length. Model exports are already unit
# vectors and receive no fitted transformation. All distances are squared
# Euclidean, equivalent to cosine ranking for these unit vectors.
#
# ### Predeclared checks and limits
#
# Gates require mean trained ARI at least 0.35, mean five-neighbor agreement
# at least 0.60, absolute mean shuffled-label ARI below 0.05, finite unit
# vectors, noncollapsed trained exports, preserved full state, and complete
# training budgets. Report each seed, means, worst seeds, and paired gains;
# gains over controls are diagnostics rather than requirements. The four-cluster
# count is known from the synthetic generator. These checks cannot establish
# an appropriate number of groups, semantic fairness, or real-world usefulness.
#
# ### Complete experiment code
#

# %%
"""P080: label-free training clusters and held-out semantic agreement."""

import copy
import gc
from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P080"
BUDGET = 600
BATCH_SIZE = 64
TRAIN_ROWS = 4096
VALIDATION_ROWS = 512
TEST_ROWS = 2048
N_CLASSES = 4
MERCHANTS = ("essentials", "household", "travel", "leisure", "services", "utilities", "shopping", "retail")
RHYTHMS = ("daily", "frequent", "weekly", "regular", "monthly", "scheduled", "irregular", "sporadic")
CENTERS = np.asarray([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
ARMS = ("untrained", "aligned", "permuted", "raw")


def sample(*, rows: int, seed: int, permuted: bool = False) -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.resize(np.arange(N_CLASSES), rows)
    rng.shuffle(labels)
    codes = []
    for _ in range(2):
        noisy = np.where(rng.random(rows) < 0.25, rng.integers(N_CLASSES, size=rows), labels)
        codes.append(2 * noisy + rng.integers(2, size=rows))
    cashflow = rng.normal(0.0, 2.0, (rows, 16))
    cashflow[:, :2] = CENTERS[labels] + rng.normal(0.0, 0.8, (rows, 2))
    merchants, rhythms = codes
    if permuted:
        merchants = merchants[rng.permutation(rows)]
        rhythms = rhythms[rng.permutation(rows)]
        cashflow = cashflow[rng.permutation(rows)]
    observations = [
        {"merchant_pattern": MERCHANTS[a], "payment_rhythm": RHYTHMS[b], "cashflow": vector.tolist()}
        for a, b, vector in zip(merchants, rhythms, cashflow, strict=True)
    ]
    return observations, labels


def records(*, rows: int, seed: int, permuted: bool = False) -> Iterator[dict]:
    # Evaluator labels never enter the data module or the model's schema.
    yield from sample(rows=rows, seed=seed, permuted=permuted)[0]


def build() -> rf.Model:
    mask = rf.Mask(rate=0.35, reconstruct=True)
    return rf.Model.xs(
        embed=True,
        reduction=rf.Attention(n_outputs=1),
        merchant_pattern=rf.Enum(values=MERCHANTS, p_unavailable=0.0, mask=mask),
        payment_rhythm=rf.Enum(values=RHYTHMS, p_unavailable=0.0, mask=mask),
        cashflow=rf.Vector(n_dim=16, mask=mask),
    )


def embedding(model: rf.Model, rows: list[dict], *, batch_size: int = 128) -> np.ndarray:
    output = model.predict(pa.Table.from_pylist(rows), batch_size=batch_size)["representations"].to_pylist()
    roots = [[record for record in row if record["path"] == [] and record["parents"] == []] for row in output]
    if len(roots) != len(rows) or any(len(root) != 1 or root[0]["output_index"] != 0 for root in roots):
        raise ValueError("P080 requires one root representation with output index zero per financial record")
    values = np.asarray([root[0]["embedding"] for root in roots], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 64:
        raise ValueError(f"P080 requires the fixed xs root width of 64, received shape {values.shape}")
    return values


def distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances without a rows-by-rows-by-width allocation."""
    values, centers = np.asarray(values, dtype=np.float64), np.asarray(centers, dtype=np.float64)
    if values.ndim != 2 or centers.ndim != 2 or values.shape[1] != centers.shape[1]:
        raise ValueError("Semantic distances require matrices with the same feature width")
    return np.maximum(
        np.square(values).sum(axis=1)[:, None] + np.square(centers).sum(axis=1) - 2 * values @ centers.T, 0
    )


def assign(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return distances(values, centers).argmin(axis=1)


def cluster(values: np.ndarray, seed: int) -> np.ndarray:
    """Fit unlabeled training vectors; select four restarts by training inertia."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < N_CLASSES or not np.isfinite(values).all():
        raise ValueError("Semantic clustering needs at least four finite training vectors")
    rng = np.random.default_rng(seed)
    best, best_inertia = None, float("inf")
    for _ in range(4):
        centers = values[rng.choice(len(values), N_CLASSES, replace=False)].copy()
        for _ in range(40):
            squared = distances(values, centers)
            groups = squared.argmin(axis=1)
            next_centers = centers.copy()
            farthest = np.argsort(squared.min(axis=1), kind="stable")[::-1]
            empty = 0
            for index in range(N_CLASSES):
                selected = values[groups == index]
                if len(selected):
                    next_centers[index] = selected.mean(axis=0)
                else:
                    next_centers[index] = values[farthest[empty]]
                    empty += 1
            if np.allclose(centers, next_centers, atol=1e-10, rtol=0.0):
                centers = next_centers
                break
            centers = next_centers
        inertia = float(distances(values, centers).min(axis=1).sum())
        if inertia < best_inertia:
            best, best_inertia = centers.copy(), inertia
    if best is None:
        raise ValueError("Semantic clustering produced no finite training solution")
    return best


def ari(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Chance-adjusted agreement, independent of arbitrary cluster names."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.ndim != 1 or predicted.shape != actual.shape or len(actual) < 2:
        raise ValueError("ARI needs aligned one-dimensional partitions with at least two records")
    _, left = np.unique(actual, return_inverse=True)
    _, right = np.unique(predicted, return_inverse=True)
    cells = np.zeros((left.max() + 1, right.max() + 1), dtype=np.int64)
    np.add.at(cells, (left, right), 1)
    observed = float(np.sum(cells * (cells - 1) / 2))
    row, column = cells.sum(axis=1), cells.sum(axis=0)
    left_pairs = float(np.sum(row * (row - 1) / 2))
    right_pairs = float(np.sum(column * (column - 1) / 2))
    expected = left_pairs * right_pairs / (len(actual) * (len(actual) - 1) / 2)
    maximum = (left_pairs + right_pairs) / 2
    return 1.0 if maximum == expected else (observed - expected) / (maximum - expected)


def evaluate(reference: np.ndarray, labels: np.ndarray, query: np.ndarray, query_labels: np.ndarray, seed: int) -> dict:
    centers = cluster(reference, seed)
    predicted = assign(query, centers)
    if len(reference) < 5 or len(labels) != len(reference) or len(query_labels) != len(query):
        raise ValueError("Semantic evaluation needs aligned reference/query labels and at least five training rows")
    agreement = []
    # Chunk queries so five-neighbor evaluation stays within the declared CPU budget.
    for start in range(0, len(query), 256):
        nearest = np.argpartition(distances(query[start : start + 256], reference), kth=4, axis=1)[:, :5]
        agreement.append(labels[nearest])
    neighbors = np.concatenate(agreement)
    rng = np.random.default_rng(seed + 1)
    shuffled = [rng.permutation(query_labels) for _ in range(32)]
    null_ari = [ari(permuted, predicted) for permuted in shuffled]
    null_neighbors = [float(np.mean(neighbors == permuted[:, None])) for permuted in shuffled]
    return {
        "ari": ari(query_labels, predicted),
        "neighbor_agreement": float(np.mean(neighbors == query_labels[:, None])),
        "shuffled_ari_mean": float(np.mean(null_ari)),
        "shuffled_ari": null_ari,
        "shuffled_neighbor_agreement": float(np.mean(null_neighbors)),
        "shuffled_neighbor_agreements": null_neighbors,
        "shuffled_repetitions": len(shuffled),
        "occupied_clusters": int(len(np.unique(predicted))),
        "train_inertia": float(distances(reference, centers).min(axis=1).sum()),
    }


def raw_features(rows: list[dict]) -> np.ndarray:
    identity = np.eye(len(MERCHANTS))
    return np.asarray(
        [
            np.concatenate(
                (
                    identity[MERCHANTS.index(row["merchant_pattern"])],
                    identity[RHYTHMS.index(row["payment_rhythm"])],
                    row["cashflow"],
                )
            )
            for row in rows
        ],
        dtype=np.float64,
    )


def same_state(left, right) -> bool:
    """Compare parameters, buffers, and extension-owned tuples, including NaNs."""
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
            return False
        try:
            torch.testing.assert_close(left.detach().cpu(), right.detach().cpu(), atol=0.0, rtol=0.0, equal_nan=True)
        except AssertionError:
            return False
        return True
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(same_state(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(same_state(a, b) for a, b in zip(left, right, strict=True))
        )
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right, equal_nan=True)
    if isinstance(left, float) and np.isnan(left):
        return isinstance(right, float) and np.isnan(right)
    return bool(left == right)


def trial(seed: int, steps: int | None, accelerator: str) -> dict:
    train, train_labels = sample(rows=TRAIN_ROWS, seed=seed * 100 + 1)
    test, test_labels = sample(rows=TEST_ROWS, seed=seed * 100 + 3)
    arms = {}
    initialized = None
    for name in ARMS[:-1]:
        lit.seed_everything(seed, workers=True)
        model = build()
        calibration = sample(rows=TRAIN_ROWS, seed=seed * 100 + 1, permuted=name == "permuted")[0]
        model.encode(pa.Table.from_pylist(calibration), strata="train", seed=seed)
        initial = copy.deepcopy(model.state_dict())
        if initialized is None:
            initialized = initial
        initialization_matches = same_state(initialized, initial)
        trained_steps = 0
        if name != "untrained":
            data = rf.SyntheticDataModule(
                train=partial(records, rows=TRAIN_ROWS, seed=seed * 100 + 1, permuted=name == "permuted"),
                validate=partial(records, rows=VALIDATION_ROWS, seed=seed * 100 + 2, permuted=name == "permuted"),
                seed=seed,
            )
            model.fit(
                data,
                batch_size=BATCH_SIZE,
                learning_rate=0.002,
                weight_decay=0.01,
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
            trained_steps = model.trainer.global_step
        model.eval().requires_grad_(False)
        model.batch_size = 128
        before = copy.deepcopy(model.state_dict())
        reference_vectors = embedding(model, train)
        test_vectors = embedding(model, test)
        measured = evaluate(reference_vectors, train_labels, test_vectors, test_labels, seed * 100 + 4)
        measured.update(
            steps=trained_steps,
            initialization_matches=initialization_matches,
            state_preserved=same_state(before, model.state_dict()),
            finite=bool(np.isfinite(reference_vectors).all() and np.isfinite(test_vectors).all()),
            norm_max_error=float(
                max(
                    np.max(np.abs(np.linalg.norm(reference_vectors, axis=1) - 1.0)),
                    np.max(np.abs(np.linalg.norm(test_vectors, axis=1) - 1.0)),
                )
            ),
            embedding_variance=float(np.var(test_vectors, axis=0).sum()),
            embedding_dimensions=test_vectors.shape[1],
        )
        arms[name] = measured
        del model
        gc.collect()
    raw_train, raw_test = raw_features(train), raw_features(test)
    mean, scale = raw_train.mean(axis=0), np.maximum(raw_train.std(axis=0), 1e-8)
    raw_train, raw_test = (raw_train - mean) / scale, (raw_test - mean) / scale
    raw_train /= np.maximum(np.linalg.norm(raw_train, axis=1, keepdims=True), 1e-12)
    raw_test /= np.maximum(np.linalg.norm(raw_test, axis=1, keepdims=True), 1e-12)
    arms["raw"] = evaluate(raw_train, train_labels, raw_test, test_labels, seed * 100 + 4)
    arms["raw"].update(
        embedding_dimensions=raw_train.shape[1], standardization="training_only", normalization="unit_l2"
    )
    return {"seed": seed, "arms": arms}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    trials = [trial(seed + offset, steps, accelerator) for offset in range(3)]
    names = ("ari", "neighbor_agreement", "shuffled_ari_mean", "shuffled_neighbor_agreement")
    means = {
        arm: {name: float(np.mean([item["arms"][arm][name] for item in trials])) for name in names} for arm in ARMS
    }
    worst = {
        arm: {name: float(min(item["arms"][arm][name] for item in trials)) for name in ("ari", "neighbor_agreement")}
        for arm in ARMS
    }
    gains = {
        control: {
            name: [item["arms"]["aligned"][name] - item["arms"][control][name] for item in trials]
            for name in ("ari", "neighbor_agreement")
        }
        for control in ("untrained", "permuted", "raw")
    }
    models = [item["arms"][arm] for item in trials for arm in ARMS[:-1]]
    checks = {
        "Mean trained held-out ARI reaches 0.35": means["aligned"]["ari"] >= 0.35,
        "Mean trained five-neighbor agreement reaches 0.60": means["aligned"]["neighbor_agreement"] >= 0.60,
        "Every arm has absolute mean shuffled-label ARI below 0.05": all(
            abs(value["shuffled_ari_mean"]) < 0.05 for value in means.values()
        ),
        "All exported model vectors are finite and unit-normalized": all(
            arm["finite"] and arm["norm_max_error"] < 1e-5 for arm in models
        ),
        "Every trained seed has noncollapsed test exports": all(
            item["arms"]["aligned"]["embedding_variance"] > 1e-6 for item in trials
        ),
        "Export preserves every parameter buffer and extension extra state": all(
            arm["state_preserved"] for arm in models
        ),
        "Paired models have identical calibrated initial state": all(arm["initialization_matches"] for arm in models),
        "Both trained arms finish 600 optimizer updates in every seed": all(
            item["arms"][arm]["steps"] == BUDGET for item in trials for arm in ("aligned", "permuted")
        ),
    }
    metrics = {
        "seeds": [item["seed"] for item in trials],
        "train_rows": TRAIN_ROWS,
        "validation_rows": VALIDATION_ROWS,
        "test_rows": TEST_ROWS,
        "cluster_count": N_CLASSES,
        "cluster_restarts": 4,
        "cluster_max_iterations": 40,
        "neighbors": 5,
        "budget": BUDGET,
        "batch_size": BATCH_SIZE,
        "learning_rate": 0.002,
        "weight_decay": 0.01,
        "trials": trials,
        "mean": means,
        "worst": worst,
        "paired_gains": gains,
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=8001)

# %% [markdown]
# </details>
