# %% [markdown]
# ---
# title: Do frozen financial embeddings retain their groups with missing evidence?
# categories:
# - Representation
# description: Reconstruct noisy financial views, fit four clusters on complete training embeddings, and keep those
#   clusters fixed while removing a view or replacing unrelated noise in independent records.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Complete evidence ARI
#   metric:
#   - summary
#   - trained
#   - complete
#   - ari_mean
#   format: number
# - label: Channel absent ARI
#   metric:
#   - summary
#   - trained
#   - one_view_missing
#   - ari_mean
#   format: number
# - label: All semantic views absent ARI
#   metric:
#   - summary
#   - trained
#   - all_semantic_views_missing
#   - ari_mean
#   format: number
# evidence-group: Representation
# ---
#
# A record may contain several clues about the same hidden pattern. Can a frozen
# representation still put similar records together when one clue disappears?
# Here four synthetic styles combine low or high amounts with low or high
# frequencies. A payment-channel code is another noisy clue, while the nuisance
# vector is unrelated to the style.
#
# ## Example
#
# ```yaml
# channel: card
# amount: 32.4
# frequency: 3.1
# nuisance: [0.3, -0.8, 1.1, 0.0, -0.2, 0.5, 0.8, -1.0, 0.2, -0.1, 0.4, 0.9, -0.6, 0.7, -0.3, 0.1]
# ```
#
# Each record has one of four hidden spending styles. Channel, amount, and
# frequency provide noisy clues; the extra vector is independent noise. The
# style label is available only to the evaluator, never to the model.
#
# The example amount 32.4 and frequency 3.1 are consistent with the low-amount,
# low-frequency style, whose centers are 30 and 3. The card channel is another
# imperfect observation; its name does not guarantee a style. All sixteen nuisance
# values are independent random noise.
#
# For the missing-channel comparison, replace only channel with null. Amount and
# frequency still provide evidence. For complete signal removal, also replace
# those two measurements with null, leaving only unrelated noise. Neither change
# supplies the hidden style as a field or label to the model.
#
# ```{typst}
# //| label: fig-evidence-semantic-robustness
# //| fig-cap: "Reconstruct visible financial measurements; export one frozen record embedding. Hidden styles are not fields."
# //| fig-alt: "Record embedding with channel Enum, amount Number, frequency Number, and an independent nuisance Vector. All four fields use sampled reconstruction."
# #tree(node("record", kind: "root", body: [Frozen embedding], children: (
#   node("channel", type: "Enum", body: [Noisy style clue]),
#   node("amount", type: "Number", body: [Low or high amount]),
#   node("frequency", type: "Number", body: [Low or high frequency]),
#   node("nuisance", type: "Vector", body: [Independent noise]),
# )))
# ```
#
# ## Comparison
#
# Fit four clusters on complete training records, then freeze the encoder,
# preprocessing, and cluster centers. Reusing those exact centers makes this a
# check of the original grouping under changed evidence, rather than fitting a
# new grouping separately for every condition.
#
# Resampling nuisance changes irrelevant details while preserving the useful
# clues. Removing channel asks whether the remaining views suffice. Removing all
# three semantic views checks whether apparent agreement survives after the
# signal is gone. Trained, untrained, and raw representations undergo the same
# conditions. ARI near one means matching hidden groups; chance-level agreement
# is near zero, independently of the cluster numbers.
#
#
# Compare reconstruction training, the same untrained encoder, and standardized
# raw fields. Fit each set of four clusters on complete training records only.
# Keep the encoder, preprocessing, and centroids fixed for complete test records,
# an absent channel, resampled nuisance, and all three semantic clues absent.
#
# ## Result
#
# {{< evidence P081 status >}}
#
# The checks distinguish complete evidence, partial evidence, and the no-signal
# control. The same known concepts and vocabulary appear across independent
# records, and the number of clusters is supplied. A later successful result
# would establish robustness in this setup, rather than discovery of new customer
# concepts or superiority to every raw representation.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P081 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P081 script >}}
#
# ### Generator and information boundary
#
# Four balanced latent styles combine amount centers 30 or 100 with frequency
# centers 3 or 12. Independent Gaussian noise has standard deviation 10 and 1.5,
# respectively; positive measurements are clipped below at 0.1. Each style has
# two channel aliases, with a 25% chance of drawing an independent replacement
# style. Sixteen independent standard-normal nuisance coordinates carry no
# style information. Labels remain outside records and the schema.
#
# Each seed has independent 4,096/512/2,048 training/validation/test rows. The
# test variants use the same underlying records and labels. Channel removal
# nulls only that field; complete signal removal nulls channel, amount, and
# frequency. Neither missingness pattern reveals a style. Nuisance replacement
# changes only the independent vector.
#
# ### Frozen protocol and predeclared checks
#
# Three paired seeds start at the requested seed. An xs model reconstructs
# fields sampled for hiding at rate 0.35 for 600 AdamW updates, learning rate
# 0.002, weight decay 0.01, batch 64, full precision, and no checkpoint selection. Both trained
# and untrained arms first calibrate numeric statistics on the same complete
# training table. No latent label is a target. Exported root vectors are frozen
# before clustering, and all persistent parameters and buffers must remain
# unchanged during export.
#
# Pure NumPy KMeans receives four groups, four seeded initializations, and at
# most 40 iterations. It chooses its initialization by training inertia only.
# Centroids fit complete, unlabeled training exports and are reused for every
# test variant. Raw features include every channel code, both measurements,
# and every nuisance coordinate; their means and scales fit training data only.
# Missing raw measurements use those training means.
#
# Adjusted Rand index (ARI) compares held-out assignments with the evaluator's
# hidden labels without aligning arbitrary cluster names. Predeclared gates
# require mean trained ARI at least 0.35 with complete evidence and at least
# 0.20 without channel, and absolute mean ARI below 0.05 without any semantic
# view. Exports must be finite and noncollapsed where semantic evidence remains;
# evaluation must preserve state and every full run must finish its budget.
# Raw and untrained comparisons, nuisance replacement, and 32 fixed shuffled
# label scores are diagnostics. Beating raw inputs is not a success condition.
# Report each seed and paired differences; do not choose seeds or thresholds
# after inspecting test results.
#
# ### Complete experiment code
#

# %%
"""P081: frozen semantic clusters under paired missingness and nuisance controls."""

import copy
import math
from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
import torch
from reporting import experiment, report

BUDGET = 600
BATCH_SIZE = 64
TRAIN_ROWS = 4096
VALIDATION_ROWS = 512
TEST_ROWS = 2048
N_CLASSES = 4
NUISANCE_DIM = 16
CHANNELS = ("card", "wallet", "direct-debit", "bank-transfer", "invoice", "wire", "cash", "voucher")
AMOUNTS = np.asarray((30.0, 30.0, 100.0, 100.0))
FREQUENCIES = np.asarray((3.0, 12.0, 3.0, 12.0))
CONDITIONS = ("complete", "one_view_missing", "nuisance_resampled", "all_semantic_views_missing")


def sample(*, rows: int, seed: int) -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.resize(np.arange(N_CLASSES), rows)
    rng.shuffle(labels)
    channel_style = np.where(rng.random(rows) < 0.25, rng.integers(N_CLASSES, size=rows), labels)
    channels = 2 * channel_style + rng.integers(2, size=rows)
    amount = np.maximum(0.1, AMOUNTS[labels] + rng.normal(0.0, 10.0, rows))
    frequency = np.maximum(0.1, FREQUENCIES[labels] + rng.normal(0.0, 1.5, rows))
    nuisance = rng.normal(size=(rows, NUISANCE_DIM))
    observations = [
        {"channel": CHANNELS[channel], "amount": float(value), "frequency": float(rate), "nuisance": noise.tolist()}
        for channel, value, rate, noise in zip(channels, amount, frequency, nuisance, strict=True)
    ]
    return observations, labels


def records(*, rows: int, seed: int) -> Iterator[dict]:
    yield from sample(rows=rows, seed=seed)[0]


def condition(rows: list[dict], name: str, seed: int) -> list[dict]:
    if name not in CONDITIONS:
        raise ValueError(f"Unknown semantic robustness condition {name!r}; expected one of {CONDITIONS}")
    selected = copy.deepcopy(rows)
    if name == "one_view_missing":
        for row in selected:
            row["channel"] = None
    elif name == "all_semantic_views_missing":
        for row in selected:
            row.update(channel=None, amount=None, frequency=None)
    elif name == "nuisance_resampled":
        nuisance = np.random.default_rng(seed).normal(size=(len(rows), NUISANCE_DIM))
        for row, vector in zip(selected, nuisance, strict=True):
            row["nuisance"] = vector.tolist()
    return selected


def build() -> rf.Model:
    mask = rf.Mask(rate=0.35, reconstruct=True)
    return rf.Model.xs(
        embed=True,
        channel=rf.Enum(values=CHANNELS, p_unavailable=0.0, mask=mask),
        amount=rf.Number(mask=mask),
        frequency=rf.Number(mask=mask),
        nuisance=rf.Vector(n_dim=NUISANCE_DIM, mask=mask),
    )


def embedding(model: rf.Model, rows: list[dict], *, batch_size: int = 256) -> np.ndarray:
    output = model.predict(pa.Table.from_pylist(rows), batch_size=batch_size)["representations"].to_pylist()
    roots = [
        sorted(
            (record for record in row if record["path"] == [] and record["parents"] == []),
            key=lambda record: record["output_index"],
        )
        for row in output
    ]
    if len(roots) != len(rows) or any(len(root) != 1 or root[0]["output_index"] != 0 for root in roots):
        raise ValueError(
            "P081 requires exactly one exported root representation with output index zero per financial record"
        )
    return np.asarray([root[0]["embedding"] for root in roots], dtype=np.float64)


def distances(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.square(values[:, None, :] - centroids[None, :, :]).sum(axis=-1)


def assign(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return distances(values, centroids).argmin(axis=1)


def cluster(values: np.ndarray, seed: int) -> np.ndarray:
    """Fit only these unlabeled rows, selecting restarts by their training inertia."""
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


def raw_features(rows: list[dict]) -> np.ndarray:
    codes = np.eye(len(CHANNELS))
    return np.asarray(
        [
            np.concatenate(
                (
                    np.zeros(len(CHANNELS)) if row["channel"] is None else codes[CHANNELS.index(row["channel"])],
                    [np.nan if row[name] is None else row[name] for name in ("amount", "frequency")],
                    row["nuisance"],
                )
            )
            for row in rows
        ],
        dtype=np.float64,
    )


def equal(left, right) -> bool:
    """Compare parameters, buffers, and extension-owned extra state without mutation."""
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            return False
        try:
            torch.testing.assert_close(left.detach().cpu(), right.detach().cpu(), rtol=0, atol=0, equal_nan=True)
        except AssertionError:
            return False
        return True
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(equal(a, b) for a, b in zip(left, right, strict=True))
        )
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right, equal_nan=left.dtype.kind in "fc")
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return True
    return bool(left == right)


def measure(training: np.ndarray, queries: dict[str, np.ndarray], labels: np.ndarray, seed: int) -> dict:
    centroids = cluster(training, seed)
    rng = np.random.default_rng(seed + 1)
    shuffled = [rng.permutation(labels) for _ in range(32)]
    measured = {}
    for name, values in queries.items():
        guessed = assign(values, centroids)
        null = [ari(permuted, guessed) for permuted in shuffled]
        measured[name] = {
            "ari": ari(labels, guessed),
            "shuffled_label_ari_mean": float(np.mean(null)),
            "shuffled_label_ari_abs_max": float(np.max(np.abs(null))),
            "shuffled_label_repetitions": len(null),
            "finite": bool(np.isfinite(values).all()),
            "embedding_variance": float(np.var(values, axis=0).sum()),
            "occupied_clusters": int(len(np.unique(guessed))),
        }
    return {
        "conditions": measured,
        "train_inertia": float(distances(training, centroids).min(axis=1).sum()),
        "centroids": centroids.tolist(),
    }


def trial(seed: int, steps: int | None, accelerator: str) -> dict:
    train, _ = sample(rows=TRAIN_ROWS, seed=seed * 100 + 1)
    test, labels = sample(rows=TEST_ROWS, seed=seed * 100 + 3)
    variants = {name: condition(test, name, seed * 100 + 4) for name in CONDITIONS}
    arms = {}
    initialized = None
    for name in ("untrained", "trained"):
        lit.seed_everything(seed, workers=True)
        model = build()
        model.encode(pa.Table.from_pylist(train), strata="train", seed=seed)
        initial = copy.deepcopy(model.state_dict())
        if initialized is None:
            initialized = initial
        initialization_matches = equal(initialized, initial)
        trained_steps = 0
        if name == "trained":
            data = rf.SyntheticDataModule(
                train=partial(records, rows=TRAIN_ROWS, seed=seed * 100 + 1),
                validate=partial(records, rows=VALIDATION_ROWS, seed=seed * 100 + 2),
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
        before = copy.deepcopy(model.state_dict())
        training = embedding(model, train)
        queries = {condition_name: embedding(model, rows) for condition_name, rows in variants.items()}
        measured = measure(training, queries, labels, seed * 100 + 5)
        measured.update(
            steps=trained_steps,
            initialization_matches=initialization_matches,
            state_preserved=equal(before, model.state_dict()),
            embedding_dimensions=training.shape[1],
            parameters=sum(parameter.numel() for parameter in model.parameters()),
            train_exports_finite=bool(np.isfinite(training).all()),
            root_norm_max_error=float(np.max(np.abs(np.linalg.norm(training, axis=1) - 1.0))),
            query_root_norm_max_error={
                condition_name: float(np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1.0)))
                for condition_name, vectors in queries.items()
            },
        )
        arms[name] = measured
    training = raw_features(train)
    mean, scale = training.mean(axis=0), np.maximum(training.std(axis=0), 1e-8)
    raw_queries = {}
    for name, rows in variants.items():
        values = raw_features(rows)
        raw_queries[name] = (np.where(np.isnan(values), mean, values) - mean) / scale
    arms["raw"] = measure((training - mean) / scale, raw_queries, labels, seed * 100 + 5)
    return {"seed": seed, "arms": arms}


@experiment("P081", seed=8101)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    trials = [trial(seed + offset, steps, accelerator) for offset in range(3)]
    summary = {}
    for arm in ("trained", "untrained", "raw"):
        summary[arm] = {}
        for name in CONDITIONS:
            values = [item["arms"][arm]["conditions"][name]["ari"] for item in trials]
            summary[arm][name] = {"ari_mean": float(np.mean(values)), "ari_min": min(values), "ari_max": max(values)}
    paired = {
        control: {
            name: [
                item["arms"]["trained"]["conditions"][name]["ari"] - item["arms"][control]["conditions"][name]["ari"]
                for item in trials
            ]
            for name in CONDITIONS
        }
        for control in ("untrained", "raw")
    }
    model_arms = [item["arms"][arm] for item in trials for arm in ("trained", "untrained")]
    observable = [
        arm["conditions"][name] for arm in model_arms for name in CONDITIONS if name != "all_semantic_views_missing"
    ]
    checks = {
        "Complete evidence reaches mean trained ARI at least 0.35": summary["trained"]["complete"]["ari_mean"] >= 0.35,
        "An absent channel retains mean trained ARI at least 0.20": summary["trained"]["one_view_missing"]["ari_mean"]
        >= 0.20,
        "Absent semantic views have absolute mean trained ARI below 0.05": abs(
            summary["trained"]["all_semantic_views_missing"]["ari_mean"]
        )
        < 0.05,
        "All exported model vectors are finite": all(
            arm["train_exports_finite"] and all(value["finite"] for value in arm["conditions"].values())
            for arm in model_arms
        ),
        "Observable evidence gives noncollapsed exports": all(
            value["embedding_variance"] > 1e-6 for value in observable
        ),
        "Export preserves every model parameter buffer and extra state": all(
            arm["state_preserved"] for arm in model_arms
        ),
        "Paired model arms have identical calibrated initial state": all(
            arm["initialization_matches"] for arm in model_arms
        ),
        "Every trained seed completes all 600 optimizer updates": all(
            item["arms"]["trained"]["steps"] == BUDGET for item in trials
        ),
    }
    metrics = {
        "seeds": [item["seed"] for item in trials],
        "train_rows": TRAIN_ROWS,
        "validation_rows": VALIDATION_ROWS,
        "test_rows": TEST_ROWS,
        "cluster_count": N_CLASSES,
        "budget": BUDGET,
        "trials": trials,
        "summary": summary,
        "paired_ari_gains": paired,
    }
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
