# %% [markdown]
# ---
# title: Do transaction histories form meaningful customer neighborhoods?
# categories:
# - Representation
# description: Cluster frozen customer embeddings using only unlabeled training histories, then score independent customers with shorter or longer histories.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Eight transactions
#   metric: [summary, reconstruction, length_8, ari_mean]
#   format: number
# - label: Twelve transactions
#   metric: [summary, reconstruction, length_12, ari_mean]
#   format: number
# - label: Untrained encoder
#   metric: [summary, untrained, length_8, ari_mean]
#   format: number
# evidence-group: Representation
# ---
#
# One observation here is an entire customer's history, rather than a single
# transaction. The model reconstructs hidden transaction fields, then exports
# one vector for the customer. We ask whether histories generated from similar
# spending patterns become neighbors in that vector space, even for new customers
# or a history length absent from training.
#
# ## Example
#
# ```yaml
# transactions:
#   - merchant: grocer
#     amount: 32.5
#     interval_days: 4.0
#     terminal: terminal-3
#   - merchant: market
#     amount: 45.0
#     interval_days: 6.0
#     terminal: terminal-8
#   - merchant: grocer
#     amount: 37.0
#     interval_days: 3.5
#     terminal: terminal-1
#   - merchant: market
#     amount: 42.0
#     interval_days: 4.5
#     terminal: terminal-5
# ```
#
# One observation is a new customer's transaction history. Four hidden spending
# styles influence the noisy merchant, amount, and timing observations. Those
# styles are kept outside the records and the model throughout training.
#
# The illustrated history contains relatively small amounts and short gaps.
# Those noisy measurements can provide repeated clues about a hidden style;
# a terminal name is independent nuisance. Two customers can therefore belong
# together without sharing a terminal or a persistent customer identifier.
# The four-row example is illustrative, not a recorded model prediction.
#
# A history is summarized by a position-free mean reduction. Reversing complete
# transactions should preserve its customer vector. A different test gives the
# same new customer's first four, eight, or twelve transactions to the frozen
# model, probing how much evidence it can use without changing the cluster fit.
#
# ```{typst}
# //| label: fig-evidence-nested-history-clusters
# //| fig-cap: "Only visible transaction fields are reconstructed. The root exports one customer embedding."
# //| fig-alt: "Customer root exporting one embedding above a mean-reduced transaction branch capped at sixteen, with merchant, amount, interval days, and an irrelevant terminal."
# #tree(node("customer", kind: "root", body: [Export frozen embedding], children: (
#   node("transactions", kind: "branch", detail: "Cap 16; mean reduction; no attention", children: (
#     node("merchant", type: "Enum", body: [Sampled reconstruction]),
#     node("amount", type: "Number", body: [Sampled reconstruction]),
#     node("interval_days", type: "Number", body: [Sampled reconstruction]),
#     node("terminal", type: "Enum", body: [Independent nuisance]),
#   )),
# )))
# ```
#
# ## Comparison
#
# Training histories contain four through eight transactions. Twelve-transaction
# test histories exceed that trained range while remaining below the schema's
# cap of sixteen. The clustering step receives four groups and fits only on
# unlabeled training histories; test labels are kept for scoring.
#
# The mixed-history control moves complete transactions between customers while
# preserving every transaction and each history length. It breaks the coherent
# customer-level pattern without scrambling fields within a transaction. The
# untrained encoder and raw field averages provide additional comparisons.
# ARI measures agreement with the hidden customer styles, so a visually pleasing
# cloud of vectors alone cannot establish meaningful neighborhoods.
#
#
# Compare masked reconstruction with the same untrained encoder, reconstruction
# after mixing transactions across customers, and raw field averages. Fit four
# clusters on unlabeled training exports once, then reuse them for every test
# history length and for reversed histories.
#
# ## Result
#
# {{< evidence P082 status >}}
#
# The checks specify grouping quality for eight- and twelve-transaction histories,
# plus unchanged export state and reversal stability. Gains over raw or untrained
# features remain separate diagnostics. The test concerns fresh customers drawn
# from four known synthetic styles, rather than discovery of new financial
# concepts or a general claim that longer histories always help.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P082 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P082 script >}}
#
# ### Population and controls
#
# Independent train/validation/test populations contain 4,096/512/2,048 customer
# histories. Styles are balanced and never enter a field, loss, cluster fit, or
# scaling fit. Training histories contain four through eight transactions,
# independently of style. Test customers supply twelve transactions; their
# prefixes of four, eight, and twelve are scored against the same fitted
# centroids. Twelve is a held-out length, within the explicit model cap of 16.
# Merchant aliases are known across populations. No persistent customer ID is
# available for memorization. Reversing eight complete records tests order
# sensitivity under the declared position-free mean reduction.
#
# The mixed-history training control preserves every whole transaction and each
# customer's history length, while destroying coherent within-customer styles.
# It shares calibrated statistics and initial parameters with reconstruction.
# The raw baseline averages every visible numeric and one-hot category coordinate
# and appends length. Its means/scales are fit on unlabeled training rows only.
#
# ### Fixed budget and scoring
#
# Each of three paired seeds (the requested seed and its next two integers) fits
# aligned and mixed-history xs encoders for 600 AdamW updates, batch 64, learning
# rate 0.002, weight decay 0.01, float32, with no checkpoint selection. Visible
# merchant, amount, and interval fields are independently masked for
# reconstruction at rate 0.35. Terminal is observed nuisance without an objective.
# All model arms receive identical train-only statistical calibration.
#
# Frozen root vectors are clustered using four Lloyd restarts, at most forty
# iterations each, selecting only training inertia. Held-out adjusted Rand index
# (ARI) ignores arbitrary cluster names. It is zero in expectation for independent
# partitions and one for exact recovery. Thirty-two permutations of test labels
# provide a null control without changing centroids. Every seed, its worst score,
# and paired control differences are retained; there is no test-based selection.
#
# Predeclared capability checks require mean reconstruction ARI at least 0.35
# with eight and twelve transactions. Contract checks require unchanged complete
# model state during export, finite noncollapsed unit vectors, the full budget,
# mean shuffled-label ARI below 0.05 in absolute value, and reversal drift below
# 0.00001. Checks may fail and are never lowered after measuring the panel.
#
# ### Complete experiment code
#

# %%
"""P082: held-out semantic clustering from finite nested transaction histories."""

from collections.abc import Iterator
from copy import deepcopy
from functools import partial

import lightning.pytorch as lit
import numpy as np
import polars as pl
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
CAPACITY = 16
MERCHANTS = ("grocer", "market", "airline", "rail", "cinema", "theater", "pharmacy", "clinic")
TERMINALS = tuple(f"terminal-{index}" for index in range(16))
CENTERS = np.asarray([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
CONDITIONS = ("length_4", "length_8", "length_12", "reversed_8")


def sample(*, rows: int, seed: int, lengths: tuple[int, int] = (4, 8), permuted: bool = False):
    if not 1 <= lengths[0] <= lengths[1] <= CAPACITY:
        raise ValueError(f"History lengths must satisfy 1 <= minimum <= maximum <= {CAPACITY}")
    rng = np.random.default_rng(seed)
    labels = np.resize(np.arange(N_CLASSES), rows)
    rng.shuffle(labels)
    counts = rng.integers(lengths[0], lengths[1] + 1, rows)
    observations = []
    for label, count in zip(labels, counts, strict=True):
        style = np.where(rng.random(count) < 0.25, rng.integers(N_CLASSES, size=count), label)
        codes = 2 * style + rng.integers(2, size=count)
        amounts = 60.0 + 20.0 * CENTERS[label, 0] + rng.normal(0.0, 15.0, count)
        intervals = np.maximum(0.1, 7.0 + 3.0 * CENTERS[label, 1] + rng.normal(0.0, 2.5, count))
        terminals = rng.integers(len(TERMINALS), size=count)
        observations.append(
            {
                "transactions": [
                    {
                        "merchant": MERCHANTS[code],
                        "amount": float(amount),
                        "interval_days": float(interval),
                        "terminal": TERMINALS[terminal],
                    }
                    for code, amount, interval, terminal in zip(codes, amounts, intervals, terminals, strict=True)
                ]
            }
        )
    if permuted:
        items = [item for row in observations for item in row["transactions"]]
        order = rng.permutation(len(items))
        offset = 0
        for row, count in zip(observations, counts, strict=True):
            row["transactions"] = [items[index] for index in order[offset : offset + count]]
            offset += count
    return observations, labels


def records(*, rows: int, seed: int, permuted: bool = False) -> Iterator[dict]:
    yield from sample(rows=rows, seed=seed, permuted=permuted)[0]


def condition(rows: list[dict], length: int, reverse: bool = False) -> list[dict]:
    if not 1 <= length <= CAPACITY:
        raise ValueError(f"History prefix must be between one and {CAPACITY}")
    result = deepcopy(rows)
    for row in result:
        items = row["transactions"][:length]
        row["transactions"] = items[::-1] if reverse else items
    return result


def build() -> rf.Model:
    mask = rf.Mask(rate=0.35, reconstruct=True)
    return rf.Model.xs(
        embed=True,
        transactions=rf.Branch(
            length=CAPACITY,
            attention=None,
            reduction=rf.Mean(),
            merchant=rf.Enum(values=MERCHANTS, p_unavailable=0.0, mask=mask),
            amount=rf.Number(mask=mask),
            interval_days=rf.Number(mask=mask),
            terminal=rf.Enum(values=TERMINALS, p_unavailable=0.0),
        ),
    )


def embedding(model: rf.Model, rows: list[dict], batch_size: int = 128) -> np.ndarray:
    outputs = pl.DataFrame(model.predict(pa.Table.from_pylist(rows), batch_size=batch_size))[
        "representations"
    ].to_list()
    selected = [[item for item in row if item["path"] == [] and item["parents"] == []] for row in outputs]
    if len(selected) != len(rows) or any(len(row) != 1 or row[0]["output_index"] != 0 for row in selected):
        raise ValueError("P082 requires exactly one root representation per customer in input row order")
    return np.asarray([row[0]["embedding"] for row in selected], dtype=np.float64)


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


def equal(left, right) -> bool:
    """Compare parameters, buffers, and extension-owned extra state without mutation."""
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.detach().cpu(), right.detach().cpu())
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
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    return bool(left == right)


def raw_features(rows: list[dict]) -> np.ndarray:
    merchant_codes, terminal_codes = np.eye(len(MERCHANTS)), np.eye(len(TERMINALS))
    result = []
    for row in rows:
        items = row["transactions"]
        values = [
            np.concatenate(
                (
                    merchant_codes[MERCHANTS.index(item["merchant"])],
                    [item["amount"], item["interval_days"]],
                    terminal_codes[TERMINALS.index(item["terminal"])],
                )
            )
            for item in items
        ]
        result.append(np.concatenate((np.mean(values, axis=0), [len(items)])))
    return np.asarray(result, dtype=np.float64)


def measure(training: np.ndarray, queries: dict[str, np.ndarray], labels: np.ndarray, seed: int) -> dict:
    centroids = cluster(training, seed)
    rng = np.random.default_rng(seed + 1)
    shuffled = [rng.permutation(labels) for _ in range(32)]
    measured = {}
    for name, values in queries.items():
        guessed = assign(values, centroids)
        measured[name] = {
            "ari": ari(labels, guessed),
            "shuffled_label_ari_mean": float(np.mean([ari(permuted, guessed) for permuted in shuffled])),
            "shuffled_label_repetitions": len(shuffled),
            "occupied_clusters": int(len(np.unique(guessed))),
            "finite": bool(np.isfinite(values).all()),
            "embedding_variance": float(np.var(values, axis=0).sum()),
        }
    return {"conditions": measured, "train_inertia": float(distances(training, centroids).min(axis=1).sum())}


def trial(seed: int, steps: int | None, accelerator: str) -> dict:
    train, _ = sample(rows=TRAIN_ROWS, seed=seed * 100 + 1)
    test, labels = sample(rows=TEST_ROWS, seed=seed * 100 + 3, lengths=(12, 12))
    variants = {f"length_{length}": condition(test, length) for length in (4, 8, 12)}
    variants["reversed_8"] = condition(test, 8, reverse=True)
    initial, arms = None, {}
    for name in ("untrained", "reconstruction", "mixed_histories"):
        lit.seed_everything(seed, workers=True)
        model = build()
        model.encode(pa.Table.from_pylist(train), strata="train", seed=seed)
        calibrated = deepcopy(model.state_dict())
        if initial is None:
            initial = calibrated
        initialization_matches = equal(initial, calibrated)
        trained_steps = 0
        if name != "untrained":
            data = rf.SyntheticDataModule(
                train=partial(records, rows=TRAIN_ROWS, seed=seed * 100 + 1, permuted=name == "mixed_histories"),
                validate=partial(
                    records, rows=VALIDATION_ROWS, seed=seed * 100 + 2, permuted=name == "mixed_histories"
                ),
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
        state = deepcopy(model.state_dict())
        training = embedding(model, train)
        queries = {key: embedding(model, value) for key, value in variants.items()}
        measured = measure(training, queries, labels, seed * 100 + 5)
        measured.update(
            steps=trained_steps,
            parameters=sum(parameter.numel() for parameter in model.parameters()),
            initialization_matches=initialization_matches,
            state_preserved=equal(state, model.state_dict()),
            embedding_dimensions=training.shape[1],
            root_norm_max_error=max(
                float(np.max(np.abs(np.linalg.norm(value, axis=1) - 1.0))) for value in (training, *queries.values())
            ),
            reversal_max_error=float(np.max(np.abs(queries["length_8"] - queries["reversed_8"]))),
        )
        arms[name] = measured
    training = raw_features(train)
    mean, scale = training.mean(axis=0), np.maximum(training.std(axis=0), 1e-8)
    arms["raw"] = measure(
        (training - mean) / scale,
        {name: (raw_features(rows) - mean) / scale for name, rows in variants.items()},
        labels,
        seed * 100 + 5,
    )
    return {"seed": seed, "arms": arms}


@experiment("P082", seed=8201)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    trials = [trial(seed + offset, steps, accelerator) for offset in range(3)]
    names = ("reconstruction", "untrained", "mixed_histories", "raw")
    summary = {
        name: {
            condition_name: {
                "ari_mean": float(
                    np.mean([item["arms"][name]["conditions"][condition_name]["ari"] for item in trials])
                ),
                "ari_min": min(item["arms"][name]["conditions"][condition_name]["ari"] for item in trials),
            }
            for condition_name in CONDITIONS
        }
        for name in names
    }
    model_arms = [item["arms"][name] for item in trials for name in names if name != "raw"]
    paired = {
        control: {
            condition_name: [
                item["arms"]["reconstruction"]["conditions"][condition_name]["ari"]
                - item["arms"][control]["conditions"][condition_name]["ari"]
                for item in trials
            ]
            for condition_name in CONDITIONS
        }
        for control in names
        if control != "reconstruction"
    }
    checks = {
        "Eight-transaction reconstruction reaches mean ARI at least 0.35": summary["reconstruction"]["length_8"][
            "ari_mean"
        ]
        >= 0.35,
        "Unseen twelve-transaction histories reach mean ARI at least 0.35": summary["reconstruction"]["length_12"][
            "ari_mean"
        ]
        >= 0.35,
        "Frozen export preserves parameters buffers and extra state": all(arm["state_preserved"] for arm in model_arms),
        "Paired arms start with identical calibrated state": all(arm["initialization_matches"] for arm in model_arms),
        "Every model export is finite and noncollapsed": all(
            all(value["finite"] and value["embedding_variance"] > 1e-6 for value in arm["conditions"].values())
            for arm in model_arms
        ),
        "Root representations are unit-normalized": all(arm["root_norm_max_error"] < 1e-5 for arm in model_arms),
        "Position-free mean reduction retains reversed histories": all(
            arm["reversal_max_error"] < 1e-5 for arm in model_arms
        ),
        "Shuffled test labels remove semantic agreement": all(
            abs(arm["conditions"][condition_name]["shuffled_label_ari_mean"]) < 0.05
            for arm in model_arms
            for condition_name in CONDITIONS
        ),
        "Every trained arm completes all 600 optimizer updates": all(
            item["arms"][name]["steps"] == BUDGET for item in trials for name in ("reconstruction", "mixed_histories")
        ),
    }
    return {
        "seeds": [item["seed"] for item in trials],
        "train_rows": TRAIN_ROWS,
        "validation_rows": VALIDATION_ROWS,
        "test_rows": TEST_ROWS,
        "known_clusters": N_CLASSES,
        "training_length_range": [4, 8],
        "test_lengths": [4, 8, 12],
        "summary": summary,
        "paired_gains": paired,
        "trials": trials,
    }, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
