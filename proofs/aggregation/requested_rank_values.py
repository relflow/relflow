# %% [markdown]
# ---
# title: Requesting a value by rank
# categories:
# - Order statistics
# proof-id: P012
# description: Use one model to recover minimum, quartiles, median, and maximum from
#   an unsorted numerical bag.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P012 status >}}
#
# ## Insights
#
# **The model can use a request to pick the minimum, maximum, median, or a quartile from an unsorted
# collection.** Each bag appears with every request, so a bag-only answer cannot solve the task. Hiding rank
# collapses those predictions, while cycling rank labels destroys accuracy against the original targets.
#
# Preserving encoded item coordinates leaves evidence available for request-conditioned decoding; it does
# not itself perform sorting. Separate gates for every rank and data shape prevent good minimum or maximum
# predictions from hiding failed quartiles. The tested quantiles select observed values from an odd-length
# bag. Even lengths, interpolation conventions, empty collections, and much larger bags are additional
# questions.
#
# ## Setup

# %%
"""Request five order statistics from one randomly ordered numeric bag.

Require endpoints, quartiles, median, and causal controls together.

Run: uv run python proofs/run.py P012"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P012"
LENGTH = 9
RANKS = ("minimum", "q25", "median", "q75", "maximum")
RANK_INDEX = dict(zip(RANKS, (0, 2, 4, 6, 8), strict=True))
SHAPES = ("symmetric", "left_skewed", "right_skewed", "duplicates")

# %% [markdown]
# ## Examples
#
# Targets use `mask=True`; the labels below are supervision hidden from the encoder.
#
# ### Request the median
#
# ```yaml
# rank: median
# items:
#   - {value: 0.4}
#   - {value: 0.1}
#   - {value: 0.9}
#   - {value: 0.3}
#   - {value: 0.7}
#   - {value: 0.2}
#   - {value: 0.8}
#   - {value: 0.6}
#   - {value: 0.5}
# answer: 0.5
# ```
#
# The fifth sorted value is 0.5, even though the input bag is unsorted.
#
# ### Request the lower quartile
#
# ```yaml
# rank: q25
# items:
#   - {value: 0.4}
#   - {value: 0.1}
#   - {value: 0.9}
#   - {value: 0.3}
#   - {value: 0.7}
#   - {value: 0.2}
#   - {value: 0.8}
#   - {value: 0.6}
#   - {value: 0.5}
# answer: 0.3
# ```
#
# For this nine-value convention, Q25 is the third sorted value: 0.3. The
# items are unchanged; only the visible rank request differs.
#
# ### Cycle the request but retain its label
#
# ```yaml
# rank: q75
# items:
#   - {value: 0.4}
#   - {value: 0.1}
#   - {value: 0.9}
#   - {value: 0.3}
#   - {value: 0.7}
#   - {value: 0.2}
#   - {value: 0.8}
#   - {value: 0.6}
#   - {value: 0.5}
# answer: 0.5  # Retained original label
# ```
#
# Cycling the median request to Q75 makes the correct requested value 0.7.
# This corruption deliberately keeps the original median label of 0.5. A
# request-sensitive model should then score worse against that retained target.
#
# ## Synthetic data and controls


# %%
def values(rng: np.random.Generator, shape: str) -> np.ndarray:
    """Draw one independently located and scaled bag with a declared shape."""
    match shape:
        case "symmetric":
            standardized = rng.normal(size=LENGTH)
        case "left_skewed":
            standardized = 1.0 - rng.exponential(size=LENGTH)
        case "right_skewed":
            standardized = rng.exponential(size=LENGTH) - 1.0
        case "duplicates":
            standardized = rng.normal(size=LENGTH)
            standardized[2] = standardized[1]
            standardized[6] = standardized[5]
        case _:
            raise ValueError(f"unknown order-statistic shape: {shape!r}")
    standardized = np.clip(standardized, -3.0, 3.0)
    scale = float(rng.uniform(0.35, 1.35))
    location = float(rng.uniform(-1.25, 1.25))
    return location + scale * standardized


def records(*, bags: int, seed: int) -> Iterator[dict]:
    """Expand each base bag into all rank requests before split assignment."""
    rng = np.random.default_rng(seed)
    for bag in range(bags):
        shape = SHAPES[bag % len(SHAPES)]
        bag_values = values(rng, shape)
        ordered = np.sort(bag_values)
        permutation = rng.permutation(LENGTH)
        items = [{"value": float(bag_values[index])} for index in permutation]
        for rank in RANKS:
            yield {
                "bag": f"{seed}:{bag}",
                "shape": shape,
                "rank": rank,
                "items": items,
                "answer": float(ordered[RANK_INDEX[rank]]),
            }


def permute_items(observations: list[dict], *, seed: int) -> list[dict]:
    """Randomize each complete bag once while retaining targets and requests."""
    rng = np.random.default_rng(seed)
    rows = deepcopy(observations)
    reordered: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        bag = str(row["bag"])
        if bag not in reordered:
            items = row["items"]
            reordered[bag] = [items[index] for index in rng.permutation(len(items))]
        row["items"] = reordered[bag]
    return rows


def validate_request_families(observations: list[dict]) -> set[str]:
    """Require one identical five-request family for every complete base bag."""
    families: dict[str, list[dict[str, object]]] = {}
    for row in observations:
        families.setdefault(str(row["bag"]), []).append(row)
    for bag, family in families.items():
        ranks = {str(row["rank"]) for row in family}
        if len(family) != len(RANKS) or ranks != set(RANKS):
            raise ValueError(f"bag {bag!r} does not contain exactly one request for every rank: {ranks!r}")
        if any((row["items"] != family[0]["items"] for row in family[1:])):
            raise ValueError(f"bag {bag!r} changes item evidence between rank requests")
    return set(families)


def cycle_rank(observations: list[dict]) -> list[dict]:
    """Break request/answer alignment while retaining every marginal exactly."""
    rows = deepcopy(observations)
    successor = {rank: RANKS[(index + 1) % len(RANKS)] for index, rank in enumerate(RANKS)}
    for row in rows:
        row["rank"] = successor[row["rank"]]
    return rows


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"], "rank": row["rank"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["/answer"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def scores(*, train: list[dict], test: list[dict], predicted: np.ndarray, key: str) -> dict[str, dict[str, float]]:
    """Compare each rank or shape with its own training-mean baseline."""
    result = {}
    for cell in sorted({row[key] for row in test}):
        train_mean = float(np.mean([row["answer"] for row in train if row[key] == cell]))
        indices = [index for index, row in enumerate(test) if row[key] == cell]
        actual = np.asarray([test[index]["answer"] for index in indices])
        baseline = rmse(actual, train_mean)
        measured = rmse(actual, predicted[indices])
        result[cell] = {"rmse": measured, "baseline_rmse": baseline, "nrmse": measured / baseline}
    return result


def overall_score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Calculate overall accuracy against per-rank training means."""
    train_rank = np.asarray([row["rank"] for row in train])
    train_answer = np.asarray([row["answer"] for row in train]).astype(np.float64)
    test_rank = np.asarray([row["rank"] for row in test])
    actual = np.asarray([row["answer"] for row in test]).astype(np.float64)
    means = {rank: float(train_answer[train_rank == rank].mean()) for rank in RANKS}
    baseline = np.asarray([means[rank] for rank in test_rank], dtype=np.float64)
    baseline_rmse = rmse(actual, baseline)
    measured_rmse = rmse(actual, predicted)
    return {"rmse": measured_rmse, "baseline_rmse": baseline_rmse, "nrmse": measured_rmse / baseline_rmse}


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-requested-rank-values
# //| fig-cap: "Both root and item branch keep all tokens with branch attention enabled; the visible rank selects among nine raw values."
# //| fig-alt: "Request contains repeated items with value inputs, visible rank, and hidden answer targets. Root reduction: Keep all tokens. Item reduction: Keep all tokens; capacity 9; branch attention MHA."
# #tree(node("request", kind: "root", width: 150pt, body: [
#     - *Reduction:* Keep all tokens
#     - *Branch attention:* MHA
#   ], children: (
#   node("rank", type: "Category"),
#   node("items", kind: "branch", repeated: true, width: 155pt, body: [
#       - *Reduction:* Keep all tokens
#       - *Branch attention:* MHA
#       - *Capacity:* 9 items
#     ], children: (
#     node("value", type: "Number"),
#   )),
#   node("answer", kind: "target", type: "Number"),
# )))
# ```
#
# ## How it works
#
# Both root and item branch use `reduction=None` to preserve encoded
# coordinates. A visible Category request chooses among `minimum`, `q25`,
# `median`, `q75`, and `maximum`. For nine values these are sorted positions
# 1, 3, 5, 7, and 9: no interpolated quantile is required, and ties retain their
# observed value.
#
# Each bag appears with all five requests in the same split. Symmetric, skewed,
# and duplicate-bearing bags vary in location and scale. Reordering complete
# items should retain predictions. Hiding rank makes the five requests identical
# and must collapse their predictions; cycling rank labels while keeping targets
# must remove request-conditioned accuracy. Endpoint success alone cannot satisfy
# the separate interior-rank gates.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    train = list(records(bags=1024, seed=seed + 1))
    validate = list(records(bags=192, seed=seed + 2))
    test = list(records(bags=384, seed=seed + 3))
    model = rf.Model(
        d_model=64,
        n_layers=3,
        n_heads=4,
        reduction=None,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.003),
        items=rf.Branch(length=LENGTH, overflow="error", n_layers=3, n_heads=4, reduction=None, value=rf.Number),
        rank=rf.Category(p_unavailable=0.0),
        answer=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        model=model,
        train=lambda: records(bags=1024, seed=seed + 1),
        validate=lambda: records(bags=192, seed=seed + 2),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=800 if steps is None else min(steps, 800),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=datamodule)
    intact_prediction = prediction(model, test)
    by_rank = scores(train=train, test=test, predicted=intact_prediction, key="rank")
    by_shape = scores(train=train, test=test, predicted=intact_prediction, key="shape")
    intact = overall_score(train=train, test=test, predicted=intact_prediction)
    permuted_prediction = prediction(model, permute_items(test, seed=seed + 4))
    target_scale = float(np.std(np.asarray([row["answer"] for row in test]).astype(np.float64)))
    permutation_drift = rmse(intact_prediction, permuted_prediction) / target_scale
    cycled_prediction = prediction(model, cycle_rank(test))
    cycled = overall_score(train=train, test=test, predicted=cycled_prediction)
    hidden_prediction = prediction(model, [{**row, "rank": None} for row in test])
    hidden = scores(train=train, test=test, predicted=hidden_prediction, key="rank")
    hidden_family_spread = float(np.max(np.ptp(hidden_prediction.reshape(-1, len(RANKS)), axis=1)))
    train_bags = validate_request_families(train)
    validate_bags = validate_request_families(validate)
    test_bags = validate_request_families(test)
    calibration = np.asarray(
        [
            target_scale,
            intact["rmse"],
            intact["baseline_rmse"],
            intact["nrmse"],
            cycled["rmse"],
            cycled["baseline_rmse"],
            cycled["nrmse"],
            permutation_drift,
            hidden_family_spread,
            *(score["baseline_rmse"] for score in by_rank.values()),
            *(score["baseline_rmse"] for score in by_shape.values()),
        ]
    )
    endpoint_passes = max((by_rank[rank]["nrmse"] for rank in ("minimum", "maximum"))) < 0.3
    interior_passes = max((by_rank[rank]["nrmse"] for rank in ("q25", "median", "q75"))) < 0.35
    shape_passes = all((by_shape[shape]["nrmse"] < 0.45 for shape in SHAPES))
    controls_pass = (
        permutation_drift < 0.1
        and cycled["nrmse"] >= 0.8
        and (cycled["nrmse"] >= intact["nrmse"] + 0.35)
        and all((hidden[rank]["nrmse"] >= 0.65 for rank in ("minimum", "maximum")))
        and (sum((score["nrmse"] >= 0.65 for score in hidden.values())) >= 3)
    )
    metrics = {
        "intact": intact,
        "by_rank": by_rank,
        "by_shape": by_shape,
        "cycled": cycled,
        "hidden": hidden,
        "permutation_drift": permutation_drift,
        "hidden_family_spread": hidden_family_spread,
        "target_scale": target_scale,
        "steps": trainer.global_step,
    }
    checks = {
        "Expected number of rank requests": bool(len(train) == 1024 * len(RANKS) and len(test) == 384 * len(RANKS)),
        "All ranks represented": bool(set(np.asarray([row["rank"] for row in test])) == set(RANKS)),
        "All bag shapes represented": bool(set(np.asarray([row["shape"] for row in test])) == set(SHAPES)),
        "Training bags separate from validation and test": bool(train_bags.isdisjoint(validate_bags | test_bags)),
        "Validation bags separate from test": bool(validate_bags.isdisjoint(test_bags)),
        "All measurements finite": bool(np.isfinite(calibration).all()),
        "Target SD above 0.80": bool(target_scale > 0.8),
        "Per-rank baseline RMSE above 0.60": bool(min((score["baseline_rmse"] for score in by_rank.values())) > 0.6),
        "Per-shape baseline RMSE above 0.80": bool(min((score["baseline_rmse"] for score in by_shape.values())) > 0.8),
        "Hidden requests yield identical predictions": bool(hidden_family_spread <= 1e-06),
        "Endpoint, interior-rank, shape, and causal-control gates": bool(
            endpoint_passes and interior_passes and shape_passes and controls_pass
        ),
    }
    return metrics, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P012 evidence >}}
#
# ## Remaining work
#
# Repeat the complete gate across seeds. Variable and even lengths need
# explicit quantile semantics; nulls, empty bags, all-equal values, heavy tails,
# and magnitude extrapolation need new cases. Permutation stability here is
# measured approximately, not guaranteed by every part of the architecture.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ## Reproduce
#
# {{< proof P012 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=3700)
