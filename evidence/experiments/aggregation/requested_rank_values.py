# %% [markdown]
# ---
# title: Can it find a requested value in an unsorted list?
# categories:
# - Order statistics
# evidence-id: P012
# description: Ask for the smallest, largest, middle, or quarter-position value without sorting the input first.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: All requests
#   metric:
#   - intact
#   - nrmse
#   format: error
# - label: Changed requests
#   metric:
#   - cycled
#   - nrmse
#   format: error
# evidence-group: Aggregation
# evidence-result:
#   accelerator: cpu
#   checks:
#     All bag shapes represented: true
#     All measurements finite: true
#     All ranks represented: true
#     Endpoint, interior-rank, shape, and causal-control gates: true
#     Expected number of rank requests: true
#     Hidden requests yield identical predictions: true
#     Per-rank baseline RMSE above 0.60: true
#     Per-shape baseline RMSE above 0.80: true
#     Target SD above 0.80: true
#     Training bags separate from validation and test: true
#     Validation bags separate from test: true
#   duration_seconds: 101.395
#   metrics:
#     by_rank:
#       maximum:
#         baseline_rmse: 0.9466038978237984
#         nrmse: 0.030806227289838364
#         rmse: 0.029161294829806863
#       median:
#         baseline_rmse: 0.8438626855475819
#         nrmse: 0.0725086661407999
#         rmse: 0.06118735773504843
#       minimum:
#         baseline_rmse: 1.0886907856680468
#         nrmse: 0.04530158390779701
#         rmse: 0.04931941697658647
#       q25:
#         baseline_rmse: 0.8870723799603668
#         nrmse: 0.051633233714214806
#         rmse: 0.04580241551591838
#       q75:
#         baseline_rmse: 0.8574627753116424
#         nrmse: 0.04656273682923559
#         rmse: 0.03992581354770197
#     by_shape:
#       duplicates:
#         baseline_rmse: 1.2044208081983223
#         nrmse: 0.03826737751004838
#         rmse: 0.04609002574828277
#       left_skewed:
#         baseline_rmse: 1.2724328485237015
#         nrmse: 0.03280378602448242
#         rmse: 0.04174061489349415
#       right_skewed:
#         baseline_rmse: 1.1207206271078058
#         nrmse: 0.03369132140736174
#         rmse: 0.037758558855749094
#       symmetric:
#         baseline_rmse: 1.2584867784356901
#         nrmse: 0.04552615866688391
#         rmse: 0.05729406875523881
#     cycled:
#       baseline_rmse: 0.9290372023301314
#       nrmse: 1.454229219898798
#       rmse: 1.3510330460015088
#     hidden:
#       maximum:
#         baseline_rmse: 0.9466038978237984
#         nrmse: 2.777319845107183
#         rmse: 2.629021790881848
#       median:
#         baseline_rmse: 0.8438626855475819
#         nrmse: 1.9600295600930049
#         rmse: 1.6539958083327286
#       minimum:
#         baseline_rmse: 1.0886907856680468
#         nrmse: 0.6928980221123755
#         rmse: 0.7543516920813578
#       q25:
#         baseline_rmse: 0.8870723799603668
#         nrmse: 1.427491963066947
#         rmse: 1.2662886930520927
#       q75:
#         baseline_rmse: 0.8574627753116424
#         nrmse: 2.402822154282153
#         rmse: 2.0603305529910743
#     hidden_family_spread: 0.0
#     intact:
#       baseline_rmse: 0.9290372023301314
#       nrmse: 0.04983691370865262
#       rmse: 0.04630034688465481
#     permutation_drift: 0.015378312926430257
#     steps: 800
#     target_scale: 1.2148096616638178
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 86aa852ff217d4eb037dedca578ea6f5a05db6987eea060922cc6890219538e7
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 2e0262893c5f
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
#     script_sha256: 7c888d2dc3ac52f586df90acfa46a2ba10b14e1b5148e9e231dacbf7fcb32349
#     source_sha256: 3677fab0ace54c43a05d1d85603d58440f5ef6eaf0e776f935581aefb25391b3
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 3700
#   started_at: '2026-11-01T01:34:00.770857+00:00'
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
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-requested-rank-values
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with request, rank, items, value, answer. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("request", kind: "root", width: 120pt, children: (
#   node("rank", type: "Category"),
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("value", type: "Number"),
#   )),
#   node("answer", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Give the same list different requests. Hide or change the requests while keeping the original answers to check that the request matters.
#
# ## Result
#
# {{< evidence P012 status >}}
#
# The model learns all five tested requests. These lists have an odd length; other lengths and rules for interpolating between values need separate tests.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P012 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P012 script >}}
#
# ### How it works
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
# ### Remaining work
#
# Repeat the complete gate across seeds. Variable and even lengths need
# explicit quantile semantics; nulls, empty bags, all-equal values, heavy tails,
# and magnitude extrapolation need new cases. Permutation stability here is
# measured approximately, not guaranteed by every part of the architecture.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""Request five order statistics from one randomly ordered numeric bag.

Require endpoints, quartiles, median, and causal controls together.

Run: uv run python evidence/experiments/run.py P012"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P012"
LENGTH = 9
RANKS = ("minimum", "q25", "median", "q75", "maximum")
RANK_INDEX = dict(zip(RANKS, (0, 2, 4, 6, 8), strict=True))
SHAPES = ("symmetric", "left_skewed", "right_skewed", "duplicates")


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
        if any(row["items"] != family[0]["items"] for row in family[1:]):
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
    return np.asarray([row["answer"]["content"] for row in output], dtype=np.float64)


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
        items=rf.Branch(length=LENGTH, overflow="error", n_layers=3, n_heads=4, reduction=None, value=rf.Number),
        rank=rf.Category(p_unavailable=0.0),
        answer=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        train=lambda: records(bags=1024, seed=seed + 1),
        validate=lambda: records(bags=192, seed=seed + 2),
        seed=seed,
    )
    model.fit(
        datamodule,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.003),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
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
    trainer = model.trainer
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
    endpoint_passes = max(by_rank[rank]["nrmse"] for rank in ("minimum", "maximum")) < 0.3
    interior_passes = max(by_rank[rank]["nrmse"] for rank in ("q25", "median", "q75")) < 0.35
    shape_passes = all(by_shape[shape]["nrmse"] < 0.45 for shape in SHAPES)
    controls_pass = (
        permutation_drift < 0.1
        and cycled["nrmse"] >= 0.8
        and (cycled["nrmse"] >= intact["nrmse"] + 0.35)
        and all(hidden[rank]["nrmse"] >= 0.65 for rank in ("minimum", "maximum"))
        and (sum(score["nrmse"] >= 0.65 for score in hidden.values()) >= 3)
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
        "Per-rank baseline RMSE above 0.60": bool(min(score["baseline_rmse"] for score in by_rank.values()) > 0.6),
        "Per-shape baseline RMSE above 0.80": bool(min(score["baseline_rmse"] for score in by_shape.values()) > 0.8),
        "Hidden requests yield identical predictions": bool(hidden_family_spread <= 1e-06),
        "Endpoint, interior-rank, shape, and causal-control gates": bool(
            endpoint_passes and interior_passes and shape_passes and controls_pass
        ),
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3700)

# %% [markdown]
# </details>
