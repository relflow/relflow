# %% [markdown]
# ---
# title: Can it learn a weighted average?
# categories:
# - Weighted aggregation
# evidence-id: P016
# description: Predict an average in which each value contributes according to its own positive weight.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Correct pairs
#   metric:
#   - intact
#   - nrmse
#   format: error
# - label: Swapped weights
#   metric:
#   - corrupted
#   - nrmse
#   format: error
# evidence-group: Aggregation
# evidence-result:
#   accelerator: cpu
#   checks:
#     Duplication drift below 0.10 target SD: true
#     Permutation drift below 0.10 target SD: true
#     Shuffled weights raise nRMSE by at least 0.25: true
#     Weight scaling drift below 0.10 target SD: true
#     Weighted mean nRMSE below 0.40: true
#   duration_seconds: 52.129
#   metrics:
#     corrupted:
#       baseline_rmse: 0.39181947854115323
#       nrmse: 0.9892419861765883
#       rmse: 0.38760427917472556
#     duplication_drift: 0.01735489301373714
#     intact:
#       baseline_rmse: 0.39181947854115323
#       nrmse: 0.042068300672942115
#       rmse: 0.016483179632784624
#     permutation_drift: 0.014192836677668286
#     scaling_drift: 0.02139593308834572
#     steps: 1100
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 70f1889787e01bb8c293f0897f412bc6931fcb75468f4d05d3b8edf79a0c829e
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 2ef2fb5d738b
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
#     script_sha256: 05caf28b0d326ca3850303e40e23e17edd0cfc97dfa8fe0b5f11af7f5bce73c8
#     source_sha256: 1b34a33325696662acd47805e575b191727a04127e198c98f3d4b99751d49295
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 3313
#   started_at: '2026-11-01T01:35:29.183831+00:00'
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
# items:
#   - {value: 0.8, weight: 0.5}
#   - {value: -0.4, weight: 1.25}
#   - {value: 0.2, weight: 2.0}
# weighted_mean: 0.08
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-variable-cardinality-weighted-mean
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, items, value, weight, weighted_mean. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("value", type: "Number"),
#     node("weight", type: "Number"),
#   )),
#   node("weighted_mean", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Reorder or repeat whole items, or scale all weights equally: the answer should stay fixed. Swapping weights between values should change it.
#
# ## Result
#
# {{< evidence P016 status >}}
#
# The model uses relative weights and approximately preserves these equal-answer changes. Zero or negative weights and much longer lists are not covered.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P016 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P016 script >}}
#
# ### How it works
#
# Learned `rf.Attention` reductions receive two through six item pairs.
# The label is `sum(value * weight) / sum(weight)`. Values vary independently;
# positive weights span 0.03 through 3.0 on a logarithmic scale, making an
# unweighted average a poor shortcut.
#
# Complete-item permutation, scaling every weight by 0.75, and duplicating all
# items should preserve the answer. Duplication is checked only for original
# lengths two and three so the result stays within branch capacity. Permuting
# weights alone must worsen error against unchanged labels: normalization does
# not remove the need to bind each weight to its own value.
#
# ### Remaining work
#
# Repeat every identity across seeds. Zero or signed total weight, missing
# weights, extreme magnitudes, and lengths outside the trained range still need
# explicit semantics and checks.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""A weighted mean should remain identifiable when collection length varies.

Learn a normalized weighted mean from two through six raw pairs.

Run: uv run python evidence/experiments/run.py P016"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P016"
ITEMS = 6


def variable_weighted_mean_records(*, rows: int, seed: int) -> Iterator[dict]:
    """Draw variable-width rows with broad positive weights."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        length = int(rng.integers(2, ITEMS + 1))
        values = rng.uniform(-1.0, 1.0, size=length)
        weights = np.exp(rng.uniform(np.log(0.03), np.log(3.0), size=length))
        yield {
            "items": [
                {"value": float(value), "weight": float(weight)} for value, weight in zip(values, weights, strict=True)
            ],
            "weighted_mean": float(np.dot(values, weights) / weights.sum()),
        }


def permute_weights(observations: list[dict], *, seed: int) -> list[dict]:
    """Break value/weight pairing within rows while preserving both marginals."""
    rng = np.random.default_rng(seed)
    rows = deepcopy(observations)
    changed = 0
    for row in rows:
        items = row["items"]
        permutation = rng.permutation(len(items))
        if np.array_equal(permutation, np.arange(len(items))):
            permutation = np.roll(permutation, 1)
        shuffled = [items[index]["weight"] for index in permutation]
        changed += sum((item["weight"] != weight for item, weight in zip(items, shuffled, strict=True)))
        row["items"] = [
            {"value": item["value"], "weight": weight} for item, weight in zip(items, shuffled, strict=True)
        ]
    if changed == 0:
        raise ValueError("weight permutation did not change any item-local associations")
    return rows


def permute_items(observations: list[dict], *, seed: int) -> list[dict]:
    """Jointly permute complete items without changing any target."""
    rng = np.random.default_rng(seed)
    rows = deepcopy(observations)
    for row in rows:
        items = row["items"]
        row["items"] = [items[index] for index in rng.permutation(len(items))]
    return rows


def scale_weights(observations: list[dict], *, factor: float) -> list[dict]:
    """Scale every raw weight and update targets according to their algebra."""
    if factor <= 0.0:
        raise ValueError(f"weight scale must be positive, got {factor}")
    rows = deepcopy(observations)
    for row in rows:
        row["items"] = [{**item, "weight": factor * item["weight"]} for item in row["items"]]
        if "weighted_sum" in row:
            row["weighted_sum"] *= factor
    return rows


def duplicate_items(observations: list[dict], *, maximum_original: int) -> list[dict]:
    """Duplicate every item on rows that remain within configured capacity."""
    rows = [row for row in deepcopy(observations) if len(row["items"]) <= maximum_original]
    if not rows:
        raise ValueError(f"no rows contain at most {maximum_original} items")
    for row in rows:
        row["items"] = [*row["items"], *row["items"]]
        if "weighted_sum" in row:
            row["weighted_sum"] *= 2.0
    return rows


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["weighted_mean"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["weighted_mean"] for row in test], dtype=np.float64)
    baseline_rmse = rmse(actual, float(np.asarray([row["weighted_mean"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline_rmse, "nrmse": measured / baseline_rmse}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(variable_weighted_mean_records(rows=1536, seed=seed + 1))
    test = list(variable_weighted_mean_records(rows=768, seed=seed + 3))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        items=rf.Branch(
            length=ITEMS, n_layers=2, reduction=rf.Attention(n_layers=2), value=rf.Number, weight=rf.Number
        ),
        weighted_mean=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        train=lambda: variable_weighted_mean_records(rows=1536, seed=seed + 1),
        validate=lambda: variable_weighted_mean_records(rows=384, seed=seed + 2),
        seed=seed,
    )
    model.fit(
        datamodule,
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=1100 if steps is None else min(steps, 1100),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer

    # Evaluate held-out answers and retain their original labels in corruption controls.
    intact_prediction = prediction(model, test)
    intact = score(train=train, test=test, predicted=intact_prediction)
    corrupted_table = permute_weights(test, seed=seed + 4)
    corrupted = score(train=train, test=test, predicted=prediction(model, corrupted_table))
    permuted_prediction = prediction(model, permute_items(test, seed=seed + 5))
    scaled_prediction = prediction(model, scale_weights(test, factor=0.75))
    original_short = [row for row in deepcopy(test) if len(row["items"]) <= ITEMS // 2]
    duplicated_prediction = prediction(model, duplicate_items(test, maximum_original=ITEMS // 2))
    original_short_prediction = prediction(model, original_short)
    target_scale = float(np.std(np.asarray([row["weighted_mean"] for row in test], dtype=np.float64)))
    permutation_drift = rmse(intact_prediction, permuted_prediction) / target_scale
    scaling_drift = rmse(intact_prediction, scaled_prediction) / target_scale
    duplication_drift = rmse(original_short_prediction, duplicated_prediction) / target_scale
    metrics = {
        "intact": intact,
        "corrupted": corrupted,
        "permutation_drift": permutation_drift,
        "scaling_drift": scaling_drift,
        "duplication_drift": duplication_drift,
        "steps": trainer.global_step,
    }
    checks = {
        "Weighted mean nRMSE below 0.40": bool(intact["nrmse"] < 0.4),
        "Shuffled weights raise nRMSE by at least 0.25": bool(corrupted["nrmse"] >= intact["nrmse"] + 0.25),
        "Permutation drift below 0.10 target SD": bool(permutation_drift < 0.1),
        "Weight scaling drift below 0.10 target SD": bool(scaling_drift < 0.1),
        "Duplication drift below 0.10 target SD": bool(duplication_drift < 0.1),
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3313)

# %% [markdown]
# </details>
