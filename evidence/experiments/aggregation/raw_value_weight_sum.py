# %% [markdown]
# ---
# title: Does each value keep its own weight?
# categories:
# - Weighted aggregation
# evidence-id: P014
# description: Predict a weighted total from values and weights stored together on each item.
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
#     Permutation drift below 0.08 target SD: true
#     Scaled-weight nRMSE below 0.35: true
#     Shuffled weights raise nRMSE by at least 0.35: true
#     Weight scaling error below 0.15 target SD: true
#     Weighted sum nRMSE below 0.30: true
#   duration_seconds: 50.364
#   metrics:
#     corrupted:
#       baseline_rmse: 1.2971704626253104
#       nrmse: 0.5792988188744012
#       rmse: 0.7514493168776029
#     intact:
#       baseline_rmse: 1.2971704626253104
#       nrmse: 0.036551052774927026
#       rmse: 0.04741294603749423
#     permutation_drift: 0.014143708115649389
#     scaled:
#       baseline_rmse: 0.9728778469689828
#       nrmse: 0.05425659371209745
#       rmse: 0.05278503807449622
#     scaling_error: 0.0306264273194869
#     steps: 900
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: f98dbeee02bddcec730530571e215f2299f6f7cf747232b3dc7de394112f428f
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 6e016c94bc5c
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
#     script_sha256: 052848ff49008eb73d97ce93f4c8211865f5d7be1edcb24cbf2dbc3d714efa0e
#     source_sha256: 5290bb90c721943b68b9b4e2380e9f6b89cb12d98452b2e0e749ee35eb9e53ed
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 3304
#   started_at: '2026-11-01T01:34:33.003677+00:00'
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
#   - {value: -0.4, weight: 1.0}
#   - {value: 0.2, weight: 1.4}
#   - {value: 0.5, weight: 0.2}
#   - {value: -0.5, weight: 0.4}
#   - {value: 0.9, weight: 0.3}
# weighted_sum: 0.45
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-raw-value-weight-sum
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, items, value, weight, weighted_sum. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("value", type: "Number"),
#     node("weight", type: "Number"),
#   )),
#   node("weighted_sum", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Swap weights between items while retaining the original total. Also reorder whole items and scale all weights together.
#
# ## Result
#
# {{< evidence P014 status >}}
#
# The model uses the pairing: swapping weights worsens accuracy. This is approximate learned arithmetic on fixed-length lists.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P014 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P014 script >}}
#
# ### How it works
#
# The item branch and root use learned `rf.Attention` reductions. Each of
# six items contains independently sampled values and positive weights. The model
# receives no product field: it must learn enough about the sibling relationship
# to predict `sum(value * weight)`.
#
# Swapping weights between items preserves both marginal distributions while
# breaking their pairing. Scoring these changed inputs against the original labels
# must worsen error. Reordering complete items should leave the answer stable;
# scaling every weight by 0.75 should scale the prediction by the same factor.
# The [supplied-contribution control](supplied-contribution-sum.html) isolates the
# summation part of this task.
#
# ### Remaining work
#
# Repeat the pairing and scaling controls across seeds. Variable-length sum
# duplication, unseen lengths, and zero, negative, missing, or extreme weights
# remain unestablished for this particular weighted-sum schema.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""Raw sibling value and weight fields should support a weighted sum.

Learn products and their sum without a derived contribution field.

Run: uv run python evidence/experiments/run.py P014"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P014"
ITEMS = 6


def weighted_records(*, rows: int, length: int, seed: int) -> Iterator[dict]:
    """Draw random value/weight pairs and expose raw fields or their products."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        values = rng.uniform(-1.0, 1.0, size=length)
        weights = rng.uniform(0.15, 1.5, size=length)
        contributions = values * weights
        items = [
            {"value": float(value), "weight": float(weight)} for value, weight in zip(values, weights, strict=True)
        ]
        yield {
            "items": items,
            "weighted_sum": float(contributions.sum()),
            "weighted_mean": float(contributions.sum() / weights.sum()),
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


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["weighted_sum"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["weighted_sum"] for row in test], dtype=np.float64)
    baseline_rmse = rmse(actual, float(np.asarray([row["weighted_sum"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline_rmse, "nrmse": measured / baseline_rmse}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(weighted_records(rows=1024, length=ITEMS, seed=seed + 1))
    test = list(weighted_records(rows=512, length=ITEMS, seed=seed + 3))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        items=rf.Branch(
            length=ITEMS, n_layers=2, reduction=rf.Attention(n_layers=2), value=rf.Number, weight=rf.Number
        ),
        weighted_sum=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        train=lambda: weighted_records(rows=1024, length=ITEMS, seed=seed + 1),
        validate=lambda: weighted_records(rows=256, length=ITEMS, seed=seed + 2),
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
        max_steps=900 if steps is None else min(steps, 900),
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
    factor = 0.75
    scaled_table = scale_weights(test, factor=factor)
    scaled_prediction = prediction(model, scaled_table)
    scaled = score(train=scale_weights(train, factor=factor), test=scaled_table, predicted=scaled_prediction)
    target_scale = float(np.std(np.asarray([row["weighted_sum"] for row in test], dtype=np.float64)))
    permutation_drift = rmse(intact_prediction, permuted_prediction) / target_scale
    scaling_error = rmse(factor * intact_prediction, scaled_prediction) / (factor * target_scale)
    metrics = {
        "intact": intact,
        "corrupted": corrupted,
        "scaled": scaled,
        "permutation_drift": permutation_drift,
        "scaling_error": scaling_error,
        "steps": trainer.global_step,
    }
    checks = {
        "Weighted sum nRMSE below 0.30": bool(intact["nrmse"] < 0.3),
        "Shuffled weights raise nRMSE by at least 0.35": bool(corrupted["nrmse"] >= intact["nrmse"] + 0.35),
        "Scaled-weight nRMSE below 0.35": bool(scaled["nrmse"] < 0.35),
        "Permutation drift below 0.08 target SD": bool(permutation_drift < 0.08),
        "Weight scaling error below 0.15 target SD": bool(scaling_error < 0.15),
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3304)

# %% [markdown]
# </details>
