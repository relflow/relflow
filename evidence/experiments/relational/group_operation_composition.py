# %% [markdown]
# ---
# title: Can it choose both a group and an operation?
# categories:
# - Category-conditioned reduction
# evidence-id: P027
# description: Ask for a group’s minimum, maximum, average, or total from the same list.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Group A, maximum
#   metric:
#   - intact/max/A/nrmse
#   format: error
# - label: Group A, average
#   metric:
#   - intact/mean/A/nrmse
#   format: error
# - label: Group A, total
#   metric:
#   - intact/sum/A/nrmse
#   format: error
# evidence-group: Relational
# evidence-result:
#   accelerator: cpu
#   checks:
#     max/A finite calibration nRMSE <= 1.60: true
#     max/A nRMSE <= 0.30: true
#     max/B finite calibration nRMSE <= 1.60: true
#     max/B nRMSE <= 0.30: true
#     max/C finite calibration nRMSE <= 1.60: true
#     max/C nRMSE <= 0.30: true
#     mean/A finite calibration nRMSE <= 1.60: true
#     mean/A nRMSE <= 0.30: true
#     mean/B finite calibration nRMSE <= 1.60: true
#     mean/B nRMSE <= 0.30: true
#     mean/C finite calibration nRMSE <= 1.60: true
#     mean/C nRMSE <= 0.30: true
#     min/A finite calibration nRMSE <= 1.60: true
#     min/A nRMSE <= 0.30: true
#     min/B finite calibration nRMSE <= 1.60: true
#     min/B nRMSE <= 0.30: true
#     min/C finite calibration nRMSE <= 1.60: true
#     min/C nRMSE <= 0.30: true
#     sum/A finite calibration nRMSE <= 1.60: true
#     sum/A nRMSE <= 0.30: true
#     sum/B finite calibration nRMSE <= 1.60: true
#     sum/B nRMSE <= 0.30: true
#     sum/C finite calibration nRMSE <= 1.60: true
#     sum/C nRMSE <= 0.30: true
#   duration_seconds: 434.577
#   metrics:
#     intact/max/A/baseline_rmse: 0.8798919617581095
#     intact/max/A/nrmse: 0.04286694995995274
#     intact/max/A/rmse: 0.03771828469484952
#     intact/max/B/baseline_rmse: 0.8645386409025352
#     intact/max/B/nrmse: 0.07849549016814521
#     intact/max/B/rmse: 0.06786238438694657
#     intact/max/C/baseline_rmse: 0.8226223328561437
#     intact/max/C/nrmse: 0.12087299298076683
#     intact/max/C/rmse: 0.09943282346514269
#     intact/mean/A/baseline_rmse: 0.8703165895406241
#     intact/mean/A/nrmse: 0.054073957974311644
#     intact/mean/A/rmse: 0.04706146268716594
#     intact/mean/B/baseline_rmse: 0.8474965036769826
#     intact/mean/B/nrmse: 0.06582768876932737
#     intact/mean/B/rmse: 0.05578873607714153
#     intact/mean/C/baseline_rmse: 0.814604977900874
#     intact/mean/C/nrmse: 0.08347560333332762
#     intact/mean/C/rmse: 0.06799964200860747
#     intact/min/A/baseline_rmse: 0.8766077148642057
#     intact/min/A/nrmse: 0.04489577314753063
#     intact/min/A/rmse: 0.03935598110591859
#     intact/min/B/baseline_rmse: 0.8365111183932998
#     intact/min/B/nrmse: 0.05399844421053883
#     intact/min/B/rmse: 0.04517029895805604
#     intact/min/C/baseline_rmse: 0.8391276941102654
#     intact/min/C/nrmse: 0.05619559117210402
#     intact/min/C/rmse: 0.047155276839410835
#     intact/sum/A/baseline_rmse: 3.4812663581624963
#     intact/sum/A/nrmse: 0.02747081470473052
#     intact/sum/A/rmse: 0.09563322306289397
#     intact/sum/B/baseline_rmse: 3.3899860147079304
#     intact/sum/B/nrmse: 0.03561627651550523
#     intact/sum/B/rmse: 0.12073867928353324
#     intact/sum/C/baseline_rmse: 3.258419911603496
#     intact/sum/C/nrmse: 0.05008986621860497
#     intact/sum/C/rmse: 0.16321381745625776
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: a252610989beb880a57873643d062754d3d98490d69abede7e46d4d4468a5c5e
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: f23cd631aa14
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
#     script_sha256: 133a05d8c784f169f3a72bcf4b46f376b278a72333dc5a985d02bb308fd8f6c1
#     source_sha256: 0ba1d8c96535bdd12e0376d31ea755f9ffb73d7ee92c89172a8ce4b8d3ef1f8f
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 16
#   started_at: '2026-11-01T01:37:21.898057+00:00'
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
# bag: 0
# items:
#   - {group: A, value: 0.1}
#   - {group: B, value: -0.1}
#   - {group: C, value: 0.0}
#   - {group: A, value: 0.4}
#   - {group: B, value: 0.3}
#   - {group: C, value: 0.5}
#   - {group: A, value: -1.0}
#   - {group: B, value: -0.9}
#   - {group: C, value: -1.2}
#   - {group: A, value: 1.2}
#   - {group: B, value: 1.3}
#   - {group: C, value: 1.0}
# selected_group: B
# operation: max
# answer: 1.3
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-group-operation-composition
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with request, items, value, group, answer, operation, selected_group. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("request", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("value", type: "Number"),
#     node("group", type: "Category"),
#   )),
#   node("answer", kind: "target", type: "Number", width: 150pt,),
#   node("operation", type: "Category"),
#   node("selected_group", type: "Category"),
# )))
# ```
#
# ## Comparison
#
# Measure every group-and-operation combination so a good score on one cannot hide a poor score on another.
#
# ## Result
#
# {{< evidence P027 status >}}
#
# All tested combinations learn. This exact experiment does not corrupt either request, so stronger evidence that both instructions are used remains needed.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P027 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P027 script >}}
#
# ### How it works
#
# Coordinate-local mixing binds groups to values, retained item slots preserve
# their associations, and visible request fields condition the decoder query.
# Independently generated bags are assigned to each split before expansion into
# requests, so related answers stay together. Measuring all twelve cells prevents
# a favorable average from hiding one failed group or operation.
#
# ### Remaining work
#
# Add label, selected-group, and operation corruptions to this composed case.
# Repeat three core seeds and ten calibration seeds, vary collection length, and
# test absent groups and missing values. Sweep reduction width independently of
# the number of requested cells. See the [filtered-mean control](filtered-mean.html).
#
# ### Complete experiment code
#

# %%
"""Compose a requested group filter with sum, mean, minimum, or maximum.

Each bag contains four items from each of three groups and produces all twelve
requests. Score every group/operation cell against its own training-mean
baseline so one easy operation cannot hide failure on another.

Run this file with --help for seed, training-budget, and reporting options.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P027"
OPERATIONS = ("sum", "mean", "min", "max")
GROUPS = ("A", "B", "C")


def reduce(values: np.ndarray, operation: str) -> float:
    """Execute one synthetic reduction outside RelFlow."""

    match operation:
        case "sum":
            return float(values.sum())
        case "mean":
            return float(values.mean())
        case "min":
            return float(values.min())
        case "max":
            return float(values.max())
        case _:
            raise ValueError(f"unknown reduction operation: {operation!r}")


def values(rng: np.random.Generator, length: int) -> np.ndarray:
    """Draw bounded asymmetric values whose reductions are usually distinct."""

    if length < 4:
        raise ValueError("reduction bags require at least four values")
    middle = rng.uniform(-0.2, 0.6, size=length - 2)
    low = rng.uniform(-1.3, -0.7, size=1)
    high = rng.uniform(0.9, 1.7, size=1)
    return np.concatenate((middle, low, high))


def records(*, bags: int, items_per_group: int, seed: int) -> Iterator[dict]:
    """Expand interleaved grouped bags into every group-operation request."""
    rng = np.random.default_rng(seed)
    length = len(GROUPS) * items_per_group
    for bag in range(bags):
        bag_values = np.concatenate([values(rng, items_per_group) + rng.uniform(-1.5, 1.5) for _ in GROUPS])
        labels = np.repeat(np.asarray(GROUPS), items_per_group)
        order = rng.permutation(length)
        items = [{"group": str(labels[index]), "value": float(bag_values[index])} for index in order]
        for group in GROUPS:
            selected = bag_values[labels == group]
            for operation in OPERATIONS:
                yield {
                    "bag": bag,
                    "selected_group": group,
                    "operation": operation,
                    "items": items,
                    "answer": reduce(selected, operation),
                }


def scores(train: list[dict], test: list[dict], predicted: np.ndarray, keys: tuple[str, ...]) -> dict:
    """Normalize each request cell against that cell's training-target mean."""
    result = {}
    cells = sorted({tuple(row[key] for key in keys) for row in test})
    for cell in cells:
        mean = float(np.mean([row["answer"] for row in train if tuple(row[key] for key in keys) == cell]))
        indices = [i for i, row in enumerate(test) if tuple(row[key] for key in keys) == cell]
        actual = np.asarray([test[i]["answer"] for i in indices])
        error = float(np.sqrt(np.mean(np.square(predicted[indices] - actual))))
        baseline = float(np.sqrt(np.mean(np.square(mean - actual))))
        result["/".join(cell)] = {"rmse": error, "baseline_rmse": baseline, "nrmse": error / baseline}
    return result


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    inputs = [{key: value for key, value in row.items() if key != "answer"} for row in rows]
    output = model.predict(inputs).to_pylist()
    return np.asarray([row["predictions"]["answer"]["content"] for row in output])


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=64,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_outputs=12, n_layers=2),
        items=rf.Branch(
            length=12,
            n_layers=2,
            reduction=None,
            value=rf.Number,
            group=rf.Category(p_unavailable=0.0),
        ),
        answer=rf.Number(mask=True, objective="mse"),
        operation=rf.Category(p_unavailable=0.0),
        selected_group=rf.Category(p_unavailable=0.0),
    )
    data = rf.SyntheticDataModule(
        train=partial(records, bags=512, items_per_group=4, seed=seed + 144),
        validate=partial(records, bags=64, items_per_group=4, seed=seed + 145),
    )
    model.fit(
        data,
        batch_size=128,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
        accelerator=accelerator,
        max_steps=1200 if steps is None else min(steps, 1200),
        max_epochs=-1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    train = list(records(bags=512, items_per_group=4, seed=seed + 144))
    test = list(records(bags=128, items_per_group=4, seed=seed + 146))
    intact_prediction = predict(model, test)
    intact = scores(train, test, intact_prediction, ("operation", "selected_group"))
    metrics = {f"intact/{cell}/{name}": value for cell, score in intact.items() for name, value in score.items()}
    checks = {
        f"{cell} finite calibration nRMSE <= 1.60": bool(np.isfinite(score["nrmse"])) and score["nrmse"] <= 1.60
        for cell, score in intact.items()
    }
    checks.update({f"{cell} nRMSE <= 0.30": score["nrmse"] <= 0.30 for cell, score in intact.items()})
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=16)

# %% [markdown]
# </details>
