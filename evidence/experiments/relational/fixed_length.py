# %% [markdown]
# ---
# title: Can an instruction change the calculation?
# categories:
# - Operation-conditioned reduction
# evidence-id: P031
# description: Ask for a list’s minimum, maximum, average, or total by changing a visible operation field.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Total, instruction visible
#   metric:
#   - intact/sum/nrmse
#   format: error
# - label: Total, instruction hidden
#   metric:
#   - hidden/sum/nrmse
#   format: error
# evidence-group: Relational
# evidence-result:
#   accelerator: cpu
#   checks:
#     At least three hidden operations have nRMSE >= 0.75: true
#     Hidden operations yield identical predictions within each bag: true
#     Permutation changes predictions by <= 0.05 target SD: true
#     max intact nRMSE <= 0.20: true
#     max permuted nRMSE <= 0.25: true
#     mean intact nRMSE <= 0.20: true
#     mean permuted nRMSE <= 0.25: true
#     min intact nRMSE <= 0.20: true
#     min permuted nRMSE <= 0.25: true
#     sum intact nRMSE <= 0.20: true
#     sum permuted nRMSE <= 0.25: true
#   duration_seconds: 75.469
#   metrics:
#     hidden/max/baseline_rmse: 0.23385342033136094
#     hidden/max/nrmse: 9.191587736963024
#     hidden/max/rmse: 2.149484230564597
#     hidden/mean/baseline_rmse: 0.07489171086510313
#     hidden/mean/nrmse: 17.047093365894707
#     hidden/mean/rmse: 1.276685987449004
#     hidden/min/baseline_rmse: 0.16915466453487707
#     hidden/min/nrmse: 6.832057840205612
#     hidden/min/rmse: 1.155674452042857
#     hidden/sum/baseline_rmse: 0.599133686920825
#     hidden/sum/nrmse: 3.5990495339513884
#     hidden/sum/rmse: 2.1563118166869724
#     hidden_request_spread: 0.0
#     intact/max/baseline_rmse: 0.23385342033136094
#     intact/max/nrmse: 0.058413740007171214
#     intact/max/rmse: 0.013660252895023844
#     intact/mean/baseline_rmse: 0.07489171086510313
#     intact/mean/nrmse: 0.07846642545117143
#     intact/mean/rmse: 0.0058764848475073
#     intact/min/baseline_rmse: 0.16915466453487707
#     intact/min/nrmse: 0.05565680791155266
#     intact/min/rmse: 0.009414608671360783
#     intact/sum/baseline_rmse: 0.599133686920825
#     intact/sum/nrmse: 0.04403293518414524
#     intact/sum/rmse: 0.026381614802822657
#     permutation_delta: 0.007678211929192353
#     permuted/max/baseline_rmse: 0.23385342033136094
#     permuted/max/nrmse: 0.05717259700314421
#     permuted/max/rmse: 0.013370007358411791
#     permuted/mean/baseline_rmse: 0.07489171086510313
#     permuted/mean/nrmse: 0.08003424237414494
#     permuted/mean/rmse: 0.005993901339192048
#     permuted/min/baseline_rmse: 0.16915466453487707
#     permuted/min/nrmse: 0.05667688235228531
#     permuted/min/rmse: 0.009587159021183516
#     permuted/sum/baseline_rmse: 0.599133686920825
#     permuted/sum/nrmse: 0.04579850124129583
#     permuted/sum/rmse: 0.02743942490414555
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 261cdefe0422a418461fd4cbf42eb1c1300bcfed16a0c6aae7e21c9d20bb84d7
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 9f4a7ffbd64d
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
#     script_sha256: fcaffe8bf940dd1b67e63d841b574e0c46b9b693a777684413914b81e32ff250
#     source_sha256: 96f077c91edf789a41b37c38c6f72ba31f4420eda3863d6d0724ea8614fb9e60
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 15
#   started_at: '2026-11-01T01:39:49.702745+00:00'
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
#   - {value: 0.1}
#   - {value: -0.1}
#   - {value: 0.4}
#   - {value: 0.2}
#   - {value: 0.5}
#   - {value: 0.0}
#   - {value: -1.0}
#   - {value: 1.2}
# operation: max
# answer: 1.2
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-fixed-length
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with request, items, value, answer, operation. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("request", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("value", type: "Number"),
#   )),
#   node("answer", kind: "target", type: "Number", width: 150pt,),
#   node("operation", type: "Category"),
# )))
# ```
#
# ## Comparison
#
# Hide the operation while keeping the different answers. Also reorder items without changing the requested calculation.
#
# ## Result
#
# {{< evidence P031 status >}}
#
# The operation matters: hiding it makes predictions collapse to the same answer. All lists have eight items, so new lengths remain untested.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P031 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P031 script >}}
#
# ### How it works
#
# The decoder must learn how the request changes the use of collection context.
# Permuting complete items should preserve a statistic; hiding the operation
# makes the four requests for a bag identical. Their predictions should then
# collapse to one value and lose accuracy on the contradictory targets.
#
# ### Remaining work
#
# Evaluate every operation at lengths 4–16, then characterize degradation through
# 32. If sum fails alone, a visible-count control can isolate lost cardinality.
# Repeat three core seeds and ten calibration seeds before fixing the supported
# range. Exact numerical reductions belong in preprocessing when exactness is
# the application contract.
#
# ### Complete experiment code
#

# %%
"""Learn four visibly requested reductions of the same eight-value bag.

Every bag produces sum, mean, minimum, and maximum requests. Permuting whole
items should preserve answers. Hiding the operation makes these requests
identical, exposing whether the requested operation drives the prediction.

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

PROOF_ID = "P031"
OPERATIONS = ("sum", "mean", "min", "max")


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


def records(*, bags: int, length: int, seed: int) -> Iterator[dict]:
    """Expand each independently drawn bag into its four reduction requests."""
    rng = np.random.default_rng(seed)
    for bag in range(bags):
        bag_values = values(rng, length)
        order = rng.permutation(length)
        items = [{"value": float(bag_values[index])} for index in order]
        for operation in OPERATIONS:
            yield {"bag": bag, "operation": operation, "items": items, "answer": reduce(bag_values, operation)}


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


def permute_items(rows: list[dict], seed: int) -> list[dict]:
    """Reorder whole item records once per bag, preserving all four requests."""
    rng = np.random.default_rng(seed)
    reordered = {}
    result = []
    for row in rows:
        bag = row["bag"]
        if bag not in reordered:
            items = row["items"]
            reordered[bag] = [items[i] for i in rng.permutation(len(items))]
        result.append({**row, "items": reordered[bag]})
    return result


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=64,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        items=rf.Branch(length=8, n_layers=2, reduction=rf.Attention(n_layers=2), value=rf.Number),
        answer=rf.Number(mask=True, objective="mse"),
        operation=rf.Category(p_unavailable=0.0),
    )
    data = rf.SyntheticDataModule(
        train=partial(records, bags=768, length=8, seed=seed + 135),
        validate=partial(records, bags=96, length=8, seed=seed + 136),
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
        max_steps=800 if steps is None else min(steps, 800),
        max_epochs=-1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    train = list(records(bags=768, length=8, seed=seed + 135))
    test = list(records(bags=192, length=8, seed=seed + 137))
    intact_prediction = predict(model, test)
    intact = scores(train, test, intact_prediction, ("operation",))
    metrics = {f"intact/{cell}/{name}": value for cell, score in intact.items() for name, value in score.items()}
    permuted_prediction = predict(model, permute_items(test, seed + 138))
    permuted = scores(train, test, permuted_prediction, ("operation",))
    hidden_prediction = predict(model, [{**row, "operation": None} for row in test])
    hidden = scores(train, test, hidden_prediction, ("operation",))
    for label, result in (("permuted", permuted), ("hidden", hidden)):
        metrics.update(
            {f"{label}/{cell}/{name}": value for cell, score in result.items() for name, value in score.items()}
        )
    scale = float(np.std([row["answer"] for row in test]))
    delta = float(np.sqrt(np.mean(np.square(intact_prediction - permuted_prediction)))) / scale
    hidden_spread = float(np.max(np.ptp(hidden_prediction.reshape(-1, len(OPERATIONS)), axis=1)))
    metrics.update(permutation_delta=delta, hidden_request_spread=hidden_spread)
    checks = {f"{cell} intact nRMSE <= 0.20": score["nrmse"] <= 0.20 for cell, score in intact.items()}
    checks.update({f"{cell} permuted nRMSE <= 0.25": score["nrmse"] <= 0.25 for cell, score in permuted.items()})
    checks.update(
        {
            "Permutation changes predictions by <= 0.05 target SD": delta <= 0.05,
            "Hidden operations yield identical predictions within each bag": hidden_spread <= 1e-6,
            "At least three hidden operations have nRMSE >= 0.75": sum(
                score["nrmse"] >= 0.75 for score in hidden.values()
            )
            >= 3,
        }
    )
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=15)

# %% [markdown]
# </details>
