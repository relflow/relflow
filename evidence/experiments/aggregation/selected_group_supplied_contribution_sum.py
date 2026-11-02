# %% [markdown]
# ---
# title: Can it total prepared contributions for one group?
# categories:
# - Grouped weighted aggregation
# evidence-id: P011
# description: Supply already calculated contributions and ask for the total belonging to one group.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Correct groups
#   metric:
#   - intact
#   - nrmse
#   format: error
# - label: Changed groups
#   metric:
#   - corrupted
#   - nrmse
#   format: error
# evidence-group: Aggregation
# evidence-result:
#   accelerator: cpu
#   checks:
#     All measurements finite: true
#     Corrupt calibration nRMSE below 1.80: true
#     Grouped contribution nRMSE below 0.50: true
#     Intact calibration nRMSE below 1.60: true
#     Permutation drift below 0.10 target SD: true
#     Rotated labels raise nRMSE by at least 0.20: true
#   duration_seconds: 83.626
#   metrics:
#     corrupted:
#       baseline_rmse: 0.9164778582723881
#       nrmse: 1.4033961925668208
#       rmse: 1.286181536871264
#     intact:
#       baseline_rmse: 0.9164778582723881
#       nrmse: 0.014767172293548844
#       rmse: 0.013533786436330995
#     permutation_drift: 0.008293799985827379
#     steps: 800
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: e205e145d072a51c29b73dc3872d7da5f22eaac9a135726f865017f5bb697cd2
#     cuda: null
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: 85af86fa3f51
#     packages:
#       lightning: 2.6.4
#       numpy: 2.4.6
#       polars: 2.0.0
#       pyarrow: 24.0.0
#       pydantic: 2.13.4
#       relflow: 0.4.2
#       tensordict: 0.12.4
#       torch: 2.12.0+cpu
#       torchmetrics: 1.9.0
#     platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.41
#     python: 3.12.16
#     relflow_path: /tmp/runtime/lib/python3.12/site-packages/relflow
#     relflow_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     relflow_source_sha256: 77953506049ed76c92adc5f32a9bcc32432a788068a4e366b2f78bcf709f29a2
#     script_sha256: d517e7abfc776b83daf6fbc8865109b70c3447fbd4f287d5a97e38ee2de23a30
#     source_sha256: 719e02229161698b2662a8b376f55c84935322abefce0b17f85cf1edc6753cf6
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 3600
#   started_at: '2026-11-02T17:47:25.068405+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41494093900-1
#   candidate:
#     payload_sha256: 2169c968a7733085889a6eed6e895bcb2590059bcc46243d49a61ffacc42f67f
#     revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   baseline:
#     payload_sha256: 2a6af4bd3869faac23f6b7327f2c59e8f5b367580b2f2fa609710958775c08ed
#     revision: 7dec5e860dc389c37c3642d8904867249e831773
#     wheel_sha256: d3da50b65e0c0c7b98059f095b82bf6bf5d0ea0c5ccd0f302015118075f1dd85
#   evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#   executor_revision: 92b3c887eabc598ee2fc6c9dba4e05b9166cb8ba
#   manifest_sha256: 4f0926f3fc366fd8d14bf728b2d06f4a475d61852679176704816c110ae51cfd
#   passed: true
#   github_run:
#     event: push
#     head_branch: main
#     head_sha: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     html_url: https://github.com/relflow/relflow/actions/runs/41494093900
#     path: .github/workflows/ci.yml
#     run_attempt: 1
#     run_id: 41494093900
#     workflow_id: 403231172
#   release: https://github.com/relflow/relflow/releases/tag/evidence-ci-41494093900-attempt-1
#   reports_sha256: b1bfee5622ffbd389b1777443dc05e6d8f08dccff9e2540a601f0c2608948594
# ---
#
# ## Example
#
# ```yaml
# selected_group: A
# items:
#   - {group: B, contribution: 0.4}
#   - {group: A, contribution: 0.6}
#   - {group: C, contribution: -0.2}
#   - {group: A, contribution: -0.1}
#   - {group: B, contribution: 0.3}
#   - {group: C, contribution: 0.8}
# answer: 0.5
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-selected-group-supplied-contribution-sum
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with request, selected_group, items, group, contribution, answer. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("request", kind: "root", width: 120pt, children: (
#   node("selected_group", type: "Category"),
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("group", type: "Category"),
#     node("contribution", type: "Number"),
#   )),
#   node("answer", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Reuse a list with different group requests, then shuffle item group labels while keeping the original answers.
#
# ## Result
#
# {{< evidence P011 status >}}
#
# The model uses group membership to select contributions. Because the products are supplied, this does not test learned multiplication.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P011 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P011 script >}}
#
# ### How it works
#
# Every item receives a precomputed value–weight product. Its group stays
# beside that contribution in the repeated branch. `reduction=None` preserves
# these coordinates, and root `rf.Attention` combines them with a visible group
# request. Every six-item bag appears once for each of A, B, and C.
#
# Rotating labels A → B → C → A changes which contributions answer the request
# while retaining all label counts and numerical values. The control keeps the
# original answer, so a group-sensitive model should score worse. Permuting
# complete items should leave predictions stable. The
# [raw-pair evidence](selected-group-raw-weighted-sum.html) adds the multiplication
# requirement that this diagnostic deliberately supplies.
#
# ### Remaining work
#
# Repeat across seeds and vary group counts independently of bag size.
# Absent groups and held-out labels need separate checks. Passing this control
# supports selection and reduction, not learned multiplication.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""Supplied products isolate category selection and fixed-width summation.

Learn filtering and reduction after item-local products are supplied.

Run: uv run python evidence/experiments/run.py P011"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P011"
GROUPS = ("A", "B", "C")
ITEMS_PER_GROUP = 2
ITEMS = len(GROUPS) * ITEMS_PER_GROUP


def records(*, bags: int, seed: int) -> Iterator[dict]:
    """Draw interleaved groups and emit one request for every selected group."""
    rng = np.random.default_rng(seed)
    for bag in range(bags):
        items: list[dict[str, object]] = []
        answers: dict[str, float] = {}
        for group in GROUPS:
            values = rng.uniform(-1.0, 1.0, size=ITEMS_PER_GROUP)
            weights = rng.uniform(0.2, 1.8, size=ITEMS_PER_GROUP)
            contributions = values * weights
            answers[group] = float(contributions.sum())
            for value, weight, contribution in zip(values, weights, contributions, strict=True):
                item: dict[str, object] = {"group": group}
                item["contribution"] = float(contribution)
                items.append(item)
        interleaved = [items[index] for index in rng.permutation(len(items))]
        for selected_group in GROUPS:
            yield {
                "bag": bag,
                "selected_group": selected_group,
                "items": interleaved,
                "answer": answers[selected_group],
            }


def rotate_group_labels(observations: list[dict]) -> list[dict]:
    """Break group/item association while preserving all group marginals."""
    successor = {group: GROUPS[(index + 1) % len(GROUPS)] for index, group in enumerate(GROUPS)}
    rows = deepcopy(observations)
    for row in rows:
        row["items"] = [{**item, "group": successor[item["group"]]} for item in row["items"]]
    return rows


def permute_items(observations: list[dict], *, seed: int) -> list[dict]:
    """Jointly permute complete items while retaining targets."""
    rng = np.random.default_rng(seed)
    rows = deepcopy(observations)
    cached: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        bag = int(row["bag"])
        if bag not in cached:
            items = row["items"]
            cached[bag] = [items[index] for index in rng.permutation(len(items))]
        row["items"] = cached[bag]
    return rows


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"selected_group": row["selected_group"], "items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["answer"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["answer"] for row in test], dtype=np.float64)
    baseline_rmse = rmse(actual, float(np.asarray([row["answer"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline_rmse, "nrmse": measured / baseline_rmse}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(records(bags=384, seed=seed + 1))
    test = list(records(bags=192, seed=seed + 3))
    model = rf.Model(
        d_model=48,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_outputs=ITEMS, n_layers=2),
        items=rf.Branch(
            length=ITEMS,
            n_layers=2,
            reduction=None,
            group=rf.Category(p_unavailable=0.0),
            contribution=rf.Number,
        ),
        selected_group=rf.Category(p_unavailable=0.0),
        answer=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        train=lambda: records(bags=384, seed=seed + 1),
        validate=lambda: records(bags=96, seed=seed + 2),
        seed=seed,
    )
    model.fit(
        datamodule,
        batch_size=96,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
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

    # Evaluate held-out answers and retain their original labels in corruption controls.
    intact_prediction = prediction(model, test)
    intact = score(train=train, test=test, predicted=intact_prediction)
    corrupted = score(train=train, test=test, predicted=prediction(model, rotate_group_labels(test)))
    permuted_prediction = prediction(model, permute_items(test, seed=seed + 4))
    target_scale = float(np.std(np.asarray([row["answer"] for row in test], dtype=np.float64)))
    permutation_drift = rmse(intact_prediction, permuted_prediction) / target_scale
    metrics = {
        "intact": intact,
        "corrupted": corrupted,
        "permutation_drift": permutation_drift,
        "steps": trainer.global_step,
    }
    checks = {
        "All measurements finite": bool(
            all(np.isfinite(value) for value in (intact["nrmse"], corrupted["nrmse"], permutation_drift))
        ),
        "Intact calibration nRMSE below 1.60": bool(intact["nrmse"] < 1.6),
        "Corrupt calibration nRMSE below 1.80": bool(corrupted["nrmse"] < 1.8),
        "Grouped contribution nRMSE below 0.50": bool(intact["nrmse"] < 0.5),
        "Rotated labels raise nRMSE by at least 0.20": bool(corrupted["nrmse"] >= intact["nrmse"] + 0.2),
        "Permutation drift below 0.10 target SD": bool(permutation_drift < 0.1),
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3600)

# %% [markdown]
# </details>
