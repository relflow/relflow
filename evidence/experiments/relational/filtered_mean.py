# %% [markdown]
# ---
# title: Can it average only the requested group?
# categories:
# - Category-conditioned reduction
# evidence-id: P026
# description: Ask for the average of one named group within a mixed list.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Group A, correct labels
#   metric:
#   - intact/A/nrmse
#   format: error
# - label: Group A, changed labels
#   metric:
#   - corrupted/A/nrmse
#   format: error
# evidence-group: Relational
# evidence-result:
#   accelerator: cpu
#   checks:
#     Group A corruption increases nRMSE by >= 0.25: true
#     Group A nRMSE <= 0.55: true
#     Group B corruption increases nRMSE by >= 0.25: true
#     Group B nRMSE <= 0.55: true
#     Group C corruption increases nRMSE by >= 0.25: true
#     Group C nRMSE <= 0.55: true
#   duration_seconds: 116.678
#   metrics:
#     corrupted/A/baseline_rmse: 0.8356639150214514
#     corrupted/A/nrmse: 0.9475733068260727
#     corrupted/A/rmse: 0.791852819352099
#     corrupted/B/baseline_rmse: 0.8356760790930771
#     corrupted/B/nrmse: 0.9863675820342399
#     corrupted/B/rmse: 0.8242837934988927
#     corrupted/C/baseline_rmse: 0.8854705200121815
#     corrupted/C/nrmse: 0.9973113827930296
#     corrupted/C/rmse: 0.8830898287358117
#     intact/A/baseline_rmse: 0.8356639150214514
#     intact/A/nrmse: 0.009866235890848288
#     intact/A/rmse: 0.008244857311071439
#     intact/B/baseline_rmse: 0.8356760790930771
#     intact/B/nrmse: 0.011445409371025454
#     intact/B/rmse: 0.009564654826793712
#     intact/C/baseline_rmse: 0.8854705200121815
#     intact/C/nrmse: 0.0076286191598551635
#     intact/C/rmse: 0.006754917374451843
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: b5d1aab4e1ffd0d8da23848c58d3057567a1cfedc2b2a7725a1d8020b3ccca7a
#     cuda: null
#     evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     gpu: null
#     host: e3cc2dd92aa0
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
#     relflow_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     relflow_source_sha256: 77953506049ed76c92adc5f32a9bcc32432a788068a4e366b2f78bcf709f29a2
#     script_sha256: 89b3c1d44fdc3d077b83d0cf4eee86f092e549bf914729d00a78e7515bda388a
#     source_sha256: bf58c84f030b0505c37bc297a3c68ec973e98f3ab472e6b598585a80c3ea1710
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   seed: 16
#   started_at: '2026-11-02T04:09:43.768754+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41386396488-1
#   candidate:
#     revision: 7dec5e860dc389c37c3642d8904867249e831773
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   baseline:
#     revision: 7dec5e860dc389c37c3642d8904867249e831773
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#   executor_revision: 349b2202e6390a5fe0cbdf09a0c5972e495351ed
#   manifest_sha256: 1f096d9b4dcdf1b2235963715e8b41c325d2a989f2cf15da7c154ddfb9bfb08d
#   passed: true
#   github_run:
#     event: workflow_dispatch
#     head_branch: main
#     head_sha: 7dec5e860dc389c37c3642d8904867249e831773
#     html_url: https://github.com/relflow/relflow/actions/runs/41386396488
#     path: .github/workflows/ci.yml
#     run_attempt: 1
#     run_id: 41386396488
#     workflow_id: 403231172
#   release: https://github.com/relflow/relflow/releases/tag/evidence-ci-41386396488-attempt-1
#   reports_sha256: fca05e4cdb79410b5090075bab03e4fc6d4f2fbc70e798f208976f4f1d8a7ff9
# ---
#
# ## Example
#
# ```yaml
# bag: 0
# items:
#   - {group: A, value: 1.0}
#   - {group: B, value: -0.4}
#   - {group: C, value: 0.5}
#   - {group: A, value: 1.2}
#   - {group: B, value: -0.2}
#   - {group: C, value: 0.7}
# selected_group: B
# operation: mean
# answer: -0.3
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-filtered-mean
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
# Reuse the list with different group requests. Shuffle group labels while retaining the original answers.
#
# ## Result
#
# {{< evidence P026 status >}}
#
# The model uses membership to choose the relevant values. Empty groups and unfamiliar group names are not tested.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P026 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P026 script >}}
#
# ### How it works
#
# Shared item coordinates let encoding bind each category to its value. The
# visible root selection can condition which evidence the answer decoder uses.
# Each underlying bag produces three requests, one per group. Corrupting only
# item labels preserves their counts and all values while breaking membership.
# The control therefore tests selection rather than aggregate statistics alone.
#
# ### Remaining work
#
# Repeat across three core seeds and ten calibration seeds. Add missing values,
# absent groups, and variable group sizes. The next rung
# [combines group selection with four operations](group-operation-composition.html).
# For an exact business calculation, compute the filtered mean in preprocessing.
#
# ### Complete experiment code
#

# %%
"""Select the mean of one requested group from interleaved items.

Every bag contains two values from each of three groups. The operation stays
fixed at mean, isolating group selection. Permuting only group labels destroys
the original group/value relationship without changing either marginal.

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

PROOF_ID = "P026"
OPERATIONS = ("sum", "mean", "min", "max")
GROUPS = ("A", "B", "C")


def records(*, bags: int, items_per_group: int, seed: int) -> Iterator[dict]:
    """Generate the smallest group-filtering rung with one fixed reduction."""
    rng = np.random.default_rng(seed)
    for bag in range(bags):
        labels = np.repeat(np.asarray(GROUPS), items_per_group)
        bag_values = np.concatenate(
            [rng.uniform(-1.5, 1.5) + rng.uniform(-0.15, 0.15, size=items_per_group) for _ in GROUPS]
        )
        order = rng.permutation(len(labels))
        items = [{"group": str(labels[index]), "value": float(bag_values[index])} for index in order]
        for group in GROUPS:
            yield {
                "bag": bag,
                "selected_group": group,
                "operation": "mean",
                "items": items,
                "answer": float(bag_values[labels == group].mean()),
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


def corrupt_labels(rows: list[dict], seed: int) -> list[dict]:
    """Permute group/value pairings once per bag; retain every original target."""
    rng = np.random.default_rng(seed)
    corrupted = {}
    result = []
    for row in rows:
        bag = row["bag"]
        if bag not in corrupted:
            items = row["items"]
            labels = rng.permutation([item["group"] for item in items]).tolist()
            corrupted[bag] = [{**item, "group": label} for item, label in zip(items, labels, strict=True)]
        result.append({**row, "items": corrupted[bag]})
    return result


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=64,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        items=rf.Branch(
            length=6,
            n_layers=2,
            reduction=rf.Attention(n_layers=2),
            value=rf.Number,
            group=rf.Category(p_unavailable=0.0),
        ),
        answer=rf.Number(mask=True, objective="mse"),
        operation=rf.Category(p_unavailable=0.0),
        selected_group=rf.Category(p_unavailable=0.0),
    )
    data = rf.SyntheticDataModule(
        train=partial(records, bags=512, items_per_group=2, seed=seed + 147),
        validate=partial(records, bags=64, items_per_group=2, seed=seed + 148),
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
    train = list(records(bags=512, items_per_group=2, seed=seed + 147))
    test = list(records(bags=128, items_per_group=2, seed=seed + 149))
    intact_prediction = predict(model, test)
    intact = scores(train, test, intact_prediction, ("selected_group",))
    metrics = {f"intact/{cell}/{name}": value for cell, score in intact.items() for name, value in score.items()}
    corrupted = scores(train, test, predict(model, corrupt_labels(test, seed + 150)), ("selected_group",))
    metrics.update(
        {f"corrupted/{cell}/{name}": value for cell, score in corrupted.items() for name, value in score.items()}
    )
    checks = {f"Group {cell} nRMSE <= 0.55": score["nrmse"] <= 0.55 for cell, score in intact.items()}
    checks.update(
        {
            f"Group {cell} corruption increases nRMSE by >= 0.25": corrupted[cell]["nrmse"] >= score["nrmse"] + 0.25
            for cell, score in intact.items()
        }
    )
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=16)

# %% [markdown]
# </details>
