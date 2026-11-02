# %% [markdown]
# ---
# title: Can one summary support comparisons within groups?
# categories:
# - Peer-relative inference
# evidence-id: P033
# description: Compress a list into one summary, then predict each item’s difference from its group average.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Correct groups
#   metric:
#   - intact_nrmse
#   format: error
# - label: Changed groups
#   metric:
#   - corrupted_nrmse
#   format: error
# evidence-group: Relational
# evidence-result:
#   accelerator: cpu
#   checks:
#     Baseline RMSE > 0.000001: true
#     Corrupted labels nRMSE >= 0.90: true
#     Finite calibration and model metrics: true
#     Label corruption changes the oracle by > 0.75 nRMSE: true
#     Label corruption increases nRMSE by >= 0.75: true
#     Single-summary grouped nRMSE <= 0.20: true
#   duration_seconds: 157.203
#   metrics:
#     baseline_rmse: 0.7508689291187257
#     corrupted_nrmse: 1.5560312544331212
#     corrupted_rmse: 1.168375521691465
#     intact_nrmse: 0.0546641230810493
#     intact_rmse: 0.041045591559081704
#     oracle_corruption_nrmse: 1.8011994379885266
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 126cbfe66349407db67808c7b00f10c1ded83adfbb72f44d6c131bae596aaee7
#     cuda: null
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: 3f469a39de2b
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
#     script_sha256: da900c1a1cd1438f9b1f0a0840268b090039b3dc8477b8b5161079a3fea391f0
#     source_sha256: 3a5714fb5215424cddcf2ebdc70fa11f78537cb2dd7d874375c00be5676e02e4
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 3720
#   started_at: '2026-11-02T17:50:07.450885+00:00'
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
# items:
#   - {group: A, value: 2.0, deviation: -1.0}
#   - {group: B, value: 1.0, deviation: 1.0}
#   - {group: C, value: -3.0, deviation: -1.0}
#   - {group: A, value: 4.0, deviation: 1.0}
#   - {group: C, value: -1.0, deviation: 1.0}
#   - {group: B, value: -1.0, deviation: -1.0}
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-single-query-preserves-group-membership
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with collection, items, value, group, deviation. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("collection", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("value", type: "Number"),
#     node("group", type: "Category"),
#     node("deviation", kind: "target", type: "Number", width: 150pt,),
#   )),
# )))
# ```
#
# ## Comparison
#
# Change group labels while retaining original answers. Each decoder still receives its own item’s group and value.
#
# ## Result
#
# {{< evidence P033 status >}}
#
# The summary and local item information together support the task. This does not show that one summary preserves every detail of an arbitrary list.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P033 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P033 script >}}
#
# ### How it works
#
# Coordinate-local encoding first mixes each item's group and value. A learned
# summary can then retain peer context used by separate, coordinate-conditioned
# target queries. Label corruption changes peer membership while retaining the
# original targets and the value and label marginals. The resulting loss of
# accuracy tests whether that membership survived in the information available
# to the decoder.
#
# ### Remaining work
#
# Repeat seed calibration and add translation and complete-item permutation
# controls to this compressed case. Sweep group sizes, simultaneous targets,
# and summary width before generalizing its capacity. A smaller representation
# alone does not establish a speed or memory improvement.
#
# ### Complete experiment code
#

# %%
"""Infer grouped deviations using one learned collection summary.

Each repeated target still receives its aligned visible group and value as
query context. The shared summary supplies peer context; this is not a claim
that one vector losslessly preserves every collection. Rotated labels test
whether the model uses the original peer memberships.

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

PROOF_ID = "P033"
ITEMS = 6
GROUPS = ("A", "B", "C")


def records(*, rows: int, seed: int) -> Iterator[dict]:
    """Generate randomly interleaved, exactly centered two-member groups."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        items: list[dict[str, object]] = []
        for group in GROUPS:
            mean = float(rng.uniform(-3.5, 3.5))
            deviation = float(rng.uniform(0.2, 1.2))
            for signed in (-deviation, deviation):
                item: dict[str, object] = {"group": group, "value": mean + signed, "deviation": signed}
                items.append(item)
        order = rng.permutation(ITEMS)
        yield {"items": [items[index] for index in order]}


def targets(rows: list[dict]) -> np.ndarray:
    return np.asarray([item["deviation"] for row in rows for item in row["items"]])


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    inputs = [
        {"items": [{key: value for key, value in item.items() if key != "deviation"} for item in row["items"]]}
        for row in rows
    ]
    output = model.predict(inputs).to_pylist()
    return np.asarray([item["deviation"]["content"] for row in output for item in row["predictions"]["items"]])


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def corrupt_groups(rows: list[dict]) -> list[dict]:
    """Rotate only labels, preserving all values, targets, and label counts."""
    result = []
    for row in rows:
        labels = [item["group"] for item in row["items"]]
        shifted = labels[1:] + labels[:1]
        result.append({"items": [{**item, "group": label} for item, label in zip(row["items"], shifted, strict=True)]})
    return result


def implied_deviation(rows: list[dict]) -> np.ndarray:
    """Measure how strongly changed peer memberships alter the correct answer."""
    values = []
    for row in rows:
        means = {
            group: float(np.mean([item["value"] for item in row["items"] if item["group"] == group]))
            for group in GROUPS
        }
        values.extend(item["value"] - means[item["group"]] for item in row["items"])
    return np.asarray(values)


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=48,
        n_layers=3,
        n_heads=4,
        reduction=None,
        items=rf.Branch(
            length=ITEMS,
            overflow="error",
            n_layers=2,
            reduction=rf.Attention(n_outputs=1, n_layers=2),
            value=rf.Number,
            group=rf.Category(p_unavailable=0.0),
            deviation=rf.Number(mask=True, objective="mse", n_linear=2),
        ),
    )
    data = rf.SyntheticDataModule(
        train=partial(records, rows=2048, seed=seed + 1),
        validate=partial(records, rows=512, seed=seed + 2),
        seed=seed,
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
    train = list(records(rows=2048, seed=seed + 1))
    test = list(records(rows=768, seed=seed + 3))
    actual = targets(test)
    baseline = rmse(actual, float(targets(train).mean()))
    predicted = predict(model, test)
    intact_error = rmse(actual, predicted)
    intact_nrmse = intact_error / baseline
    metrics = {"intact_rmse": intact_error, "baseline_rmse": baseline, "intact_nrmse": intact_nrmse}
    checks = {"Baseline RMSE > 0.000001": baseline > 1e-6}
    corrupted = corrupt_groups(test)
    corrupted_error = rmse(actual, predict(model, corrupted))
    corrupted_nrmse = corrupted_error / baseline
    oracle = rmse(implied_deviation(corrupted), actual) / baseline
    metrics.update(corrupted_rmse=corrupted_error, corrupted_nrmse=corrupted_nrmse, oracle_corruption_nrmse=oracle)
    checks["Label corruption changes the oracle by > 0.75 nRMSE"] = oracle > 0.75
    checks.update(
        {
            "Single-summary grouped nRMSE <= 0.20": intact_nrmse <= 0.20,
            "Corrupted labels nRMSE >= 0.90": corrupted_nrmse >= 0.90,
            "Label corruption increases nRMSE by >= 0.75": corrupted_nrmse >= intact_nrmse + 0.75,
        }
    )
    checks["Finite calibration and model metrics"] = bool(np.isfinite(list(metrics.values())).all())
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3720)

# %% [markdown]
# </details>
