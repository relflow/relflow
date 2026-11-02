# %% [markdown]
# ---
# title: Can it recover when a whole branch is hidden?
# categories:
# - Masking
# evidence-id: P053
# description: Provide two redundant measurements of the same signal and sometimes hide an entire branch.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: First view hidden
#   metric:
#   - patterns
#   - hiddenvisible
#   - nrmse
#   format: error
# - label: Second view hidden
#   metric:
#   - patterns
#   - visiblehidden
#   - nrmse
#   format: error
# - label: Both views hidden
#   metric:
#   - patterns
#   - hiddenhidden
#   - nrmse
#   format: error
# evidence-group: Masking
# evidence-result:
#   accelerator: cpu
#   checks:
#     Shuffling available evidence removes accuracy: true
#     'backup: ancestor and local skips compose': true
#     'hiddenhidden: expected information boundary': true
#     'hiddenhidden: skipped values cannot affect predictions': true
#     'hiddenvisible: expected information boundary': true
#     'hiddenvisible: skipped values cannot affect predictions': true
#     'reading: ancestor and local skips compose': true
#     'visiblehidden: expected information boundary': true
#     'visiblehidden: skipped values cannot affect predictions': true
#     'visiblevisible: expected information boundary': true
#     'visiblevisible: skipped values cannot affect predictions': true
#   duration_seconds: 15.929
#   metrics:
#     baseline_rmse: 0.8548844813618697
#     patterns:
#       hiddenhidden:
#         hidden_value_drift: 0.0
#         nrmse: 1.0020721030893713
#       hiddenvisible:
#         hidden_value_drift: 0.0
#         nrmse: 0.019111931082546274
#       visiblehidden:
#         hidden_value_drift: 0.0
#         nrmse: 0.019112357378374644
#       visiblevisible:
#         hidden_value_drift: 0.0
#         nrmse: 0.01907755794178149
#     shuffled_nrmse: 1.4018123645812262
#     steps: 500
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: c2471d582c3ff05c20efb49da64a96a9c7da0d4d5e3ca213393cc4867cc7826b
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 32b92f63175d
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
#     script_sha256: f3c91ed4c4547f3343ce72868a84a53bc7e7202035131befc3813b1245cf4dc5
#     source_sha256: d496097f52c093daddff66d3b69f49084cc4a1211322e396d8e3fc8fa3d31f3b
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 5301
#   started_at: '2026-11-01T01:47:19.591856+00:00'
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
# views:
#   - {reading: 0.4, backup: 0.9, hidden: true, hide_backup: false}
#   - {reading: 0.4, backup: 0.9, hidden: false, hide_backup: true}
# target: 0.4
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-branch-ablation
# //| fig-cap: "Amber cards are hidden prediction targets. hidden skips a whole view and its descendants."
# //| fig-alt: "Model tree with record, views, reading, backup, target. Amber cards are hidden prediction targets. hidden skips a whole view and its descendants."
# #tree(node("record", kind: "root", children: (
#   node("views", kind: "branch", repeated: true, body: [Skip when `hidden`], children: (
#     node("reading", type: "Number"),
#     node("backup", type: "Number", body: [Also skip when `hide_backup`]),
#   )),
#   node("target", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Test each branch alone, both visible, and both hidden. Change hidden descendant values to check they cannot affect the answer.
#
# ## Result
#
# {{< evidence P053 status >}}
#
# Either view supports prediction. When both are hidden, the model falls back near the constant baseline; the test shows redundancy, rather than causal attribution.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P053 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P053 script >}}
#
# ### Data and model
#
# Each independent record draws a latent `z ~ Uniform(-1, 1)`. Both views expose
# `reading = z` and `backup = 2*z + 0.1`; the hidden target is `1.5*z - 0.2`.
# The latent variable is never a separate input. Independent branch selectors
# produce all four visibility patterns, independently of `z`. Leaf selectors
# also hide each backup with probability 0.5. Train/validation/test use
# 4,096/512/2,048 independent records.
#
# ### Training and controls
#
# The model uses the [xs preset](../../concepts/model-tree.qmd#choose-a-size).
#
# Fit for 500 AdamW updates at learning rate 0.002. On the same test observations,
# set each of the four branch-mask patterns deterministically. At least one
# visible view must give nRMSE below 0.30; no visible views must give nRMSE above
# 0.85. Poison all skipped descendants and the supervised target: prediction
# drift must stay below 1e-5. Permuting visible views between records must remove
# accuracy. The encoded presence bits must equal the union of ancestor and
# local skip selectors. Unselected backups are explicitly left visible when
# their branch is available.
#
# ### Complete experiment code
#

# %%
"""P053: redundant-view learning under composed branch and leaf skip policies."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P053"
BUDGET = 500


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        value = float(rng.uniform(-1.0, 1.0))
        yield {
            "views": [
                {
                    "reading": value,
                    "backup": 2 * value + 0.1,
                    "hidden": bool(rng.random() < 0.5),
                    "hide_backup": bool(rng.random() < 0.5),
                }
                for _ in range(2)
            ],
            "target": 1.5 * value - 0.2,
        }


def build() -> rf.Model:
    return rf.Model.xs(
        views=rf.Branch(
            length=2,
            mask=rf.Mask(query="hidden", skip=True, dropout=False),
            reading=rf.Number,
            backup=rf.Number(mask=rf.Mask(query="hide_backup", skip=True, dropout=False)),
        ),
        target=rf.Number(mask=True, objective="mse"),
    )


def prediction(model: rf.Model, rows: list[dict]) -> np.ndarray:
    output = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    return np.asarray([row["target"]["content"] for row in output], dtype=np.float64)


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    data = rf.SyntheticDataModule(
        train=partial(records, rows=4096, seed=seed + 1),
        validate=partial(records, rows=512, seed=seed + 2),
        seed=seed,
    )
    model.fit(
        data,
        batch_size=64,
        learning_rate=0.002,
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
    trainer = model.trainer
    model.eval()
    train = list(records(rows=4096, seed=seed + 1))
    test = list(records(rows=2048, seed=seed + 3))
    actual = np.asarray([row["target"] for row in test])
    baseline = float(np.sqrt(np.mean((actual - np.mean([row["target"] for row in train])) ** 2)))
    metrics = {"steps": trainer.global_step, "baseline_rmse": baseline, "patterns": {}}
    checks = {}
    for pattern in ((False, False), (True, False), (False, True), (True, True)):
        name = "".join("hidden" if hidden else "visible" for hidden in pattern)
        selected = [
            {
                **row,
                "views": [
                    {**view, "hidden": hidden, "hide_backup": False}
                    for view, hidden in zip(row["views"], pattern, strict=True)
                ],
            }
            for row in test
        ]
        predicted = prediction(model, selected)
        poisoned = [
            {
                "target": 1000.0,
                "views": [
                    {**view, "reading": 1000.0, "backup": -1000.0} if view["hidden"] else view for view in row["views"]
                ],
            }
            for row in selected
        ]
        measured = {
            "nrmse": float(np.sqrt(np.mean((predicted - actual) ** 2))) / baseline,
            "hidden_value_drift": float(np.max(np.abs(predicted - prediction(model, poisoned)))),
        }
        metrics["patterns"][name] = measured
        checks[f"{name}: expected information boundary"] = (
            measured["nrmse"] > 0.85 if all(pattern) else measured["nrmse"] < 0.30
        )
        checks[f"{name}: skipped values cannot affect predictions"] = measured["hidden_value_drift"] < 1e-5
    order = np.random.default_rng(seed + 4).permutation(len(test))
    shuffled = [
        {
            "target": row["target"],
            "views": [{**view, "hidden": False, "hide_backup": False} for view in test[index]["views"]],
        }
        for row, index in zip(test, order, strict=True)
    ]
    metrics["shuffled_nrmse"] = float(np.sqrt(np.mean((prediction(model, shuffled) - actual) ** 2))) / baseline
    checks["Shuffling available evidence removes accuracy"] = metrics["shuffled_nrmse"] > 0.85
    fields = model.encode(pa.Table.from_pylist(test), strata="predict")
    for name in ("reading", "backup"):
        expected = np.asarray(
            [
                [not (view["hidden"] or (name == "backup" and view["hide_backup"])) for view in row["views"]]
                for row in test
            ]
        )
        present = fields[f"/views/{name}"].present.cpu().numpy().reshape(expected.shape)
        checks[f"{name}: ancestor and local skips compose"] = bool(np.array_equal(present, expected))
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=5301)

# %% [markdown]
# </details>
