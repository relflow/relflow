# %% [markdown]
# ---
# title: Do harmless schema edits preserve predictions?
# categories:
# - Schema mutation
# evidence-id: P046
# description: Train a useful predictor, then make edits that should not alter its visible information or learned
#   state.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Trained model
#   metric:
#   - source
#   - nrmse
#   format: error
# - label: Completely reset
#   metric:
#   - controls
#   - complete_reset
#   - nrmse
#   format: error
# evidence-group: Mutations
# evidence-result:
#   accelerator: cpu
#   checks:
#     Complete reset loses the learned relationship: true
#     Duplicate-name edit is rejected: true
#     Edited checkpoint preserves learned state: true
#     Edited checkpoint preserves predictions: true
#     Edited checkpoint preserves schema: true
#     Source learns the relationship below 0.25 nRMSE: true
#     'cycle 1 inactive deletion: all original state and vocabulary survive': true
#     'cycle 1 inactive deletion: predictions survive': true
#     'cycle 1 inactive extension: all original state and vocabulary survive': true
#     'cycle 1 inactive extension: predictions survive': true
#     'cycle 1 metadata: all original state and vocabulary survive': true
#     'cycle 1 metadata: predictions survive': true
#     'cycle 2 inactive deletion: all original state and vocabulary survive': true
#     'cycle 2 inactive deletion: predictions survive': true
#     'cycle 2 inactive extension: all original state and vocabulary survive': true
#     'cycle 2 inactive extension: predictions survive': true
#     'cycle 2 metadata: all original state and vocabulary survive': true
#     'cycle 2 metadata: predictions survive': true
#     'cycle 3 inactive deletion: all original state and vocabulary survive': true
#     'cycle 3 inactive deletion: predictions survive': true
#     'cycle 3 inactive extension: all original state and vocabulary survive': true
#     'cycle 3 inactive extension: predictions survive': true
#     'cycle 3 metadata: all original state and vocabulary survive': true
#     'cycle 3 metadata: predictions survive': true
#     'equivalent source rebind: all original state and vocabulary survive': true
#     'equivalent source rebind: predictions survive': true
#     'rejected duplicate extension: all original state and vocabulary survive': true
#     'rejected duplicate extension: predictions survive': true
#     'source rebind restoration: all original state and vocabulary survive': true
#     'source rebind restoration: predictions survive': true
#   duration_seconds: 16.397
#   metrics:
#     controls:
#       complete_reset:
#         baseline_rmse: 0.8103651362027426
#         nrmse: 0.9959644305458001
#         rmse: 0.8070948514123343
#     downstream_interpretable: true
#     edits:
#     - changed_or_missing_state_entries: []
#       edit: cycle 1 metadata
#       max_prediction_drift: 0.0
#       nrmse: 0.02379013795466527
#       vocabulary_preserved: true
#     - changed_or_missing_state_entries: []
#       edit: cycle 1 inactive extension
#       max_prediction_drift: 0.0
#       nrmse: 0.02379013795466527
#       vocabulary_preserved: true
#     - changed_or_missing_state_entries: []
#       edit: cycle 1 inactive deletion
#       max_prediction_drift: 0.0
#       nrmse: 0.02379013795466527
#       vocabulary_preserved: true
#     - changed_or_missing_state_entries: []
#       edit: cycle 2 metadata
#       max_prediction_drift: 0.0
#       nrmse: 0.02379013795466527
#       vocabulary_preserved: true
#     - changed_or_missing_state_entries: []
#       edit: cycle 2 inactive extension
#       max_prediction_drift: 0.0
#       nrmse: 0.02379013795466527
#       vocabulary_preserved: true
#     - changed_or_missing_state_entries: []
#       edit: cycle 2 inactive deletion
#       max_prediction_drift: 0.0
#       nrmse: 0.02379013795466527
#       vocabulary_preserved: true
#     - changed_or_missing_state_entries: []
#       edit: cycle 3 metadata
#       max_prediction_drift: 0.0
#       nrmse: 0.02379013795466527
#       vocabulary_preserved: true
#     - changed_or_missing_state_entries: []
#       edit: cycle 3 inactive extension
#       max_prediction_drift: 0.0
#       nrmse: 0.02379013795466527
#       vocabulary_preserved: true
#     - changed_or_missing_state_entries: []
#       edit: cycle 3 inactive deletion
#       max_prediction_drift: 0.0
#       nrmse: 0.02379013795466527
#       vocabulary_preserved: true
#     - changed_or_missing_state_entries: []
#       edit: equivalent source rebind
#       max_prediction_drift: 0.0
#       nrmse: 0.02379013795466527
#       vocabulary_preserved: true
#     - changed_or_missing_state_entries: []
#       edit: source rebind restoration
#       max_prediction_drift: 0.0
#       nrmse: 0.02379013795466527
#       vocabulary_preserved: true
#     - changed_or_missing_state_entries: []
#       edit: rejected duplicate extension
#       max_prediction_drift: 0.0
#       nrmse: 0.02379013795466527
#       vocabulary_preserved: true
#     optimizer_policy: Fresh AdamW for source training; no post-edit fitting
#     source:
#       baseline_rmse: 0.8103651362027426
#       nrmse: 0.02379013795466527
#       rmse: 0.01927869838391436
#     source_prerequisite_met: true
#     source_steps: 512
#     source_vocabulary:
#     - code-1
#     - code-2
#     - code-3
#     - code-0
#     test_rows: 1024
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: a8e8c255181b72039b7147deba8b80d29ff42afa460a9f53c2dd64f36f1a1cf3
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 130482dad006
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
#     script_sha256: 2633992e073008e4553c768cc1ec418d6b3952955021e6a39dbe4916879bea21
#     source_sha256: e85509d12302e9edffe27b9cd14399a230d47d3503b40cd58dc7c8e6c123c504
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 4601
#   started_at: '2026-11-01T01:46:27.157259+00:00'
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
# x: 0.5
# code: code-0
# y: -0.25
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-mutations-neutral-edits
# //| fig-alt: "Model tree with record, x, code, y, unused. Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number", body: [Source key becomes *renamed_x*]),
#   node("code", type: "Category"),
#   node("y", type: "Number", kind: "target", body: [Always hidden]),
#   node("unused", type: "Number", body: [Added inactive, then deleted]),
# )))
# ```
#
# ## Comparison
#
# Repeat metadata changes, inactive-field additions and deletions, equivalent input-key changes, and save/load. Compare with a complete reset.
#
# ## Result
#
# {{< evidence P046 status >}}
#
# These edits preserve predictions; a reset loses the learned relationship. Changes to visible context or model size need separate checks.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P046 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P046 script >}}
#
# ### Remaining work
#
# Three GPU seeds support these edits. Calibrate on ten separate seeds before
# freezing numerical gates. Test inactive nested branches separately.
# Renaming schema addresses and resizing learned tensors require separate
# experiments; matching physical input values does not make those neutral edits.
#
# ### Complete experiment code
#

# %%
"""P046: preserve a trained function across neutral public schema edits."""

from collections.abc import Iterator
from copy import deepcopy
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P046"
OFFSETS = (-0.75, -0.25, 0.25, 0.75)


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        x = float(rng.uniform(-1, 1))
        code = int(rng.integers(len(OFFSETS)))
        yield {"x": x, "code": f"code-{code}", "y": x + OFFSETS[code]}


def prediction(model: rf.Model, rows: list[dict], *, key: str = "x") -> np.ndarray:
    output = model.predict([{key: row["x"], "code": row["code"]} for row in rows])["predictions"].to_pylist()
    return np.asarray([row["y"]["content"] for row in output], dtype=np.float64)


def equal(first, second) -> bool:
    """Compare learned tensors and extension-owned nested state exactly."""
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return (
            isinstance(second, dict)
            and first.keys() == second.keys()
            and all(equal(value, second[name]) for name, value in first.items())
        )
    if isinstance(first, (tuple, list)):
        return (
            type(first) is type(second)
            and len(first) == len(second)
            and all(equal(a, b) for a, b in zip(first, second, strict=True))
        )
    return type(first) is type(second) and first == second


def errors(actual: np.ndarray, predicted: np.ndarray, baseline: float) -> dict:
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    return {"rmse": rmse, "baseline_rmse": baseline, "nrmse": rmse / baseline}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    budget = 512 if steps is None else min(steps, 512)
    train = list(records(rows=2048, seed=seed + 1))
    test = list(records(rows=1024, seed=seed + 3))
    actual = np.asarray([row["y"] for row in test])
    baseline = float(np.sqrt(np.mean((actual - np.mean([row["y"] for row in train])) ** 2)))
    scale = float(np.std([row["y"] for row in train]))
    model = rf.Model.xs(
        x=rf.Number,
        code=rf.Category(p_unavailable=0.0),
        y=rf.Number(mask=True, objective="mse"),
    )
    data = rf.SyntheticDataModule(
        train=partial(records, rows=2048, seed=seed + 1),
        validate=partial(records, rows=512, seed=seed + 2),
        seed=seed,
    )
    model.fit(
        data,
        batch_size=128,
        learning_rate=0.003,
        fused=False,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=budget,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer
    model.eval()
    reference = prediction(model, test)
    state = deepcopy(model.state_dict())
    vocabulary = list(rf.Category.vocabulary(model, rf.Address("code")))
    root = rf.where("address") == "/"
    source = rf.where("address") == "/x"
    measured = errors(actual, reference, baseline)
    edits = []
    checks = {"Source learns the relationship below 0.25 nRMSE": measured["nrmse"] < 0.25}

    def observe(label: str, *, key: str = "x") -> None:
        result = prediction(model, test, key=key)
        current = model.state_dict()
        missing = [name for name, value in state.items() if name not in current or not equal(value, current[name])]
        preserved = bool(np.allclose(reference, result, rtol=1e-5, atol=1e-6 * scale))
        same_vocabulary = list(rf.Category.vocabulary(model, rf.Address("code"))) == vocabulary
        edits.append(
            {
                "edit": label,
                "max_prediction_drift": float(np.max(np.abs(reference - result))),
                "nrmse": errors(actual, result, baseline)["nrmse"],
                "changed_or_missing_state_entries": missing,
                "vocabulary_preserved": same_vocabulary,
            }
        )
        checks[f"{label}: predictions survive"] = preserved
        checks[f"{label}: all original state and vocabulary survive"] = not missing and same_vocabulary

    for cycle in range(3):
        model.update(source, description=f"Equivalent input, cycle {cycle + 1}")
        observe(f"cycle {cycle + 1} metadata")
        model.extend(root, unused=rf.Number(active=False))
        observe(f"cycle {cycle + 1} inactive extension")
        model.delete(rf.where("address") == "/unused")
        observe(f"cycle {cycle + 1} inactive deletion")
    model.update(source, query="renamed_x")
    observe("equivalent source rebind", key="renamed_x")
    model.update(source, query=None)
    observe("source rebind restoration")
    rejected = False
    try:
        model.extend(root, x=rf.Number)
    except ValueError:
        rejected = True
    checks["Duplicate-name edit is rejected"] = rejected
    observe("rejected duplicate extension")

    with TemporaryDirectory(prefix="relflow-mutation-") as directory:
        path = Path(directory) / "edited.ckpt"
        model.save(path)
        loaded = rf.Model.load(path).to(model.device).eval()
        restored = prediction(loaded, test)
        checks["Edited checkpoint preserves schema"] = loaded.schema.model_dump() == model.schema.model_dump()
        checks["Edited checkpoint preserves learned state"] = equal(model.state_dict(), loaded.state_dict())
        checks["Edited checkpoint preserves predictions"] = bool(
            np.allclose(reference, restored, rtol=1e-5, atol=1e-6 * scale)
        )
        lit.seed_everything(seed + 100, workers=True)
        loaded.reset(root, descendants=True)
        reset = errors(actual, prediction(loaded.eval(), test), baseline)
    checks["Complete reset loses the learned relationship"] = (
        reset["nrmse"] > 0.8 and reset["nrmse"] > 3 * measured["nrmse"]
    )
    return {
        "source": measured,
        "source_prerequisite_met": measured["nrmse"] < 0.25,
        "downstream_interpretable": measured["nrmse"] < 0.25,
        "source_steps": trainer.global_step,
        "edits": edits,
        "controls": {"complete_reset": reset},
        "source_vocabulary": vocabulary,
        "test_rows": len(test),
        "optimizer_policy": "Fresh AdamW for source training; no post-edit fitting",
    }, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=4601)

# %% [markdown]
# </details>
