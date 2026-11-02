# %% [markdown]
# ---
# title: Does admitting a label teach it to predict that label?
# categories:
# - Vocabulary
# evidence-id: P059
# description: Introduce new output labels to a model trained on an earlier pair of classes.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: New labels, just admitted
#   metric:
#   - new_accuracy_immediately_after_admission
#   format: percent
# - label: New labels, after fitting
#   metric:
#   - final_new_accuracy
#   format: percent
# - label: Old labels, after fitting
#   metric:
#   - final_old_accuracy
#   format: percent
# evidence-group: Vocabulary
# evidence-result:
#   accelerator: cpu
#   checks:
#     Admission alone preserves existing learned parameter rows: true
#     Cold prediction leaves the vocabulary empty: true
#     Continued checkpoint preserves mapping, counts, and predictions: true
#     Continued fitting learns new labels above 0.95 accuracy: true
#     Continued fitting preserves all admitted indices: true
#     Empty output vocabulary returns no label and zero probability: true
#     Evaluation cannot admit new output labels: true
#     Evaluation leaves exposure counts and numerical moments frozen: true
#     Initial checkpoint preserves mapping, counts, and predictions: true
#     Initial known-label accuracy reaches 0.95: true
#     Rehearsal retains old labels above 0.95 accuracy: true
#     Training admission appends both labels without moving old indices: true
#     Unadmitted labels cannot be predicted: true
#   duration_seconds: 14.85
#   metrics:
#     continuation_steps: 400
#     final_new_accuracy: 1.0
#     final_old_accuracy: 1.0
#     final_vocabulary:
#     - positive
#     - negative
#     - far-positive
#     - far-negative
#     initial_accuracy: 1.0
#     initial_steps: 300
#     initial_vocabulary:
#     - positive
#     - negative
#     new_accuracy_before_admission: 0.0
#     new_accuracy_immediately_after_admission: 0.0
#     two_class_chance: 0.5
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 1fc9339c3fb4d36dc96fc723987494ba61217106bc246aac4a90df6125655d79
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: d5cd716baa24
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
#     script_sha256: 2d7cfc5a5de2da3e6f8527ea4d32c9c89fa0e6e9e6e1684cab8f8d8b8ff043ae
#     source_sha256: 9ba161d7c59d8d2688c5516f97c752ae5b2ae3c439124df048b782bf51489eef
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 5901
#   started_at: '2026-11-01T01:47:53.990201+00:00'
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
# x: -0.9
# label: negative
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-vocabulary-admission-training
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with record, x, label. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number", body: [Four separated numerical regions]),
#   node("label", kind: "target", type: "Category", body: [Two labels, growing to four]),
# )))
# ```
#
# ## Comparison
#
# Measure before admission, immediately after admission without training, and after continued fitting. Rehearse the old classes too.
#
# ## Result
#
# {{< evidence P059 status >}}
#
# Admission creates storage; further training teaches the new distinctions. Old-label retention here depends on continued examples of the old classes.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P059 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P059 script >}}
#
# ### Data and model
#
# Initially, x lies within 0.15 of -1 or +1 and predicts negative or positive.
# New classes lie within 0.15 of -3 or +3. Each split balances its classes and
# shuffles them with an independent local RNG. Each fitting phase uses 4,096
# training and 512 validation rows; the old and new tests each use 1,024 rows.
# Storage initially learns two output labels, then grows to hold four.
#
# ### Training and controls
#
# The model uses the [xs preset](../../concepts/model-tree.qmd#choose-a-size).
#
# Use 300 initial and 400 continuation AdamW updates at learning rate 0.002,
# one device, and no loader workers. Require 0.95 accuracy on the initial task,
# then on both old and new held-out classes after continued fitting. Chance is
# 0.5 on each two-class test. Before admission, new-class accuracy is necessarily
# zero. Validation, test, and prediction must not admit labels or change counts
# or numerical moments. Explicit train encoding must append labels without
# changing existing parameter rows; new rows, normalization, and count buffers may change.
# Save/load must preserve labels, counts, and predictions in both phases.
# Immediate accuracy after admission is diagnostic, not an acceptance gate.
#
# ### Complete experiment code
#

# %%
"""P059: vocabulary persistence and new-label admission followed by measured learning."""

from collections.abc import Iterator
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P059"
BUDGET = 300
CONTINUATION = 400
LABELS = ("negative", "positive", "far-negative", "far-positive")
CENTERS = (-1.0, 1.0, -3.0, 3.0)


def records(*, rows: int, seed: int, phase: str = "initial") -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    indices = {"initial": (0, 1), "new": (2, 3), "mixed": (0, 1, 2, 3)}[phase]
    classes = np.resize(np.asarray(indices), rows)
    rng.shuffle(classes)
    noise = rng.uniform(-0.15, 0.15, rows)
    for index, deviation in zip(classes, noise, strict=True):
        yield {"x": float(CENTERS[index] + deviation), "label": LABELS[index]}


def build() -> rf.Model:
    return rf.Model.xs(
        x=rf.Number,
        label=rf.Category(p_unavailable=0.0, topk=[3], mask=True),
    )


def fit(model: rf.Model, phase: str, seed: int, budget: int, accelerator: str) -> int:
    data = rf.SyntheticDataModule(
        train=partial(records, rows=4096, seed=seed + 1, phase=phase),
        validate=partial(records, rows=512, seed=seed + 2, phase=phase),
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
        max_steps=budget,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer
    model.eval()
    return trainer.global_step


def prediction(model: rf.Model, rows: list[dict]) -> list[dict]:
    output = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    return [row["label"]["content"] for row in output]


def accuracy(output: list[dict], rows: list[dict]) -> float:
    return float(np.mean([value["value"] == row["label"] for value, row in zip(output, rows, strict=True)]))


def roundtrip(model: rf.Model, rows: list[dict], path: Path) -> tuple[rf.Model, bool]:
    before = prediction(model, rows)
    vocabulary = rf.Category.vocabulary(model, "/label")
    counts = rf.Category.counts(model, "/label")
    model.save(path)
    loaded = rf.Model.load(path).to(model.device)
    loaded.eval()
    after = prediction(loaded, rows)
    matches = rf.Category.vocabulary(loaded, "/label") == vocabulary
    matches &= rf.Category.counts(loaded, "/label") == counts
    matches &= all(
        a["value"] == b["value"] and abs(a["probability"] - b["probability"]) < 1e-5
        for a, b in zip(before, after, strict=True)
    )
    return loaded, matches


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    old = list(records(rows=1024, seed=seed + 3))
    new = list(records(rows=1024, seed=seed + 4, phase="new"))
    cold = prediction(model, new)
    checks = {
        "Cold prediction leaves the vocabulary empty": rf.Category.vocabulary(model, "/label") == (),
        "Empty output vocabulary returns no label and zero probability": all(
            row["value"] is None and row["probability"] == 0.0 for row in cold
        ),
    }
    initial_steps = fit(model, "initial", seed, BUDGET if steps is None else min(steps, BUDGET), accelerator)
    initial_accuracy = accuracy(prediction(model, old), old)
    vocabulary = rf.Category.vocabulary(model, "/label")
    counts = rf.Category.counts(model, "/label")
    normalization = rf.Number.normalization(model, "/x")
    checks["Initial known-label accuracy reaches 0.95"] = initial_accuracy >= 0.95
    with TemporaryDirectory(prefix="relflow-vocabulary-") as directory:
        model, matches = roundtrip(model, old, Path(directory) / "initial.pt")
        checks["Initial checkpoint preserves mapping, counts, and predictions"] = matches
        before_admission = accuracy(prediction(model, new), new)
        source = pa.Table.from_pylist(new)
        for strata in ("validate", "test", "predict"):
            model.encode(source, strata=strata)
        checks["Evaluation cannot admit new output labels"] = rf.Category.vocabulary(model, "/label") == vocabulary
        checks["Evaluation leaves exposure counts and numerical moments frozen"] = (
            rf.Category.counts(model, "/label") == counts and rf.Number.normalization(model, "/x") == normalization
        )
        checks["Unadmitted labels cannot be predicted"] = before_admission == 0.0
        parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
        admission = pa.Table.from_pylist(list(records(rows=32, seed=seed + 10, phase="new")))
        model.encode(admission, strata="train")
        admitted = rf.Category.vocabulary(model, "/label")
        checks["Training admission appends both labels without moving old indices"] = (
            admitted[:2] == vocabulary and len(admitted) == 4 and set(admitted) == set(LABELS)
        )
        checks["Admission alone preserves existing learned parameter rows"] = all(
            torch.equal(parameters[name], value[: parameters[name].shape[0]])
            for name, value in model.named_parameters()
        )
        after_admission = accuracy(prediction(model, new), new)
        continuation_steps = fit(
            model, "mixed", seed + 20, CONTINUATION if steps is None else min(steps, CONTINUATION), accelerator
        )
        final_old = accuracy(prediction(model, old), old)
        final_new = accuracy(prediction(model, new), new)
        checks["Continued fitting learns new labels above 0.95 accuracy"] = final_new >= 0.95
        checks["Rehearsal retains old labels above 0.95 accuracy"] = final_old >= 0.95
        checks["Continued fitting preserves all admitted indices"] = rf.Category.vocabulary(model, "/label") == admitted
        model, matches = roundtrip(model, old + new, Path(directory) / "continued.pt")
        checks["Continued checkpoint preserves mapping, counts, and predictions"] = matches
    return {
        "initial_steps": initial_steps,
        "continuation_steps": continuation_steps,
        "initial_vocabulary": vocabulary,
        "final_vocabulary": admitted,
        "initial_accuracy": initial_accuracy,
        "new_accuracy_before_admission": before_admission,
        "new_accuracy_immediately_after_admission": after_admission,
        "final_old_accuracy": final_old,
        "final_new_accuracy": final_new,
        "two_class_chance": 0.5,
    }, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=5901)

# %% [markdown]
# </details>
