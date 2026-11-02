# %% [markdown]
# ---
# title: Does declaring a class teach its meaning?
# categories:
# - Enum
# evidence-id: P070
# description: Declare all four classes, but leave one out of initial training.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Withheld class, before examples
#   metric:
#   - initial_withheld
#   - accuracy
#   format: percent
# - label: Withheld class, after examples
#   metric:
#   - final_withheld
#   - accuracy
#   format: percent
# - label: Old classes, after continuation
#   metric:
#   - final_seen
#   - accuracy
#   format: percent
# evidence-group: Representation
# evidence-result:
#   accelerator: cpu
#   checks:
#     All-classes-trained control reaches 0.95 on both populations: true
#     Checkpoint preserves fixed class storage and exposures: true
#     Checkpoint preserves withheld predictions: true
#     Cold prediction exposes every declared candidate: true
#     Cold prediction leaves exposure counts at zero: true
#     Continuation adds exposure to every declared class: true
#     Continuation leaves class IDs and all row counts unchanged: true
#     Continuation reaches 0.95 withheld-class accuracy: true
#     Final evaluation preserves counts: true
#     Initial training reaches 0.95 seen-class accuracy: true
#     Initial training retains the complete fixed declaration: true
#     Prediction of withheld records cannot teach the class: true
#     Rehearsal retains 0.95 seen-class accuracy: true
#     Untrained class retains output support: true
#     Withheld class has zero initial training exposure: true
#     Written class probabilities sum to one: true
#   duration_seconds: 20.551
#   metrics:
#     cold_withheld:
#       accuracy: 0.0
#       full_declared_support: true
#       mean_withheld_probability: 0.19286564979847753
#       probability_sum_error: 1.4901161193847656e-07
#     continuation_steps: 300
#     control_seen:
#       accuracy: 1.0
#       full_declared_support: true
#       mean_withheld_probability: 3.608925033038446e-06
#       probability_sum_error: 1.577382136019878e-07
#     control_steps: 300
#     control_withheld:
#       accuracy: 1.0
#       full_declared_support: true
#       mean_withheld_probability: 0.9999877552618273
#       probability_sum_error: 5.968740879325196e-08
#     final_counts:
#       amber: 11227
#       blue: 11179
#       coral: 11212
#       green: 4782
#     final_seen:
#       accuracy: 1.0
#       full_declared_support: true
#       mean_withheld_probability: 1.4679822186863811e-06
#       probability_sum_error: 1.638068454212771e-07
#     final_withheld:
#       accuracy: 1.0
#       full_declared_support: true
#       mean_withheld_probability: 0.999990523676388
#       probability_sum_error: 5.964022875559749e-08
#     initial_counts:
#       amber: 6431
#       blue: 6376
#       coral: 6393
#       green: 0
#     initial_seen:
#       accuracy: 1.0
#       full_declared_support: true
#       mean_withheld_probability: 1.0260045238408289e-06
#       probability_sum_error: 1.5371654171758564e-07
#     initial_seen_majority_accuracy: 0.33349609375
#     initial_steps: 300
#     initial_withheld:
#       accuracy: 0.0
#       full_declared_support: true
#       mean_withheld_probability: 5.314755078533739e-07
#       probability_sum_error: 1.1784038633777527e-07
#     storage:
#       count_rows: 4
#       decoder_rows: 4
#       embedding_rows: 4
#       labels:
#       - amber
#       - blue
#       - coral
#       - green
#     test_rows_per_population: 2048
#     train_rows_per_stage: 4096
#     uniform_four_class_accuracy: 0.25
#     validation_rows_per_stage: 512
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 7043cf182648b5d0bb8831df62a2ebb6067fb6f5caea76dd8ad618da6c2787ab
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 6b485b4ade90
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
#     script_sha256: ea753661af589ceb96101d1722bb63bbd2a67aa8b3e0a155081913d0d00a7dec
#     source_sha256: fd8dad6670536f33140f00b26bcd5a3fe37cb497246da2dddc1cd9303f3eecb3
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 7001
#   started_at: '2026-11-01T01:48:56.335146+00:00'
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
# x: -0.94
# label: blue
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-enum-declared-without-examples
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with record, x, label. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number", body: [Four separated numerical regions]),
#   node("label", kind: "target", type: "Enum", body: [Four fixed classes; one initially withheld]),
# )))
# ```
#
# ## Comparison
#
# Measure the withheld class before and after continued fitting that includes it. Continue supplying old classes to check retention.
#
# ## Result
#
# {{< evidence P070 status >}}
#
# A declared output slot alone supplies no reliable understanding. Examples teach the withheld class; the old classes are retained with rehearsal.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P070 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P070 script >}}
#
# ### Data and observability
#
# A class index selects one of four separated numerical centers, and uniform
# noise in `[-0.15, 0.15]` prevents exact repeated numerical observations. The
# target is that region's opaque label. The initial stage includes the first
# three classes; continuation rehearses them equally alongside the fourth.
# Each phase uses 4,096 training and 512 validation rows with independent seeds.
# Seen-class and withheld-class tests each contain 2,048 fresh observations.
# Rows are the split unit. The finite, separated regions make the Bayes error
# zero; no irreducible target noise is added to this controlled omission task.
#
# Continued fitting also rehearses this familiar region. Accuracy on both
# familiar and withheld regions guards against simply always predicting green.
#
# ### Training and controls
#
# The `xs` model receives 300 initial updates and 300 continuation updates,
# with a fresh AdamW optimizer at learning rate 0.002 for each fitting stage,
# batch size 64, one device, and no loader workers. A paired model starts from
# the same seed and sees all four classes during its own 300-update phase.
# This control checks that the withheld region is learnable within one stage.
# It matches the phase budget, rather than the cumulative 600-update budget.
# No checkpoint or hyperparameter is selected using the test sets.
#
# Uniform guessing across the four-class head scores 0.25 on either population;
# the three-class initial task also has a 1/3 majority baseline. A constant
# prediction of the withheld label would score 1.0 on its isolated population
# and 0.0 on the seen population, so both accuracies are essential gates.
#
# Predeclared gates require at least 0.95 initial seen accuracy, at least 0.95
# on both populations after continuation, and at least 0.95 for both populations
# in the all-classes-trained control. The initial withheld exposure must be zero;
# continuation must make it positive. Head size, count-buffer size, and declaration
# order remain fixed. Pre-continuation withheld accuracy has no acceptance gate.
#
# ### Complete experiment code
#

# %%
"""P070: separate a declared Enum class from learning its relationship to context."""

from collections.abc import Iterator
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P070"
BUDGET = 300
LABELS = ("amber", "blue", "coral", "green")
CENTERS = (-3.0, -1.0, 1.0, 3.0)


def records(*, rows: int, seed: int, phase: str = "initial") -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    indices = {"initial": (0, 1, 2), "withheld": (3,), "mixed": (0, 1, 2, 3)}[phase]
    classes = np.resize(np.asarray(indices), rows)
    rng.shuffle(classes)
    noise = rng.uniform(-0.15, 0.15, rows)
    for index, deviation in zip(classes, noise, strict=True):
        yield {"x": float(CENTERS[index] + deviation), "label": LABELS[index]}


def build() -> rf.Model:
    return rf.Model.xs(
        x=rf.Number,
        label=rf.Enum(values=LABELS, p_unavailable=0.0, topk=[4], mask=True),
    )


def storage(model: rf.Model) -> dict:
    """Read the fixed mapping and row counts independently from exposure values."""
    node = model.nodes[rf.Address("label")]
    return {
        "labels": rf.Enum.vocabulary(model, rf.Address("label")),
        "embedding_rows": node.embedder.embeddings["content"].num_embeddings,
        "count_rows": node.embedder.counters["content"].counts.numel(),
        "decoder_rows": node.decoder.linears["content"].out_features,
    }


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


def measure(output: list[dict], rows: list[dict]) -> dict:
    probabilities = [{candidate["value"]: candidate["probability"] for candidate in row["topk"]} for row in output]
    return {
        "accuracy": float(np.mean([value["value"] == row["label"] for value, row in zip(output, rows, strict=True)])),
        "mean_withheld_probability": float(np.mean([row[LABELS[-1]] for row in probabilities])),
        "full_declared_support": all(set(row) == set(LABELS) for row in probabilities),
        "probability_sum_error": float(max(abs(sum(row.values()) - 1.0) for row in probabilities)),
    }


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    budget = BUDGET if steps is None else min(steps, BUDGET)
    model = build()
    initial_storage = storage(model)
    seen = list(records(rows=2048, seed=seed + 3))
    withheld = list(records(rows=2048, seed=seed + 4, phase="withheld"))
    cold = measure(prediction(model, withheld), withheld)
    checks = {
        "Cold prediction exposes every declared candidate": cold["full_declared_support"],
        "Cold prediction leaves exposure counts at zero": all(
            value == 0 for value in rf.Enum.counts(model, "/label").values()
        ),
    }
    initial_steps = fit(model, "initial", seed, budget, accelerator)
    initial_counts = rf.Enum.counts(model, "/label")
    initial_seen = measure(prediction(model, seen), seen)
    initial_withheld = measure(prediction(model, withheld), withheld)
    checks["Initial training reaches 0.95 seen-class accuracy"] = initial_seen["accuracy"] >= 0.95
    checks["Withheld class has zero initial training exposure"] = initial_counts[LABELS[-1]] == 0
    checks["Initial training retains the complete fixed declaration"] = storage(model) == initial_storage
    checks["Prediction of withheld records cannot teach the class"] = rf.Enum.counts(model, "/label") == initial_counts
    checks["Untrained class retains output support"] = initial_withheld["full_declared_support"]
    continuation_steps = fit(model, "mixed", seed + 20, budget, accelerator)
    final_counts = rf.Enum.counts(model, "/label")
    final_seen = measure(prediction(model, seen), seen)
    final_output = prediction(model, withheld)
    final_withheld = measure(final_output, withheld)
    checks["Continuation reaches 0.95 withheld-class accuracy"] = final_withheld["accuracy"] >= 0.95
    checks["Rehearsal retains 0.95 seen-class accuracy"] = final_seen["accuracy"] >= 0.95
    checks["Continuation adds exposure to every declared class"] = all(
        final_counts[label] > initial_counts[label] for label in LABELS
    )
    checks["Continuation leaves class IDs and all row counts unchanged"] = storage(model) == initial_storage
    checks["Final evaluation preserves counts"] = rf.Enum.counts(model, "/label") == final_counts
    with TemporaryDirectory(prefix="relflow-enum-continuation-") as directory:
        path = Path(directory) / "continued.pt"
        model.save(path)
        restored = rf.Model.load(path).to(model.device)
        restored.eval()
        restored_output = prediction(restored, withheld)
        checks["Checkpoint preserves fixed class storage and exposures"] = (
            storage(restored) == initial_storage and rf.Enum.counts(restored, "/label") == final_counts
        )
        checks["Checkpoint preserves withheld predictions"] = all(
            before["value"] == after["value"] and abs(before["probability"] - after["probability"]) < 1e-5
            for before, after in zip(final_output, restored_output, strict=True)
        )
    lit.seed_everything(seed, workers=True)
    control = build()
    control_steps = fit(control, "mixed", seed, budget, accelerator)
    control_seen = measure(prediction(control, seen), seen)
    control_withheld = measure(prediction(control, withheld), withheld)
    checks["All-classes-trained control reaches 0.95 on both populations"] = (
        min(control_seen["accuracy"], control_withheld["accuracy"]) >= 0.95
    )
    checks["Written class probabilities sum to one"] = (
        max(
            value["probability_sum_error"]
            for value in (
                cold,
                initial_seen,
                initial_withheld,
                final_seen,
                final_withheld,
                control_seen,
                control_withheld,
            )
        )
        < 1e-5
    )
    return {
        "train_rows_per_stage": 4096,
        "validation_rows_per_stage": 512,
        "test_rows_per_population": 2048,
        "initial_steps": initial_steps,
        "continuation_steps": continuation_steps,
        "control_steps": control_steps,
        "uniform_four_class_accuracy": 0.25,
        "initial_seen_majority_accuracy": 683 / 2048,
        "cold_withheld": cold,
        "initial_seen": initial_seen,
        "initial_withheld": initial_withheld,
        "final_seen": final_seen,
        "final_withheld": final_withheld,
        "control_seen": control_seen,
        "control_withheld": control_withheld,
        "initial_counts": initial_counts,
        "final_counts": final_counts,
        "storage": storage(model),
    }, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7001)

# %% [markdown]
# </details>
