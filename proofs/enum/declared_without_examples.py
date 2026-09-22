# %% [markdown]
# ---
# title: Declaring a class does not teach its meaning
# categories: [Enum]
# proof-id: P070
# description: A declared class has output support before receiving examples; continued fitting with rehearsal teaches its numerical region without growing the head.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P070 status >}}
#
# ## Insights
#
# Enum creates every declared class row immediately, including classes absent
# from training. This is structural support, not semantic zero-shot learning.
# The proof withholds one class, records its accuracy and probability without
# assuming they must be zero, then measures learning and old-class retention
# after balanced continued fitting. The declaration and head never change.
#
# All ten CPU seeds reached 100% on the seen classes initially and on both
# populations after continued fitting. The withheld class scored 0% after
# initial training, although untrained random models ranged from 0% to 100%
# on that single-class population. Declaration alone therefore supplies no
# reliable accuracy. All gates passed on those seeds and in a separate
# RTX 3090 run, with the original gates and training budget unchanged.

# %%
"""P070: separate a declared Enum class from learning its relationship to context."""

from collections.abc import Iterator
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
from reporting import report

import relflow as rf

PROOF_ID = "P070"
BUDGET = 300
LABELS = ("amber", "blue", "coral", "green")
CENTERS = (-3.0, -1.0, 1.0, 3.0)

# %% [markdown]
# ## Data and observability
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
# ```yaml
# x: -0.94
# label: blue
# ```
#
# This is an initial-task example.
#
# ```yaml
# x: 3.08
# label: green
# ```
#
# This region is withheld until continued fitting. `green` nevertheless appears
# in the declaration, the output head, and every full-width candidate list from
# the start. Its accuracy and probability before examples are diagnostics;
# initialization or accidental extrapolation can produce either correct or
# incorrect answers without teaching a general semantic relationship.
#
# ```yaml
# x: 1.06
# label: coral
# ```
#
# Continued fitting also rehearses this familiar region. Accuracy on both
# familiar and withheld regions guards against simply always predicting green.
#
# ```{typst}
# //| label: fig-proof-enum-declared-without-examples
# //| fig-cap: "Four output classes exist throughout, while training exposure grows from three regions to four."
# //| fig-alt: "Record contains Number x with four separated regions and hidden Enum label with all four labels declared before training."
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number", body: [Four separated numerical regions]),
#   node("label", kind: "target", type: "Enum", body: [Four fixed classes; one initially withheld]),
# )))
# ```


# %%
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
        batch_size=64,
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
    model.optimizer = rf.adamw(learning_rate=0.002)
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, rows=4096, seed=seed + 1, phase=phase),
        validate=partial(records, rows=512, seed=seed + 2, phase=phase),
        seed=seed,
    )
    trainer = lit.Trainer(
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
    trainer.fit(model, datamodule=data)
    model.eval()
    return trainer.global_step


def prediction(model: rf.Model, rows: list[dict]) -> list[dict]:
    output = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    return [row["/label"]["content"] for row in output]


def measure(output: list[dict], rows: list[dict]) -> dict:
    probabilities = [{candidate["value"]: candidate["probability"] for candidate in row["topk"]} for row in output]
    return {
        "accuracy": float(np.mean([value["value"] == row["label"] for value, row in zip(output, rows, strict=True)])),
        "mean_withheld_probability": float(np.mean([row[LABELS[-1]] for row in probabilities])),
        "full_declared_support": all(set(row) == set(LABELS) for row in probabilities),
        "probability_sum_error": float(max(abs(sum(row.values()) - 1.0) for row in probabilities)),
    }


# %% [markdown]
# ## Training and controls
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


# %%
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


# %% [markdown]
# ## Evidence and remaining work
#
# {{< proof P070 evidence >}}
#
# The ten-seed calibration is recorded with the retained thresholds. This proof
# does not promise semantic zero-shot learning, retention without rehearsal,
# open-world label admission, or calibrated probabilities for unobserved classes. The initial withheld
# probability is reported as a diagnostic, not an uncertainty guarantee.
#
# ## Reproduce
#
# {{< proof P070 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=7001)
