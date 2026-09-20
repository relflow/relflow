# %% [markdown]
# ---
# title: Admitting a label is not learning to predict it
# categories: [Vocabulary and OOV]
# proof-id: P059
# description: Follow an output vocabulary through cold prediction, training, checkpoints, new-label admission, and continued fitting with rehearsal.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P059 status >}}
#
# ## Insights
#
# Prediction cannot teach a vocabulary. Training-time encoding can admit a
# label through automatic storage growth without taking an optimizer step; admission
# alone does not establish learned behavior. Check both mapping retention and
# held-out accuracy after continued fitting. Rehearse original classes so that
# old-task retention is an explicit part of the experiment.
#
# ## Setup

# %%
"""P059: vocabulary persistence and new-label admission followed by measured learning."""

from collections.abc import Iterator
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P059"
BUDGET = 300
CONTINUATION = 400
LABELS = ("negative", "positive", "far-negative", "far-positive")
CENTERS = (-1.0, 1.0, -3.0, 3.0)

# %% [markdown]
# ## Data and model
#
# Initially, x lies within 0.15 of -1 or +1 and predicts negative or positive.
# New classes lie within 0.15 of -3 or +3. Each split balances its classes and
# shuffles them with an independent local RNG. Each fitting phase uses 4,096
# training and 512 validation rows; the old and new tests each use 1,024 rows.
# Storage initially learns two output labels, then grows to hold four.
#
# ```yaml
# x: -0.9
# label: negative
# ```
#
# The initial task fills only part of the output vocabulary.
#
# ```yaml
# x: -3.1
# label: far-negative
# ```
#
# Evaluation does not allocate a slot for this new answer.
#
# ```yaml
# x: 3.1
# label: far-positive
# ```
#
# Explicit training encoding can admit both new labels. Continued fitting
# then sees a balanced mixture of all four classes, including the old ones.
#
# ```{typst}
# //| label: fig-proof-vocabulary-admission-training
# //| fig-cap: "Output-vocabulary admission and optimizer learning are separate events."
# //| fig-alt: "Record contains Number x and hidden Category label, whose vocabulary grows from two to four discovered labels during continued training."
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number", body: [Four separated numerical regions]),
#   node("label", kind: "target", type: "Category", body: [Two labels, growing to four]),
# )))
# ```


# %%
def records(*, rows: int, seed: int, phase: str = "initial") -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    indices = {"initial": (0, 1), "new": (2, 3), "mixed": (0, 1, 2, 3)}[phase]
    classes = np.resize(np.asarray(indices), rows)
    rng.shuffle(classes)
    noise = rng.uniform(-0.15, 0.15, rows)
    for index, deviation in zip(classes, noise, strict=True):
        yield {"x": float(CENTERS[index] + deviation), "label": LABELS[index]}


def build() -> rf.Model:
    return rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        batch_size=64,
        x=rf.Number,
        label=rf.Category(p_unavailable=0.0, topk=[3], mask=True),
    )


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


# %% [markdown]
# ## Training and controls
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


# %%
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


# %% [markdown]
# ## Evidence and remaining work
#
# {{< proof P059 evidence >}}
#
# Gates are provisional. This exercises automatic growth of a trained head.
# It does not claim retention without rehearsal,
# optimal continual learning, or vocabulary synchronization across workers or
# distributed ranks. Admission may change buffers and the output softmax
# denominator, so preserved old rows do not imply unchanged predictions.
#
# ## Reproduce
#
# {{< proof P059 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=5901)
