# %% [markdown]
# ---
# title: Does reconstruction accuracy include unfamiliar answers?
# categories:
# - Vocabulary
# description: Hide a categorical answer and reconstruct it from numeric context, while varying how many test answers
#   belong to the learned vocabulary.
# evidence-readout:
# - label: Known-answer accuracy, low coverage
#   metric:
#   - known_102
#   - accuracy.content
#   format: percent
# - label: Answers covered
#   metric:
#   - known_102
#   - coverage.content
#   format: percent
# - label: All-answer accuracy
#   metric:
#   - known_102
#   - accuracy.all
#   format: percent
# evidence-group: Vocabulary
# evidence-diagram:
#   caption: Selector notes identify hidden values. hide chooses which code answers are hidden.
#   alt: Model tree with record, x, code. Selector notes identify hidden values. hide chooses which code answers are
#     hidden.
# evidence-review:
#   question: Does reconstruction accuracy include unfamiliar answers?
#   modeling: Learn to reconstruct a hidden Category answer from numeric context. At evaluation, replace some expected
#     target names with unseen labels while keeping the exact same inputs and selected positions. Compare accuracy
#     among representable answers with accuracy across every selected answer.
#   comparison:
#     kind: Recorded conditions
#     metric: Accuracy
#     conditions:
#     - label: Known-answer accuracy, low coverage
#       path:
#       - known_102
#       - accuracy.content
#       format: percent
#     - label: All-answer accuracy
#       path:
#       - known_102
#       - accuracy.all
#       format: percent
#     definition: Fraction of predictions matching the expected answer, over the population named in each condition.
#     direction: Higher is better
#     domain:
#     - 0
#     - 1
#   note: The chart uses the approximately 10%-coverage condition. Known-answer scores exclude unavailable labels;
#     coverage and all-answer accuracy expose that exclusion.
#   criteria:
#     prefixes:
#       '0': No known targets
#       '102': 102 known targets
#       '512': 512 known targets
#       '1024': 1,024 known targets
# evidence-plot:
#   mode: coverage
#   target: code
#   focus: Can high conditional accuracy conceal missing target coverage?
#   comparison: Plot coverage, known-target accuracy, and all-target accuracy together as known targets are removed.
#   limit: Conditional accuracy is undefined at zero coverage and remains visibly missing.
# evidence-cost:
#   baseline: 18
#   candidate: 18
# ---
#
# Reconstruction quality has two denominators: the answers the model can represent and all answers it
# was asked to reconstruct. This experiment holds the numeric evidence and masking positions fixed
# while changing how many selected answer names belong to the learned vocabulary. It makes the
# difference visible without making the original classification rule harder.
#
# ## Example
#
# ```yaml
# x: 2.0
# code: class-2
# hide: true
# ```
#
# code is hidden wherever hide is true. Unknown code names cannot be emitted by the learned output
# vocabulary.
#
# ```typst
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number", detail: "Class coordinate"),
#   node("code", type: "Category", detail: "Reconstruct where hide is true"),
# )))
# ```
#
# The illustrative `x = 2` identifies `class-2`, and `hide: true` selects that code for
# reconstruction. The auxiliary hide value controls masking; it is not a predictive schema field.
# Replacing the hidden target name with `novel-class-2` leaves the visible numeric evidence unchanged,
# but the learned output vocabulary still contains only the original eight class names.
#
# ## Comparison
#
# Compare known-answer accuracy, vocabulary coverage, and accuracy over all selected answers.
#
# Exactly 1,024 evaluation answers are selected. Keeping only 102 of their original names gives about
# 10% coverage: even perfect recovery of every representable answer would recover only 102 of the
# 1,024 requested names. Reordering those same records changes which batches contain known targets,
# allowing the experiment to check that batch composition does not distort epoch-level counts or
# scores.
#
# ## Result
#
# Known-only accuracy stays high even as coverage falls. Excluding unfamiliar answers from the content
# score does not mean they were recovered.
#
# Known-answer accuracy measures conditional quality; coverage states how often that condition holds;
# all-answer accuracy counts unfamiliar targets as wrong. When no selected answers are known,
# conditional loss and accuracy are undefined and are recorded as missing values. A blank conditional
# score must not be read as zero error or as successful recovery of unknown names.
#
# ## Method
#
# ### Training and controls
#
# Train for 400 updates on 8,192 rows; storage grows to admit all eight labels.
# Evaluation selects exactly 1,024 of 2,048 independent records. Compare the
# original set with copies retaining 50%, 10%, or 0% of selected target names.
# All copies preserve the exact visible inputs and query-selected masks.
#
# Repeat the 10%-coverage case with unknown targets grouped into separate
# batches. Count-weighted epoch metrics must agree despite all-OOV batches.
# Scores with no known targets must be undefined, not zero reconstruction
# error. Naive accuracy over all selected valued targets counts OOV as wrong.
#

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "lightning",
#     "numpy",
#     "pyyaml>=6",
#     "relflow>=0.5.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "reliability"
# types = ["Category"]
# conditions = ["unseen-values", "transformations"]
# ///

"""P062: report stable reconstruction quality alongside pretraining coverage."""

import sys
from collections.abc import Iterator
from functools import partial
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import relflow as rf
from reporting import experiment, report

BUDGET = 400
LABELS = tuple(f"class-{index}" for index in range(8))


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for index, label in enumerate(rng.integers(0, len(LABELS), rows)):
        yield {"x": float(label), "code": LABELS[label], "hide": index % 2 == 0}


class PredictionModel(rf.Model):
    x = rf.Number
    code = rf.Category(mask=rf.Mask(query="hide", reconstruct=True))


def build() -> rf.Model:
    model = PredictionModel(d_model=32, n_layers=1, n_heads=4, dropout=0.0)
    return model


def evaluate(trainer: lit.Trainer, model: rf.Model, rows: list[dict]) -> dict[str, float]:
    data = rf.SyntheticDataModule(validate=lambda: iter(rows), seed=0)
    result = trainer.validate(model, datamodule=data, verbose=False)[0]
    prefix = ".code/validate."
    return {name.removeprefix(prefix): float(value) for name, value in result.items() if name.startswith(prefix)}


@experiment("P062", seed=7621)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    data = rf.SyntheticDataModule(
        train=partial(records, rows=8192, seed=seed + 1),
        validate=partial(records, rows=1024, seed=seed + 2),
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
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer
    original = list(records(rows=2048, seed=seed + 3))
    vocabulary = rf.Category.vocabulary(model, "/code")
    counts = rf.Category.counts(model, "/code")
    metrics, checks = {}, {}
    for keep in (1024, 512, 102, 0):
        selected = 0
        rows = []
        for row in original:
            unavailable = row["hide"] and selected >= keep
            rows.append({**row, "code": f"novel-{row['code']}" if unavailable else row["code"]})
            selected += int(row["hide"])
        measured = evaluate(trainer, model, rows)
        # Keep undefined quantities explicit without nonstandard JSON NaNs.
        metrics[f"known_{keep}"] = {name: value if np.isfinite(value) else None for name, value in measured.items()}
        checks[f"{keep}: exact known and unavailable target counts"] = (
            measured.get("targets.known") == keep and measured.get("targets.unavailable") == 1024 - keep
        )
        checks[f"{keep}: coverage matches selected targets"] = (
            abs(measured.get("coverage.content", -1) - keep / 1024) < 1e-7
        )
        if keep:
            checks[f"{keep}: known accuracy exceeds 0.95"] = measured.get("accuracy.content", -1) > 0.95
            checks[f"{keep}: all-target accuracy includes unavailable errors"] = (
                abs(measured.get("accuracy.all", -1) - measured.get("accuracy.content", 0) * keep / 1024) < 1e-7
            )
        else:
            checks["All-OOV conditional scores are undefined"] = all(
                name in measured and np.isnan(measured[name])
                for name in ("accuracy.content", "nll.content", "loss.content")
            )
            checks["All-OOV all-target accuracy is zero"] = measured.get("accuracy.all") == 0
        if keep == 102:
            rng = np.random.default_rng(seed + 4)
            permuted = [rows[index] for index in rng.permutation(len(rows))]
            regrouped = evaluate(trainer, model, permuted)
            names = (
                "targets.known",
                "targets.unavailable",
                "coverage.content",
                "accuracy.content",
                "accuracy.all",
                "nll.content",
                "loss.content",
            )
            drift = (
                max(abs(measured[name] - regrouped[name]) for name in names)
                if all(name in measured and name in regrouped for name in names)
                else None
            )
            metrics["batch_partition_drift"] = drift
            metrics["reported_content_loss_drift"] = abs(measured["loss.content"] - regrouped["loss.content"])
            checks["Rebatching preserves counts and scores"] = drift is not None and drift < 1e-5
    checks["Validation leaves mapping and exposures frozen"] = (
        rf.Category.vocabulary(model, "/code") == vocabulary and rf.Category.counts(model, "/code") == counts
    )
    return metrics, checks


if __name__ == "__main__":
    report(run)
