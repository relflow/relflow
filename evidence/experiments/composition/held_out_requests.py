# %% [markdown]
# ---
# title: Can familiar instructions work in new combinations?
# categories:
# - Compositional generalization
# description: Train on group and operation requests, deliberately leaving out three combinations, such as coral’s
#   maximum.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Withheld coral maximum, restricted training
#   metric:
#   - restricted
#   - cells
#   - coral/max
#   - normalized_rmse
#   format: error
# - label: Same request, complete training
#   metric:
#   - complete
#   - cells
#   - coral/max
#   - normalized_rmse
#   format: error
# evidence-group: Composition
# ---
#
# The request contains two instructions: which group to inspect and which operation to apply. Every
# group and every operation occurs during fitting, but three particular pairings are deliberately
# absent. The task is to combine familiar instructions on a fresh bag of values rather than memorize
# an answer for an old record.
#
# ## Example
#
# ```yaml
# selected_group: amber
# operation: mean
# items:
#   - {group: blue, value: -0.4}
#   - {group: amber, value: 0.2}
#   - {group: coral, value: 1.1}
#   - {group: amber, value: 0.8}
#   - {group: blue, value: 0.6}
#   - {group: coral, value: 1.5}
# answer: 0.51
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-held-out-requests
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with request, selected_group, operation, items, group, value, answer. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("request", kind: "root", children: (
#   node("selected_group", type: "Enum"),
#   node("operation", type: "Enum"),
#   node("items", kind: "branch", repeated: true, children: (
#     node("group", type: "Enum"), node("value", type: "Number"),
#   )),
#   node("answer", kind: "target", type: "Number"),
# )))
# ```
#
# In the illustrative bag, amber has values 0.2 and 0.8, so its mean is 0.5; the shown answer 0.51
# includes a possible small noise draw. Asking for amber's maximum would instead give 0.8 before
# noise. Amber/mean is one of the withheld training combinations, making this example a test of
# composition in the restricted arm.
#
# ## Comparison
#
# Test the missing combinations and compare with a model trained on all combinations. Change either request to check that it matters.
#
# The complete-training model receives examples of all nine group/operation cells and checks that the
# task is learnable with the same total row and update budgets. The restricted model sees only six
# cells. Changing a request while keeping the original answer fixed should worsen error if the model
# actually follows that instruction; it is not scored against a recomputed answer.
#
# ## Result
#
# {{< evidence P071 status >}}
#
# Read the selected cell scores separately: familiar requests, withheld combinations, and corrupted instructions answer different questions. The criteria above show which requirements this recorded run meets.
#
# Each cell has its own error score, so strong familiar-cell performance cannot conceal one failed
# withheld combination. Normalized RMSE compares with a training-only operation baseline, pooling
# exchangeable groups without using withheld answers. Even success on these bounded two-item groups
# would not establish exact arithmetic, unseen operations, or generalization to larger collections.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P071 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P071 script >}}
#
# ### Synthetic process and split
#
# Each fresh bag contains two numbers per group. Each group's center is drawn
# independently from Uniform(-1.5, 1.5), with two independent Uniform(-0.8, 0.8)
# deviations. The visible request selects one group's mean, minimum, or maximum.
# Target noise is Normal(0, 0.02). Items are randomly interleaved in every row.
# The model never receives group centers or the generator's selected values.
#
# Training excludes amber/mean, blue/min, and coral/max. Every group and every
# operation still occurs in two observed combinations. Independently generated
# bags supply 4,608 training rows, 576 validation rows, and 2,304 test rows;
# test has 256 rows per cell. No bag is reused across splits or requests.
# Validation for the restricted model also excludes withheld combinations.
#
# ### Training, controls, and gates
#
# Both arms use the xs preset, retaining all item-branch tokens, batch size 128,
# and 600 AdamW updates at learning rate 0.002. Initialization is paired. The
# complete arm has the same row count and update budget, spread over all nine
# combinations. It is a task-learnability control, not a matched cell-frequency
# control. No architecture, checkpoint, or threshold is chosen on test results.
#
# The positive control must reach normalized RMSE below 0.35 in every cell.
# The restricted arm must reach 0.35 on every observed cell and 0.50 on every
# withheld cell. Each denominator uses a training-only operation mean pooled
# across exchangeable groups; no withheld target supplies a baseline estimate.
# Group corruption must give overall normalized RMSE above 0.80 and exceed
# intact error by 0.25. Operation corruption must add at least 0.10 to overall
# normalized RMSE. Both control predictions are compared with the original
# answers, so these controls measure reliance on the requests, not accuracy on
# the altered instructions. Report their cell scores as well as aggregate gaps.
#
# ### Complete experiment code
#

# %%
"""P071: test composition on group-operation pairs excluded from training."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
from relflow.presets.base import resolve
from reporting import experiment, report

BUDGET = 600
GROUPS = ("amber", "blue", "coral")
OPERATIONS = ("mean", "min", "max")
CELLS = tuple((group, operation) for group in GROUPS for operation in OPERATIONS)
WITHHELD = tuple(zip(GROUPS, OPERATIONS, strict=True))
TRAIN_ROWS = 4608
TEST_ROWS = 2304


def records(*, rows: int, seed: int, complete: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    cells = CELLS if complete else tuple(cell for cell in CELLS if cell not in WITHHELD)
    requests = np.resize(np.arange(len(cells)), rows)
    rng.shuffle(requests)
    for index in requests:
        group, operation = cells[index]
        values = rng.uniform(-1.5, 1.5, (3, 1)) + rng.uniform(-0.8, 0.8, (3, 2))
        selected = values[GROUPS.index(group)]
        answer = {"mean": np.mean, "min": np.min, "max": np.max}[operation](selected)
        order = rng.permutation(6)
        yield {
            "selected_group": group,
            "operation": operation,
            "items": [{"group": GROUPS[i // 2], "value": float(values.flat[i])} for i in order],
            "answer": float(answer + rng.normal(0, 0.02)),
        }


def build() -> rf.Model:
    # Freeze the experimental architecture independently of SDK preset changes.
    control_preset = rf.presets.Preset(
        name="xs",
        d_model=64,
        root=rf.presets.BranchDefaults(n_layers=1, n_heads=2, reduction=rf.Attention()),
        branch=rf.presets.BranchDefaults(n_layers=1, n_heads=2, reduction=rf.Attention()),
        leaf=rf.presets.LeafDefaults(n_heads=2),
    )
    model = rf.Model(
        schema=resolve(
            control_preset,
            selected_group=rf.Enum(values=GROUPS, p_unavailable=0.0),
            operation=rf.Enum(values=OPERATIONS, p_unavailable=0.0),
            items=rf.Branch(
                length=6,
                reduction=None,
                group=rf.Enum(values=GROUPS, p_unavailable=0.0),
                value=rf.Number,
            ),
            answer=rf.Number(mask=True, objective="mse"),
        )
    )
    model.preset = control_preset
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{key: value for key, value in row.items() if key != "answer"} for row in rows]
    output = pl.DataFrame(model.predict(pa.Table.from_pylist(visible)))["predictions"].to_list()
    return np.asarray([row["answer"]["content"] for row in output])


def corrupt(rows: list[dict], field: str) -> list[dict]:
    """Cycle one request to a different declared value without changing its bag or target."""
    values = GROUPS if field == "selected_group" else OPERATIONS
    return [{**row, field: values[(values.index(row[field]) + 1) % len(values)]} for row in rows]


def scores(train: list[dict], test: list[dict], predicted: np.ndarray) -> dict:
    """Use the observed-operation training mean; groups are exchangeable by construction."""
    means = {
        operation: np.mean([row["answer"] for row in train if row["operation"] == operation])
        for operation in OPERATIONS
    }
    result = {}
    for group, operation in CELLS:
        indices = [
            index for index, row in enumerate(test) if (row["selected_group"], row["operation"]) == (group, operation)
        ]
        target = np.asarray([test[index]["answer"] for index in indices])
        rmse = float(np.sqrt(np.mean((predicted[indices] - target) ** 2)))
        baseline = float(np.sqrt(np.mean((means[operation] - target) ** 2)))
        result[f"{group}/{operation}"] = {
            "rows": len(indices),
            "withheld": (group, operation) in WITHHELD,
            "rmse": rmse,
            "baseline_rmse": baseline,
            "normalized_rmse": rmse / baseline,
        }
    return result


@experiment("P071", seed=7101)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    test = list(records(rows=TEST_ROWS, seed=seed * 100 + 3, complete=True))
    metrics = {"train_rows": TRAIN_ROWS, "test_rows": TEST_ROWS, "noise_rmse": 0.02}
    checks = {}
    for complete in (False, True):
        lit.seed_everything(seed, workers=True)
        model = build()
        data = rf.SyntheticDataModule(
            train=partial(records, rows=TRAIN_ROWS, seed=seed * 100 + 1, complete=complete),
            validate=partial(records, rows=576, seed=seed * 100 + 2, complete=complete),
            seed=seed,
        )
        model.fit(
            data,
            batch_size=128,
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
        train = list(records(rows=TRAIN_ROWS, seed=seed * 100 + 1, complete=complete))
        output = predict(model, test)
        cells = scores(train, test, output)
        arm = "complete" if complete else "restricted"
        measured = {"steps": trainer.global_step, "cells": cells}
        for cell, score in cells.items():
            gate = 0.50 if score["withheld"] and not complete else 0.35
            checks[f"{arm}: {cell} normalized RMSE below {gate:.2f}"] = score["normalized_rmse"] < gate
        baseline = float(np.sqrt(np.mean([score["baseline_rmse"] ** 2 for score in cells.values()])))
        target = np.asarray([row["answer"] for row in test])
        intact = float(np.sqrt(np.mean((output - target) ** 2)) / baseline)
        measured["normalized_rmse"] = intact
        for field in ("selected_group", "operation"):
            control = predict(model, corrupt(test, field))
            error = float(np.sqrt(np.mean((control - target) ** 2)) / baseline)
            measured[f"corrupt_{field}"] = {
                "normalized_rmse": error,
                "gap": error - intact,
                "cells": scores(train, test, control),
            }
            checks[f"{arm}: corrupting {field} increases normalized RMSE"] = error - intact > (
                0.25 if field == "selected_group" else 0.10
            )
            if field == "selected_group":
                checks[f"{arm}: corrupted group normalized RMSE exceeds 0.80"] = error > 0.80
            checks[f"{arm}: {field} control predictions are finite"] = bool(np.isfinite(control).all())
        checks[f"{arm}: intact predictions are finite"] = bool(np.isfinite(output).all())
        metrics[arm] = measured
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
