# %% [markdown]
# ---
# title: Familiar Instructions in Unseen Combinations
# categories: [Compositional generalization]
# proof-id: P071
# description: Withhold group-operation pairs while retaining every individual instruction during training.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P071 status >}}
#
# ## Insights
#
# A model can fit every observed request without learning a reusable operation.
# This experiment withholds three group-operation combinations and measures
# every request cell separately. A model trained on all combinations supplies
# a positive control for ordinary task learnability. Corrupting either request
# field tests whether predictions use both instructions.
#
# Across three full CPU seeds, two met every gate. Complete-combination
# training reached normalized RMSE 0.052–0.274 across all cells and seeds;
# the restricted model reached 0.083–0.272 on observed cells. Withheld cells
# ranged from 0.133 to 1.330: seed 7103 failed coral/max, performing worse than
# its constant baseline despite fitting the observed requests. Both request
# corruption controls met their gates on every seed. This supports using both
# instructions, while reliable transfer to all unseen pairs remains unproven.
# The original gates and budget are unchanged; thresholds remain provisional.

# %%
"""P071: test composition on group-operation pairs excluded from training."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
from reporting import report

import relflow as rf

PROOF_ID = "P071"
BUDGET = 600
GROUPS = ("amber", "blue", "coral")
OPERATIONS = ("mean", "min", "max")
CELLS = tuple((group, operation) for group in GROUPS for operation in OPERATIONS)
WITHHELD = tuple(zip(GROUPS, OPERATIONS, strict=True))
TRAIN_ROWS = 4608
TEST_ROWS = 2304

# %% [markdown]
# ## Synthetic process and split
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
# This withheld request has noiseless answer 0.5. The same instruction and
# example values with a familiar operation instead give maximum 0.8:
#
# ```yaml
# selected_group: amber
# operation: max
# items:
#   - {group: blue, value: -0.4}
#   - {group: amber, value: 0.2}
#   - {group: coral, value: 1.1}
#   - {group: amber, value: 0.8}
#   - {group: blue, value: 0.6}
#   - {group: coral, value: 1.5}
# answer: 0.79
# ```
#
# Changing the group to blue while retaining the first answer is a deliberately
# incorrect request corruption; the correct blue mean would be 0.1:
#
# ```yaml
# selected_group: blue
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
# ```{typst}
# //| label: fig-proof-held-out-requests
# //| fig-cap: "Familiar group and operation instructions request a statistic from six interleaved items."
# //| fig-alt: "Request has Enum group and operation, a repeated branch containing group and Number value, and a hidden Number answer."
# #tree(node("request", kind: "root", children: (
#   node("selected_group", type: "Enum"),
#   node("operation", type: "Enum"),
#   node("items", kind: "branch", repeated: true, children: (
#     node("group", type: "Enum"), node("value", type: "Number"),
#   )),
#   node("answer", kind: "target", type: "Number"),
# )))
# ```


# %%
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
    model = rf.Model.xs(
        batch_size=128,
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
    model.optimizer = rf.adamw(learning_rate=0.002)
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{key: value for key, value in row.items() if key != "answer"} for row in rows]
    output = model.predict(pa.Table.from_pylist(visible))["predictions"].to_pylist()
    return np.asarray([row["/answer"]["content"] for row in output])


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


# %% [markdown]
# ## Training, controls, and gates
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


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    test = list(records(rows=TEST_ROWS, seed=seed * 100 + 3, complete=True))
    metrics = {"train_rows": TRAIN_ROWS, "test_rows": TEST_ROWS, "noise_rmse": 0.02}
    checks = {}
    for complete in (False, True):
        lit.seed_everything(seed, workers=True)
        model = build()
        data = rf.SyntheticDataModule(
            model=model,
            train=partial(records, rows=TRAIN_ROWS, seed=seed * 100 + 1, complete=complete),
            validate=partial(records, rows=576, seed=seed * 100 + 2, complete=complete),
            seed=seed,
        )
        trainer = lit.Trainer(
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
        trainer.fit(model, datamodule=data)
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


# %% [markdown]
# ## Evidence and limitations
#
# {{< proof P071 evidence >}}
#
# This is a finite compositional split with two items per group. It does not
# establish arbitrary instructions, variable-length extrapolation, new group
# meanings, or order invariance. Random interleaving discourages positional
# shortcuts but does not constitute an exact permutation-invariance gate.
# The three-seed panel exposes a seed-dependent failure on an unseen pair.
# No ten-seed calibration has been run, so these thresholds remain provisional.
#
# ## Reproduce
#
# {{< proof P071 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=7101)
