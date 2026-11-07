# %% [markdown]
# ---
# title: Can new requests work on unfamiliar group sizes?
# categories:
# - Composition
# description: A request selects a group and operation from a mixed list, with a hidden numeric answer.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: New requests and lengths
#   metric:
#   - restricted_training
#   - unseen_lengths
#   - unseen_requests
#   - normalized_rmse
#   format: number
# - label: Complete-training reference
#   metric:
#   - complete_training
#   - unseen_lengths
#   - unseen_requests
#   - normalized_rmse
#   format: number
# evidence-group: Composition
# ---
#
# The model receives an unsorted mixed list and a request naming one group and one operation. In the
# illustration, max refers only to amber, so the blue value 0.9 is irrelevant and the answer is 0.7. At
# evaluation, both the instruction pairing and the number of items in each group can differ from training.
#
# ## Example
#
# ```yaml
# selected_group: amber
# operation: max
# items:
#   - {group: blue, value: 0.9}
#   - {group: amber, value: 0.2}
#   - {group: amber, value: 0.7}
# answer: 0.7
# ```
#
# ```{typst}
# //| label: fig-evidence-p094
# //| fig-cap: "Visible inputs and the hidden prediction target for this controlled task."
# //| fig-alt: "A request selects a group and operation from a mixed list, with a hidden numeric answer."
# #tree(node("request", kind: "root", children: (node("selected_group", type: "Enum"), node("operation", type: "Enum"), node("items", kind: "branch", children: (node("group", type: "Enum"), node("value", type: "Number"))), node("answer", kind: "target", type: "Number"),)))
# ```
#
# ## Comparison
#
# Compare training on all nine group-operation pairs with training that withholds amber/mean, blue/min, and
# coral/max. Every group and operation is still observed individually. For each arm, report familiar versus
# withheld request combinations crossed with familiar versus larger group sizes. Cycling the requested
# group while keeping the original answer tests whether apparent accuracy actually depends on the
# instruction.
#
# ## Result
#
# {{< evidence P094 status >}}
#
# These checks define a new experiment. A full recorded run is required before claiming that the model
# meets them. Lower normalized error is better; a value of one matches the stated constant reference. The
# numerical gates are provisional and must be calibrated on separate validation seeds rather than adjusted
# after seeing acceptance-test outcomes.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P094 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P094 script >}}
#
# ### Process, controls, and held-out evaluation
#
# There are three declared groups and three operations: mean, min, max. Each group has an independent
# center Uniform(-1,1) plus per-item Uniform(-0.8,0.8) variation. Training and validation groups contain
# 1-3 items; shifted groups contain 4-6, within a schema cap 18. Complete records are shuffled, preserving
# item fields. Each training arm has 4608 rows,576 validation rows, and 600 AdamW updates, batch 64,
# learning rate 0.002, weight decay 0.01, FP32, identical initialization, no checkpoint selection. Each of
# two test conditions has 1152 independent rows balanced across all nine requests. Error is normalized
# against each operation's training-mean answer; all four request/length cells receive separate quality
# gates. The complete-training reference identifies whether a failure is specific to missing combinations.
# The stronger combined shift remains a provisional capability requirement, not an assumption inherited
# from separate length and composition proofs.
#
# ### Complete experiment code

# %%
"""P094: separate held-out request composition from unfamiliar collection cardinality."""

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
import torch
from reporting import experiment, report

BUDGET = 600
TRAIN_ROWS = 4608
TEST_ROWS = 1152
GROUPS = ("amber", "blue", "coral")
OPERATIONS = ("mean", "min", "max")
CELLS = tuple((group, operation) for group in GROUPS for operation in OPERATIONS)
WITHHELD = tuple(zip(GROUPS, OPERATIONS, strict=True))


def records(*, rows: int, seed: int, complete: bool = False, long: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    cells = CELLS if complete else tuple(cell for cell in CELLS if cell not in WITHHELD)
    requests = np.resize(np.arange(len(cells)), rows)
    rng.shuffle(requests)
    for index in requests:
        selected_group, operation = cells[index]
        items, selected = [], []
        for group in GROUPS:
            count = int(rng.integers(4, 7) if long else rng.integers(1, 4))
            values = rng.uniform(-1.0, 1.0) + rng.uniform(-0.8, 0.8, count)
            items.extend({"group": group, "value": float(value)} for value in values)
            if group == selected_group:
                selected = values
        order = rng.permutation(len(items))
        answer = {"mean": np.mean, "min": np.min, "max": np.max}[operation](selected)
        yield {
            "selected_group": selected_group,
            "operation": operation,
            "items": [items[i] for i in order],
            "answer": float(answer),
        }


def build() -> rf.Model:
    return rf.Model.xs(
        selected_group=rf.Enum(values=GROUPS, p_unavailable=0.0),
        operation=rf.Enum(values=OPERATIONS, p_unavailable=0.0),
        items=rf.Branch(
            length=18,
            overflow="error",
            reduction=None,
            group=rf.Enum(values=GROUPS, p_unavailable=0.0),
            value=rf.Number,
        ),
        answer=rf.Number(mask=True, objective="mse"),
    )


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{key: value for key, value in row.items() if key != "answer"} for row in rows]
    output = model.predict(pa.Table.from_pylist(visible), batch_size=128)["predictions"].to_pylist()
    return np.asarray([row["answer"]["content"] for row in output])


def score(output: np.ndarray, rows: list[dict], means: dict[str, float]) -> dict:
    result = {}
    for withheld in (False, True):
        selected = [
            i for i, row in enumerate(rows) if ((row["selected_group"], row["operation"]) in WITHHELD) == withheld
        ]
        target = np.asarray([rows[i]["answer"] for i in selected])
        reference = np.asarray([means[rows[i]["operation"]] for i in selected])
        error = float(np.sqrt(np.mean((output[selected] - target) ** 2)))
        result["unseen_requests" if withheld else "familiar_requests"] = {
            "rows": len(selected),
            "rmse": error,
            "normalized_rmse": error / float(np.sqrt(np.mean((reference - target) ** 2))),
        }
    return result


def fit(model: rf.Model, train: list[dict], validate: list[dict], *, seed: int, budget: int, accelerator: str) -> int:
    """Own the fixed optimizer budget and independent synthetic splits."""
    data = rf.ArrowDataModule(
        train=pa.Table.from_pylist(train),
        validate=pa.Table.from_pylist(validate),
        seed=seed,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    model.fit(
        data,
        batch_size=64,
        learning_rate=0.002,
        weight_decay=0.01,
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
    model.eval()
    return int(model.trainer.global_step)


@experiment("P094", seed=9401)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    tests = {
        "familiar_lengths": list(records(rows=TEST_ROWS, seed=seed * 100 + 3, complete=True)),
        "unseen_lengths": list(records(rows=TEST_ROWS, seed=seed * 100 + 4, complete=True, long=True)),
    }
    metrics, checks = {}, {}
    for complete in (False, True):
        train = list(records(rows=TRAIN_ROWS, seed=seed * 100 + 1, complete=complete))
        validate = list(records(rows=576, seed=seed * 100 + 2, complete=complete))
        means = {
            operation: float(np.mean([row["answer"] for row in train if row["operation"] == operation]))
            for operation in OPERATIONS
        }
        lit.seed_everything(seed, workers=True)
        model = build()
        completed = fit(
            model,
            train,
            validate,
            seed=seed,
            budget=BUDGET if steps is None else min(steps, BUDGET),
            accelerator=accelerator,
        )
        before = deepcopy(model.state_dict())
        arm = "complete_training" if complete else "restricted_training"
        measured = {"steps": completed}
        checks[f"{arm}: full budget completed"] = completed == BUDGET
        for lengths, rows in tests.items():
            output = predict(model, rows)
            measured[lengths] = score(output, rows, means)
            changed = [
                {**row, "selected_group": GROUPS[(GROUPS.index(row["selected_group"]) + 1) % len(GROUPS)]}
                for row in rows
            ]
            measured[lengths]["corrupted_group_request"] = score(predict(model, changed), rows, means)
            checks[f"{arm}/{lengths}: finite predictions"] = bool(np.isfinite(output).all())
            for requests in ("familiar_requests", "unseen_requests"):
                maximum = 0.55 if lengths == "familiar_lengths" else 0.75
                checks[f"{arm}/{lengths}/{requests}: normalized RMSE below {maximum}"] = (
                    measured[lengths][requests]["normalized_rmse"] < maximum
                )
                checks[f"{arm}/{lengths}/{requests}: request corruption raises RMSE"] = (
                    measured[lengths]["corrupted_group_request"][requests]["rmse"]
                    > 1.2 * measured[lengths][requests]["rmse"]
                )
        checks[f"{arm}: evaluation preserves full model state"] = all(
            (
                torch.equal(value, model.state_dict()[name])
                if isinstance(value, torch.Tensor)
                else value == model.state_dict()[name]
            )
            for name, value in before.items()
        )
        metrics[arm] = measured
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
