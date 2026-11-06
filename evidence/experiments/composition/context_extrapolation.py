# %% [markdown]
# ---
# title: Can an inferred rule work beyond its examples?
# categories:
# - Composition
# description: A set of numeric examples defines a fresh rule; the model predicts a hidden answer at a requested
#   query.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Four examples, outside range
#   metric:
#   - context
#   - count_4/outside
#   - normalized_rmse
#   format: number
# - label: Eight examples, outside range
#   metric:
#   - context
#   - count_8/outside
#   - normalized_rmse
#   format: number
# evidence-group: Composition
# ---
#
# Each record supplies examples from a new linear rule, but never supplies its slope or intercept. The
# shown examples follow y=2*x+0.5; at 1.75 the answer would be 4.0. That query lies beyond every supplied
# x, unlike interpolation between examples. The fitted model must infer and apply the current record's rule
# without changing its weights. The worked slope illustrates the arithmetic; measured slopes range from
# 0.75 to 1.75 in magnitude.
#
# ## Example
#
# ```yaml
# examples:
#   - {x: -1.0, y: -1.5}
#   - {x: -0.333333, y: -0.166666}
#   - {x: 0.333333, y: 1.166666}
#   - {x: 1.0, y: 2.5}
# query: 1.75
# answer: 4.0
# ```
#
# ```{typst}
# //| label: fig-evidence-p097
# //| fig-cap: "Visible inputs and the hidden prediction target for this controlled task."
# //| fig-alt: "A set of numeric examples defines a fresh rule; the model predicts a hidden answer at a requested query."
# #tree(node("task", kind: "root", children: (node("examples", kind: "branch", children: (node("x", type: "Number"), node("y", type: "Number"))), node("query", type: "Number"), node("answer", kind: "target", type: "Number"),)))
# ```
#
# ## Comparison
#
# Train only with four examples and queries inside their span. Test two, four, and eight examples,
# separately for interior queries and queries outside all anchors. Compare a query-only model and
# whole-context swapping, and fit a least-squares reference from each record's visible examples. This
# separates missing information from a failure to use it, and separates new query range from new context
# length.
#
# ## Result
#
# {{< evidence P097 status >}}
#
# These checks define a new experiment. A full recorded run is required before claiming that the model
# meets them. Lower normalized error is better; a value of one matches the stated constant reference. The
# numerical gates are provisional and must be calibrated on separate validation seeds rather than adjusted
# after seeing acceptance-test outcomes.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P097 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P097 script >}}
#
# ### Process, controls, and held-out evaluation
#
# Each independent record samples a signed slope with magnitude 0.75-1.75, intercept Uniform(-1,1), equally
# spaced x anchors from-1 to 1, and independent Gaussian observation/target noise with standard deviation
# 0.01. Training/validation have 4096/512 fresh rules, four examples, queries Uniform(-0.8,0.8). Six test
# conditions each have 1024 fresh rules: counts 2/4/8 crossed with interior queries or signed query
# magnitudes 1.5-2.0. Example order is shuffled. Each xs arm receives 600 AdamW steps, batch 64, learning
# rate 0.002, weight decay 0.01, FP32, identical initialization and no checkpoint selection. The query-only
# arm removes the example branch, so it is an information control rather than a parameter-matched
# architecture. Every condition retains a distinct normalized-error and corruption gate. Full model state
# must remain unchanged across inference. These provisional checks concern this bounded affine family; they
# do not establish arbitrary algorithm induction.
#
# ### Complete experiment code

# %%
"""P097: infer fresh affine rules outside their example range and at new context lengths."""

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
import torch
from reporting import experiment, report

BUDGET = 600
TRAIN_ROWS = 4096
TEST_ROWS = 1024


def records(*, rows: int, seed: int, count: int = 4, extrapolate: bool = False) -> Iterator[dict]:
    if count not in (2, 4, 8):
        raise ValueError("P097 context count must be 2, 4, or 8")
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        slope = float(rng.choice((-1, 1)) * rng.uniform(0.75, 1.75))
        intercept = float(rng.uniform(-1, 1))
        x = np.linspace(-1.0, 1.0, count)
        y = slope * x + intercept + rng.normal(0, 0.01, count)
        query = float(rng.choice((-1, 1)) * rng.uniform(1.5, 2.0)) if extrapolate else float(rng.uniform(-0.8, 0.8))
        order = rng.permutation(count)
        yield {
            "examples": [{"x": float(x[i]), "y": float(y[i])} for i in order],
            "query": query,
            "answer": float(slope * query + intercept + rng.normal(0, 0.01)),
        }


def build(context: bool) -> rf.Model:
    fields = {"query": rf.Number, "answer": rf.Number(mask=True, objective="mse")}
    if context:
        fields["examples"] = rf.Branch(length=8, overflow="error", reduction=None, x=rf.Number, y=rf.Number)
    return rf.Model.xs(fields=fields)


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{key: value for key, value in row.items() if key != "answer"} for row in rows]
    output = model.predict(pa.Table.from_pylist(visible), batch_size=128)["predictions"].to_pylist()
    return np.asarray([row["answer"]["content"] for row in output])


def least_squares(rows: list[dict]) -> np.ndarray:
    output = []
    for row in rows:
        x = np.asarray([item["x"] for item in row["examples"]])
        y = np.asarray([item["y"] for item in row["examples"]])
        centered = x - x.mean()
        slope = float(np.dot(centered, y) / np.dot(centered, centered))
        output.append(float(y.mean() + slope * (row["query"] - x.mean())))
    return np.asarray(output)


def score(output: np.ndarray, rows: list[dict], mean: float) -> dict:
    target = np.asarray([row["answer"] for row in rows])
    error = float(np.sqrt(np.mean((output - target) ** 2)))
    return {"rmse": error, "normalized_rmse": error / float(np.sqrt(np.mean((target - mean) ** 2)))}


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


@experiment("P097", seed=9701)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train = list(records(rows=TRAIN_ROWS, seed=seed * 100 + 1))
    validate = list(records(rows=512, seed=seed * 100 + 2))
    tests = {
        f"count_{count}/{'outside' if outside else 'inside'}": list(
            records(rows=TEST_ROWS, seed=seed * 100 + 10 + count + int(outside), count=count, extrapolate=outside)
        )
        for count in (2, 4, 8)
        for outside in (False, True)
    }
    mean = float(np.mean([row["answer"] for row in train]))
    metrics, checks = {}, {}
    for context in (True, False):
        lit.seed_everything(seed, workers=True)
        model = build(context)
        completed = fit(
            model,
            train,
            validate,
            seed=seed,
            budget=BUDGET if steps is None else min(steps, BUDGET),
            accelerator=accelerator,
        )
        before = deepcopy(model.state_dict())
        arm = "context" if context else "query_only"
        measured = {"steps": completed}
        checks[f"{arm}: full budget completed"] = completed == BUDGET
        for condition, rows in tests.items():
            output = predict(model, rows)
            measured[condition] = score(output, rows, mean)
            checks[f"{arm}/{condition}: finite predictions"] = bool(np.isfinite(output).all())
            if context:
                oracle = score(least_squares(rows), rows, mean)
                measured[condition]["least_squares"] = oracle
                checks[f"{condition}: visible-example oracle RMSE below 0.06"] = oracle["rmse"] < 0.06
                order = np.random.default_rng(seed + 19).permutation(len(rows))
                changed = [{**row, "examples": rows[i]["examples"]} for row, i in zip(rows, order, strict=True)]
                measured[condition]["swapped_context"] = score(predict(model, changed), rows, mean)
        checks[f"{arm}: evaluation preserves full state"] = all(
            (
                torch.equal(value, model.state_dict()[name])
                if isinstance(value, torch.Tensor)
                else value == model.state_dict()[name]
            )
            for name, value in before.items()
        )
        metrics[arm] = measured
    for condition in tests:
        measured = metrics["context"][condition]
        maximum = 0.55 if condition.endswith("outside") else 0.40
        checks[f"{condition}: context normalized RMSE below {maximum}"] = measured["normalized_rmse"] < maximum
        checks[f"{condition}: context halves query-only RMSE"] = (
            measured["rmse"] < 0.5 * metrics["query_only"][condition]["rmse"]
        )
        checks[f"{condition}: context swapping doubles RMSE"] = (
            measured["swapped_context"]["rmse"] > 2 * measured["rmse"]
        )
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
