# %% [markdown]
# ---
# title: Can an inferred rule work beyond its examples?
# description: A set of numeric examples defines a fresh rule; the model predicts a hidden answer at a requested query.
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
# evidence-review:
#   question: Can an inferred rule work beyond its examples?
#   modeling: Give each record examples from a fresh straight-line rule and hide its answer at a query x. Train with
#     four examples and queries inside their span. Compare a context model with a query-only control on queries beyond
#     the examples, using the same four-example test records.
#   comparison:
#     metric: Normalized RMSE
#     kind: Controlled comparison
#     conditions:
#     - label: Query only · outside example range
#       path:
#       - query_only
#       - count_4/outside
#       - normalized_rmse
#       format: number
#     - label: Four visible examples · outside range
#       path:
#       - context
#       - count_4/outside
#       - normalized_rmse
#       format: number
#     definition: Root mean squared error divided by the reference scale defined in this experiment. Zero is exact;
#       one equals that reference scale.
#     direction: Lower error is better
#   note: Two- and eight-example conditions are checked separately. This tests a bounded affine family, not arbitrary
#     rule induction or unlimited extrapolation.
# evidence-plot:
#   target: answer
#   focus: Where do context examples stop supporting extrapolation?
#   comparison: Separate familiar, outside-range, sparse-context, and altered-example conditions with least squares.
#   limit: Context count, query range, and consistency are distinct information boundaries.
#   renderer: plot
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
# These checks define a new experiment. A full recorded run is required before claiming that the model
# meets them. Lower normalized error is better; a value of one matches the stated constant reference. The
# numerical gates are provisional and must be calibrated on separate validation seeds rather than adjusted
# after seeing acceptance-test outcomes.
#
# ## Method
#
# ### Process, controls, and held-out evaluation
#
# Each independent record samples a signed slope with magnitude 0.75-1.75, intercept Uniform(-1,1), equally
# spaced x anchors from-1 to 1, and independent Gaussian observation/target noise with standard deviation
# 0.01. Training/validation have 4096/512 fresh rules, four examples, queries Uniform(-0.8,0.8). Six test
# conditions each have 1024 fresh rules: counts 2/4/8 crossed with interior queries or signed query
# magnitudes 1.5-2.0. Example order is shuffled. Each frozen control arm receives 600 AdamW steps, batch 64, learning
# rate 0.002, weight decay 0.01, FP32, identical initialization and no checkpoint selection. The query-only
# arm removes the example branch, so it is an information control rather than a parameter-matched
# architecture. Every condition retains a distinct normalized-error and corruption gate. Full model state
# must remain unchanged across inference. These provisional checks concern this bounded affine family; they
# do not establish arbitrary algorithm induction.
#

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.10",
#     "lightning",
#     "numpy",
#     "polars",
#     "pyarrow",
#     "pyyaml>=6",
#     "relflow>=0.5.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "computation"
# types = []
# conditions = ["longer-collections", "extrapolation", "missing-inputs", "changed-associations", "varying-collections", "limited-information"]
# ///

"""P097: infer fresh affine rules outside their example range and at new context lengths."""

import sys
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
import torch
from policies import CONTROL
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


class Model(rf.Model):
    query = rf.Number
    answer = rf.Number(mask=True, objective="mse")


class ContextModel(Model):
    examples = rf.Branch(length=8, overflow="error", reduction=None, x=rf.Number, y=rf.Number)


def build(context: bool) -> rf.Model:
    model_type = ContextModel if context else Model
    return model_type(preset=CONTROL.name)


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{key: value for key, value in row.items() if key != "answer"} for row in rows]
    output = pl.DataFrame(model.predict(pa.Table.from_pylist(visible), batch_size=128))["predictions"].to_list()
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


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    from evidence.diagnostics.plots import COLORS, MARKERS, axes_pair

    m = bundle.record["metrics"]
    counts = [2, 4, 8]
    fig, axes = axes_pair(width=10, height=4.8)
    for ax, region, title in zip(
        axes,
        ["inside", "outside"],
        ["Queries inside the context range", "Queries outside the context range"],
        strict=True,
    ):
        series = {
            "Intact context": [m["context"][f"count_{n}/{region}"]["normalized_rmse"] for n in counts],
            "Context swapped across tasks": [
                m["context"][f"count_{n}/{region}"]["swapped_context"]["normalized_rmse"] for n in counts
            ],
            "Query-only model": [m["query_only"][f"count_{n}/{region}"]["normalized_rmse"] for n in counts],
            "Least-squares reference": [
                m["context"][f"count_{n}/{region}"]["least_squares"]["normalized_rmse"] for n in counts
            ],
        }
        for index, (name, values) in enumerate(series.items()):
            ax.plot(counts, values, marker=MARKERS[index], color=COLORS[index], label=name, lw=1.7)
        ax.set(
            xticks=counts,
            xlabel="Visible context examples",
            ylabel="RMSE / task baseline RMSE",
            title=title,
            ylim=(0, 1.65),
        )
        ax.axhline(1, color="gray", ls=":")
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8)
    return (
        fig,
        "Markers are the three evaluated context counts; both panels share the same error scale. The least-squares reference uses the supplied examples directly. The query-only control and swapped context isolate dependence on task information. Extrapolation is scored separately from interpolation, including its unmet checks.",
        {},
    )


if __name__ == "__main__":
    report(run)
