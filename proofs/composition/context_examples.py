# %% [markdown]
# ---
# title: Infer an Affine Rule From Context Examples
# categories: [Learning from context]
# proof-id: P072
# description: A frozen model receives examples of a new per-record numerical rule and answers a fresh query.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P072 status >}}
#
# ## Insights
#
# Every record defines a fresh rule using four input/output examples. The model
# must infer its rule at prediction time with fixed weights. A held-out region
# of task parameters distinguishes this from new rows of already sampled tasks;
# context swapping and a separately trained query-only model test whether the
# examples supply useful information. Least squares on the visible examples
# establishes how much information is actually available.
#
# Across three full CPU seeds, familiar-quadrant normalized RMSE was
# 0.039–0.066, versus 0.976–0.989 for query-only models and 1.372–1.399 after
# swapping examples. Weights stayed fixed during prediction. The withheld
# positive/positive quadrant was unstable: normalized RMSE ranged from 0.362
# to 0.907, and only seed 7203 met every gate. Seeds 7201 and 7202 failed both
# the withheld error threshold and the requirement to halve query-only error.
# Visible-example least squares achieved source-unit RMSE 0.0114–0.0119 on
# both splits, showing the required information was present in the examples.
#
# These results support inference within familiar parts of a bounded affine
# family, while reliable transfer to the excluded task region remains unproven.
# They do not establish arbitrary program induction. The original gates and
# training budget are unchanged, and thresholds remain provisional.

# %%
"""P072: infer a fresh bounded affine rule from noisy examples without weight updates."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P072"
BUDGET = 600
TRAIN_ROWS = 4096
TEST_ROWS = 2048
QUADRANTS = ((-1, -1), (-1, 1), (1, -1))

# %% [markdown]
# ## Rules, observations, and splits
#
# Independently for each record, sample slope magnitude Uniform(0.75, 1.75)
# and intercept magnitude Uniform(0.25, 1). Training draws the three sign
# combinations (-,-), (-,+), and (+,-). The entirely withheld quadrant (+,+)
# is used only for testing. Thus each sign and each magnitude is familiar,
# but their positive/positive combination is absent from training and validation.
# Four example x values are jittered anchors around -1, -1/3, 1/3, and 1;
# their y values equal `slope*x + intercept + Normal(0, 0.01)`.
# Query x is Uniform(-0.8, 0.8), independently drawn; the hidden answer uses
# the same rule and independent Normal(0, 0.01) noise. Neither slope nor
# intercept is included in the observation. The query lies between examples.
#
# ```yaml
# examples:
#   - {x: -1.0, y: -0.5}
#   - {x: -0.3, y: 0.2}
#   - {x: 0.3, y: 0.8}
#   - {x: 1.0, y: 1.5}
# query: 0.6
# answer: 1.1
# ```
#
# These examples encode a withheld positive/positive rule, approximately
# `y=x+0.5`. Another record gives a different answer for the identical query:
#
# ```yaml
# examples:
#   - {x: -1.0, y: 1.5}
#   - {x: -0.3, y: 0.8}
#   - {x: 0.3, y: 0.2}
#   - {x: 1.0, y: -0.5}
# query: 0.6
# answer: -0.1
# ```
#
# Swapping the context while keeping the first query and answer deliberately
# breaks the per-record rule. The evaluation control has this form:
#
# ```yaml
# examples:
#   - {x: -1.0, y: 1.5}
#   - {x: -0.3, y: 0.8}
#   - {x: 0.3, y: 0.2}
#   - {x: 1.0, y: -0.5}
# query: 0.6
# answer: 1.1
# ```
#
# Independent streams supply 4,096 training rules, 512 validation rules,
# 2,048 familiar-quadrant test rules, and 2,048 withheld-quadrant test rules.
# No task or example is reused across splits. A continuous rule family avoids
# a finite task lookup table, while the quadrant split supplies a concrete
# extrapolation of task combinations rather than extrapolation of query range.
#
# ```{typst}
# //| label: fig-proof-context-examples
# //| fig-cap: "Four examples describe a fresh rule, and the query requests its answer."
# //| fig-alt: "A task has four repeated examples containing Number x and y, a visible Number query, and a hidden Number answer."
# #tree(node("task", kind: "root", children: (
#   node("examples", kind: "branch", repeated: true, children: (
#     node("x", type: "Number"), node("y", type: "Number"),
#   )),
#   node("query", type: "Number"),
#   node("answer", kind: "target", type: "Number"),
# )))
# ```


# %%
def records(*, rows: int, seed: int, withheld: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    signs = ((1, 1),) if withheld else QUADRANTS
    quadrants = np.resize(np.arange(len(signs)), rows)
    rng.shuffle(quadrants)
    for quadrant in quadrants:
        slope_sign, intercept_sign = signs[quadrant]
        slope = slope_sign * rng.uniform(0.75, 1.75)
        intercept = intercept_sign * rng.uniform(0.25, 1.0)
        example_x = np.linspace(-1.0, 1.0, 4) + rng.uniform(-0.04, 0.04, 4)
        example_y = slope * example_x + intercept + rng.normal(0.0, 0.01, 4)
        order = rng.permutation(4)
        query = rng.uniform(-0.8, 0.8)
        yield {
            "examples": [{"x": float(example_x[i]), "y": float(example_y[i])} for i in order],
            "query": float(query),
            "answer": float(slope * query + intercept + rng.normal(0.0, 0.01)),
        }


def build(*, context: bool = True) -> rf.Model:
    fields = {"query": rf.Number, "answer": rf.Number(mask=True, objective="mse")}
    if context:
        fields["examples"] = rf.Branch(length=4, reduction=None, x=rf.Number, y=rf.Number)
    model = rf.Model.xs(
        batch_size=128,
        fields=fields,
    )
    model.optimizer = rf.adamw(learning_rate=0.002)
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{key: value for key, value in row.items() if key != "answer"} for row in rows]
    output = model.predict(pa.Table.from_pylist(visible))["predictions"].to_pylist()
    return np.asarray([row["/answer"]["content"] for row in output])


def swap(rows: list[dict], seed: int) -> list[dict]:
    """Exchange complete example sets; preserve each query and its original target."""
    order = np.random.default_rng(seed).permutation(len(rows))
    return [{**row, "examples": rows[index]["examples"]} for row, index in zip(rows, order, strict=True)]


def least_squares(rows: list[dict]) -> np.ndarray:
    """Fit each rule only from its visible examples, without generator parameters."""
    x = np.asarray([[example["x"] for example in row["examples"]] for row in rows])
    y = np.asarray([[example["y"] for example in row["examples"]] for row in rows])
    centered = x - x.mean(axis=1, keepdims=True)
    slope = (centered * y).sum(axis=1) / (centered**2).sum(axis=1)
    query = np.asarray([row["query"] for row in rows])
    return y.mean(axis=1) + slope * (query - x.mean(axis=1))


def score(predicted: np.ndarray, rows: list[dict], training_mean: float) -> dict:
    target = np.asarray([row["answer"] for row in rows])
    rmse = float(np.sqrt(np.mean((predicted - target) ** 2)))
    baseline = float(np.sqrt(np.mean((training_mean - target) ** 2)))
    return {"rmse": rmse, "baseline_rmse": baseline, "normalized_rmse": rmse / baseline}


# %% [markdown]
# ## Training and predeclared gates
#
# The context model and query-only model each receive 600 xs AdamW updates at
# learning rate 0.002, batch size 128, with identical observations and seeds.
# Removing the example branch reduces the control's parameters; it intentionally
# measures what the query alone permits, rather than matching unused capacity.
# The final update is evaluated once without checkpoint selection or adaptation.
#
# Normalize RMSE by the training-mean predictor evaluated on each test split.
# Require context normalized RMSE below 0.35 on familiar quadrants and 0.40 on
# the withheld quadrant. On familiar quadrants, context swapping must exceed
# 0.80 and double the intact error, and the query-only model must exceed 0.75.
# Both splits must improve on query-only error by at least a factor of two.
# Least-squares RMSE must be below 0.03 in source units on both splits, confirming
# observability independently of neural optimization. Swapping within the
# withheld quadrant remains diagnostic: its slope and intercept signs are
# constant, so such a swap leaves useful information intact.
#
# Snapshot every model parameter before prediction and require exact equality
# afterward. This checks the limited no-weight-update claim, alongside finite
# outputs. It does not describe a training-free system: meta-training on many
# affine tasks is what may produce this behavior.


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    tests = {
        "familiar": list(records(rows=TEST_ROWS, seed=seed * 100 + 3)),
        "withheld": list(records(rows=TEST_ROWS, seed=seed * 100 + 4, withheld=True)),
    }
    training_mean = float(np.mean([row["answer"] for row in records(rows=TRAIN_ROWS, seed=seed * 100 + 1)]))
    metrics = {"train_rows": TRAIN_ROWS, "test_rows_per_split": TEST_ROWS, "training_mean": training_mean}
    checks = {}
    for context in (True, False):
        lit.seed_everything(seed, workers=True)
        model = build(context=context)
        data = rf.SyntheticDataModule(
            model=model,
            train=partial(records, rows=TRAIN_ROWS, seed=seed * 100 + 1),
            validate=partial(records, rows=512, seed=seed * 100 + 2),
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
        snapshot = [parameter.detach().clone() for parameter in model.parameters()]
        arm = "context" if context else "query_only"
        measured = {"steps": trainer.global_step, "parameters": sum(value.numel() for value in snapshot)}
        for name, rows in tests.items():
            output = predict(model, rows)
            measured[name] = score(output, rows, training_mean)
            checks[f"{arm}: {name} predictions are finite"] = bool(np.isfinite(output).all())
            if context:
                control = predict(model, swap(rows, seed=seed * 100 + 5))
                measured[name]["swapped_context"] = score(control, rows, training_mean)
                oracle = score(least_squares(rows), rows, training_mean)
                measured[name]["least_squares"] = oracle
                checks[f"{name}: visible-example least-squares RMSE below 0.03"] = oracle["rmse"] < 0.03
                checks[f"{name}: swapped predictions are finite"] = bool(np.isfinite(control).all())
        checks[f"{arm}: prediction leaves every weight unchanged"] = all(
            torch.equal(before, after) for before, after in zip(snapshot, model.parameters(), strict=True)
        )
        metrics[arm] = measured
    for name, gate in (("familiar", 0.35), ("withheld", 0.40)):
        context_error = metrics["context"][name]["normalized_rmse"]
        control_error = metrics["query_only"][name]["normalized_rmse"]
        checks[f"{name}: context normalized RMSE below {gate:.2f}"] = context_error < gate
        checks[f"{name}: context more than halves query-only RMSE"] = context_error < control_error / 2
    familiar = metrics["context"]["familiar"]
    swapped = familiar["swapped_context"]["normalized_rmse"]
    checks["Familiar context swapping exceeds 0.80 normalized RMSE"] = swapped > 0.80
    checks["Familiar context swapping more than doubles RMSE"] = swapped > 2 * familiar["normalized_rmse"]
    checks["Familiar query-only normalized RMSE exceeds 0.75"] = (
        metrics["query_only"]["familiar"]["normalized_rmse"] > 0.75
    )
    return metrics, checks


# %% [markdown]
# ## Evidence and limitations
#
# {{< proof P072 evidence >}}
#
# The oracle is deliberately specific to affine rules; neural success would
# establish bounded family inference, not discovery of arbitrary mathematical
# structure. Four examples are always available, query values interpolate their
# range, and the experiment does not establish adaptation to nonlinear rules,
# examples with outliers, or longer reasoning chains. Three CPU seeds have
# been measured; no ten-seed calibration has been run. The withheld-region
# failures remain recorded under the original gates.
#
# ## Reproduce
#
# {{< proof P072 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=7201)
