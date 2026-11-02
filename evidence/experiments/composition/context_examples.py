# %% [markdown]
# ---
# title: Can it infer a new rule from examples in the record?
# categories:
# - Learning from context
# evidence-id: P072
# description: Each record supplies four examples of a fresh straight-line rule and asks for the answer at a new input.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Familiar rules
#   metric:
#   - context
#   - familiar
#   - normalized_rmse
#   format: error
# - label: Excluded rules
#   metric:
#   - context
#   - withheld
#   - normalized_rmse
#   format: error
# - label: Excluded rules, direct line fit
#   metric:
#   - context
#   - withheld
#   - least_squares
#   - normalized_rmse
#   format: error
# evidence-group: Composition
# evidence-result:
#   accelerator: cpu
#   checks:
#     Familiar context swapping exceeds 0.80 normalized RMSE: true
#     Familiar context swapping more than doubles RMSE: true
#     Familiar query-only normalized RMSE exceeds 0.75: true
#     'context: familiar predictions are finite': true
#     'context: prediction leaves every weight unchanged': true
#     'context: withheld predictions are finite': true
#     'familiar: context more than halves query-only RMSE': true
#     'familiar: context normalized RMSE below 0.35': true
#     'familiar: swapped predictions are finite': true
#     'familiar: visible-example least-squares RMSE below 0.03': true
#     'query_only: familiar predictions are finite': true
#     'query_only: prediction leaves every weight unchanged': true
#     'query_only: withheld predictions are finite': true
#     'withheld: context more than halves query-only RMSE': true
#     'withheld: context normalized RMSE below 0.40': true
#     'withheld: swapped predictions are finite': true
#     'withheld: visible-example least-squares RMSE below 0.03': true
#   duration_seconds: 64.518
#   metrics:
#     context:
#       familiar:
#         baseline_rmse: 0.8586269648305972
#         least_squares:
#           baseline_rmse: 0.8586269648305972
#           normalized_rmse: 0.01348474940825404
#           rmse: 0.011578369455910358
#         normalized_rmse: 0.080137333036159
#         rmse: 0.06880807503445595
#         swapped_context:
#           baseline_rmse: 0.8586269648305972
#           normalized_rmse: 1.413318910084246
#           rmse: 1.213513726103324
#       parameters: 588114
#       steps: 600
#       withheld:
#         baseline_rmse: 1.0593012958262034
#         least_squares:
#           baseline_rmse: 1.0593012958262034
#           normalized_rmse: 0.01078544031707928
#           rmse: 0.011425030903938259
#         normalized_rmse: 0.2000332098012014
#         rmse: 0.21189543835068744
#         swapped_context:
#           baseline_rmse: 1.0593012958262034
#           normalized_rmse: 0.3491833337533002
#           rmse: 0.36989035792578456
#     query_only:
#       familiar:
#         baseline_rmse: 0.8586269648305972
#         normalized_rmse: 0.9822884104150658
#         rmse: 0.8434193164229599
#       parameters: 410446
#       steps: 600
#       withheld:
#         baseline_rmse: 1.0593012958262034
#         normalized_rmse: 1.0701923399036597
#         rmse: 1.1336561324432235
#     test_rows_per_split: 2048
#     train_rows: 4096
#     training_mean: -0.20962522122372595
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 6f22ddf16a211935a98a2e87a67d383f6feaa84120ebb404b8d394c44f6f7633
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: a9f2482c435a
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
#     script_sha256: 0918d23026eff6a105f9e5f43f3f5ae08c3da98daf4d3b877d08ec8d000a6d2b
#     source_sha256: 7f8fe30ceca2a5fcfc9ce92e171e74bb3c2d1c5c9a957b079263b5e1340542a3
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 7201
#   started_at: '2026-11-01T01:49:04.973955+00:00'
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
# examples:
#   - {x: -1.0, y: -0.5}
#   - {x: -0.3, y: 0.2}
#   - {x: 0.3, y: 0.8}
#   - {x: 1.0, y: 1.5}
# query: 0.6
# answer: 1.1
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-context-examples
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with task, examples, x, y, query, answer. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("task", kind: "root", children: (
#   node("examples", kind: "branch", repeated: true, children: (
#     node("x", type: "Number"), node("y", type: "Number"),
#   )),
#   node("query", type: "Number"),
#   node("answer", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Swap the examples between records, remove them, and compare with fitting a line directly. Test a region of rules excluded from training.
#
# ## Result
#
# {{< evidence P072 status >}}
#
# Examples help in familiar regions, but only one of three seeds passes every check. Reliable transfer to the excluded rule region remains unestablished.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P072 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P072 script >}}
#
# ### Rules, observations, and splits
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
# Independent streams supply 4,096 training rules, 512 validation rules,
# 2,048 familiar-quadrant test rules, and 2,048 withheld-quadrant test rules.
# No task or example is reused across splits. A continuous rule family avoids
# a finite task lookup table, while the quadrant split supplies a concrete
# extrapolation of task combinations rather than extrapolation of query range.
#
# ### Training and predeclared gates
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
#
# ### Complete experiment code
#

# %%
"""P072: infer a fresh bounded affine rule from noisy examples without weight updates."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P072"
BUDGET = 600
TRAIN_ROWS = 4096
TEST_ROWS = 2048
QUADRANTS = ((-1, -1), (-1, 1), (1, -1))


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
        fields=fields,
    )
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{key: value for key, value in row.items() if key != "answer"} for row in rows]
    output = model.predict(pa.Table.from_pylist(visible))["predictions"].to_pylist()
    return np.asarray([row["answer"]["content"] for row in output])


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
            train=partial(records, rows=TRAIN_ROWS, seed=seed * 100 + 1),
            validate=partial(records, rows=512, seed=seed * 100 + 2),
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


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7201)

# %% [markdown]
# </details>
