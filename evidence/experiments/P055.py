# %% [markdown]
# ---
# title: Is an unfamiliar category the same as a null?
# description: Compare familiar labels, new labels, and missing values in a task where identity and numeric context
#   both matter.
# evidence-readout:
# - label: Error on paired new identities
#   metric:
#   - unknown
#   - paired_rmse
#   format: number
# - label: Unavoidable error without identity
#   metric:
#   - unknown
#   - information_limit_rmse
#   format: number
# evidence-group: Vocabulary
# evidence-review:
#   question: Is an unfamiliar category the same as a null?
#   modeling: Generate numeric context x and a category-specific effect, then hide 2x + effect. At test time, introduce
#     two unseen categories with opposite effects. Both map to unknown content, so the model can use x but cannot
#     distinguish those effects. Compare its error with that unavoidable information limit.
#   comparison:
#     kind: Baseline reference
#     metric: RMSE
#     conditions:
#     - label: Unavoidable error without identity
#       path:
#       - unknown
#       - information_limit_rmse
#       format: number
#     - label: Error on paired new identities
#       path:
#       - unknown
#       - paired_rmse
#       format: number
#     definition: Root mean squared prediction error, in the target’s original units. Zero is an exact prediction.
#     direction: Lower error is better
#   note: Unknown category content remains different from a null. The reference is a mathematical information limit,
#     not another trained model.
# evidence-plot:
#   target: target
#   focus: What prediction is available for an unseen input label?
#   comparison: Compare known labels, unknown-label renaming, and null inputs against the fixed baseline.
#   limit: An unknown label can use shared context but has no learned persistent identity row.
#   renderer: plot
# evidence-cost:
#   baseline: 22
#   candidate: 22
# ---
#
# The question is what remains predictable when a categorical identity is unfamiliar. Each row
# supplies a number `x` and a `code`; the hidden numeric target combines a slope shared by every row
# with an effect belonging to that code. This separates learning a general relationship from
# remembering individual labels.
#
# ## Example
#
# ```yaml
# x: 0.4
# code: right
# target: 2.3
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
#
# The illustrated values are a worked calculation, not a measured prediction: `2 × 0.4 + 1.5 = 2.3`
# for `right`. At the same `x`, `left` would give −0.7 and null would give −2.2. A new code has no
# learned individual effect. For two indistinguishable new codes assigned opposite effects, the best
# shared answer is their midpoint, 0.8.
#
# ## Comparison
#
# Rename new labels without changing other inputs. Train with some input labels made unavailable so the model can learn a fallback.
#
# The unknown-code renaming control keeps `x` fixed, so any prediction change would come from an
# identity distinction the vocabulary has never learned. Replacing the code with null is a different
# intervention: null appeared during training and has its own target relationship. Training-time
# unavailable content gives the model practice using the shared numeric fallback.
#
# ## Result
#
# Different new labels share unavailable identity content, while null remains distinct. Other context can support a fallback, but the model cannot infer the new label’s individual effect.
#
# RMSE measures error in target units. The 1.5 unknown-identity limit is the error left even by the
# correct midpoint prediction on this paired probe; getting below the familiar-label error is not the
# appropriate goal. Evaluation also checks that looking at new labels does not silently teach them to
# the vocabulary.
#
# ## Method
#
# ### Data and model
#
# Independent rows draw `x ~ Uniform(-1, 1)` and a code with probabilities
# 0.4 left, 0.4 right, and 0.2 null. The target is `2*x + effect(code)`.
# Training/validation/test use 4,096/512/2,048 rows. At test time, two never-seen
# codes have opposite effects of -1.5 and +1.5. The model cannot distinguish
# their identities, so `2*x` is the paired unknown-code conditional mean.
#
# ### Training and controls
#
# The model uses the frozen Evidence control architecture; its settings are shown in the code.
#
# Use 600 AdamW updates at learning rate 0.002. Known-code and null predictions
# must each achieve nRMSE below 0.25 against their training-mean baseline.
# Unknown predictions must approximate `2*x` below 0.25 nRMSE and stay within
# 15% of the 1.5 RMSE information limit on paired novel identities. Renaming
# an unknown code cannot change predictions; replacing it with null must.
# All evaluation uses frozen vocabularies and normalization statistics.
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
#     "relflow>=0.6.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "reliability"
# types = ["Category"]
# conditions = ["unseen-values", "missing-inputs", "value-states"]
# ///

"""P055: known category effects, unknown-content fallback, and distinct null semantics."""

import sys
from collections.abc import Iterator
from functools import partial
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
from policies import CONTROL
from reporting import experiment, report

BUDGET = 600
EFFECTS = {"left": -1.5, "right": 1.5, None: -3.0}


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    codes = np.asarray(["left", "right", None], dtype=object)
    for x, index in zip(rng.uniform(-1, 1, rows), rng.choice(3, rows, p=[0.4, 0.4, 0.2]), strict=True):
        code = codes[index]
        yield {"x": float(x), "code": code, "target": float(2 * x + EFFECTS[code])}


class Model(rf.Model):
    x = rf.Number
    code = rf.Category(p_unavailable=0.3)
    target = rf.Number(predict=True, objective="mse")


def build() -> rf.Model:

    model = Model(preset=CONTROL.name)
    return model


def prediction(model: rf.Model, rows: list[dict]) -> np.ndarray:
    result = pl.DataFrame(model.predict(pa.Table.from_pylist(rows)))["predictions"].to_list()
    return np.asarray([row["target"]["content"] for row in result], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


@experiment("P055", seed=5501)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    data = rf.SyntheticDataModule(
        train=partial(records, rows=4096, seed=seed + 1),
        validate=partial(records, rows=512, seed=seed + 2),
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
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer
    model.eval()
    train = list(records(rows=4096, seed=seed + 1))
    test = list(records(rows=2048, seed=seed + 3))
    vocabulary = rf.Category.vocabulary(model, "/code")
    counts = rf.Category.counts(model, "/code")
    predicted = prediction(model, test)
    metrics = {"steps": trainer.global_step, "vocabulary": vocabulary}
    checks = {}
    for name, is_null in (("known", False), ("null", True)):
        selected = np.asarray([(row["code"] is None) == is_null for row in test])
        actual = np.asarray([row["target"] for row in test])[selected]
        mean = float(np.mean([row["target"] for row in train if (row["code"] is None) == is_null]))
        baseline = rmse(actual, mean)
        metrics[name] = {
            "rmse": rmse(actual, predicted[selected]),
            "baseline_rmse": baseline,
            "nrmse": rmse(actual, predicted[selected]) / baseline,
        }
        checks[f"{name}: held-out nRMSE below 0.25"] = metrics[name]["nrmse"] < 0.25
    novel = [{**row, "code": "novel-left"} for row in test]
    renamed = [{**row, "code": "novel-right"} for row in test]
    nulls = [{**row, "code": None} for row in test]
    first = prediction(model, novel)
    second = prediction(model, renamed)
    expected = 2 * np.asarray([row["x"] for row in test])
    baseline = rmse(expected, float(np.mean([2 * row["x"] for row in train])))
    metrics["unknown"] = {
        "fallback_nrmse": rmse(expected, first) / baseline,
        "paired_rmse": rmse(np.concatenate((expected - 1.5, expected + 1.5)), np.concatenate((first, second))),
        "information_limit_rmse": 1.5,
        "renaming_drift": float(np.max(np.abs(first - second))),
        "null_prediction_gap": float(np.mean(np.abs(first - prediction(model, nulls)))),
    }
    field = model.encode(pa.Table.from_pylist(novel), strata="test")["/code"]
    checks.update(
        {
            "Unknown codes retain valued state": bool(field.state.eq(rf.Tokens.valued).all()),
            "Unknown content uses the allocation-independent sentinel": bool(field.content.eq(-1).all()),
            "Unknown fallback nRMSE below 0.25": metrics["unknown"]["fallback_nrmse"] < 0.25,
            "Paired novel identities stay near their information limit": 1.5 - 1e-6
            <= metrics["unknown"]["paired_rmse"]
            < 1.5 * 1.15,
            "Unknown spelling cannot alter predictions": metrics["unknown"]["renaming_drift"] < 1e-5,
            "Null remains distinct from unknown content": metrics["unknown"]["null_prediction_gap"] > 2.0,
            "Evaluation does not admit labels or change counts": rf.Category.vocabulary(model, "/code") == vocabulary
            and rf.Category.counts(model, "/code") == counts,
        }
    )
    return metrics, checks


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    import numpy as np

    from evidence.diagnostics.plots import numeric

    pairs = bundle.pairs("target")
    chosen = []
    labels = {
        "novel": "Unseen label: conditional mean",
        "renamed": "Unseen spelling changed",
        "nulls": "Null input: its own conditional mean",
        "test": "Known and null inputs: observed answers",
    }
    for argument in ["novel", "renamed", "nulls", "test"]:
        pair = next((item for item in pairs if item["call"] == f"prediction(model, {argument})"))
        pair = dict(pair, label=labels[argument])
        if argument != "test":
            pair["actual"] = 2 * np.asarray([row["x"] for row in pair["rows"]]) - (3 if argument == "nulls" else 0)
        chosen.append(pair)
    run = next((frame for frame in bundle.frames if frame["function"] == "run"))
    baseline = bundle.value(run["values"]["baseline"])
    measured = np.sqrt(np.mean((chosen[0]["actual"] - chosen[0]["predicted"]) ** 2)) / baseline
    if not np.isclose(measured, bundle.record["metrics"]["unknown"]["fallback_nrmse"]):
        raise ValueError("Unknown fallback plot disagrees with the proof's conditional-mean score")
    fig, note, interactive = numeric(bundle, spec, pairs=chosen)
    note += " For unseen labels the reference is the generator's conditional mean 2 × x; for null it is 2 × x − 3. Copied historical target labels are not the oracle for these changed inputs."
    return (fig, note, interactive)


if __name__ == "__main__":
    report(run)
