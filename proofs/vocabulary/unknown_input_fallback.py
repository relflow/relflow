# %% [markdown]
# ---
# title: Unknown categories are not nulls
# categories: [Vocabulary and OOV]
# proof-id: P055
# description: Learn a fallback from remaining evidence while distinguishing unknown valued categories from nulls.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P055 status >}}
#
# ## Insights
#
# Different unseen Category labels share unavailable content, but retain the
# valued state. They cannot carry different learned semantics. Training with
# `p_unavailable` can teach a fallback from other fields; it does not make
# unseen labels recognizable. Nulls have a separate state and may mean something
# different. This proof measures that boundary, not automatic OOD detection.
#
# ## Setup

# %%
"""P055: known category effects, unknown-content fallback, and distinct null semantics."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
from reporting import report

import relflow as rf

PROOF_ID = "P055"
BUDGET = 600
EFFECTS = {"left": -1.5, "right": 1.5, None: -3.0}

# %% [markdown]
# ## Data and model
#
# Independent rows draw `x ~ Uniform(-1, 1)` and a code with probabilities
# 0.4 left, 0.4 right, and 0.2 null. The target is `2*x + effect(code)`.
# Training/validation/test use 4,096/512/2,048 rows. At test time, two never-seen
# codes have opposite effects of -1.5 and +1.5. The model cannot distinguish
# their identities, so `2*x` is the paired unknown-code conditional mean.
#
# ```yaml
# x: 0.4
# code: right
# target: 2.3
# ```
#
# Known right contributes +1.5.
#
# ```yaml
# x: 0.4
# code: novel-right
# target: 2.3
# ```
#
# The spelling does not teach +1.5. Novel-left and novel-right must produce
# identical predictions for the same x, despite their different labels.
#
# ```yaml
# x: 0.4
# code: null
# target: -2.2
# ```
#
# A null contributes -3.0 in this process; it is not the unknown-valued fallback.
#
# ```{typst}
# //| label: fig-proof-unknown-input-fallback
# //| fig-cap: "The numeric input supports fallback when category content is unavailable."
# //| fig-alt: "Record has Number x, Category code with training-only unavailable-content simulation, and a hidden Number target. Null code remains a separate state."
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number"),
#   node("code", type: "Category", body: [30% unavailable content in training]),
#   node("target", kind: "target", type: "Number"),
# )))
# ```


# %%
def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    codes = np.asarray(["left", "right", None], dtype=object)
    for x, index in zip(rng.uniform(-1, 1, rows), rng.choice(3, rows, p=[0.4, 0.4, 0.2]), strict=True):
        code = codes[index]
        yield {"x": float(x), "code": code, "target": float(2 * x + EFFECTS[code])}


def build() -> rf.Model:
    return rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        batch_size=64,
        x=rf.Number,
        code=rf.Category(p_unavailable=0.3),
        target=rf.Number(mask=True, objective="mse"),
    )


def prediction(model: rf.Model, rows: list[dict]) -> np.ndarray:
    result = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    return np.asarray([row["/target"]["content"] for row in result], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


# %% [markdown]
# ## Training and controls
#
# Use 600 AdamW updates at learning rate 0.002. Known-code and null predictions
# must each achieve nRMSE below 0.25 against their training-mean baseline.
# Unknown predictions must approximate `2*x` below 0.25 nRMSE and stay within
# 15% of the 1.5 RMSE information limit on paired novel identities. Renaming
# an unknown code cannot change predictions; replacing it with null must.
# All evaluation uses frozen vocabularies and normalization statistics.


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    model.optimizer = rf.adamw(learning_rate=0.002)
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, rows=4096, seed=seed + 1),
        validate=partial(records, rows=512, seed=seed + 2),
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


# %% [markdown]
# ## Evidence and remaining work
#
# {{< proof P055 evidence >}}
#
# Gates are provisional, declared before full runs. The fallback is supported by
# balanced training effects and simulated unavailable content, not arbitrary
# unknown-category semantics. Cluster uses a different unavailable representation
# and is not covered by this Category experiment.
#
# ## Reproduce
#
# {{< proof P055 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=5501)
