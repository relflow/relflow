# %% [markdown]
# ---
# title: How does Group handle unseen identities?
# categories: [Group]
# description: Controlled research on Group behavior with repeated identities.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-requires:
# - A relflow wheel exporting the unreleased Group datatype.
# evidence-readout:
# - label: Held-out normalized error
#   metric: [unseen_error]
#   format: error
# evidence-group: Representation
# ---
#
# Train with 48 identities and test both those familiar identities and 16 identities never admitted during training. Also test null identities.
#
# ## Example
#
# ```yaml
# entity: entity-0063
# x: 0.5
# y: -6.3
# ```
#
# For the default seed, this identity follows `y = -6 - 0.6x` initially, with
# independently sampled noise omitted. The model sees `entity` and `x` and
# predicts `y`. The evaluator knows the generator; the model does not.
#
# ```{typst}
# //| label: fig-evidence-p104
# //| fig-cap: "A repeated identity and numeric setting predict a response."
# //| fig-alt: "Group identity and numeric input with numeric target."
# #tree(node("observation", kind: "root", children: (
#   node("entity", type: "Group"),
#   node("x", type: "Number"),
#   node("y", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Compare unavailable-content dropout 0 and 0.10. New names share the unavailable assignment, so this measures fallback prediction, not discovery of unseen identity rules. Validation is restricted to the training vocabulary.
# Normalized RMSE divides error by the error of predicting the training mean;
# lower is better, and 1 matches that baseline. ARI of 1 indicates exact
# partition agreement without regard to slot labels.
#
# ## Result
#
# {{< evidence P104 status >}}
#
# This is a provisional research hypothesis, not a recorded CI result. A
# full run uses 2,400 updates per training stage, 4,096 training, 1,024
# validation, and 2,048 held-out rows, 32-bit CPU precision, AdamW at 0.003,
# batch size 128, and EMA 0.99. The four rules have intercepts -6, -2, 2, 6
# and slopes -0.6, 0.6, -0.6, 0.6, with independent noise standard deviation
# 0.1. Experiments use fresh numeric observations across splits.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P104 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P104 script >}}
#
# ### Complete experiment code
#
# %%
"""P104: How does Group handle unseen identities?"""

from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import experiment, report

BUDGET = 2400
BATCH_SIZE = 128
TRAIN_ROWS = 4096
VALID_ROWS = 1024
TEST_ROWS = 2048
INTERCEPTS = np.asarray((-6.0, -2.0, 2.0, 6.0))
SLOPES = np.asarray((-0.6, 0.6, -0.6, 0.6))


def partition(seed):
    return np.random.default_rng(seed).permutation(np.arange(64) % 4)


def records(rows, seed, partition_seed, *, population="balanced", phase="original", shuffled=False):
    rng = np.random.default_rng(seed)
    if population == "rare":
        entities = np.repeat(np.arange(64), np.where(np.arange(64) < 32, 120, 8))
        if rows != len(entities):
            raise ValueError("Rare training population requires 4096 rows")
    elif population == "known":
        entities = np.resize(np.arange(48), rows)
    elif population == "unseen":
        entities = np.resize(np.arange(48, 64), rows)
    else:
        entities = np.resize(np.arange(64), rows)
    rng.shuffle(entities)
    groups = partition(partition_seed)[entities].copy()
    if phase == "changed":
        changed = entities % 4 == 0
        groups[changed] = (groups[changed] + 1) % 4
    x = rng.uniform(-1, 1, rows)
    y = INTERCEPTS[groups] + SLOPES[groups] * x + rng.normal(0, 0.1, rows)
    if shuffled:
        entities = rng.permutation(entities)
    return [
        {"entity": f"entity-{i:04d}", "x": float(value), "y": float(target)}
        for i, value, target in zip(entities, x, y, strict=True)
    ]


def build(kind="group", groups=4, unavailable=0.0):
    fields = {"x": rf.Number, "y": rf.Number(mask=True, objective="mse")}
    if kind == "group":
        fields["entity"] = rf.Group(n_groups=groups, p_unavailable=unavailable)
    elif kind == "category":
        fields["entity"] = rf.Category(p_unavailable=unavailable)
    return rf.Model.xs(**fields)


def train(model, train_rows, valid_rows, seed, steps, accelerator):
    data = rf.SyntheticDataModule(train=partial(iter, train_rows), validate=partial(iter, valid_rows), seed=seed)
    model.fit(
        data,
        batch_size=BATCH_SIZE,
        learning_rate=0.003,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_steps=BUDGET if steps is None else min(steps, BUDGET),
        max_epochs=-1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=[rf.EMA(decay=0.99)],
    )
    model.eval()
    return model


def predictions(model, rows):
    visible = [{"entity": row["entity"], "x": row["x"]} for row in rows]
    return np.asarray(
        [row["y"]["content"] for row in model.predict(pa.Table.from_pylist(visible))["predictions"].to_pylist()]
    )


def scores(model, rows, training_mean):
    target = np.asarray([row["y"] for row in rows])
    predicted = predictions(model, rows)
    rmse = float(np.sqrt(np.mean((target - predicted) ** 2)))
    baseline = float(np.sqrt(np.mean((target - training_mean) ** 2)))
    return {"rmse": rmse, "normalized_rmse": rmse / baseline, "baseline_rmse": baseline, "rows": len(rows)}


def assignments(model):
    return rf.Group.assignments(model, rf.Address("entity"))


def adjusted_rand(actual, predicted):
    _, left = np.unique(actual, return_inverse=True)
    _, right = np.unique(predicted, return_inverse=True)
    cells = np.zeros((left.max() + 1, right.max() + 1), dtype=np.int64)
    np.add.at(cells, (left, right), 1)

    def pairs(values):
        return float(np.sum(values * (values - 1) / 2))

    joint, a, b = pairs(cells), pairs(cells.sum(axis=1)), pairs(cells.sum(axis=0))
    expected = a * b / (len(actual) * (len(actual) - 1) / 2)
    maximum = (a + b) / 2
    return 1.0 if maximum == expected else (joint - expected) / (maximum - expected)


def assignment_scores(model, partition_seed):
    values = assignments(model)
    ordered = [values.get(f"entity-{i:04d}", {}).get("group", -1) for i in range(64)]
    return {
        "adjusted_rand": adjusted_rand(partition(partition_seed), np.asarray(ordered)),
        "occupied_groups": len(set(ordered) - {-1}),
        "assigned": sum(x >= 0 for x in ordered),
        "assignments": ordered,
        "expected_partition": partition(partition_seed).tolist(),
    }


@experiment("P104", seed=10401)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    ps = seed * 100
    train_rows = records(TRAIN_ROWS, seed * 100 + 1, ps, population="known")
    valid_rows = records(VALID_ROWS, seed * 100 + 2, ps, population="known")
    known = records(TEST_ROWS, seed * 100 + 3, ps, population="known")
    unseen = records(TEST_ROWS, seed * 100 + 4, ps, population="unseen")
    null = [dict(r, entity=None) for r in unseen]
    mean = float(np.mean([r["y"] for r in train_rows]))
    metrics = {}
    for arm, unavailable in [("no_dropout", 0.0), ("dropout", 0.10)]:
        lit.seed_everything(seed, workers=True)
        model = train(build(unavailable=unavailable), train_rows, valid_rows, seed, steps, accelerator)
        before = assignments(model)
        known_result, unseen_result, null_result = (
            scores(model, known, mean),
            scores(model, unseen, mean),
            scores(model, null, mean),
        )
        after = assignments(model)
        probe = [{"entity": "unseen-a", "x": 0.25, "y": 0.0}, {"entity": "unseen-b", "x": 0.25, "y": 0.0}]
        pair = predictions(model, probe)
        metrics[arm] = {
            "known": known_result,
            "unseen": unseen_result,
            "null": null_result,
            "vocabulary_before": len(before),
            "vocabulary_after": len(after),
            "unknown_pair_difference": float(abs(pair[0] - pair[1])),
            "assignments_preserved": before == after,
        }
    metrics["unseen_error"] = metrics["dropout"]["unseen"]["normalized_rmse"]
    checks = {
        "Unknown identities do not grow or mutate vocabulary": all(
            v["vocabulary_before"] == 48 and v["assignments_preserved"] for v in metrics.values() if isinstance(v, dict)
        ),
        "Unknown names share the unavailable prediction": all(
            v["unknown_pair_difference"] < 1e-6 for v in metrics.values() if isinstance(v, dict)
        ),
        "Unavailable dropout keeps unseen error at most 1.25 baseline": metrics["unseen_error"] <= 1.25,
    }
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
