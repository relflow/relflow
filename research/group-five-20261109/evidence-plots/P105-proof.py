# %% [markdown]
# ---
# title: Do Group assignments affect predictions?
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
#   metric: [intervened_error]
#   format: error
# evidence-group: Representation
# ---
#
# Train on four stable hidden rules, then temporarily move each familiar identity to a different learned slot.
#
# ## Example
#
# ```yaml
# entity: entity-0001
# x: 0.5
# y: 1.7
# ```
#
# For the default seed, this identity follows `y = 2 - 0.6x` initially, with
# independently sampled noise omitted. The model sees `entity` and `x` and
# predicts `y`. The evaluator knows the generator; the model does not.
#
# ```{typst}
# //| label: fig-evidence-p105
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
# Compare predictions before, during, and after the override, including a
# representative of the destination slot at the same setting. Rotating
# arbitrary slot IDs does not assume an intrinsic ordering.
# Normalized RMSE divides error by the error of predicting the training mean;
# lower is better, and 1 matches that baseline. ARI of 1 indicates exact
# partition agreement without regard to slot labels.
#
# ## Result
#
# {{< evidence P105 status >}}
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
# {{< evidence P105 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P105 script >}}
#
# ### Complete experiment code
#
# %%
"""P105: Do Group assignments affect predictions?"""

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


def stream(rows):
    yield from rows


def train(model, train_rows, valid_rows, seed, steps, accelerator):
    data = rf.SyntheticDataModule(train=partial(stream, train_rows), validate=partial(stream, valid_rows), seed=seed)
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


@experiment("P105", seed=10501)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    ps = seed * 100
    train_rows = records(TRAIN_ROWS, seed * 100 + 1, ps)
    valid_rows = records(VALID_ROWS, seed * 100 + 2, ps)
    test = records(TEST_ROWS, seed * 100 + 3, ps)
    mean = float(np.mean([r["y"] for r in train_rows]))
    lit.seed_everything(seed, workers=True)
    model = train(build(), train_rows, valid_rows, seed, steps, accelerator)
    before_assignments = assignments(model)
    before = predictions(model, test)
    original = scores(model, test, mean)
    mapping = {label: (item["group"] + 1) % 4 for label, item in before_assignments.items()}
    representatives = {}
    for label, item in before_assignments.items():
        representatives.setdefault(item["group"], label)
    matched = [i for i, row in enumerate(test) if mapping.get(row["entity"]) in representatives]
    references = [dict(test[i], entity=representatives[mapping[test[i]["entity"]]]) for i in matched]
    reference = predictions(model, references) if references else np.asarray([])
    with rf.Group.override(model, rf.Address("entity"), mapping):
        altered = predictions(model, test)
        intervention = scores(model, test, mean)
    after = predictions(model, test)
    target = np.asarray([r["y"] for r in test])
    metrics = {
        "original": original,
        "intervention": intervention,
        "intervened_error": intervention["normalized_rmse"],
        "mean_absolute_prediction_change": float(np.mean(abs(altered - before))),
        "destination_reference_rows": len(matched),
        "destination_max_difference": float(np.max(abs(altered[matched] - reference))) if matched else None,
        "restoration_max_difference": float(np.max(abs(after - before))),
        "assignment_state_restored": before_assignments == assignments(model),
        "partition": assignment_scores(model, ps),
        "mean_absolute_target": float(np.mean(abs(target))),
    }
    checks = {
        "Intervention changes predictions": metrics["mean_absolute_prediction_change"] > 0.1,
        "Intervention worsens held-out normalized error by 0.1": intervention["normalized_rmse"]
        >= original["normalized_rmse"] + 0.1,
        "Predictions follow destination slot representatives": len(matched) == len(test)
        and metrics["destination_max_difference"] is not None
        and metrics["destination_max_difference"] < 1e-5,
        "Override restores original predictions and assignments": metrics["restoration_max_difference"] < 1e-6
        and metrics["assignment_state_restored"],
    }
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
