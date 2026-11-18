# %% [markdown]
# ---
# title: Can ordered outputs fill ranks never shown during training?
# description: Compare Ordinal, rounded Number, and Category targets when integer ranks or declared intermediate labels
#   are withheld from training.
# evidence-readout:
# - label: Unseen integer accuracy
#   metric:
#   - domains
#   - integer
#   - ordinal
#   - unseen
#   - exact_accuracy
#   format: percent
# - label: Unseen ordered-label accuracy
#   metric:
#   - domains
#   - labels
#   - ordinal
#   - unseen
#   - exact_accuracy
#   format: percent
# - label: Number integer accuracy
#   metric:
#   - domains
#   - integer
#   - number
#   - unseen
#   - exact_accuracy
#   format: percent
# evidence-group: Representation
# evidence-review:
#   question: Can ordered outputs fill ranks never shown during training?
#   modeling: Give a noisy numeric clue to an ordered answer and withhold intermediate target ranks during training.
#     Compare Ordinal, rounded Number, and Category on the same unseen integer ranks. All test ranks lie within the
#     observed numeric range.
#   comparison:
#     metric: Accuracy
#     kind: Controlled comparison
#     conditions:
#     - label: Category · unseen integer ranks
#       path:
#       - domains
#       - integer
#       - category
#       - unseen
#       - exact_accuracy
#       format: percent
#     - label: Rounded Number · unseen integer ranks
#       path:
#       - domains
#       - integer
#       - number
#       - unseen
#       - exact_accuracy
#       format: percent
#     - label: Ordinal · unseen integer ranks
#       path:
#       - domains
#       - integer
#       - ordinal
#       - unseen
#       - exact_accuracy
#       format: percent
#     definition: Fraction of predictions matching the expected answer, over the population named in each condition.
#     direction: Higher is better
#     domain:
#     - 0
#     - 1
#   note: This is interpolation. Declared ordered labels are a separate condition, and Category cannot emit labels
#     absent from its learned output vocabulary.
# evidence-plot:
#   target: rank
#   focus: Can an ordered decoder fill ranks absent from training?
#   comparison: Separate familiar and withheld ranks for Ordinal, rounded Number, and Category outputs.
#   limit: Withheld ranks lie inside the observed range. One seed does not establish arbitrary extrapolation; Category
#     has no output classes for the withheld answers.
#   renderer: plot
# ---
#
# Compare Ordinal, rounded Number, and Category targets when integer ranks or declared intermediate
# labels are withheld from training.
#
# ## Example
#
# ```yaml
# signal: 2.02
# rank: 2
# ```
#
# In the integer task, training answers are odd ranks from -9 to 9. A new signal near 2 should still
# permit the even rank 2. It is not an unknown vocabulary label for an implicit Ordinal.
#
# The second task declares five ordered labels but trains only on positions zero, two, and four.
# Positions one and three are held out. Merely declaring an order does not guarantee
# the model will learn to interpolate it.
#
# ## Comparison
#
# Keep numeric evidence, initialization seed, sample counts, and updates matched across Ordinal, rounded
# Number, and Category decoders. Category can name only the target classes actually learned, providing
# an explicit output-vocabulary information boundary. Evaluate familiar and withheld ranks separately.
#
# ## Result
#
# The numerical checks below are provisional, predeclared hypotheses, not measured successes. No full
# learning result is claimed until a matching run is recorded. Smoke runs establish execution only.
#
# ## Method
#
# ### Population, controls, and scoring
#
# Each of six fits receives 4,096 training, 512 validation, and 2,048 independent test rows per
# condition, with 400 updates. Inputs equal the generating rank plus Gaussian noise of standard
# deviation 0.03. Integer training ranks are odd and test ranks even, all within the observed numeric
# range; this is interpolation, not arbitrary extrapolation. Declared labels have ranks 0 through 4 with
# gaps preserved. Number outputs are rounded with ties to even, matching Ordinal decoding. Report exact
# accuracy and rank error. These gates are provisional and do not establish that order is appropriate
# for every categorical domain.
#

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.10",
#     "lightning",
#     "numpy",
#     "pyarrow",
#     "pyyaml>=6",
#     "relflow>=0.5.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "representation"
# types = ["Number", "Ordinal", "Category"]
# conditions = ["unseen-values", "representations"]
# ///

"""P100: interpolate integer ranks and declared ordered labels absent from training."""

import sys
from collections.abc import Iterator
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from policies import CONTROL
from reporting import experiment, report

BUDGET = 400
TRAIN_ROWS = 4096
TEST_ROWS = 2048
LABELS = ("very_low", "low", "medium", "high", "very_high")


def records(*, rows: int, seed: int, domain: str, novel: bool, arm: str) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    ranks = np.arange(-8, 9, 2) if novel else np.arange(-9, 10, 2)
    if domain == "labels":
        ranks = np.asarray([1, 3] if novel else [0, 2, 4])
    selected = rng.choice(ranks, rows)
    for rank, error in zip(selected, rng.normal(0, 0.03, rows), strict=True):
        value = int(rank)
        target = LABELS[value] if domain == "labels" and arm != "number" else value
        if arm == "category" and domain == "integer":
            target = f"rank_{value}"
        yield {"signal": float(value + error), "rank": target}


def build(domain: str, arm: str) -> rf.Model:
    if arm == "ordinal":
        target = (
            rf.Ordinal(values=LABELS, mask=True, objective="mse")
            if domain == "labels"
            else rf.Ordinal(mask=True, objective="mse")
        )
    elif arm == "number":
        target = rf.Number(mask=True, objective="mse")
    else:
        target = rf.Category(mask=True)

    class Model(rf.Model):
        signal = rf.Number
        rank = target

    model = Model(preset=CONTROL.name)
    return model


def prediction(model: rf.Model, rows: list[dict], domain: str, arm: str) -> np.ndarray:
    output = model.predict(pa.Table.from_pylist([{"signal": row["signal"]} for row in rows]), batch_size=256)[
        "predictions"
    ].to_list()
    values = [row["rank"]["content"] for row in output]
    if arm == "category":
        values = [item["value"] for item in values]
    if domain == "labels" and arm != "number":
        return np.asarray([LABELS.index(value) for value in values], dtype=np.int64)
    if arm == "category":
        return np.asarray([int(value.removeprefix("rank_")) for value in values], dtype=np.int64)
    return np.rint(np.asarray(values, dtype=np.float64)).astype(np.int64)


def actual(rows: list[dict], domain: str, arm: str) -> np.ndarray:
    if domain == "labels" and arm != "number":
        return np.asarray([LABELS.index(row["rank"]) for row in rows], dtype=np.int64)
    if arm == "category":
        return np.asarray([int(row["rank"].removeprefix("rank_")) for row in rows], dtype=np.int64)
    return np.asarray([row["rank"] for row in rows], dtype=np.int64)


def evaluate(model: rf.Model, rows: list[dict], domain: str, arm: str) -> dict:
    """Score each held-out rank and retain the same readouts for diagnostic plots."""
    predicted = prediction(model, rows, domain, arm)
    expected = actual(rows, domain, arm)
    return {
        "exact_accuracy": float(np.mean(predicted == expected)),
        "rank_mae": float(np.mean(np.abs(predicted - expected))),
        "ranks": {
            str(rank): {
                "rows": int(np.sum(expected == rank)),
                "exact_accuracy": float(np.mean(predicted[expected == rank] == rank)),
                "rank_mae": float(np.mean(np.abs(predicted[expected == rank] - rank))),
            }
            for rank in np.unique(expected)
        },
    }


def fit(model: rf.Model, train: pa.Table, validate: pa.Table, seed: int, steps: int | None, accelerator: str) -> int:
    """Fit the declared fixed-data protocol without checkpoint selection."""
    data = rf.ArrowDataModule(train=train, validate=validate, num_workers=0, persistent_workers=False, pin_memory=False)
    model.fit(
        data,
        batch_size=64,
        learning_rate=0.003,
        weight_decay=0.01,
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
    model.eval()
    return int(model.trainer.global_step)


@experiment("P100", seed=10001)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    metrics = {"budget": BUDGET, "domains": {}}
    checks = {}
    for domain in ("integer", "labels"):
        metrics["domains"][domain] = {}
        for arm in ("ordinal", "number", "category"):
            lit.seed_everything(seed, workers=True)
            model = build(domain, arm)
            train = list(records(rows=TRAIN_ROWS, seed=seed + 1, domain=domain, novel=False, arm=arm))
            validation = list(records(rows=512, seed=seed + 2, domain=domain, novel=False, arm=arm))
            updates = fit(
                model, pa.Table.from_pylist(train), pa.Table.from_pylist(validation), seed, steps, accelerator
            )
            scores = {"steps": updates}
            for condition, novel in (("seen", False), ("unseen", True)):
                rows = list(records(rows=TEST_ROWS, seed=seed + 3, domain=domain, novel=novel, arm=arm))
                scores[condition] = evaluate(model, rows, domain, arm)
            metrics["domains"][domain][arm] = scores
            checks[f"{domain}, {arm}: full budget retained"] = updates == BUDGET
            checks[f"{domain}, {arm}: seen accuracy at least 0.85"] = scores["seen"]["exact_accuracy"] >= 0.85
            if arm != "category":
                checks[f"{domain}, {arm}: unseen rank accuracy at least 0.80"] = (
                    scores["unseen"]["exact_accuracy"] >= 0.80
                )
            else:
                checks[f"{domain}: Category cannot name unseen target classes"] = (
                    scores["unseen"]["exact_accuracy"] == 0.0
                )
    return metrics, checks


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    from evidence.diagnostics.data import human
    from evidence.diagnostics.plots import dots

    "Show familiar and withheld rank accuracy without averaging away individual gaps."
    measured = bundle.record["metrics"]["domains"]
    arms = ("ordinal", "number", "category")
    labels = ("Ordinal", "Rounded Number", "Category")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained", width_ratios=(1, 1.5))
    for row, domain in enumerate(("integer", "labels")):
        scores = measured[domain]
        dots(
            axes[row, 0],
            ("Familiar ranks", "Withheld ranks"),
            {
                label: [scores[arm][condition]["exact_accuracy"] for condition in ("seen", "unseen")]
                for arm, label in zip(arms, labels, strict=True)
            },
            xlabel="Exact rank accuracy",
            percent=True,
        )
        axes[row, 0].set(
            title="Integer ranks" if domain == "integer" else "Declared ordered labels", xlim=(-0.03, 1.03)
        )
        ranks = sorted(
            {int(rank) for condition in ("seen", "unseen") for rank in scores["ordinal"][condition]["ranks"]}
        )
        withheld = {int(rank) for rank in scores["ordinal"]["unseen"]["ranks"]}
        matrix = np.array(
            [
                [
                    scores[arm]["unseen" if rank in withheld else "seen"]["ranks"][str(rank)]["exact_accuracy"]
                    for rank in ranks
                ]
                for arm in arms
            ]
        )
        image = axes[row, 1].imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axes[row, 1].set(
            yticks=range(3),
            yticklabels=labels,
            xticks=range(len(ranks)),
            xticklabels=[str(rank) + ("*" if rank in withheld else "") for rank in ranks],
            xlabel="Actual rank (* withheld from training)",
            title="Where does interpolation succeed?",
        )
        axes[row, 1].tick_params(axis="x", labelsize=8)
        fig.colorbar(
            image, ax=axes[row, 1], fraction=0.035, format=PercentFormatter(1), label="Exact accuracy within rank"
        )
    examples = []
    for frame in bundle.frames:
        if frame["function"] != "evaluate":
            continue
        context = {
            **bundle.context(frame),
            **{key: bundle.value(frame["values"].get(key)) for key in ("domain", "arm")},
        }
        predicted, expected, rows = bundle.values(frame, "predicted", "expected", "rows")
        if predicted is None or expected is None or rows is None:
            raise ValueError(
                f"{bundle.manifest['id']}: evaluated ranks must retain predictions, expected ranks and source rows"
            )
        examples.append(
            {
                "label": f"{human(context['domain'])} · {human(context['arm'])} · {human(context['condition'])}",
                "field": "rank",
                "population": "test",
                "n": len(expected),
                "records": [
                    {"coordinate": [i], "input": rows[i], "actual": int(expected[i]), "predicted": int(predicted[i])}
                    for i in range(min(16, len(expected)))
                ],
            }
        )
    if len(examples) != 12:
        raise ValueError(f"{bundle.manifest['id']}: expected all twelve held-out rank populations, got {len(examples)}")
    note = "Each decoder receives the same signal and 400 updates. Each familiar/withheld test population has 2,048 rows; heatmap cells score every row at that rank. Asterisks mark answers excluded from training. Category cannot emit those absent target classes. Rank positions 0–4 map to the five declared labels. This tests interpolation within the training range, with one seed and no uncertainty interval."
    return (fig, note, {"kind": "numeric", "conditions": examples})


if __name__ == "__main__":
    report(run)
