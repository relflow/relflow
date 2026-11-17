# %% [markdown]
# ---
# title: Can perfect accuracy hide answers it cannot name?
# categories:
# - Vocabulary
# description: Ask a Category output to predict a mix of familiar and unfamiliar target labels.
# evidence-readout:
# - label: Known-label accuracy
#   metric:
#   - known_accuracy
#   format: percent
# - label: All-row accuracy
#   metric:
#   - all_row_accuracy
#   format: percent
# - label: Answers covered
#   metric:
#   - coverage
#   format: percent
# evidence-group: Vocabulary
# evidence-diagram:
#   caption: Amber cards are hidden prediction targets. Other fields provide the input.
#   alt: Model tree with record, x, hint, label. Amber cards are hidden prediction targets. Other fields provide the
#     input.
# evidence-review:
#   question: Can perfect accuracy hide answers relflow cannot name?
#   modeling: Train a Category output to name two classes from numeric evidence. In half the test rows, replace the
#     expected name with an unseen target label while keeping the evidence unchanged. Compare accuracy on known answers
#     with accuracy across every test row.
#   comparison:
#     kind: Recorded conditions
#     metric: Accuracy
#     conditions:
#     - label: Known-label accuracy
#       path:
#       - known_accuracy
#       format: percent
#     - label: All-row accuracy
#       path:
#       - all_row_accuracy
#       format: percent
#     definition: Fraction of predictions matching the expected answer, over the population named in each condition.
#     direction: Higher is better
#     domain:
#     - 0
#     - 1
#   note: These scores have different denominators. Perfect known-answer accuracy can coexist with zero ability to
#     emit the missing target names.
# evidence-plot:
#   target: label
#   focus: How does an unavailable target change total accuracy?
#   comparison: Separate known-target accuracy, all-row accuracy, unavailable-target behavior, and shuffled hints.
#   limit: Conditional accuracy omits unavailable answers; all-row accuracy includes them.
#   renderer: plot
# evidence-cost:
#   baseline: 16
#   candidate: 16
# ---
#
# A classifier can be good at choosing among its available answers while lacking the correct answer
# altogether. This experiment deliberately separates those two properties. The sign of `x` identifies
# a binary class during training, while an independent `hint` field contains strings that may look
# like answer names but carry no useful signal.
#
# ## Example
#
# ```yaml
# x: 0.7
# hint: novel-negative
# label: positive
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```typst
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number"),
#   node("hint", type: "Category", body: [All four strings; independent]),
#   node("label", kind: "target", type: "Category", body: [Only two populated labels]),
# )))
# ```
#
# In the illustrative row, positive `x = 0.7` supports `positive`. The hint `novel-negative` does not
# override that rule: hints are drawn independently of the sign. In the shifted test set, half the
# target names are replaced by their novel counterparts while the numeric inputs stay the same. The
# input pattern remains familiar, but those answer names were never output classes.
#
# ## Comparison
#
# Compare accuracy on known labels with accuracy over every row, and count how many answers the vocabulary can represent.
#
# Coverage counts the fraction of test answers present in the output vocabulary. Known-label accuracy
# scores only that covered fraction; all-row accuracy keeps every test row in the denominator.
# Shuffling `x` preserves its distribution and the label counts while removing the relationship the
# model should have learned. The independent hint should not rescue that broken relationship.
#
# ## Result
#
# Known-only accuracy can look perfect while half the answers are impossible to emit. Report coverage and all-row accuracy alongside it; confidence need not warn about the missing labels.
#
# Confidence is normalized across the two populated output labels. It can therefore be high even when
# neither name is the required novel answer. Recognizing a string in the separate hint vocabulary does
# not add it to the output vocabulary. These measurements concern answer coverage, rather than
# recognizing that a prediction should be rejected.
#
# ## Method
#
# ### Data and model
#
# Training labels are negative or positive according to the sign of x, with
# `abs(x) ~ Uniform(0.2, 1)`. The independent hint takes all four strings,
# including names never used as training targets. Splits have 4,096 training,
# 512 validation, and 2,048 test rows, with exactly balanced signs. On half of
# test rows, rename the target to its novel counterpart without changing x.
# The output vocabulary automatically stores its two discovered training labels.
#
# ### Training and controls
#
# The model uses the frozen Evidence control architecture; its settings are shown in the code.
#
# Fit for 350 AdamW updates at learning rate 0.002. Require at least 0.95
# known-only accuracy, exactly 50% output-label coverage, and zero accuracy on
# OOV targets. All-row accuracy must therefore be at most 0.5. Unknown targets
# should still attract confident known-label predictions on these deliberately
# unchanged inputs. Permuting x must reduce known-label accuracy toward chance.
# Call the actual test loop to compare its content metric with external scoring.
# Vocabularies, counts, and numerical normalization must stay frozen throughout.
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
# capability = "reliability"
# types = ["Category"]
# conditions = ["unseen-values", "changed-associations"]
# ///

"""P056: a learned classifier exposes the difference between accuracy and vocabulary coverage."""

import sys
from collections.abc import Iterator
from functools import partial
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
from policies import CONTROL
from reporting import experiment, report

BUDGET = 350
LABELS = ("negative", "positive", "novel-negative", "novel-positive")


def records(*, rows: int, seed: int, shifted: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    signs = np.arange(rows) % 2
    rng.shuffle(signs)
    magnitudes = rng.uniform(0.2, 1.0, rows)
    hints = rng.integers(0, len(LABELS), rows)
    for index, (sign, magnitude, hint) in enumerate(zip(signs, magnitudes, hints, strict=True)):
        offset = 2 if shifted and index % 2 else 0
        yield {"x": float((2 * sign - 1) * magnitude), "hint": LABELS[hint], "label": LABELS[sign + offset]}


class Model(rf.Model):
    x = rf.Number
    hint = rf.Category(p_unavailable=0.0)
    label = rf.Category(p_unavailable=0.0, topk=[2], mask=True)


def build() -> rf.Model:

    model = Model(preset=CONTROL.name)
    return model


@experiment("P056", seed=5601)
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
    vocabulary = rf.Category.vocabulary(model, "/label")
    hints = rf.Category.vocabulary(model, "/hint")
    counts = rf.Category.counts(model, "/label")
    normalization = rf.Number.normalization(model, "/x")
    test = list(records(rows=2048, seed=seed + 3, shifted=True))
    source = pa.Table.from_pylist(test)
    output = pl.DataFrame(model.predict(source))["predictions"].to_list()
    content = [row["label"]["content"] for row in output]
    actual = np.asarray([row["label"] for row in test])
    predicted = np.asarray([row["value"] for row in content])
    known = np.isin(actual, vocabulary)
    probability = np.asarray([row["probability"] for row in content])
    evaluated = trainer.test(
        model,
        datamodule=rf.SyntheticDataModule(test=partial(records, rows=2048, seed=seed + 3, shifted=True)),
        verbose=False,
    )[0]
    builtin = float(evaluated[".label/test.accuracy.content"])
    order = np.random.default_rng(seed + 4).permutation(len(test))
    corrupted = [{**row, "x": test[index]["x"]} for row, index in zip(test, order, strict=True)]
    broken = pl.DataFrame(model.predict(pa.Table.from_pylist(corrupted)))["predictions"].to_list()
    broken_values = np.asarray([row["label"]["content"]["value"] for row in broken])
    field = model.encode(source, strata="test")["/label"]
    metrics = {
        "steps": trainer.global_step,
        "output_vocabulary": vocabulary,
        "hint_vocabulary": hints,
        "coverage": float(known.mean()),
        "builtin_content_accuracy": builtin,
        "known_accuracy": float((predicted[known] == actual[known]).mean()),
        "all_row_accuracy": float((predicted == actual).mean()),
        "unknown_accuracy": float((predicted[~known] == actual[~known]).mean()),
        "unknown_mean_confidence": float(probability[~known].mean()),
        "shuffled_known_accuracy": float((broken_values[known] == actual[known]).mean()),
        "majority_baseline": float(max(np.mean(actual == label) for label in vocabulary)),
    }
    candidate_labels = {candidate["value"] for row in content for candidate in row["topk"]}
    return metrics, {
        "Known-only content accuracy reaches 0.95": builtin >= 0.95 and metrics["known_accuracy"] >= 0.95,
        "Built-in accuracy agrees with known-only external scoring": abs(builtin - metrics["known_accuracy"]) < 0.02,
        "Coverage is exactly one half": metrics["coverage"] == 0.5,
        "All-row accuracy exposes the coverage ceiling": metrics["all_row_accuracy"] <= 0.5,
        "Unseen output labels cannot be emitted": metrics["unknown_accuracy"] == 0.0,
        "Unknown answers still receive confident known-label predictions": metrics["unknown_mean_confidence"] > 0.8,
        "Shuffling x removes known-label accuracy": 0.40 < metrics["shuffled_known_accuracy"] < 0.60,
        "Predictions and candidates use populated output labels only": set(predicted) <= set(vocabulary)
        and candidate_labels == set(vocabulary),
        "Candidate probabilities normalize over populated labels": all(
            abs(sum(candidate["probability"] for candidate in row["topk"]) - 1.0) < 1e-5 for row in content
        ),
        "Labels known in another field remain unknown here": set(LABELS[2:]) <= set(hints)
        and not set(LABELS[2:]) & set(vocabulary),
        "Unknown target content uses the unavailable sentinel": bool(
            (field.targets[rf.TensorKey.content].reshape(-1).cpu().numpy()[~known] == -1).all()
        ),
        "Evaluation preserves output vocabulary and exposure counts": rf.Category.vocabulary(model, "/label")
        == vocabulary
        and rf.Category.counts(model, "/label") == counts,
        "Evaluation preserves hint vocabulary and numerical moments": rf.Category.vocabulary(model, "/hint") == hints
        and rf.Number.normalization(model, "/x") == normalization,
    }


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    from evidence.diagnostics.plots import COLORS, MARKERS, axes_pair, dots, finite

    m = bundle.record["metrics"]
    fig, axes = axes_pair()
    keys = sorted(
        [key for key in m if key.startswith("known_") and key.split("_")[-1].isdigit()],
        key=lambda key: -int(key.split("_")[1]),
    )
    if keys:
        x = finite([m[key]["coverage.content"] for key in keys])
        for index, (key, label) in enumerate(
            [("accuracy.content", "Accuracy on known answers"), ("accuracy.all", "Accuracy on all answers")]
        ):
            values = [m[k].get(key) for k in keys]
            present = [i for i, value in enumerate(values) if value is not None]
            axes[0].plot(
                x[present], [values[i] for i in present], marker=MARKERS[index], color=COLORS[index], label=label
            )
        axes[0].plot(x, x, ls=":", color="gray", label="Coverage ceiling for perfect known answers")
        axes[0].set(
            title="Coverage and conditional accuracy differ",
            xlabel="Fraction of targets in the vocabulary",
            ylabel="Observed accuracy",
            xlim=(-0.03, 1.03),
            ylim=(-0.03, 1.03),
        )
        axes[0].xaxis.set_major_formatter(PercentFormatter(1))
        axes[0].yaxis.set_major_formatter(PercentFormatter(1))
        axes[0].legend(fontsize=8)
        known = finite([m[key]["targets.known"] for key in keys])
        unknown = finite([m[key]["targets.unavailable"] for key in keys])
        axes[1].barh(range(len(keys)), known, color=COLORS[0], label="Known answers")
        axes[1].barh(range(len(keys)), unknown, left=known, color=COLORS[1], label="Unavailable answers")
        axes[1].set(
            yticks=range(len(keys)),
            yticklabels=[f"{value:.0%} coverage" for value in x],
            xlabel="Selected target count",
            title="Which answers contribute to each score?",
        )
        axes[1].invert_yaxis()
        axes[1].legend(fontsize=8)
        return (
            fig,
            "Undefined conditional accuracy at zero coverage is omitted, rather than replaced by zero. Counts come from the proof's selected-target measurements.",
            {},
        )
    labels = ["Known targets", "All targets", "Unavailable targets", "Known targets, shuffled hints"]
    values = [m["known_accuracy"], m["all_row_accuracy"], m["unknown_accuracy"], m["shuffled_known_accuracy"]]
    dots(axes[0], labels, {"Accuracy": values}, xlabel="Fraction correctly predicted", percent=True)
    axes[0].set_xlim(-0.02, 1.02)
    run = next((frame for frame in reversed(bundle.frames) if frame["function"] == "run"))
    probability, known = bundle.values(run, "probability", "known")
    for mask, label, color in [(known, "Known answers", COLORS[0]), (~known, "Unavailable answers", COLORS[1])]:
        axes[1].hist(probability[mask], bins=np.linspace(0, 1, 26), histtype="step", lw=1.8, color=color, label=label)
    axes[1].set(
        title="Confidence does not establish availability",
        xlabel="Top-answer confidence",
        ylabel="Record count",
        xlim=(0, 1),
    )
    axes[1].legend(fontsize=8)
    return (
        fig,
        "Accuracy conditions and confidence populations stay separate. An unavailable true label cannot be recovered merely by assigning high probability to an available label.",
        {},
    )


if __name__ == "__main__":
    report(run)
