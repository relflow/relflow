# %% [markdown]
# ---
# title: Can perfect accuracy hide answers it cannot name?
# categories:
# - Vocabulary
# description: Ask a Category output to predict a mix of familiar and unfamiliar target labels.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
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
# ```{typst}
# //| label: fig-evidence-unknown-target-coverage
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with record, x, hint, label. Amber cards are hidden prediction targets. Other fields provide the input."
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
# {{< evidence P056 status >}}
#
# Known-only accuracy can look perfect while half the answers are impossible to emit. Report coverage and all-row accuracy alongside it; confidence need not warn about the missing labels.
#
# Confidence is normalized across the two populated output labels. It can therefore be high even when
# neither name is the required novel answer. Recognizing a string in the separate hint vocabulary does
# not add it to the output vocabulary. These measurements concern answer coverage, rather than
# recognizing that a prediction should be rejected.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P056 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P056 script >}}
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
# The model uses the [xs preset](../../concepts/model-tree.qmd#choose-a-size).
#
# Fit for 350 AdamW updates at learning rate 0.002. Require at least 0.95
# known-only accuracy, exactly 50% output-label coverage, and zero accuracy on
# OOV targets. All-row accuracy must therefore be at most 0.5. Unknown targets
# should still attract confident known-label predictions on these deliberately
# unchanged inputs. Permuting x must reduce known-label accuracy toward chance.
# Call the actual test loop to compare its content metric with external scoring.
# Vocabularies, counts, and numerical normalization must stay frozen throughout.
#
# ### Complete experiment code
#

# %%
"""P056: a learned classifier exposes the difference between accuracy and vocabulary coverage."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
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


def build() -> rf.Model:
    return rf.Model.xs(
        x=rf.Number,
        hint=rf.Category(p_unavailable=0.0),
        label=rf.Category(p_unavailable=0.0, topk=[2], mask=True),
    )


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


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
