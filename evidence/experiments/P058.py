# %% [markdown]
# ---
# title: What happens to Set members never seen in training?
# description: Train a Set field on known members without ever teaching it how unavailable members should affect prediction.
# evidence-readout:
# - label: Prediction for empty input
#   metric:
#   - boundary_predictions
#   - empty
#   format: number
# - label: Prediction for all-unknown input
#   metric:
#   - boundary_predictions
#   - all_unknown
#   format: number
# evidence-group: Vocabulary
# evidence-review:
#   question: What does an untrained Set fallback do with entirely new members?
#   modeling: Train on known set members, empty sets, and nulls, without simulating unavailable members. Predict a
#     numeric function of membership. Then compare an empty input with an input containing only unseen member names.
#   comparison:
#     kind: Recorded conditions
#     metric: Predicted value
#     conditions:
#     - label: Prediction for empty input
#       path:
#       - boundary_predictions
#       - empty
#       format: number
#     - label: Prediction for all-unknown input
#       path:
#       - boundary_predictions
#       - all_unknown
#       format: number
#     definition: Model output for each input condition, in target units. These are predictions, not prediction errors.
#     direction: Task-specific interpretation
#   note: 'The checks deliberately expose the untrained fallback: empty and all-unknown inputs can receive the same
#     prediction, although null is distinct. P064 trains that fallback explicitly.'
# evidence-plot:
#   target: labels
#   focus: How do unknown and duplicate set members affect the result?
#   comparison: Inspect numeric predictions and reconstructed sets alongside boundary probes and shuffled labels.
#   limit: Unknown-member fallback and exact set recovery are separate behaviors.
#   renderer: plot
# evidence-cost:
#   baseline: 18
#   candidate: 18
# ---
#
# A Set carries membership rather than order or repetition count. This experiment first teaches a
# small membership rule using only `red`, `blue`, empty sets, and null. It then asks what the model
# retains when an input contains a member absent from the learned vocabulary, while a separate hidden
# Set head tries to reconstruct the members.
#
# ## Example
#
# ```yaml
# tags: [red, never-seen]
# labels: [red, never-seen]
# target: 1.0
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
#
# The illustrated row is a deliberate prediction probe. The known member `red` contributes +1, and
# `never-seen` supplies no learned membership bit, giving the reference numeric answer 1. For familiar
# training patterns, `blue` contributes −1.5, both colors give −0.5, and the empty set gives 0. Null
# is a separate case whose target is 3.
#
# ## Comparison
#
# Compare empty, all-unknown, and partly known sets. Check which member names the output can actually emit.
#
# Adding unknown members and repeating known members preserves the available known-membership
# information. The experiment compares those augmented rows with their originals and also tests empty,
# all-unknown, and null inputs directly. Shuffling whole tag sets between records preserves how often
# each pattern occurs but breaks its connection to the original numeric targets.
#
# ## Result
#
# The untrained fallback treats all-unknown input like empty input here. Unknown output members also cannot be named. The next fallback test shows how training can distinguish nonempty input.
#
# Exact-set accuracy requires the entire predicted membership set to match, rather than rewarding a
# few correct members. An output vocabulary containing only known colors cannot emit `never-seen`.
# Because unavailable members were never simulated during training, equality of the empty and
# all-unknown predictions describes this untrained boundary; it does not establish a useful fallback
# for every deployment.
#
# ## Method
#
# ### Data and model
#
# Each record samples one of five equally likely patterns: empty, red, blue,
# both, or null. `target = 1*red - 1.5*blue` for valued sets and 3 for null.
# A separate masked Set head reconstructs the original membership. Inputs and
# targets have independent, automatically grown vocabularies. Disable simulated
# unavailability to isolate actual OOV behavior. Splits contain 4,096 training,
# 512 validation, and 2,048 test records.
#
# ### Training and controls
#
# Fit for 400 AdamW updates at learning rate 0.002. Known membership must give
# regression nRMSE below 0.15 and at least 0.95 exact-set accuracy on valued
# targets. Shuffling tags removes regression accuracy. Adding OOV members and
# duplicates must leave predictions unchanged. An all-OOV target and an empty
# target must have the same encoded membership bits, without learning labels
# from evaluation. Null versus empty predictions must remain well separated.
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
# types = ["Set"]
# conditions = ["unseen-values", "missing-inputs", "value-states", "changed-associations", "transformations"]
# ///

"""P058: learn known membership while exposing empty/all-OOV and null boundaries."""

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
from reporting import experiment, report

BUDGET = 400
PATTERNS = ((), ("red",), ("blue",), ("red", "blue"), None)


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for index in rng.integers(len(PATTERNS), size=rows):
        pattern = PATTERNS[index]
        tags = None if pattern is None else list(pattern)
        target = 3.0 if pattern is None else float(("red" in pattern) - 1.5 * ("blue" in pattern))
        yield {"tags": tags, "labels": None if tags is None else list(tags), "target": target}


class PredictionModel(rf.Model):
    tags = rf.Set(p_unavailable=0.0)
    labels = rf.Set(p_unavailable=0.0, threshold=0.5, predict=True)
    target = rf.Number(predict=True, objective="mse")


def build() -> rf.Model:
    return PredictionModel(d_model=32, n_layers=1, n_heads=4, dropout=0.0)


def prediction(model: rf.Model, rows: list[dict]) -> tuple[np.ndarray, list[set[str]]]:
    output = pl.DataFrame(model.predict(pa.Table.from_pylist(rows)))["predictions"].to_list()
    values = np.asarray([row["target"]["content"] for row in output], dtype=np.float64)
    labels = [{candidate["value"] for candidate in row["labels"]["content"]} for row in output]
    return values, labels


@experiment("P058", seed=5801)
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
    vocabularies = {name: rf.Set.vocabulary(model, f"/{name}") for name in ("tags", "labels")}
    train = list(records(rows=4096, seed=seed + 1))
    test = list(records(rows=2048, seed=seed + 3))
    actual = np.asarray([row["target"] for row in test])
    baseline = float(np.sqrt(np.mean((actual - np.mean([row["target"] for row in train])) ** 2)))
    values, labels = prediction(model, test)
    augmented = [
        {**row, "tags": None if row["tags"] is None else [*row["tags"], *row["tags"], "never-seen"]} for row in test
    ]
    repeated, repeated_labels = prediction(model, augmented)
    order = np.random.default_rng(seed + 4).permutation(len(test))
    shuffled = [{**row, "tags": test[index]["tags"]} for row, index in zip(test, order, strict=True)]
    broken, _ = prediction(model, shuffled)
    probes = [
        {"tags": [], "labels": [], "target": 0.0},
        {"tags": ["never-seen"], "labels": ["never-seen"], "target": 0.0},
        {"tags": None, "labels": None, "target": 3.0},
        {"tags": ["red", "never-seen"], "labels": ["red", "never-seen"], "target": 1.0},
    ]
    boundary, boundary_labels = prediction(model, probes)
    fields = model.encode(pa.Table.from_pylist(probes), strata="test")
    input_field = fields["/tags"]
    target_bits = fields["/labels"].targets[rf.TensorKey.content]["indices"]
    metrics = {
        "steps": trainer.global_step,
        "vocabularies": vocabularies,
        "baseline_rmse": baseline,
        "nrmse": float(np.sqrt(np.mean((values - actual) ** 2))) / baseline,
        "known_exact_set_accuracy": float(
            np.mean(
                [
                    predicted == set(row["labels"])
                    for row, predicted in zip(test, labels, strict=True)
                    if row["labels"] is not None
                ]
            )
        ),
        "shuffled_nrmse": float(np.sqrt(np.mean((broken - actual) ** 2))) / baseline,
        "unknown_and_duplicate_drift": float(np.max(np.abs(values - repeated))),
        "boundary_predictions": dict(
            zip(("empty", "all_unknown", "null", "partial_unknown"), boundary.tolist(), strict=True)
        ),
        "all_unknown_target_exact_match": boundary_labels[1] == {"never-seen"},
    }
    return metrics, {
        "Known membership regression nRMSE below 0.15": metrics["nrmse"] < 0.15,
        "Known valued sets reconstruct above 0.95 exact accuracy": metrics["known_exact_set_accuracy"] >= 0.95,
        "Shuffled tags remove predictive accuracy": metrics["shuffled_nrmse"] > 0.85,
        "Unknown and duplicate members leave regression unchanged": metrics["unknown_and_duplicate_drift"] < 1e-5,
        "Unknown and duplicate members leave reconstructed sets unchanged": labels == repeated_labels,
        "All-unknown and empty inputs have identical known membership bits": bool(
            input_field.state[:2].eq(rf.Tokens.valued).all()
        )
        and bool((input_field.content["indices"][0] == input_field.content["indices"][1]).all()),
        "All-unknown and empty targets have identical membership bits": bool((target_bits[0] == target_bits[1]).all()),
        "Untrained fallback leaves all-unknown and empty predictions identical": abs(boundary[0] - boundary[1]) < 1e-5
        and boundary_labels[0] == boundary_labels[1],
        "Null remains distinct from valued empty content": abs(boundary[2] - boundary[0]) > 2.0,
        "Unknown target labels cannot be recovered": not metrics["all_unknown_target_exact_match"]
        and all("never-seen" not in row for row in boundary_labels),
        "Evaluation does not grow either vocabulary": all(
            rf.Set.vocabulary(model, f"/{name}") == vocabulary for name, vocabulary in vocabularies.items()
        ),
    }


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    from matplotlib.ticker import PercentFormatter

    from evidence.diagnostics.data import human
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
    dots(
        axes[0],
        ["Intact membership", "Shuffled membership"],
        {"Normalized RMSE": [m["nrmse"], m["shuffled_nrmse"]]},
        xlabel="Numeric prediction RMSE / baseline RMSE",
        reference=(1, "Constant guess"),
    )
    names = ["empty", "all_unknown", "null", "partial_unknown"]
    run = next((frame for frame in reversed(bundle.frames) if frame["function"] == "run"))
    probes = bundle.value(run["values"]["probes"])
    dots(
        axes[1],
        [human(name) for name in names],
        {
            "Prediction": [m["boundary_predictions"][name] for name in names],
            "Declared probe target": [row["target"] for row in probes],
        },
        xlabel="Numeric answer for the fixed membership probes",
    )
    return (
        fig,
        f"Known-valued sets have {m['known_exact_set_accuracy']:.1%} exact reconstruction accuracy. Boundary probes distinguish null from valued empty content. An unavailable target label has no decoder row, even when its input produces a valid numeric answer.",
        {},
    )


if __name__ == "__main__":
    report(run)
