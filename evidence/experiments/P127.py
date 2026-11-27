# %% [markdown]
# ---
# title: Does withholding rows preserve the distance between two visible events?
# description: Classify the distance between two markers when all intervening events have been withheld.
# evidence-group: Masking
# evidence-readout:
# - {label: Original coordinates, metric: [lowest_accuracy], format: percent}
# - {label: Physically filtered, metric: [filtered_accuracy], format: percent}
# evidence-review:
#   question: Can a model distinguish adjacent visible markers from markers separated by withheld rows?
#   modeling: In every matched pair the two live markers have the same values and order. Only their positions differ. Train a Boolean classifier to recognize two withheld rows between the markers.
#   comparison:
#     kind: Controlled comparison
#     metric: Accuracy
#     conditions:
#     - {label: Coordinates retained, path: [lowest_accuracy], format: percent}
#     - {label: Rows filtered, path: [filtered_accuracy], format: percent}
#     definition: Fraction of balanced labels predicted correctly. The original-coordinate readout reports the lowest accuracy across three independent fits.
#     direction: Higher accuracy indicates that the original spacing is available.
#   note: Matched pairs share the collection length and the first marker position. The proof tests two source distances.
# evidence-plot:
#   renderer: plot
#   focus: Is the distance between markers recoverable after intervening events are withheld?
#   comparison: Score retained coordinates and physically filtered rows against the same held-out labels.
#   limit: Filtering removes all information that distinguishes the two labels.
# ---
#
# ## Example
#
# ```yaml
# events:
#   - {marker: false, hidden: false}
#   - {marker: true, hidden: true}
#   - {marker: false, hidden: true}
#   - {marker: true, hidden: false}
# separated: true
# ```
#
# The visible markers occupy positions zero and three. Removing the two hidden
# records would make them adjacent. An adjacent pair has the opposite label.
#
# ## Comparison
#
# Flip every hidden marker value while preserving its row and measure prediction
# drift. Then physically filter the same hidden rows. Every filtered example
# becomes the same two-event sequence, with half the original labels true.
#
# ## Result
#
# A model that uses spacing must exceed 95 percent accuracy and classify both
# members of more than 90 percent of held-out pairs correctly on each seed.
# Physically filtering the rows must give 50 percent accuracy and flipping
# their hidden values must change probabilities by less than 1e-5.
#
# ## Method
#
# Each of three fits uses independently generated splits of 1,024 training,
# 192 validation, and 512 test pairs. The generator picks a length from six to
# ten and a first marker position that fits both alternatives. It fills the
# other positions with independent fair bits shared across the pair. A preprocessor
# replaces each flagged list element with a null slot on every split. One marker
# pair is adjacent and the other is three positions apart. Only the marker
# value enters the encoder on visible rows. Fit a width-32 model for 320 AdamW
# updates with learning rate 0.002 and batch size 32.
#
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["lightning", "matplotlib>=3.10", "numpy", "polars", "pyyaml>=6", "relflow>=0.5.0", "torch"]
#
# [tool.relflow.evidence]
# capability = "structure"
# types = ["Boolean"]
# conditions = ["hidden-values", "missing-inputs"]
# ///

"""Isolate original marker spacing from visible value, order and sequence length."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
from policies import CONTROL
from reporting import experiment, report
from visibility import TARGET

BUDGET = 320


@rf.preprocess
def redact(frame: pl.DataFrame) -> pl.DataFrame:
    event = pl.element()
    return frame.with_columns(pl.col("events").list.eval(pl.when(~event.struct.field("hidden")).then(event)))


class Model(rf.Model):
    events = rf.Branch(length=10, marker=rf.Boolean)
    separated = rf.Boolean(**TARGET)


def records(seed, pairs):
    rng = np.random.default_rng(seed)
    for _ in range(pairs):
        length = int(rng.integers(6, 11))
        anchor = int(rng.integers(length - 3))
        filler = rng.integers(2, size=length)
        for distance in (1, 3):
            events = [dict(marker=bool(x), hidden=True) for x in filler]
            events[anchor] = dict(marker=False, hidden=False)
            events[anchor + distance] = dict(marker=True, hidden=False)
            yield dict(events=events, separated=distance == 3)


def probabilities(model, rows):
    output = model.predict(pl.DataFrame({"events": [row["events"] for row in rows]}), preprocess=redact, batch_size=64)
    return np.asarray(
        output["predictions"].struct.field("separated").struct.field("content").struct.field("probability")
    )


def trial(seed, steps, accelerator):
    lit.seed_everything(seed, workers=True)
    train = pl.DataFrame(list(records(seed + 1, 1024))).sample(fraction=1, shuffle=True, seed=seed)
    val = pl.DataFrame(list(records(seed + 2, 192)))
    test = list(records(seed + 3, 512))
    model = Model(preset=CONTROL.name, d_model=32)
    data = rf.PolarsDataModule(
        preprocessor=redact,
        train=train,
        validate=val,
        seed=seed,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    model.fit(
        data,
        max_steps=BUDGET if steps is None else min(steps, BUDGET),
        max_epochs=-1,
        batch_size=32,
        learning_rate=0.002,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        accelerator=accelerator,
        strategy="auto",
        devices=1,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    model.eval()
    flipped = [
        {"events": [{**e, "marker": not e["marker"]} if e["hidden"] else e for e in row["events"]]} for row in test
    ]
    filtered = [{"events": [e for e in row["events"] if not e["hidden"]]} for row in test]
    expected = np.array([row["separated"] for row in test])
    intact = probabilities(model, test)
    correct = (intact >= 0.5) == expected
    collapsed = probabilities(model, filtered)
    return dict(
        seed=seed,
        steps=model.trainer.global_step,
        accuracy=float(correct.mean()),
        paired_accuracy=float(correct.reshape(-1, 2).all(axis=1).mean()),
        filtered_accuracy=float(((collapsed >= 0.5) == expected).mean()),
        filtered_spread=float(np.ptp(collapsed)),
        hidden_drift=float(np.max(np.abs(intact - probabilities(model, flipped)))),
    )


@experiment("P127", seed=12601)
def run(seed, steps, accelerator):
    trials = [trial(seed + 10 * i, steps, accelerator) for i in range(3)]
    lowest = min(t["accuracy"] for t in trials)
    return {
        "trials": trials,
        "lowest_accuracy": lowest,
        "filtered_accuracy": float(np.mean([t["filtered_accuracy"] for t in trials])),
    }, {
        "Original coordinates identify the distance for every seed": lowest > 0.95,
        "Both members of each held-out pair are distinguishable": all(t["paired_accuracy"] > 0.90 for t in trials),
        "Physical filtering collapses the balanced classes to chance": all(
            t["filtered_accuracy"] == 0.50 and t["filtered_spread"] < 1e-5 for t in trials
        ),
        "Flipping hidden values preserves every probability": all(t["hidden_drift"] < 1e-5 for t in trials),
    }


def plot(bundle, spec):
    import matplotlib.pyplot as plt

    from evidence.diagnostics.plots import dots

    trials = bundle.record["metrics"]["trials"]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    dots(
        ax,
        [f"Seed {t['seed']}" for t in trials],
        {
            "Original coordinates": [t["accuracy"] for t in trials],
            "Physically filtered": [t["filtered_accuracy"] for t in trials],
        },
        xlabel="Accuracy on balanced test pairs",
        reference=(0.5, "Chance"),
        percent=True,
    )
    return (
        fig,
        "Each pair has identical visible marker values. Physical filtering discards the distance that distinguishes the two labels.",
        {},
    )


if __name__ == "__main__":
    report(run)
