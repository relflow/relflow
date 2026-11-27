# %% [markdown]
# ---
# title: Can a model reconstruct consecutive readings in their original rows?
# description: Hide adjacent readings in an ordered series and test whether each answer still uses its own reference.
# evidence-group: Masking
# evidence-readout:
# - label: Aligned reference
#   metric: [worst_aligned_nrmse]
#   format: error
# - label: Reassigned reference
#   metric: [best_reassigned_nrmse]
#   format: error
# evidence-review:
#   question: Can a pair of withheld readings be recovered at its original positions?
#   modeling: Each row carries a reference and two calibrated readings. Train with sampled two-row spans, row masking, and occasional explicit gaps. At test time withhold both readings across a two-row gap.
#   comparison:
#     kind: Controlled comparison
#     metric: Normalized RMSE
#     conditions:
#     - {label: Aligned reference, path: [worst_aligned_nrmse], format: number}
#     - {label: Reassigned reference, path: [best_reassigned_nrmse], format: number}
#     definition: RMSE divided by the standard deviation of the corresponding training readings. Each summary reports the least favorable result across both channels and three seeds.
#     direction: Lower error is better with aligned references.
#   note: Gaps include the beginning, middle and end of variable-length collections. The reference at each withheld position remains available.
# evidence-plot:
#   renderer: plot
#   focus: Does moving the reference within each record break recovery of the missing readings?
#   comparison: Compare the aligned and reassigned references using the same fitted model and hidden answers.
#   limit: This tests reconstruction with a local reference in an ordered series.
# ---
#
# ## Example
#
# ```yaml
# series:
#   - {reference: 0.5, reading: 1.1, inverse: -0.7, held: true}
#   - {reference: -0.4, reading: -0.7, inverse: 0.2, held: true}
#   - {reference: 0.0, reading: 0.1, inverse: -0.2, held: false}
# ```
#
# The two readings at position 1 can be recovered from its reference of -0.4.
# A null item elsewhere in the collection still occupies its original position.
#
# ## Comparison
#
# Keep the test rows and targets fixed. Rotate the reference values among real
# items in each collection to break their alignment. In a second intervention,
# overwrite both withheld answers and measure the largest change in predictions.
#
# ## Result
#
# Both channels must recover each selected row, with normalized error below
# 0.30 for every seed. Reassigning the references must raise error above 0.85.
# The output must retain every collection length, null slot, and inferred flag.
#
# ## Method
#
# For each of three model seeds, independently generate 1,024 training, 192
# validation, and 320 test collections. Lengths vary from five to eight. A
# source flag selects an adjacent pair in one third of training collections
# and in every test collection. A second branch policy samples spans of two
# eligible rows at rate 0.35. Both reconstruction rules exempt the reference.
# A training-only row mask hides both readings together on flagged rows.
# Audit both validation channels for shared, repeatable span selections that stop
# at nulls and eligibility boundaries and exempt the reference.
# Fit a width-32 model for 320 AdamW steps at learning rate 0.002 and batch
# size 32. Each channel uses its training standard deviation as its error scale.
#
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["lightning", "matplotlib>=3.10", "numpy", "polars", "pyyaml>=6", "relflow>=0.5.0", "torch"]
#
# [tool.relflow.evidence]
# capability = "training"
# types = ["Number"]
# conditions = ["hidden-values", "changed-associations", "objectives"]
# ///

"""Recover two adjacent, coordinate-aligned answers across structural holes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
from policies import CONTROL
from reporting import experiment, report

CHANNELS = ("reading", "inverse")
BUDGET = 320


class Sample(rf.Branch):
    reference = rf.Number
    reading = rf.Number(objective="mse")
    inverse = rf.Number(objective="mse")


class Model(rf.Model):
    series = Sample(
        length=8,
        reduction=None,
        hide="redacted",
        ignore="reference",
        task=(
            rf.reconstruct(query="eligible", rate=0.35, span=2, ignore="reference"),
            rf.reconstruct(query="held", ignore="reference"),
        ),
    )


def records(seed, count, *, test=False):
    rng = np.random.default_rng(seed)
    for index in range(count):
        length = int(rng.integers(5, 9))
        start = (0, int(rng.integers(1, length - 2)), length - 2)[index % 3]
        gap = test or rng.random() < 1 / 3
        spare = [j for j in range(length) if j not in (start, start + 1)]
        hole = int(rng.choice(spare)) if index % 2 else -1
        series = []
        for j, x in enumerate(rng.uniform(-1, 1, length)):
            series.append(
                None
                if j == hole
                else dict(
                    reference=x,
                    reading=2 * x + 0.1,
                    inverse=-x - 0.2,
                    held=gap and j in (start, start + 1),
                    eligible=j % 4 != 3,
                    redacted=rng.random() < 0.1,
                )
            )
        yield {"series": series}


def readout(model, rows):
    output = model.predict(pl.DataFrame(rows), batch_size=64)["predictions"].struct.field("series").to_list()
    predictions = []
    aligned = True
    for row, items in zip(rows, output, strict=True):
        for source, result in zip(row["series"], items, strict=True):
            if source is None or result is None:
                aligned &= source is None and result is None
                continue
            aligned &= all(result[k]["inferred"] == source["held"] for k in CHANNELS)
            if source["held"]:
                predictions.append([result[k]["content"] for k in CHANNELS])
    return np.asarray(predictions), bool(aligned)


def audit_spans(model, rows, seed):
    # Isolate sampled spans from explicitly requested gaps on these same collections.
    series = [[{**v, "held": False} if v else None for v in r["series"]] for r in rows]
    frame = pl.DataFrame({"series": series})

    def selections(split, epoch):
        batch = model.encode(frame, strata=split, seed=seed, epoch=epoch)
        selected = batch["/series/reading"].trainable.reshape(len(rows), -1)
        paired = batch["/series/inverse"].trainable.reshape(selected.shape)
        ref = batch["/series/reference"]
        return selected.numpy(), bool((selected == paired).all() and not ref.trainable.any())

    chosen, shared = selections("validate", 0)
    replay, _ = selections("validate", 7)
    eligible = np.zeros_like(chosen)
    for i, items in enumerate(series):
        eligible[i, : len(items)] = [bool(v and v["eligible"]) for v in items]
    starts = chosen & ~np.pad(chosen[:, :-1], ((0, 0), (1, 0)))
    extendible = starts[:, :-1] & eligible[:, 1:]
    return dict(
        selected=int(chosen.sum()),
        starts_with_room=int(extendible.sum()),
        paired=shared,
        boundary=not bool((chosen & ~eligible).any()),
        extended=bool(chosen[:, 1:][extendible].all()),
        replay=bool(np.array_equal(chosen, replay)),
    )


def trial(seed, steps, accelerator):
    lit.seed_everything(seed, workers=True)
    train, val = list(records(seed + 1, 1024)), list(records(seed + 2, 192))
    test = list(records(seed + 3, 320, test=True))
    model = Model(preset=CONTROL.name, d_model=32)
    data = rf.PolarsDataModule(
        train=pl.DataFrame(train),
        validate=pl.DataFrame(val),
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
    reassigned, poisoned = [], []
    for row in test:
        items = row["series"]
        references = iter(np.roll([item["reference"] for item in items if item is not None], 1))
        reassigned.append(
            {"series": [None if item is None else {**item, "reference": next(references)} for item in items]}
        )
        poisoned.append(
            {
                "series": [
                    {**item, "reading": 999.0, "inverse": -999.0} if item and item["held"] else item for item in items
                ]
            }
        )
    expected = np.array(
        [[item[k] for k in CHANNELS] for row in test for item in row["series"] if item and item["held"]]
    )
    scale = np.std([[item[k] for k in CHANNELS] for row in train for item in row["series"] if item], axis=0)
    intact, aligned = readout(model, test)
    moved, moved_aligned = readout(model, reassigned)
    replaced, poison_aligned = readout(model, poisoned)
    return dict(
        seed=seed,
        steps=model.trainer.global_step,
        coordinates=len(expected),
        spans=audit_spans(model, val, seed),
        aligned_nrmse=(np.sqrt(np.mean((intact - expected) ** 2, axis=0)) / scale).tolist(),
        reassigned_nrmse=(np.sqrt(np.mean((moved - expected) ** 2, axis=0)) / scale).tolist(),
        drift=float(np.max(np.abs(intact - replaced))),
        aligned=aligned and moved_aligned and poison_aligned,
    )


@experiment("P129", seed=12401)
def run(seed, steps, accelerator):
    trials = [trial(seed + 10 * i, steps, accelerator) for i in range(3)]
    aligned = max(max(t["aligned_nrmse"]) for t in trials)
    moved = min(min(t["reassigned_nrmse"]) for t in trials)
    return {"trials": trials, "worst_aligned_nrmse": aligned, "best_reassigned_nrmse": moved}, {
        "Both channels recover the correct rows for every seed": aligned < 0.30,
        "Reassigned references break reconstruction for every seed": moved > 0.85,
        "Sampled spans extend to the next eligible row and respect every boundary": all(
            t["spans"]["starts_with_room"] > 0 and t["spans"]["extended"] and t["spans"]["boundary"] for t in trials
        ),
        "Validation repeats both channels’ shared spans and exempts the reference": all(
            t["spans"]["paired"] and t["spans"]["replay"] for t in trials
        ),
        "Withheld answers cannot influence either prediction": all(t["drift"] < 1e-5 for t in trials),
        "Every gap and null slot retains its coordinate": all(t["aligned"] and t["coordinates"] == 640 for t in trials),
    }


def plot(bundle, spec):
    """Show both readings for each fitted seed and matched control."""
    from evidence.diagnostics.plots import axes_pair, dots

    trials = bundle.record["metrics"]["trials"]
    labels = [f"Seed {t['seed']}" for t in trials]
    fig, axes = axes_pair()
    for j, (ax, name) in enumerate(zip(axes, ("Reading", "Inverse"), strict=True)):
        dots(
            ax,
            labels,
            {
                "Aligned": [t["aligned_nrmse"][j] for t in trials],
                "Reassigned": [t["reassigned_nrmse"][j] for t in trials],
            },
            xlabel="Normalized RMSE on withheld rows",
            reference=(0.30, "Recovery criterion"),
        )
        ax.set_title(name)
    return (
        fig,
        "Each pair uses the same model, withheld coordinates and answers. Only the row supplying the reference changes.",
        {},
    )


if __name__ == "__main__":
    report(run)
