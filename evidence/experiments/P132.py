# %% [markdown]
# ---
# title: Can reconstruction learn when each batch teaches just two fields?
# description: Recover six calibrated sensor readings while rotating which two supply answers on each training batch.
# evidence-group: Masking
# evidence-readout:
# - {label: Aligned sensors, metric: [worst_aligned_nrmse], format: error}
# - {label: Reassigned sensors, metric: [best_reassigned_nrmse], format: error}
# evidence-review:
#   question: Can all six readouts learn from two ablated fields per batch and four accumulated batches per update?
#   modeling: Each sensor measures a common signal on its own scale. Two entire sensor columns supply reconstruction targets in each training batch. The reference and remaining sensors provide context.
#   comparison:
#     kind: Controlled comparison
#     metric: Normalized RMSE
#     conditions:
#     - {label: Aligned sensors, path: [worst_aligned_nrmse], format: number}
#     - {label: Reassigned sensors, path: [best_reassigned_nrmse], format: number}
#     definition: Divide each sensor's RMSE by its training standard deviation. Report the worst aligned and best reassigned result over sensors and seeds.
#     direction: Lower error indicates more accurate recovery of the hidden readings.
#   note: The six sensors have a common scalar signal and no measurement noise.
# evidence-plot:
#   renderer: plot
#   focus: Does each hidden reading follow the visible sensors from the same observation?
#   comparison: Reassign the visible context across test records and score the original hidden answers.
#   limit: This experiment measures reconstruction on correlated numeric fields.
# ---
#
# ## Example
#
# The north meter doubles the reference. The south meter reports 0.25
# minus the reference. This observation requests both readings.
#
# ```yaml
# reference: 0.4
# north: 0.8
# south: -0.15
# east: 0.7
# west: -0.25
# upper: 1.6
# lower: 0.95
# request_north: true
# request_south: true
# request_east: false
# request_west: false
# request_upper: false
# request_lower: false
# ```
#
# ## Comparison
#
# Request the same pair of meters for each held-out record. First overwrite
# both hidden readings. Then rotate the reference and the four visible readings
# together across records and score the original answers.
#
# ## Result
#
# Every meter must recover its held-out readings to within half a training
# standard deviation. Reassigning the visible context must raise every error
# above 0.8 standard deviations. Overwriting hidden answers must preserve the
# corresponding predictions.
#
# ## Method
#
# Draw 1,024 training, 192 validation and 480 test records per seed.
# Use six fixed gains and offsets. Train for 160 AdamW updates with four
# independently sampled 16-row batches per update. Cover all 15 pairs equally
# on the test split and repeat with three independent seeds.
#
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["lightning", "matplotlib>=3.10", "numpy", "polars", "pyyaml>=6", "relflow>=0.6.0", "torch"]
#
# [tool.relflow.evidence]
# capability = "training"
# types = ["Number"]
# conditions = ["hidden-values", "changed-associations", "objectives"]
# ///

"""Measure recovery from a fixed budget of sampled field objectives."""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
from policies import CONTROL
from reporting import experiment, report

METERS = ("north", "south", "east", "west", "upper", "lower")
GAINS = np.array([2.0, -1.0, 0.5, -2.5, 1.5, -0.75])
PAIRS = tuple(combinations(range(len(METERS)), 2))


class Sensors(rf.Model):
    reference = rf.Number
    north = rf.Number(objective="mse", task=rf.reconstruct(query="request_north", ablate=True))
    south = rf.Number(objective="mse", task=rf.reconstruct(query="request_south", ablate=True))
    east = rf.Number(objective="mse", task=rf.reconstruct(query="request_east", ablate=True))
    west = rf.Number(objective="mse", task=rf.reconstruct(query="request_west", ablate=True))
    upper = rf.Number(objective="mse", task=rf.reconstruct(query="request_upper", ablate=True))
    lower = rf.Number(objective="mse", task=rf.reconstruct(query="request_lower", ablate=True))


def observations(seed, count, *, request=False):
    for j, signal in enumerate(np.random.default_rng(seed).uniform(-1, 1, count)):
        record = {"reference": signal}
        for i, (name, gain) in enumerate(zip(METERS, GAINS, strict=True)):
            record[name] = gain * signal + i / 4
            record[f"request_{name}"] = request and i in PAIRS[j % len(PAIRS)]
        yield record


def read(model, rows):
    result = model.predict(rows, batch_size=64)["predictions"].struct
    return np.column_stack([result.field(name).struct.field("content").to_numpy() for name in METERS])


def trial(seed, steps, accelerator):
    lit.seed_everything(seed, workers=True)
    train = pl.DataFrame(observations(seed + 1, 1024))
    val = pl.DataFrame(observations(seed + 2, 192))
    test = pl.DataFrame(observations(seed + 3, 480, request=True))
    model = Sensors(preset=CONTROL.name, d_model=32, task=rf.reconstruct(fields=2, ignore="reference", ablate=True))
    data = rf.PolarsDataModule(
        train=train, validate=val, seed=seed, num_workers=0, persistent_workers=False, pin_memory=False
    )

    class Draws(lit.Callback):
        def __init__(self):
            self.fields = []
            self.aligned = True

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            selected = []
            for name in METERS:
                field = batch.tensors["/" + name]
                if field.trainable.any():
                    selected.append(name)
                    self.aligned &= bool(field.trainable.all() and not field.present.any())
                else:
                    self.aligned &= bool(field.present.all())
            self.fields.append(selected)

    draws = Draws()
    model.fit(
        data,
        callbacks=[draws],
        max_steps=160 if steps is None else min(steps, 160),
        max_epochs=-1,
        accumulate_grad_batches=4,
        batch_size=16,
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
    expected = test.select(METERS).to_numpy()
    selected = test.select([f"request_{n}" for n in METERS]).to_numpy()
    scale = train.select(METERS).to_numpy().std(axis=0)
    aligned = read(model, test)
    poisoned = test.with_columns(
        pl.when(pl.col(f"request_{n}")).then(987.0).otherwise(pl.col(n)).alias(n) for n in METERS
    )
    shifted = test.with_columns(
        pl.Series("reference", np.roll(test["reference"].to_numpy(), 1)),
        *[
            pl.Series(name, np.where(selected[:, i], expected[:, i], np.roll(expected[:, i], 1)))
            for i, name in enumerate(METERS)
        ],
    )
    reassigned = read(model, shifted)

    def errors(pred):
        return (
            np.sqrt(np.sum(np.where(selected, (pred - expected) ** 2, 0.0), axis=0) / selected.sum(axis=0)) / scale
        ).tolist()

    return dict(
        seed=seed,
        steps=model.trainer.global_step,
        batches=len(draws.fields),
        heads_per_batch=sorted({len(batch) for batch in draws.fields}),
        aligned_inputs=draws.aligned,
        distinct=sorted({n for batch in draws.fields for n in batch}),
        aligned=errors(aligned),
        reassigned=errors(reassigned),
        hidden_drift=float(np.max(np.abs(read(model, poisoned)[selected] - aligned[selected]))),
    )


@experiment("P132", seed=13101)
def run(seed, steps, accelerator):
    trials = [trial(seed + 10 * i, steps, accelerator) for i in range(3)]
    aligned = max(max(t["aligned"]) for t in trials)
    reassigned = min(min(t["reassigned"]) for t in trials)
    return dict(trials=trials, worst_aligned_nrmse=aligned, best_reassigned_nrmse=reassigned), {
        "Each training batch teaches two fields and every field participates": all(
            t["heads_per_batch"] == [2] and t["distinct"] == sorted(METERS) and t["aligned_inputs"] for t in trials
        ),
        "Each optimizer update accumulates four sampled batches": all(t["batches"] == 4 * t["steps"] for t in trials),
        "Each readout recovers its own hidden reading on every seed": aligned < 0.5,
        "Reassigning the visible sensors harms each readout": reassigned > 0.8,
        "Overwriting requested answers preserves their predictions": all(t["hidden_drift"] < 1e-6 for t in trials),
    }


def plot(bundle, spec):
    """Show every hidden reading's error with its own visible sensors and with reassigned sensors."""
    from evidence.diagnostics.plots import axes_pair, dots

    trials = bundle.record["metrics"]["trials"]
    labels = ["North", "South", "East", "West", "Upper", "Lower"]
    fig, (aligned, reassigned) = axes_pair()
    dots(
        aligned,
        labels,
        {f"Seed {t['seed']}": t["aligned"] for t in trials},
        xlabel="Normalized RMSE on requested readings",
        reference=(0.5, "Recovery criterion"),
    )
    aligned.set_title("Sensors from the same observation")
    dots(
        reassigned,
        labels,
        {f"Seed {t['seed']}": t["reassigned"] for t in trials},
        xlabel="Normalized RMSE on requested readings",
        reference=(0.8, "Control criterion"),
    )
    reassigned.set_title("Sensors reassigned across records")
    return (
        fig,
        "Each reading is scored only where it was requested. Reassigning the visible sensors keeps the hidden answers fixed.",
        {},
    )


if __name__ == "__main__":
    report(run)
