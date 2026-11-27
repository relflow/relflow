# %% [markdown]
# ---
# title: Can a model use a restored history after whole-collection ablation?
# description: Remove entire histories during training and measure predictions with missing, restored, and mismatched histories.
# evidence-group: Masking
# evidence-readout:
# - {label: Restored history, metric: [worst_restored_nrmse], format: error}
# - {label: Mismatched history, metric: [best_mismatched_nrmse], format: error}
# evidence-review:
#   question: Does a model trained with collection ablation use history when present and recover the conditional mean when it is absent?
#   modeling: A root target sums two independent signals. The first is visible at the root and the second is repeated in a child history. Half the training histories are ablated. Each present child also has a supervised target.
#   comparison:
#     kind: Controlled comparison
#     metric: Normalized RMSE
#     conditions:
#     - {label: Restored history, path: [worst_restored_nrmse], format: number}
#     - {label: Mismatched history, path: [best_mismatched_nrmse], format: number}
#     definition: Root prediction RMSE divided by the training target standard deviation. The summaries select the least favorable result across three independent seeds.
#     direction: Lower error is better with the correct history.
#   note: The source contains redundant readings of a single history signal. The empty-history oracle is the available root signal.
# evidence-plot:
#   renderer: plot
#   focus: Does restoring a matching collection recover its independent contribution to the target?
#   comparison: Compare the intact history with a history taken from another test record.
#   limit: The experiment uses independent, centered signals and random training-time collection loss.
# ---
#
# ## Example
#
# A record has an offset of 0.3 and two history readings of 0.5. Its root
# answer is 0.8. Each history reading has its own supervised answer of 1.0.
# If the entire history disappears, 0.3 is the best estimate of the root
# answer under squared error. Both child answers disappear with the history.
#
# ```yaml
# offset: 0.3
# history: [{reading: 0.5, doubled: 1.0}, {reading: 0.5, doubled: 1.0}]
# total: 0.8
# ```
#
# ## Comparison
#
# Score the same fitted model and test records with the correct history,
# an empty history, and another record’s history. Removing history leaves
# an exact conditional-mean oracle. Reassigning it preserves the history
# distribution while breaking its relationship with the root answer.
#
# ## Result
#
# With history restored, both the root and the child answers must achieve
# normalized RMSE below 0.30. With history absent, the model must approach
# the offset-only oracle within 0.30 training standard deviations. Moving
# history between records must raise root error above 0.75.
#
# ## Method
#
# Generate independent training, validation, and test sets of 1,024, 192,
# and 512 records for each of three model seeds. Both signals are uniform
# on [-1, 1]. One to four child rows carry the same history signal. Fit a
# width-32 model for 320 AdamW steps with learning rate 0.002 and batch size
# 32. Audit the training and validation encodings on 512 test records using
# a separate model. Each dropped parent must lose all of its child objectives
# while retaining its root objective. Validation must restore every real child.
#
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["lightning", "matplotlib>=3.10", "numpy", "polars", "pyyaml>=6", "relflow>=0.5.0", "torch"]
#
# [tool.relflow.evidence]
# capability = "reliability"
# types = ["Number"]
# conditions = ["missing-inputs", "changed-associations", "objectives"]
# ///

"""Compare missing and restored histories under a known conditional-mean oracle."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
from policies import CONTROL
from reporting import experiment, report

BUDGET = 320


class Reading(rf.Branch):
    reading = rf.Number
    doubled = rf.Number(predict=True, objective="mse")


class Model(rf.Model):
    offset = rf.Number
    history = Reading(length=4, ablate=0.5)
    total = rf.Number(predict=True, objective="mse")


def records(seed, count):
    rng = np.random.default_rng(seed)
    for a, b in rng.uniform(-1, 1, (count, 2)):
        yield dict(offset=a, total=a + b, history=[dict(reading=b, doubled=2 * b) for _ in range(rng.integers(1, 5))])


def readout(model, rows):
    # Retain the list element type even when every collection is empty.
    schema = {"offset": pl.Float64, "history": pl.List(pl.Struct({"reading": pl.Float64}))}
    data = pl.DataFrame(
        [{"offset": r["offset"], "history": [{"reading": v["reading"]} for v in r["history"]]} for r in rows],
        schema=schema,
    )
    values = model.predict(data, batch_size=64)["predictions"].to_list()
    return np.array([r["total"]["content"] for r in values]), np.array(
        [v["doubled"]["content"] for r in values for v in r["history"]]
    )


def trial(seed, steps, accelerator):
    lit.seed_everything(seed, workers=True)
    train, val, test = list(records(seed + 1, 1024)), list(records(seed + 2, 192)), list(records(seed + 3, 512))
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
    actual = np.array([r["total"] for r in test])
    child_actual = np.array([v["doubled"] for r in test for v in r["history"]])
    scale = np.std([r["total"] for r in train])
    child_scale = np.std([v["doubled"] for r in train for v in r["history"]])
    full, child = readout(model, test)
    missing, _ = readout(model, [{**r, "history": []} for r in test])
    shuffled, _ = readout(model, [{**r, "history": test[i - 1]["history"]} for i, r in enumerate(test)])
    audit = Model(preset=CONTROL.name, d_model=32)
    lengths = np.array([len(r["history"]) for r in test])
    source = pl.DataFrame(test)
    retained = {}
    routing = True
    for split in ("train", "validate", "test"):
        batch = audit.encode(source, strata=split, seed=seed)
        sizes = batch["/"]["/history"].lengths.flatten().tolist()
        kept = np.array(sizes) > 0
        counts = batch["/history/doubled"].trainable.reshape(len(test), -1).sum(-1).tolist()
        routing &= (
            np.array_equal(sizes, lengths * kept)
            and np.array_equal(counts, sizes)
            and bool(batch["/total"].trainable.all())
        )
        retained[split] = int(kept.sum())
    return dict(
        seed=seed,
        steps=model.trainer.global_step,
        retained=retained,
        routing=routing,
        restored_nrmse=float(np.sqrt(np.mean((actual - full) ** 2)) / scale),
        child_nrmse=float(np.sqrt(np.mean((child_actual - child) ** 2)) / child_scale),
        mismatched_nrmse=float(np.sqrt(np.mean((actual - shuffled) ** 2)) / scale),
        empty_oracle_nrmse=float(np.sqrt(np.mean((missing - np.array([r["offset"] for r in test])) ** 2)) / scale),
    )


@experiment("P130", seed=12501)
def run(seed, steps, accelerator):
    trials = [trial(seed + i * 10, steps, accelerator) for i in range(3)]
    full = max(t["restored_nrmse"] for t in trials)
    mismatched = min(t["mismatched_nrmse"] for t in trials)
    return {"trials": trials, "worst_restored_nrmse": full, "best_mismatched_nrmse": mismatched}, {
        "Restored history predicts the full root answer": full < 0.30,
        "Restored children recover their own targets": all(t["child_nrmse"] < 0.30 for t in trials),
        "An empty history recovers the offset-only oracle": all(t["empty_oracle_nrmse"] < 0.30 for t in trials),
        "Mismatched histories break the root prediction": mismatched > 0.75,
        "Child objectives follow whole collections and root objectives survive": all(t["routing"] for t in trials),
        "Ablation affects about half the training histories and none at evaluation": all(
            0.40 * 512 < t["retained"]["train"] < 0.60 * 512
            and t["retained"]["validate"] == t["retained"]["test"] == 512
            for t in trials
        ),
    }


def plot(bundle, spec):
    """Keep each seed and error reference visible in the comparison."""
    from evidence.diagnostics.plots import axes_pair, dots

    trials = bundle.record["metrics"]["trials"]
    labels = [f"Seed {t['seed']}" for t in trials]
    fig, axes = axes_pair()
    dots(
        axes[0],
        labels,
        {"Restored": [t["restored_nrmse"] for t in trials], "Mismatched": [t["mismatched_nrmse"] for t in trials]},
        xlabel="Normalized error against the full root answer",
    )
    dots(
        axes[1],
        labels,
        {"Empty history": [t["empty_oracle_nrmse"] for t in trials]},
        xlabel="Normalized error against the offset-only oracle",
        reference=(0.30, "Oracle criterion"),
    )
    return (
        fig,
        "The left panel scores the full answer for the same held-out records. The right panel scores the expected answer given an empty history.",
        {},
    )


if __name__ == "__main__":
    report(run)
