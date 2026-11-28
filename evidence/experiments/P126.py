# %% [markdown]
# ---
# title: Can an Index give attention the time between records?
# description: Decide whether a payment arrived more than two days after its invoice when only an Index carries record
#   timing, against ordinal positions, a prepared elapsed-hours Number, and donor timelines.
# evidence-readout:
# - label: Ordinal positions
#   metric:
#   - median
#   - ordinal
#   - accuracy
#   format: percent
# - label: Prepared hours
#   metric:
#   - median
#   - hours
#   - accuracy
#   format: percent
# - label: Index
#   metric:
#   - median
#   - index
#   - accuracy
#   format: percent
# evidence-group: Temporal
# evidence-review:
#   question: Can an Index give branch attention the elapsed time between records that their order cannot?
#   modeling: Each history holds one invoice, one payment, and other events at irregular hours. Predict whether the
#     payment came more than 48 hours after the invoice. The number of events between them is independent of the answer.
#     Compare an Index over event timestamps with ordinal positions alone, a prepared hours-since-first-event Number,
#     and an Index trained on donor timelines, using matched seeds, budgets, and held-out histories.
#   comparison:
#     metric: Held-out accuracy
#     kind: Controlled comparison
#     conditions:
#     - label: Donor timelines
#       path:
#       - median
#       - donor
#       - accuracy
#       format: percent
#     - label: Ordinal positions
#       path:
#       - median
#       - ordinal
#       - accuracy
#       format: percent
#     - label: Prepared hours
#       path:
#       - median
#       - hours
#       - accuracy
#       format: percent
#     - label: Index
#       path:
#       - median
#       - index
#       - accuracy
#       format: percent
#     definition: Share of held-out histories whose late-payment probability falls on the correct side of 0.5, median
#       of three seeds. Half of the histories are late, so 50% is chance.
#     direction: Higher accuracy is better
#   note: Histories are synthetic and the rule depends only on one elapsed time. Donor timelines keep each history's
#     record order and timestamp distribution but take their times from an independent history.
# evidence-plot:
#   mode: summary
#   focus: Can an Index give branch attention the elapsed time between records that their order cannot?
#   comparison: Compare held-out accuracy for the Index, prepared hours, ordinal positions, and donor timelines on every
#     seed.
#   limit: One synthetic elapsed-time rule; the plot does not show how an Index behaves with many interacting intervals.
# evidence-cost:
#   candidate: 240
# ---
#
# A history lists the events of one account: exactly one `invoice`, exactly one `payment`, and two
# to six `other` events. Each event has a `kind` and a timestamp `at`. The hidden answer `late` is
# true when the payment came more than 48 hours after the invoice. Event times are irregular, so
# counting the records between the invoice and the payment says nothing about the hours between
# them: the generator draws that count independently of the answer.
#
# ## Example
#
# ```yaml
# events:
# - {kind: other, at: "2026-03-02T01:00:00Z"}
# - {kind: invoice, at: "2026-03-02T06:00:00Z"}
# - {kind: other, at: "2026-03-03T20:00:00Z"}
# - {kind: payment, at: "2026-03-05T02:00:00Z"}
# late: true
# ```
#
# The payment arrives 68 hours after the invoice, more than 48, so the answer is `true`. Moving the
# payment to `2026-03-03T22:00:00Z` would make the gap 40 hours and the answer `false` without
# changing the order or number of records. `late` is hidden when the model predicts.
#
# ## Comparison
#
# Every arm reads the same events in time order and predicts `late` from a reduced `events`
# branch. The **Index** arm declares `at = rf.Index(granularity="hours")`, so branch attention
# rotates each event by its hour instead of its row; `at` is never an input value. The
# **ordinal** arm drops the Index, leaving only record order. The **prepared hours** arm instead
# supplies hours since the history's first event as an ordinary `rf.Number`, the feature a user
# would otherwise compute upstream. The **donor** arm keeps the Index but trains on timestamps
# taken from an independent history with the same record count, keeping the order and the
# distribution of times while breaking their link to the answer; it is scored on true timelines.
#
# Two interventions then probe the Index model on the same held-out histories: shuffling the
# events within each history, and moving every timestamp 400 days later. Neither changes any
# elapsed time, so an Index should leave every prediction in place.
#
# ## Result
#
# Accuracy is the share of held-out histories classified on the correct side of 0.5; 50% is
# chance because half of the histories are late. Ordinal positions and donor timelines should
# stay near chance, because neither gives the model the elapsed time. An Index should match the
# prepared Number without any upstream preparation, while keeping its predictions fixed under
# reordering and a common shift. Histories near the 48-hour boundary are reported separately;
# they show how finely the learned threshold resolves time. One synthetic interval rule does not
# establish behavior with many interacting intervals, longer histories, or other granularities.
#
# ## Method
#
# Generate 2,048 training, 512 validation and 1,024 test histories for each of three seeds. The
# gap between invoice and payment is uniform on 2–94 hours; zero to three `other` events fall
# between them and one to three fall elsewhere in a 120-hour window. Each history starts on a
# random day within 300 days of 2026-01-01. Events are supplied in time order. Every arm uses the
# Evidence control architecture, a reduced `events` branch capped at eight records, a Boolean
# `late` target, 600 AdamW updates at batch size 64 and learning rate 0.003, and identical seeds.
#
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["lightning", "numpy", "polars", "pyyaml>=6", "relflow>=0.6.0", "torch"]
#
# [tool.relflow.evidence]
# capability = "structure"
# types = ["Index"]
# conditions = ["representations", "changed-associations", "transformations", "varying-collections"]
# ///

"""P126: an Index gives branch attention elapsed time that record order cannot."""

import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from itertools import repeat
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
import torch
from policies import CONTROL
from reporting import experiment, report

BUDGET = 600
TRAIN_ROWS = 2048
VALIDATION_ROWS = 512
TEST_ROWS = 1024
PANEL = (0, 100, 200)
ANCHOR = datetime(2026, 1, 1, tzinfo=timezone.utc)
WINDOW = 120.0
THRESHOLD = 48.0
NEAR = 8.0
SHIFT = timedelta(days=400)
ARMS = ("index", "hours", "ordinal", "donor")
SCHEMA = {
    "events": pl.List(pl.Struct({"kind": pl.String, "at": pl.Datetime("us", "UTC"), "hours": pl.Float64})),
    "late": pl.Boolean,
}
KINDS = ("invoice", "payment", "other")


class IndexedEvent(rf.Branch):
    at = rf.Index(granularity="hours")
    kind = rf.Enum(*KINDS)


class PreparedEvent(rf.Branch):
    hours = rf.Number
    kind = rf.Enum(*KINDS)


class OrdinalEvent(rf.Branch):
    kind = rf.Enum(*KINDS)


class IndexedModel(rf.Model):
    events = IndexedEvent(length=8)
    late = rf.Boolean(predict=True)


class PreparedModel(rf.Model):
    events = PreparedEvent(length=8)
    late = rf.Boolean(predict=True)


class OrdinalModel(rf.Model):
    events = OrdinalEvent(length=8)
    late = rf.Boolean(predict=True)


MODELS = {"index": IndexedModel, "hours": PreparedModel, "ordinal": OrdinalModel, "donor": IndexedModel}


def timeline(rng: np.random.Generator, *, gap: float, between: int, outside: int) -> list[tuple[str, float]]:
    """Place an invoice and payment ``gap`` hours apart among other events in a 120-hour window."""
    start = float(rng.uniform(0, WINDOW - gap))
    times = [("invoice", start), ("payment", start + gap)]
    times += [("other", float(rng.uniform(start, start + gap))) for _ in range(between)]
    for _ in range(outside):
        before = rng.uniform() < start / (WINDOW - gap)
        times.append(("other", float(rng.uniform(0, start) if before else rng.uniform(start + gap, WINDOW))))
    return sorted(times, key=lambda item: item[1])


def records(*, rows: int, seed: int):
    rng = np.random.default_rng(seed)
    for index in range(rows):
        gap = float(rng.uniform(2, WINDOW - 26))
        between, outside = int(rng.integers(0, 4)), int(rng.integers(1, 4))
        base = ANCHOR + timedelta(days=int(rng.integers(-300, 301)))
        events = [
            {"kind": kind, "at": base + timedelta(hours=hours)}
            for kind, hours in timeline(rng, gap=gap, between=between, outside=outside)
        ]
        # A donor timeline matches the record count but draws its own gap and placements.
        donor = timeline(rng, gap=float(rng.uniform(2, WINDOW - 26)), between=between, outside=outside)
        yield {
            "case": f"{seed}:{index}",
            "events": events,
            "gap": gap,
            "late": gap > THRESHOLD,
            "donor": [base + timedelta(hours=hours) for _, hours in donor],
        }


def frame(
    rows: list[dict],
    *,
    labels: bool = True,
    donor: bool = False,
    order: np.random.Generator | None = None,
    shift: timedelta = timedelta(0),
) -> pl.DataFrame:
    histories = []
    for row in rows:
        events = [dict(event) for event in row["events"]]
        if donor:
            for event, at in zip(events, row["donor"], strict=True):
                event["at"] = at
        first = min(event["at"] for event in events)
        for event in events:
            event["hours"] = (event["at"] - first).total_seconds() / 3600
            event["at"] = event["at"] + shift
        if order is not None:
            events = [events[int(position)] for position in order.permutation(len(events))]
        histories.append({"events": events, "late": row["late"] if labels else None})
    return pl.DataFrame(histories, schema=SCHEMA)


def fit(model: rf.Model, train: pl.DataFrame, validate: pl.DataFrame, seed: int, steps: int | None, accelerator: str):
    data = rf.PolarsDataModule(
        train=train, validate=validate, seed=seed, num_workers=0, persistent_workers=False, pin_memory=False
    )
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


def probabilities(model: rf.Model, data: pl.DataFrame) -> np.ndarray:
    output = pl.DataFrame(model.predict(data, batch_size=256))
    late = output["predictions"].struct.field("late").struct.field("content").struct.field("probability")
    values = late.to_numpy().astype(np.float64)
    if len(values) != len(data) or not np.isfinite(values).all():
        raise ValueError("late probabilities are nonfinite or misaligned")
    return values


def auc(actual: np.ndarray, scores: np.ndarray) -> float:
    ranks = np.empty(len(scores))
    ranks[np.argsort(scores, kind="stable")] = np.arange(1, len(scores) + 1)
    positive = int(actual.sum())
    negative = len(actual) - positive
    return float((ranks[actual].sum() - positive * (positive + 1) / 2) / (positive * negative))


def trial(seed: int, steps: int | None, accelerator: str) -> dict:
    """Train and score every arm on one seed's independent splits."""
    train = list(records(rows=TRAIN_ROWS, seed=seed + 1))
    validate = list(records(rows=VALIDATION_ROWS, seed=seed + 2))
    test = list(records(rows=TEST_ROWS, seed=seed + 3))
    actual = np.asarray([row["late"] for row in test])
    gaps = np.asarray([row["gap"] for row in test])
    near = np.abs(gaps - THRESHOLD) <= NEAR
    result: dict = {
        "seed": seed,
        "late_share": float(actual.mean()),
        "test": {"case": [row["case"] for row in test], "gap": gaps, "late": actual},
        "arms": {},
    }
    for arm in ARMS:
        lit.seed_everything(seed, workers=True)
        model = MODELS[arm](preset=CONTROL.name)
        donor = arm == "donor"
        updates = fit(model, frame(train, donor=donor), frame(validate, donor=donor), seed, steps, accelerator)
        scores = probabilities(model, frame(test, labels=False))
        reordered = probabilities(model, frame(test, labels=False, order=np.random.default_rng(seed + 4)))
        shifted = probabilities(model, frame(test, labels=False, shift=SHIFT))
        predicted = scores > 0.5
        result["arms"][arm] = {
            "steps": updates,
            "accuracy": float(np.mean(predicted == actual)),
            "near_accuracy": float(np.mean(predicted[near] == actual[near])),
            "far_accuracy": float(np.mean(predicted[~near] == actual[~near])),
            "auc": auc(actual, scores),
            "reordered_accuracy": float(np.mean((reordered > 0.5) == actual)),
            "reorder_drift": float(np.max(np.abs(reordered - scores))),
            "shift_drift": float(np.max(np.abs(shifted - scores))),
            "probabilities": scores,
        }
    return result


@experiment("P126", seed=12601)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    seeds = [seed + offset for offset in PANEL]
    if accelerator == "cpu":
        with ProcessPoolExecutor(
            max_workers=2,
            mp_context=get_context("spawn"),
            initializer=torch.set_num_threads,
            initargs=(torch.get_num_threads(),),
        ) as pool:
            trials = list(pool.map(trial, seeds, repeat(steps), repeat(accelerator)))
    else:
        trials = [trial(item, steps, accelerator) for item in seeds]

    def values(arm: str, key: str) -> list[float]:
        return [result["arms"][arm][key] for result in trials]

    keys = ("accuracy", "near_accuracy", "far_accuracy", "auc", "reordered_accuracy", "reorder_drift", "shift_drift")
    median = {arm: {key: float(np.median(values(arm, key))) for key in keys} for arm in ARMS}
    advantage = [
        index - ordinal
        for index, ordinal in zip(values("index", "accuracy"), values("ordinal", "accuracy"), strict=True)
    ]
    metrics = {
        "budget": BUDGET,
        "rows": {"train": TRAIN_ROWS, "validation": VALIDATION_ROWS, "test": TEST_ROWS},
        "threshold_hours": THRESHOLD,
        "near_hours": NEAR,
        "trials": trials,
        "median": median,
        "worst": {
            "index_accuracy": min(values("index", "accuracy")),
            "index_over_ordinal": min(advantage),
            "index_reorder_drift": max(values("index", "reorder_drift")),
            "index_shift_drift": max(values("index", "shift_drift")),
        },
    }
    checks = {
        "Every arm keeps its full budget on every seed": all(
            result["arms"][arm]["steps"] == BUDGET for result in trials for arm in ARMS
        ),
        "Index reaches held-out accuracy of at least 95% on every seed": min(values("index", "accuracy")) >= 0.95,
        "Ordinal positions stay near chance (median accuracy at most 60%)": median["ordinal"]["accuracy"] <= 0.60,
        "Donor timelines stay near chance (median accuracy at most 60%)": median["donor"]["accuracy"] <= 0.60,
        "Index beats ordinal positions by at least 30 points on every seed": min(advantage) >= 0.30,
        "Index matches prepared hours within 2 points (median)": (
            median["index"]["accuracy"] >= median["hours"]["accuracy"] - 0.02
        ),
        "Index predictions ignore record order on every seed": max(values("index", "reorder_drift")) <= 1e-4,
        "Index predictions ignore a common 400-day shift on every seed": max(values("index", "shift_drift")) <= 1e-6,
    }
    return metrics, checks


if __name__ == "__main__":
    report(run)
