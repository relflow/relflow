# %% [markdown]
# ---
# title: Can a summary retain the winning item’s value?
# description: Predict the value attached to the highest-scoring candidate after compressing the candidates into learned
#   summaries.
# evidence-readout:
# - label: Correct pairs
#   metric:
#   - intact_nrmse
#   format: error
# - label: Broken pairs
#   metric:
#   - broken_nrmse
#   format: error
# evidence-group: Relational
# evidence-review:
#   question: Can a summary retain the winning item’s value?
#   modeling: Give each candidate a score and a separate payload value. Hide the payload belonging to the highest
#     score. Compress the candidate collection into learned summaries, then compare correct records with copies that
#     rotate payloads between candidates while retaining the original answer.
#   comparison:
#     kind: Controlled comparison
#     metric: Normalized RMSE
#     conditions:
#     - label: Broken pairs
#       path:
#       - broken_nrmse
#       format: number
#     - label: Correct pairs
#       path:
#       - intact_nrmse
#       format: number
#     definition: Root mean squared error divided by the reference scale defined in this experiment. Zero is exact;
#       one equals that reference scale.
#     direction: Lower error is better
#   note: The control preserves the separate score and payload distributions. It breaks their association, which is
#     the information needed to retrieve the winner.
# evidence-plot:
#   mode: regression
#   target: answer
#   focus: Can a compressed relational route recover the requested numeric payload?
#   comparison: Compare intact and broken identity links using the same original answers.
#   limit: Low intact error and a broken-link gap are separate requirements.
# evidence-cost:
#   baseline: 47
#   candidate: 46
# ---
#
# Each candidate has a score and a separate numeric payload. The hidden answer is the payload
# belonging to the highest score, not the highest payload or the score itself. This version
# compresses candidate information into learned summaries before producing the answer.
#
# ## Example
#
# ```yaml
# items:
#   - {score: -0.4, payload: 2.0}
#   - {score: 1.7, payload: -3.0}
#   - {score: 0.2, payload: 5.0}
# answer: -3.0
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# The highest illustrated score is 1.7, whose payload is −3.0, so the answer is −3.0 even though
# another candidate carries payload 5.0. A predictor that returns the largest payload, an average
# payload, or the winning score would be answering a different question.
#
# ## Comparison
#
# Shuffle values between candidates while keeping the original answer. Scores and values
# individually stay the same.
#
# Scores and payloads are sampled independently, making a score alone uninformative about the
# answer's numeric value. The control rotates payloads among the same three candidates while
# retaining the original target. Both collections of numbers survive unchanged, but the payload
# attached to the winner changes. Independent train and test records prevent copying a previously
# memorized triple from solving this comparison.
#
# ## Result
#
# The tested summaries retain useful score–value relationships. Exact retrieval, larger lists, and
# speed or memory savings are not established.
#
# The error percentages compare predictions with always guessing the mean training payload answer.
# Low intact error and a substantial rise after breaking pairs support retention of the
# score–payload association through the summaries. They do not show an exact argmax operator,
# isolate the minimum necessary summary width, or measure memory or speed savings against the
# pass-through route.
#
# ## Method
#
# ### How it works
#
# Item encoding can bind the score and payload before reduction. The learned
# summaries must retain enough of those associations for the root decoder to
# recover the winning payload. Rotating payloads while retaining the original
# answer tests whether success depends on that binding rather than statistics
# of the collection.
#
# ### Remaining work
#
# Calibrate across three core seeds and ten calibration seeds. Sweep reduction
# width and candidate count before treating summary capacity as established;
# argmin, category payloads, and group-filtered retrieval remain unimplemented.
# The [pass-through evidence](pass-through-number-payload.html) provides the simpler
# route comparison. Use exact preprocessing when retrieval must be exact.
#

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "lightning",
#     "numpy",
#     "polars",
#     "pyyaml>=6",
#     "relflow>=0.5.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "structure"
# types = []
# conditions = ["changed-associations"]
# ///

"""Retrieve the payload paired with the largest of three scores.

The root retains three learned Attention outputs and the item branch uses one.
Rotating payloads without rotating targets tests whether the score/payload
pairing, rather than their separate distributions, drives the prediction.

Run this file with --help for seed, training-budget, and reporting options.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from functools import partial
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
import torch
from reporting import experiment, report

LENGTH = 3


def records(*, rows: int, seed: int, break_pairs: bool = False) -> Iterator[dict]:
    """Rotate only visible payloads for the control, retaining original answers."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(-2.0, 2.0, size=(rows, LENGTH))
    payloads = rng.normal(0.0, 1.0, size=(rows, LENGTH))
    answers = payloads[np.arange(rows), scores.argmax(axis=1)]
    visible = np.roll(payloads, shift=1, axis=1) if break_pairs else payloads
    for row_scores, row_payloads, answer in zip(scores, visible, answers, strict=True):
        yield {
            "items": [
                {"score": float(score), "payload": float(payload)}
                for score, payload in zip(row_scores, row_payloads, strict=True)
            ],
            "answer": float(answer),
        }


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    output = pl.DataFrame(model.predict([{"items": row["items"]} for row in rows])).to_dicts()
    return np.asarray([row["predictions"]["answer"]["content"] for row in output])


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


class Item(rf.Branch):
    score = rf.Number
    payload = rf.Number


class PredictionModel(rf.Model):
    items = Item(length=LENGTH, overflow="error", n_layers=2, n_heads=4, reduction=rf.Attention())
    answer = rf.Number(mask=True)


@experiment("P022", seed=23)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = PredictionModel(d_model=48, n_layers=1, n_heads=4, reduction=rf.Attention(n_outputs=LENGTH))
    data = rf.SyntheticDataModule(
        train=partial(records, rows=4096, seed=seed + 1),
        validate=partial(records, rows=1024, seed=seed + 2),
        seed=seed,
    )
    model.fit(
        data,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.003),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
        accelerator=accelerator,
        max_steps=-1 if steps is None else steps,
        max_epochs=25,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    train = list(records(rows=4096, seed=seed + 1))
    test = list(records(rows=2048, seed=seed + 3))
    broken = list(records(rows=2048, seed=seed + 3, break_pairs=True))
    actual = np.asarray([row["answer"] for row in test])
    train_actual = np.asarray([row["answer"] for row in train])
    baseline = rmse(actual, float(train_actual.mean()))
    measured = rmse(actual, predict(model, test))
    intact_nrmse = measured / baseline
    broken_nrmse = rmse(actual, predict(model, broken)) / baseline
    metrics = {"rmse": measured, "baseline_rmse": baseline, "intact_nrmse": intact_nrmse, "broken_nrmse": broken_nrmse}

    checks = {
        "Intact retrieval nRMSE <= 0.35": intact_nrmse <= 0.35,
        "Broken pairing nRMSE >= 0.90": broken_nrmse >= 0.90,
        "Breaking pairing increases nRMSE by >= 0.35": broken_nrmse >= intact_nrmse + 0.35,
    }
    return metrics, checks


if __name__ == "__main__":
    report(run)
