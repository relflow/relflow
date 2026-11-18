# %% [markdown]
# ---
# title: Can it return the winning item’s value?
# description: Predict the value attached to the highest score while retaining the encoded candidates.
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
#   question: Can relflow return the winning item’s value?
#   modeling: Give each candidate a score and payload, retain the encoded candidates, and hide the payload with the
#     highest score. Compare correct records with copies that rotate payloads while leaving scores and expected answers
#     fixed.
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
#   note: P022 asks the same retrieval question after compression. This page tests the route that retains candidates.
# evidence-plot:
#   mode: regression
#   target: answer
#   focus: Does preserving relational tokens support payload retrieval?
#   comparison: Inspect intact and identity-corrupted queries alongside the training diagnostic.
#   limit: Training fit alone is not held-out retrieval evidence.
# evidence-cost:
#   baseline: 42
#   candidate: 41
# ---
#
# The model must return the numeric payload attached to the largest score among three candidates.
# Here the architecture keeps the encoded candidate information instead of reducing it into learned
# summaries. Keeping that information available does not itself specify the correct selection rule;
# the rule still has to be learned.
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
# The second illustrated candidate wins because 1.7 is the largest score. Its payload is −3.0, which
# is therefore the hidden answer. The payload 5.0 belongs to a lower-scoring candidate and must not
# win simply because it is numerically larger.
#
# ## Comparison
#
# Shuffle values between candidates while keeping the original answer.
#
# Training and test use new independent triples. Rotating only the payloads breaks their pairing
# with scores, while leaving the original answer in place for scoring. A model using the winning
# association should now disagree with that old answer. The comparison complements intact accuracy:
# a constant predictor might ignore the changed inputs, while a useful but pairing-insensitive
# shortcut would fail to respond appropriately.
#
# ## Result
#
# Keeping the pairs together supports the learned selection. This test uses three candidates and
# does not guarantee exact selection.
#
# The intact and broken percentages share a constant training-mean baseline; training error is also
# recorded separately. Low intact error plus worse broken-pair error supports learned selection in
# this small setting. This is not a controlled speed or memory comparison with the summary model,
# and it does not establish exact retrieval or larger-candidate generalization.
#
# ## Method
#
# ### How it works
#
# The shared item coordinate binds each score to its payload. Retained item
# context lets the decoder learn selection by score followed by value retrieval.
# The paired control rotates only payloads while keeping scores and original
# answers. Both marginal distributions remain intact; the winning association
# does not.
#
# ### Remaining work
#
# Repeat across three core seeds and ten calibration seeds, then vary candidate
# count, selection rule, and payload type. This establishes a three-candidate
# learning behavior; exact application retrieval belongs in preprocessing.
# Compare the [attention route](attention-number-payload.html).
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

"""Retrieve an argmax payload while preserving every candidate token.

Both root and item reductions are disabled. Compare held-out and training error,
then rotate only visible payloads to test whether the learned answer depends on
which payload belongs to the largest score.

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
    items = Item(length=LENGTH, overflow="error", n_layers=2, n_heads=4, reduction=None)
    answer = rf.Number(mask=True)


@experiment("P023", seed=23)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = PredictionModel(d_model=48, n_layers=1, n_heads=4, reduction=None)
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
    metrics["train_nrmse"] = rmse(train_actual, predict(model, train)) / rmse(train_actual, float(train_actual.mean()))
    checks = {
        "Intact retrieval nRMSE <= 0.35": intact_nrmse <= 0.35,
        "Broken pairing nRMSE >= 0.90": broken_nrmse >= 0.90,
        "Breaking pairing increases nRMSE by >= 0.35": broken_nrmse >= intact_nrmse + 0.35,
    }
    return metrics, checks


if __name__ == "__main__":
    report(run)
