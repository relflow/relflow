# %% [markdown]
# ---
# title: Can it return the winning item’s value?
# categories:
# - Argmax retrieval
# proof-id: P023
# description: Predict the value attached to the highest score while retaining the encoded candidates.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: Correct pairs
#   metric:
#   - intact_nrmse
#   format: error
# - label: Broken pairs
#   metric:
#   - broken_nrmse
#   format: error
# ---
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
# ```{typst}
# //| label: fig-proof-pass-through-number-payload
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with retrieval, items, score, payload, answer. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("retrieval", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("score", type: "Number"),
#     node("payload", type: "Number"),
#   )),
#   node("answer", kind: "target", type: "Number", width: 150pt,),
# )))
# ```
#
# ## Comparison
#
# Shuffle values between candidates while keeping the original answer.
#
# ## Result
#
# {{< proof P023 status >}}
#
# Keeping the pairs together supports the learned selection. This test uses three candidates and does not guarantee exact selection.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P023 evidence >}}
#
# ### Run this experiment
#
# {{< proof P023 script >}}
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
# ### Complete experiment code
#

# %%
"""Retrieve an argmax payload while preserving every candidate token.

Both root and item reductions are disabled. Compare held-out and training error,
then rotate only visible payloads to test whether the learned answer depends on
which payload belongs to the largest score.

Run this file with --help for seed, training-budget, and reporting options.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P023"
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
    output = model.predict([{"items": row["items"]} for row in rows]).to_pylist()
    return np.asarray([row["predictions"]["/answer"]["content"] for row in output])


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=48,
        n_layers=1,
        n_heads=4,
        reduction=None,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        items=rf.Branch(
            length=LENGTH,
            overflow="error",
            n_layers=2,
            n_heads=4,
            reduction=None,
            score=rf.Number,
            payload=rf.Number,
        ),
        answer=rf.Number(mask=True),
    )
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, rows=4096, seed=seed + 1),
        validate=partial(records, rows=1024, seed=seed + 2),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        max_steps=-1 if steps is None else steps,
        max_epochs=25,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model, datamodule=data)
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
    report(PROOF_ID, run, seed=23)

# %% [markdown]
# </details>
