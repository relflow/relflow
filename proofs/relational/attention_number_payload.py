# %% [markdown]
# ---
# title: Can a summary retain the winning item’s value?
# categories:
# - Argmax retrieval
# proof-id: P022
# description: Predict the value attached to the highest-scoring candidate after compressing the candidates
#   into learned summaries.
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
# //| label: fig-proof-attention-number-payload
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
# Shuffle values between candidates while keeping the original answer. Scores and values individually stay the same.
#
# ## Result
#
# {{< proof P022 status >}}
#
# The tested summaries retain useful score–value relationships. Exact retrieval, larger lists, and speed or memory savings are not established.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P022 evidence >}}
#
# ### Run this experiment
#
# {{< proof P022 script >}}
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
# The [pass-through proof](pass-through-number-payload.html) provides the simpler
# route comparison. Use exact preprocessing when retrieval must be exact.
#
# ### Complete experiment code
#

# %%
"""Retrieve the payload paired with the largest of three scores.

The root retains three learned Attention outputs and the item branch uses one.
Rotating payloads without rotating targets tests whether the score/payload
pairing, rather than their separate distributions, drives the prediction.

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

PROOF_ID = "P022"
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
        reduction=rf.Attention(n_outputs=LENGTH),
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        items=rf.Branch(
            length=LENGTH,
            overflow="error",
            n_layers=2,
            n_heads=4,
            reduction=rf.Attention(),
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
