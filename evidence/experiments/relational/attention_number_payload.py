# %% [markdown]
# ---
# title: Can a summary retain the winning item’s value?
# categories:
# - Argmax retrieval
# evidence-id: P022
# description: Predict the value attached to the highest-scoring candidate after compressing the candidates into learned
#   summaries.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
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
# evidence-result:
#   accelerator: cpu
#   checks:
#     Breaking pairing increases nRMSE by >= 0.35: true
#     Broken pairing nRMSE >= 0.90: true
#     Intact retrieval nRMSE <= 0.35: true
#   duration_seconds: 53.346
#   metrics:
#     baseline_rmse: 0.9911584043638657
#     broken_nrmse: 1.335483480792639
#     intact_nrmse: 0.19251400565879598
#     rmse: 0.19081187466646843
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 4a9c323274863bffa2585091cd2def3f7b4dc8014d6dd624db1ad2a50e73ac7e
#     cuda: null
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: 110f3d31b319
#     packages:
#       lightning: 2.6.4
#       numpy: 2.4.6
#       polars: 2.0.0
#       pyarrow: 24.0.0
#       pydantic: 2.13.4
#       relflow: 0.4.2
#       tensordict: 0.12.4
#       torch: 2.12.0+cpu
#       torchmetrics: 1.9.0
#     platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.41
#     python: 3.12.16
#     relflow_path: /tmp/runtime/lib/python3.12/site-packages/relflow
#     relflow_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     relflow_source_sha256: 77953506049ed76c92adc5f32a9bcc32432a788068a4e366b2f78bcf709f29a2
#     script_sha256: db40edf2e590a9c9385d509ff252a4d74b251b37ef3e49a68d13294f751cf4e4
#     source_sha256: 6ed80570ca1a1b2015176d20497d3e05179358f809fa30f3bf01a3686aaff4fd
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 23
#   started_at: '2026-11-02T17:48:32.965438+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41494093900-1
#   candidate:
#     payload_sha256: 2169c968a7733085889a6eed6e895bcb2590059bcc46243d49a61ffacc42f67f
#     revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   baseline:
#     payload_sha256: 2a6af4bd3869faac23f6b7327f2c59e8f5b367580b2f2fa609710958775c08ed
#     revision: 7dec5e860dc389c37c3642d8904867249e831773
#     wheel_sha256: d3da50b65e0c0c7b98059f095b82bf6bf5d0ea0c5ccd0f302015118075f1dd85
#   evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#   executor_revision: 92b3c887eabc598ee2fc6c9dba4e05b9166cb8ba
#   manifest_sha256: 4f0926f3fc366fd8d14bf728b2d06f4a475d61852679176704816c110ae51cfd
#   passed: true
#   github_run:
#     event: push
#     head_branch: main
#     head_sha: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     html_url: https://github.com/relflow/relflow/actions/runs/41494093900
#     path: .github/workflows/ci.yml
#     run_attempt: 1
#     run_id: 41494093900
#     workflow_id: 403231172
#   release: https://github.com/relflow/relflow/releases/tag/evidence-ci-41494093900-attempt-1
#   reports_sha256: b1bfee5622ffbd389b1777443dc05e6d8f08dccff9e2540a601f0c2608948594
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
# //| label: fig-evidence-attention-number-payload
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
# {{< evidence P022 status >}}
#
# The tested summaries retain useful score–value relationships. Exact retrieval, larger lists, and speed or memory savings are not established.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P022 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P022 script >}}
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
import relflow as rf
import torch
from reporting import report

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
    return np.asarray([row["predictions"]["answer"]["content"] for row in output])


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=48,
        n_layers=1,
        n_heads=4,
        reduction=rf.Attention(n_outputs=LENGTH),
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
    report(PROOF_ID, run, seed=23)

# %% [markdown]
# </details>
