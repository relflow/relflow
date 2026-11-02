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
#   duration_seconds: 51.465
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
#     evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     gpu: null
#     host: 45ea5bc411ed
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
#     relflow_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     relflow_source_sha256: 77953506049ed76c92adc5f32a9bcc32432a788068a4e366b2f78bcf709f29a2
#     script_sha256: 6a73b67e5766564c4923718a5f2951e3402fc194ff551af99567b95f6b2f10ab
#     source_sha256: 0c908510fd81c628ef751eafe8709b02de4ca76162747a1a68448d737a34ed8f
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   seed: 23
#   started_at: '2026-11-02T04:08:52.266422+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41386396488-1
#   candidate:
#     revision: 7dec5e860dc389c37c3642d8904867249e831773
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   baseline:
#     revision: 7dec5e860dc389c37c3642d8904867249e831773
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#   executor_revision: 349b2202e6390a5fe0cbdf09a0c5972e495351ed
#   manifest_sha256: 1f096d9b4dcdf1b2235963715e8b41c325d2a989f2cf15da7c154ddfb9bfb08d
#   passed: true
#   github_run:
#     event: workflow_dispatch
#     head_branch: main
#     head_sha: 7dec5e860dc389c37c3642d8904867249e831773
#     html_url: https://github.com/relflow/relflow/actions/runs/41386396488
#     path: .github/workflows/ci.yml
#     run_attempt: 1
#     run_id: 41386396488
#     workflow_id: 403231172
#   release: https://github.com/relflow/relflow/releases/tag/evidence-ci-41386396488-attempt-1
#   reports_sha256: fca05e4cdb79410b5090075bab03e4fc6d4f2fbc70e798f208976f4f1d8a7ff9
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
