# %% [markdown]
# ---
# title: Can it average prepared squared differences?
# categories:
# - Distribution statistics
# evidence-id: P009
# description: Supply each value’s squared distance from the average, then predict the average of those distances.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Average prediction
#   metric:
#   - measured
#   - nrmse
#   format: error
# evidence-group: Aggregation
# evidence-result:
#   accelerator: cpu
#   checks:
#     Variance nRMSE below 0.20: true
#   duration_seconds: 8.424
#   metrics:
#     measured:
#       baseline_rmse: 0.4800716415268894
#       nrmse: 0.012960816013301747
#       rmse: 0.0062221202190337645
#     steps: 350
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 281247993c5e441318ad3f1cff3c48c98cd2d7a947dcad5ccdad54601d4d14d9
#     cuda: null
#     evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     gpu: null
#     host: 4725e7a57374
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
#     script_sha256: 46582a7257c1ccb1743e6813d4eed7f5a6fb471b226e74d43faba38a9ba0bc7a
#     source_sha256: b5187a1ae0a022cf5812744968803b1c6a2bd540f17e0949bfea3845558963d7
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   seed: 3500
#   started_at: '2026-11-02T04:05:52.243326+00:00'
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
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
# variance: 1.0
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-supplied-squared-deviations
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, items, squared_deviation, variance. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("squared_deviation", width: 150pt, type: "Number"),
#   )),
#   node("variance", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Compare the prediction with the correct variance on new records.
#
# ## Result
#
# {{< evidence P009 status >}}
#
# This tests the final averaging step. The inputs already contain the difficult parts of calculating variance.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P009 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P009 script >}}
#
# ### How it works
#
# The generator computes `(value - mean(value)) ** 2` before encoding. The
# item branch uses `attention=None` and `rf.Mean()`; the root uses learned
# attention to predict the average of eight supplied squared deviations.
#
# Mean averages encoded tokens, so the decoder still has to learn their numerical
# interpretation. Passing this diagnostic supports reduction and decoding of
# sufficient statistics. It does not prove learned centering or squaring; the
# [raw-value variance evidence](raw-value-variance.html) checks those requirements.
#
# ### Remaining work
#
# Repeat this control across seeds and broaden collection lengths and edge
# cases. Keep population-versus-sample variance semantics explicit, and retain the
# raw-value comparison when claiming that dispersion was learned from inputs.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""Supplied squared deviations isolate collection averaging.

Mean learns population variance once squared deviations are supplied.

Run: uv run python evidence/experiments/run.py P009"""

from __future__ import annotations

from collections.abc import Iterator

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P009"
ITEMS = 8


def dispersion_records(*, rows: int, seed: int) -> Iterator[dict]:
    """Draw rows whose location and spread are statistically independent."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        basis = rng.normal(size=ITEMS)
        basis -= basis.mean()
        basis /= np.sqrt(np.mean(np.square(basis)))
        location = float(rng.uniform(-0.8, 0.8))
        scale = float(rng.uniform(0.12, 1.25))
        values = location + scale * basis
        items = [{"squared_deviation": float(np.square(value - location))} for value in values]
        yield {"items": items, "variance": scale * scale}


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["variance"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["variance"] for row in test], dtype=np.float64)
    baseline = rmse(actual, float(np.asarray([row["variance"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline, "nrmse": measured / baseline}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(dispersion_records(rows=768, seed=seed + 1))
    test = list(dispersion_records(rows=384, seed=seed + 3))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        items=rf.Branch(length=ITEMS, n_layers=2, attention=None, reduction=rf.Mean(), squared_deviation=rf.Number),
        variance=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        train=lambda: dispersion_records(rows=768, seed=seed + 1),
        validate=lambda: dispersion_records(rows=192, seed=seed + 2),
        seed=seed,
    )
    model.fit(
        datamodule,
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=350 if steps is None else min(steps, 350),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer

    # Evaluate held-out answers and retain their original labels in corruption controls.
    measured = score(train=train, test=test, predicted=prediction(model, test))
    metrics = {"measured": measured, "steps": trainer.global_step}
    checks = {"Variance nRMSE below 0.20": bool(measured["nrmse"] < 0.2)}
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3500)

# %% [markdown]
# </details>
