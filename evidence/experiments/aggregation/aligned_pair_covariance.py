# %% [markdown]
# ---
# title: Can it learn how two measurements vary together?
# categories:
# - Distribution statistics
# evidence-id: P006
# description: 'Predict covariance: whether paired measurements tend to rise together or move in opposite directions.'
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Correct pairs
#   metric:
#   - intact
#   - nrmse
#   format: error
# - label: Broken pairs
#   metric:
#   - corrupted
#   - nrmse
#   format: error
# evidence-group: Aggregation
# evidence-result:
#   accelerator: cpu
#   checks:
#     All measurements finite: true
#     Baseline RMSE above 0.35: true
#     Corruption preserves baseline: true
#     Covariance nRMSE below 0.35: true
#     Shuffled pairing raises nRMSE by at least 0.35: true
#   duration_seconds: 64.077
#   metrics:
#     corrupted:
#       baseline_rmse: 0.47722980830253015
#       nrmse: 1.3635274958463686
#       rmse: 0.6507159654579915
#     intact:
#       baseline_rmse: 0.47722980830253015
#       nrmse: 0.07266366666222125
#       rmse: 0.0346772677117708
#     steps: 1100
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 8adc85f341991f6ac967c300de453e309157e7f6bbee19ff7ec1729049f50cde
#     cuda: null
#     evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     gpu: null
#     host: e472c06f4c72
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
#     script_sha256: 16cedd4f3f6a272e73849066b2fd1f5a83d716c522d3c974977158d8d7210f22
#     source_sha256: d95e2c6bc3f1e5bd626967214d683bae43ff7474e574c9be97118338c1932e84
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   seed: 3508
#   started_at: '2026-11-02T04:05:33.868624+00:00'
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
#   - {x: -1, y: -1.4}
#   - {x: 1, y: 0.2}
#   - {x: -1, y: -0.2}
#   - {x: 1, y: 1.4}
#   - {x: -1, y: -1.4}
#   - {x: 1, y: 0.2}
#   - {x: -1, y: -0.2}
#   - {x: 1, y: 1.4}
# covariance: 0.8
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-aligned-pair-covariance
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, items, x, y, covariance. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("x", type: "Number"),
#     node("y", type: "Number"),
#   )),
#   node("covariance", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Compare correct pairs with pairs whose second measurements are shuffled, keeping the original answers. The separate values stay the same.
#
# ## Result
#
# {{< evidence P006 status >}}
#
# Accuracy depends on keeping each pair together. This tests covariance, not correlation, which also accounts for each measurement’s spread.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P006 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P006 script >}}
#
# ### How it works
#
# The repeated branch keeps `x` and `y` together and uses `reduction=None`
# to preserve their encoded coordinates. Learned root attention can combine that
# evidence before scalar decoding. Independent locations, scales, and a random
# correlation coefficient prevent either marginal alone from supplying covariance.
#
# The negative control circularly shifts only `y`, preserving the exact values in
# both fields but breaking their alignment. Labels stay unchanged, so error must
# increase when the model responds to the changed pairs. The target is population
# covariance, the mean centered product; Pearson correlation is not tested here.
# [Supplied cross-deviations](supplied-cross-deviations.html) isolate its reduction.
#
# ### Remaining work
#
# Repeat across seeds and check complete-pair permutation. Variable lengths,
# missing values, zero variance, and correlation normalization remain separate
# questions. Coordinate-mixing ablations would distinguish which mechanisms are
# necessary for the observed result.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""Aligned sibling fields should support covariance.

Learn the centered cross-moment and respond to pairing corruption.

Run: uv run python evidence/experiments/run.py P006"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P006"
ITEMS = 8


def covariance_records(*, rows: int, seed: int) -> Iterator[dict]:
    """Draw paired fields with controlled means, scales, and covariance."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        x_basis = rng.normal(size=ITEMS)
        x_basis -= x_basis.mean()
        x_basis /= np.sqrt(np.mean(np.square(x_basis)))
        orthogonal = rng.normal(size=ITEMS)
        orthogonal -= orthogonal.mean()
        orthogonal -= np.mean(orthogonal * x_basis) * x_basis
        orthogonal /= np.sqrt(np.mean(np.square(orthogonal)))
        correlation = float(rng.uniform(-0.88, 0.88))
        y_basis = correlation * x_basis + np.sqrt(1.0 - correlation * correlation) * orthogonal
        x_scale = float(rng.uniform(0.35, 1.45))
        y_scale = float(rng.uniform(0.35, 1.45))
        x_location = float(rng.uniform(-0.8, 0.8))
        y_location = float(rng.uniform(-0.8, 0.8))
        x_values = x_location + x_scale * x_basis
        y_values = y_location + y_scale * y_basis
        yield {
            "items": [{"x": float(x), "y": float(y)} for x, y in zip(x_values, y_values, strict=True)],
            "covariance": correlation * x_scale * y_scale,
            "correlation": correlation,
        }


def shuffle_y(observations: list[dict], *, seed: int) -> list[dict]:
    """Break within-row pairing while preserving every marginal value."""
    rng = np.random.default_rng(seed)
    rows = deepcopy(observations)
    changed = 0
    for row in rows:
        items = row["items"]
        offset = int(rng.integers(1, len(items)))
        shifted = np.roll([item["y"] for item in items], offset)
        changed += sum((item["y"] != y for item, y in zip(items, shifted, strict=True)))
        row["items"] = [{"x": item["x"], "y": float(y)} for item, y in zip(items, shifted, strict=True)]
    if changed == 0:
        raise ValueError("pair shuffle did not change any item-local associations")
    return rows


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["covariance"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["covariance"] for row in test], dtype=np.float64)
    baseline = rmse(actual, float(np.asarray([row["covariance"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline, "nrmse": measured / baseline}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(covariance_records(rows=1536, seed=seed + 1))
    test = list(covariance_records(rows=768, seed=seed + 3))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        items=rf.Branch(length=ITEMS, n_layers=2, reduction=None, x=rf.Number, y=rf.Number),
        covariance=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        train=lambda: covariance_records(rows=1536, seed=seed + 1),
        validate=lambda: covariance_records(rows=384, seed=seed + 2),
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
        max_steps=1100 if steps is None else min(steps, 1100),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer

    # Evaluate held-out answers and retain their original labels in corruption controls.
    intact = score(train=train, test=test, predicted=prediction(model, test))
    shuffled = shuffle_y(test, seed=seed + 4)
    corrupted = score(train=train, test=shuffled, predicted=prediction(model, shuffled))
    calibration = np.asarray(
        [
            intact["rmse"],
            intact["baseline_rmse"],
            intact["nrmse"],
            corrupted["rmse"],
            corrupted["baseline_rmse"],
            corrupted["nrmse"],
        ]
    )
    metrics = {"intact": intact, "corrupted": corrupted, "steps": trainer.global_step}
    checks = {
        "All measurements finite": bool(np.isfinite(calibration).all()),
        "Baseline RMSE above 0.35": bool(intact["baseline_rmse"] > 0.35),
        "Corruption preserves baseline": bool(
            np.isclose(intact["baseline_rmse"], corrupted["baseline_rmse"], atol=1e-12, rtol=0.0)
        ),
        "Covariance nRMSE below 0.35": bool(intact["nrmse"] < 0.35),
        "Shuffled pairing raises nRMSE by at least 0.35": bool(corrupted["nrmse"] >= intact["nrmse"] + 0.35),
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3508)

# %% [markdown]
# </details>
