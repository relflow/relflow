# %% [markdown]
# ---
# title: Can it infer an average and compare each item?
# categories:
# - Peer-relative inference
# evidence-id: P035
# description: Predict each item’s difference from the list average without supplying that average.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Original lists
#   metric:
#   - intact_nrmse
#   format: error
# - label: Shifted lists
#   metric:
#   - translated_nrmse
#   format: error
# evidence-group: Relational
# evidence-result:
#   accelerator: cpu
#   checks:
#     Baseline RMSE > 0.000001: true
#     Common-translated nRMSE < 0.45: true
#     Finite calibration and model metrics: true
#     Normalized permutation drift < 0.20: true
#     Permutation preserves the target multiset: true
#     Training deviations are exactly centered: true
#     Translation preserves all targets: true
#     Translation preserves target shape: true
#     Ungrouped deviation nRMSE < 0.35: true
#   duration_seconds: 86.654
#   metrics:
#     baseline_rmse: 0.731698187191497
#     intact_nrmse: 0.059908823835349456
#     intact_rmse: 0.043835177797099945
#     permutation_drift: 0.008894648616558797
#     translated_nrmse: 0.14538617005420465
#     translated_rmse: 0.10637879707137625
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: f5bea0518d91abec222f96d8c2283ad9a8b64e467a0ddfc296f382bae9202188
#     cuda: null
#     evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     gpu: null
#     host: 6862e0ca0101
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
#     script_sha256: 98bdba9fac6f605cb1446e2e2f69c260f4906d411d381edf929de17d715c3b36
#     source_sha256: 6d2fc8abf15cd07a9a407e55117cc5fdf7bc869d89755c9ab97db8fb65fa6bc6
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   seed: 3710
#   started_at: '2026-11-02T04:16:35.948836+00:00'
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
#   - {value: 1.5, deviation: -1.0}
#   - {value: 1.9, deviation: -0.6}
#   - {value: 2.3, deviation: -0.2}
#   - {value: 2.7, deviation: 0.2}
#   - {value: 3.1, deviation: 0.6}
#   - {value: 3.5, deviation: 1.0}
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-ungrouped-peer-deviation
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with collection, items, value, deviation. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("collection", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("value", type: "Number"),
#     node("deviation", kind: "target", type: "Number", width: 150pt,),
#   )),
# )))
# ```
#
# ## Comparison
#
# Shift the whole list equally: differences should stay fixed. Reorder items: predictions should move with their items.
#
# ## Result
#
# {{< evidence P035 status >}}
#
# The model approximately combines list information with each item’s value. This result covers six-item lists, rather than arbitrary lengths or exact arithmetic.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P035 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P035 script >}}
#
# ### How it works
#
# The learned summary can provide collection context to a decoder conditioned
# on each item's visible value. Wide random shifts between rows make an item's
# value alone a poor predictor of its deviation. Adding one constant to all six
# values preserves every target, and permuting whole items should reorder the
# predictions in exactly the same way.
#
# ### Remaining work
#
# Repeat three core seeds and ten calibration seeds. Test varying cardinality,
# empty collections, and missing values, and tighten the equivariance contract.
# The [grouped case](grouped-peer-deviation.html) adds membership selection;
# the [supplied-mean control](supplied-peer-mean-control.html) removes aggregation.
#
# ### Complete experiment code
#

# %%
"""Infer every item's deviation from the mean of six raw peer values.

Wide random row locations make the item's own value a poor shortcut. A shared
summary must support the repeated predictions. Common translations preserve
targets, while whole-item permutations should permute the answers with them.

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

PROOF_ID = "P035"
ITEMS = 6


def records(*, rows: int, seed: int) -> Iterator[dict]:
    """Generate collections with independent locations and centered residuals."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        basis = rng.normal(size=ITEMS)
        basis -= basis.mean()
        basis /= np.sqrt(np.mean(np.square(basis)))
        scale = float(rng.uniform(0.25, 1.1))
        mean = float(rng.uniform(-3.5, 3.5))
        deviations = scale * basis
        values = mean + deviations
        items: list[dict[str, float]] = []
        for value, deviation in zip(values, deviations, strict=True):
            item = {"value": float(value), "deviation": float(deviation)}

            items.append(item)
        yield {"items": items}


def targets(rows: list[dict]) -> np.ndarray:
    return np.asarray([item["deviation"] for row in rows for item in row["items"]])


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    inputs = [
        {"items": [{key: value for key, value in item.items() if key != "deviation"} for item in row["items"]]}
        for row in rows
    ]
    output = model.predict(inputs).to_pylist()
    return np.asarray([item["deviation"]["content"] for row in output for item in row["predictions"]["items"]])


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def translate(rows: list[dict], seed: int) -> list[dict]:
    """Shift visible values while retaining the original relative targets."""
    rng = np.random.default_rng(seed)
    result = []
    for row in rows:
        offset = float(rng.uniform(-2.5, 2.5))
        result.append({"items": [{**item, "value": item["value"] + offset} for item in row["items"]]})
    return result


def permute_items(rows: list[dict], seed: int) -> tuple[list[dict], np.ndarray]:
    """Reorder complete records and retain indices for comparing predictions."""
    rng = np.random.default_rng(seed)
    result, flattened = [], []
    for index, row in enumerate(rows):
        order = rng.permutation(ITEMS)
        result.append({"items": [row["items"][i] for i in order]})
        flattened.extend(index * ITEMS + order)
    return result, np.asarray(flattened)


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=48,
        n_layers=3,
        n_heads=4,
        reduction=None,
        items=rf.Branch(
            length=ITEMS,
            overflow="error",
            n_layers=2,
            reduction=rf.Attention(n_outputs=1, n_layers=2),
            value=rf.Number,
            deviation=rf.Number(mask=True, objective="mse", n_linear=2),
        ),
    )
    data = rf.SyntheticDataModule(
        train=partial(records, rows=1536, seed=seed + 1),
        validate=partial(records, rows=384, seed=seed + 2),
        seed=seed,
    )
    model.fit(
        data,
        batch_size=128,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
        accelerator=accelerator,
        max_steps=900 if steps is None else min(steps, 900),
        max_epochs=-1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    train = list(records(rows=1536, seed=seed + 1))
    test = list(records(rows=768, seed=seed + 3))
    actual = targets(test)
    baseline = rmse(actual, float(targets(train).mean()))
    predicted = predict(model, test)
    intact_error = rmse(actual, predicted)
    intact_nrmse = intact_error / baseline
    metrics = {"intact_rmse": intact_error, "baseline_rmse": baseline, "intact_nrmse": intact_nrmse}
    checks = {"Baseline RMSE > 0.000001": baseline > 1e-6}
    translated = translate(test, seed + 4)
    translated_error = rmse(targets(translated), predict(model, translated))
    translated_nrmse = translated_error / baseline
    permuted, order = permute_items(test, seed + 5)
    drift = rmse(predict(model, permuted), predicted[order]) / baseline
    metrics.update(translated_rmse=translated_error, translated_nrmse=translated_nrmse, permutation_drift=drift)
    checks.update(
        {
            "Training deviations are exactly centered": abs(float(targets(train).mean())) < 1e-12,
            "Translation preserves target shape": targets(translated).shape == actual.shape,
            "Translation preserves all targets": rmse(targets(translated), actual) == 0.0,
            "Permutation preserves the target multiset": sorted(targets(permuted).tolist()) == sorted(actual.tolist()),
            "Normalized permutation drift < 0.20": drift < 0.20,
        }
    )
    checks.update(
        {
            "Ungrouped deviation nRMSE < 0.35": intact_nrmse < 0.35,
            "Common-translated nRMSE < 0.45": translated_nrmse < 0.45,
        }
    )
    checks["Finite calibration and model metrics"] = bool(np.isfinite(list(metrics.values())).all())
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3710)

# %% [markdown]
# </details>
