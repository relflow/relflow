# %% [markdown]
# ---
# title: Does each subtotal stay with its own item?
# categories:
# - Item alignment
# evidence-id: P030
# description: Predict quantity times price separately for every real item in an order.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Correct item answers
#   metric:
#   - aligned_nrmse
#   format: error
# - label: Shuffled item answers
#   metric:
#   - permuted_target_nrmse
#   format: error
# evidence-group: Relational
# evidence-result:
#   accelerator: cpu
#   checks:
#     Aligned subtotal nRMSE <= 0.25: true
#     Inferred flags identify every real coordinate: true
#     Output retains each real item without padding: true
#     Permuting targets increases nRMSE by >= 0.50: true
#     Target permutation preserves visible inputs: true
#   duration_seconds: 71.635
#   metrics:
#     aligned_nrmse: 0.013806636856808791
#     baseline_rmse: 1.0618151651706766
#     permuted_target_nrmse: 1.1500412680652632
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: c965b8fccf7d01de5712e0b942740f4b7b4435689123ca9277224ca13cbd37f6
#     cuda: null
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: f130d4b8a517
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
#     script_sha256: 3e80fa3e1f31be6d059cd23f3212f54666ce81ea37b6ebc5f7da5fbdbbcb4f89
#     source_sha256: 54745b0d865d7b5f0c68f280b48a3f76f36663cf472d5ed26298337bd639b9d3
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 31
#   started_at: '2026-11-02T17:49:32.984278+00:00'
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
#   - {quantity: 1.5, unit_price: 1.2, subtotal: 1.8}
#   - {quantity: 0.5, unit_price: -1.0, subtotal: -0.5}
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-repeated-subtotal-reconstruction
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with order, items, quantity, unit_price, subtotal. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("order", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("quantity", type: "Number"),
#     node("unit_price", type: "Number"),
#     node("subtotal", kind: "target", type: "Number", width: 150pt,),
#   )),
# )))
# ```
#
# ## Comparison
#
# Shuffle the expected subtotals between items while keeping visible inputs fixed. Also check empty orders and retained item positions.
#
# ## Result
#
# {{< evidence P030 status >}}
#
# Predictions follow the correct items, and empty orders stay empty. The arithmetic is approximate and should not replace exact billing calculations.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P030 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P030 script >}}
#
# ### How it works
#
# Shared coordinates expose each target's own visible siblings to the decoder.
# A paired control keeps every visible order unchanged but permutes target
# subtotals between items. Predictions are therefore identical, while comparison
# against the corrupted targets should become substantially worse.
#
# Lengths cycle from zero through five. Each output retains exactly the input's
# item count without padding; every real item must be marked `inferred=True`.
#
# ### Remaining work
#
# Report accuracy separately by nonempty length and repeat three paired core
# seeds plus ten calibration seeds. The evidence guide also flags a mismatch between
# its exact-product generator and the noisy-product proposal; that broader noisy
# task has not been established by this case.
#
# ### Complete experiment code
#

# %%
"""Reconstruct quantity times unit price at each original item coordinate.

Rows range from empty to five items. Permuting only target subtotals tests local
alignment; output lengths and inferred flags identify the retained coordinates.

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

PROOF_ID = "P030"
LENGTH = 5


def records(*, rows: int, seed: int, permute_targets: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    target_rng = np.random.default_rng(seed + 10000)
    for row in range(rows):
        length = row % (LENGTH + 1)
        quantity = rng.uniform(0.25, 2.0, size=length)
        unit_price = rng.uniform(-1.5, 1.5, size=length)
        subtotal = quantity * unit_price
        if permute_targets and length > 1:
            subtotal = subtotal[target_rng.permutation(length)]
        yield {
            "items": [
                {"quantity": float(item_quantity), "unit_price": float(item_price), "subtotal": float(item_subtotal)}
                for item_quantity, item_price, item_subtotal in zip(quantity, unit_price, subtotal, strict=True)
            ]
        }


def inputs(rows: list[dict]) -> list[dict]:
    return [
        {"items": [{key: value for key, value in item.items() if key != "subtotal"} for item in row["items"]]}
        for row in rows
    ]


def targets(rows: list[dict]) -> np.ndarray:
    return np.asarray([item["subtotal"] for row in rows for item in row["items"]])


def prediction(model: rf.Model, rows: list[dict]) -> list[list[dict]]:
    output = model.predict(inputs(rows)).to_pylist()
    return [[item["subtotal"] for item in row["predictions"]["items"]] for row in output]


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model.xs(
        items=rf.Branch(
            length=LENGTH,
            overflow="error",
            n_layers=2,
            quantity=rf.Number,
            unit_price=rf.Number,
            subtotal=rf.Number(mask=True, objective="mse"),
        ),
    )
    data = rf.SyntheticDataModule(
        train=partial(records, rows=4096, seed=seed + 1),
        validate=partial(records, rows=1024, seed=seed + 2),
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
    broken = list(records(rows=2048, seed=seed + 3, permute_targets=True))
    coordinates = prediction(model, test)
    lengths = [len(row["items"]) for row in test]
    predicted = np.asarray([value["content"] for row in coordinates for value in row])
    actual = targets(test)
    baseline = rmse(actual, float(targets(train).mean()))
    aligned = rmse(actual, predicted) / baseline
    broken_error = rmse(targets(broken), predicted) / baseline
    inferred = [[bool(value["inferred"]) for value in row] for row in coordinates]
    expected = [[True] * length for length in lengths]
    metrics = {"aligned_nrmse": aligned, "permuted_target_nrmse": broken_error, "baseline_rmse": baseline}
    checks = {
        "Target permutation preserves visible inputs": inputs(test) == inputs(broken),
        "Output retains each real item without padding": [len(row) for row in coordinates] == lengths,
        "Inferred flags identify every real coordinate": inferred == expected,
        "Aligned subtotal nRMSE <= 0.25": aligned <= 0.25,
        "Permuting targets increases nRMSE by >= 0.50": broken_error >= aligned + 0.50,
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=31)

# %% [markdown]
# </details>
