# %% [markdown]
# ---
# title: Can new IDs connect two lists?
# categories:
# - Sibling entity transfer
# evidence-id: P037
# description: Transfer values between separately ordered lists using IDs that were absent from training.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Items retained
#   metric:
#   - preserved_nrmse
#   format: error
# - label: Compressed route
#   metric:
#   - compressed_nrmse
#   format: error
# - label: Changed IDs, items retained
#   metric:
#   - preserved_broken_nrmse
#   format: error
# evidence-group: Relational
# evidence-result:
#   accelerator: cpu
#   checks:
#     Broken identities nRMSE >= 0.80: true
#     Finite errors: true
#     Identity corruption increases nRMSE by >= 0.50: true
#     Preserved transfer calibration nRMSE <= 1.10: true
#     Preserved transfer nRMSE <= 0.25: true
#     Preserving Hash tokens improves nRMSE by >= 0.15: true
#   duration_seconds: 100.883
#   metrics:
#     compressed_broken_nrmse: 0.7253847793525894
#     compressed_nrmse: 0.72597079594237
#     preserved_broken_nrmse: 1.4417964021251342
#     preserved_nrmse: 0.12011518617091581
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 1093be31b1fc06f20485354b9c2b2163750fe41833c38920dcfa373bd405615a
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 9eb50728247d
#     packages:
#       lightning: 2.6.4
#       numpy: 2.4.6
#       polars: 2.0.0
#       pyarrow: 24.0.0
#       pydantic: 2.13.4
#       relflow: 0.4.1
#       tensordict: 0.12.4
#       torch: 2.12.0+cpu
#       torchmetrics: 1.9.0
#     platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.41
#     python: 3.12.16
#     relflow_path: /tmp/runtime/lib/python3.12/site-packages/relflow
#     relflow_revision: 41d6f5fb74c0ba9f0be104c896913a55adb125f5
#     relflow_source_sha256: 00fdf397aeefb3a4e705d431850e0a6294eef142989513fddcb903bb1aefa873
#     script_sha256: a153f7508ba12b32d4cdb2c04367cdbb3aad05da37744201dca9b3f87aef5b78
#     source_sha256: 4ec968757c5fe290af175b5fb81eabcc1b77494062a22f043772e764f2177789
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 17
#   started_at: '2026-11-01T01:44:54.449218+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41261482839-1
#   candidate:
#     revision: 41d6f5fb74c0ba9f0be104c896913a55adb125f5
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   baseline:
#     revision: 1add6d8919e686dc63b7fa6c4cdbdc437be2b570
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#   manifest_sha256: f4b3bfe706e43e38785ab5bed94c3f51eb44fe7314d0a1b2f0dd3c6f3f7bef92
#   passed: true
#   github_run:
#     event: pull_request
#     head_branch: dev/evidence-in-docs
#     head_sha: 8bb7764d8186c743eb554cb4a91dedb205ad4cb9
#     html_url: https://github.com/relflow/relflow/actions/runs/41261482839
#     pull_requests:
#     - base_sha: 1add6d8919e686dc63b7fa6c4cdbdc437be2b570
#       head_sha: 8bb7764d8186c743eb554cb4a91dedb205ad4cb9
#       number: 64
#     run_attempt: 1
#     run_id: 41261482839
#   release: https://github.com/relflow/docs/releases/tag/ci-41261482839-attempt-1
#   reports_sha256: bc67a1cb3725cbf9c5a8deb65844297d70f6d2122cd3b7de5c1d15f56603864f
# ---
#
# ## Example
#
# ```yaml
# source:
#   - {entity_id: row-42-K1, value: 0.7}
#   - {entity_id: row-42-K2, value: -0.2}
# target:
#   - {entity_id: row-42-K2, value: -0.2}
#   - {entity_id: row-42-K1, value: 0.7}
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-hash-identity
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with association, source, entity_id, value, target, entity_id, value. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("association", kind: "root", width: 120pt, children: (
#   node("source", kind: "branch", repeated: true, width: 120pt, children: (
#     node("entity_id", type: "Hash"),
#     node("value", type: "Number"),
#   )),
#   node("target", kind: "branch", repeated: true, width: 120pt, children: (
#     node("entity_id", type: "Hash"),
#     node("value", kind: "target", type: "Number", width: 150pt,),
#   )),
# )))
# ```
#
# ## Comparison
#
# Compare keeping encoded items with compressing them. Change target IDs while retaining original answers.
#
# ## Result
#
# {{< evidence P037 status >}}
#
# Keeping items works better here, and changed IDs break the lookup. This small learned example does not guarantee an exact join or define missing and duplicate-ID behavior.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P037 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P037 script >}}
#
# ### How it works
#
# Hash preserves equality within an encoded batch without a learned vocabulary.
# Coordinate-local mixing binds source identity to value, and retained source
# context remains available to queries conditioned on visible target identities.
# The key-rotation control leaves target values and source records unchanged
# while breaking the intended match.
#
# ### Remaining work
#
# Repeat three core seeds and ten calibration seeds, then sweep 2, 4, 8, and 16
# entities. Add explicit permutation/equivariance tests, absent sources, duplicate
# keys, padding, and Hash collisions. The [Category control](category-identity.html)
# uses recurring identities and succeeds under the tested compression. Exact
# production retrieval can use a preprocessing join.
#
# ### Complete experiment code
#

# %%
"""Transfer values between sibling branches using fresh Hash identities.

Every split uses unseen names and independent source/target order. Compare
compressed and preserved routes, then rotate query keys while retaining targets
to test whether improved transfer really depends on identity.

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

PROOF_ID = "P037"
PAIR_COUNT = 2


def records(*, rows: int, seed: int, namespace: str, broken_identity: bool = False) -> Iterator[dict]:
    """Generate observation-local key/value associations in sibling branches."""
    rng = np.random.default_rng(seed)
    for row in range(rows):
        keys = [f"{namespace}-{row:06d}-{pair:02d}" for pair in range(PAIR_COUNT)]
        values = rng.uniform(-1.0, 1.0, size=PAIR_COUNT)
        query_keys = list(keys)
        if broken_identity:
            query_keys = query_keys[1:] + query_keys[:1]
        source_order = rng.permutation(PAIR_COUNT)
        query_order = rng.permutation(PAIR_COUNT)
        source = [{"entity_id": keys[index], "value": float(values[index])} for index in source_order]
        target = [{"entity_id": query_keys[index], "value": float(values[index])} for index in query_order]
        yield {"source": source, "target": target}


def normalized_rmse(model: rf.Model, records: list[dict]) -> float:
    actual = np.asarray([item["value"] for row in records for item in row["target"]])
    output = model.predict(records).to_pylist()
    predicted = np.asarray([item["value"]["content"] for row in output for item in row["predictions"]["target"]])
    return float(np.sqrt(np.mean(np.square(actual - predicted)) / np.mean(np.square(actual))))


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    test = list(records(rows=1024, seed=seed + 3, namespace="test"))
    broken = list(records(rows=1024, seed=seed + 3, namespace="test", broken_identity=True))
    metrics = {}
    # Fit both routes from the same seed and data, changing only reduction.
    for route, reduction in (("compressed", rf.Attention()), ("preserved", None)):
        lit.seed_everything(seed, workers=True)
        model = rf.Model(
            d_model=64,
            n_layers=2,
            n_heads=4,
            reduction=reduction,
            source=rf.Branch(
                length=PAIR_COUNT,
                n_layers=2,
                reduction=reduction,
                entity_id=rf.Hash(n_hashes=4, n_bands=8),
                value=rf.Number,
            ),
            target=rf.Branch(
                length=PAIR_COUNT,
                n_layers=2,
                reduction=reduction,
                entity_id=rf.Hash(n_hashes=4, n_bands=8),
                value=rf.Number(mask=True, objective="mse"),
            ),
        )
        data = rf.SyntheticDataModule(
            train=partial(records, rows=2048, seed=seed + 1, namespace="train"),
            validate=partial(records, rows=512, seed=seed + 2, namespace="validate"),
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
            max_steps=600 if steps is None else min(steps, 600),
            max_epochs=-1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            deterministic=True,
            check_val_every_n_epoch=5,
        )
        metrics[f"{route}_nrmse"] = normalized_rmse(model, test)
        metrics[f"{route}_broken_nrmse"] = normalized_rmse(model, broken)
    compressed = metrics["compressed_nrmse"]
    intact = metrics["preserved_nrmse"]
    broken_error = metrics["preserved_broken_nrmse"]
    checks = {
        "Finite errors": bool(np.isfinite(list(metrics.values())).all()),
        "Preserved transfer calibration nRMSE <= 1.10": intact <= 1.10,
        "Preserved transfer nRMSE <= 0.25": intact <= 0.25,
        "Broken identities nRMSE >= 0.80": broken_error >= 0.80,
        "Identity corruption increases nRMSE by >= 0.50": broken_error - intact >= 0.50,
        "Preserving Hash tokens improves nRMSE by >= 0.15": intact <= compressed - 0.15,
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=17)

# %% [markdown]
# </details>
