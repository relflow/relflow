# %% [markdown]
# ---
# title: Is it copying by position instead of by ID?
# categories:
# - Associative recall
# evidence-id: P024
# description: Place source values and questions in matching positions, then ask the model to copy the values.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Matching IDs
#   metric:
#   - intact_nrmse
#   format: error
# - label: Changed IDs
#   metric:
#   - broken_nrmse
#   format: error
# evidence-group: Relational
# evidence-result:
#   accelerator: cpu
#   checks:
#     Aligned copying nRMSE <= 0.25: true
#     Broken identities retain copying nRMSE <= 0.25: true
#     Identity change alters nRMSE by <= 0.10: true
#   duration_seconds: 111.238
#   metrics:
#     broken_nrmse: 0.009778064240644495
#     corruption_gap: 0.001680398371379144
#     intact_nrmse: 0.008097665869265351
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 53ad8260b404253160952072bb12a450b0d6df32d6192f4a65bd7c5f4f7a98eb
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 8df5d8222351
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
#     script_sha256: c87185709344b4768fbfa7fff3753ad5bcea02b181c430653837a5c9a2493ebd
#     source_sha256: 184065700f85e3d713473e35f54b40d58cff7946fcba65a8d6e776b746f0b8de
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 17
#   started_at: '2026-11-01T01:36:26.388952+00:00'
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
# memory:
#   - {role: source, is_query: false, entity_id: K1, value: 0.4}
#   - {role: source, is_query: false, entity_id: K2, value: -0.8}
#   - {role: query, is_query: true, entity_id: K1, value: 0.4}
#   - {role: query, is_query: true, entity_id: K2, value: -0.8}
# ```
#
# Source values are visible. Query values are the hidden answers, selected by is_query.
#
# ```{typst}
# //| label: fig-evidence-aligned-position-control
# //| fig-cap: "Selector notes identify hidden values. Source values are visible; query values are hidden."
# //| fig-alt: "Model tree with association, memory, entity_id, role, is_query, value. Selector notes identify hidden values. Source values are visible; query values are hidden."
# #tree(node("association", kind: "root", width: 120pt, children: (
#   node("memory", kind: "branch", repeated: true, width: 120pt, children: (
#     node("entity_id", type: "Hash"),
#     node("role", type: "Category"),
#     node("is_query", type: "Boolean"),
#     node("value", type: "Number", width: 155pt, detail: "Source visible; query hidden"),
#   )),
# )))
# ```
#
# ## Comparison
#
# Change question IDs while leaving their positions and answers fixed. If accuracy survives, matching IDs were unnecessary.
#
# ## Result
#
# {{< evidence P024 status >}}
#
# Accuracy survives the changed IDs. Success on aligned records can therefore hide a positional shortcut.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P024 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P024 script >}}
#
# ### How it works
#
# The decoder has positional information, so it can learn to copy the first
# source to the first query and the second source to the second query. Rotating
# query identities leaves these target positions unchanged. A model following
# position should therefore remain accurate after this intervention.
#
# ### Remaining work
#
# Repeat seed calibration and retain this control alongside identity evidences.
# Its successful predictions cannot establish unseen-key lookup: the required
# information is already available through a fixed position relationship.
#
# ### Complete experiment code
#

# %%
"""Show why aligned copying is insufficient evidence of identity lookup.

The source and query halves have matching positions. A compressed model can
copy values by position even after query keys are changed. Successful copying
on both versions is the expected control behavior.

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

PROOF_ID = "P024"
PAIR_COUNT = 2


def records(*, rows: int, seed: int, namespace: str, broken_identity: bool = False) -> Iterator[dict]:
    """Generate unseen, observation-local key/value associations."""
    rng = np.random.default_rng(seed)
    for row in range(rows):
        keys = [f"{namespace}-{row:06d}-{pair:02d}" for pair in range(PAIR_COUNT)]
        values = rng.uniform(-1.0, 1.0, size=PAIR_COUNT)
        query_keys = list(keys)
        if broken_identity:
            query_keys = query_keys[1:] + query_keys[:1]
        source_order = rng.permutation(PAIR_COUNT)
        query_order = source_order
        source = [{"entity_id": keys[index], "value": float(values[index])} for index in source_order]
        target = [{"entity_id": query_keys[index], "value": float(values[index])} for index in query_order]
        memory = [{**item, "role": "source", "is_query": False} for item in source]
        memory.extend({**item, "role": "query", "is_query": True} for item in target)
        yield {"memory": memory}


def normalized_rmse(model: rf.Model, records: list[dict]) -> float:
    """Score only query coordinates; source values remain visible to the model."""
    output = model.predict(records).to_pylist()
    actual, predicted = [], []
    for row, result in zip(records, output, strict=True):
        for item, prediction in zip(row["memory"], result["predictions"]["memory"], strict=True):
            if item["is_query"]:
                actual.append(item["value"])
                predicted.append(prediction["value"]["content"])
    actual = np.asarray(actual)
    return float(np.sqrt(np.mean(np.square(np.asarray(predicted) - actual)) / np.mean(np.square(actual))))


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=64,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(),
        memory=rf.Branch(
            length=2 * PAIR_COUNT,
            n_layers=2,
            reduction=rf.Attention(),
            entity_id=rf.Hash(n_hashes=4, n_bands=8),
            role=rf.Category(p_unavailable=0.0),
            is_query=rf.Boolean,
            value=rf.Number(mask=rf.Mask(query="is_query", dropout=False, reconstruct=True), objective="mse"),
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
        max_steps=800 if steps is None else min(steps, 800),
        max_epochs=-1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        check_val_every_n_epoch=5,
    )
    test = list(records(rows=1024, seed=seed + 3, namespace="test"))
    broken = list(records(rows=1024, seed=seed + 3, namespace="test", broken_identity=True))
    intact_nrmse = normalized_rmse(model, test)
    broken_nrmse = normalized_rmse(model, broken)
    metrics = {
        "intact_nrmse": intact_nrmse,
        "broken_nrmse": broken_nrmse,
        "corruption_gap": broken_nrmse - intact_nrmse,
    }
    checks = {
        "Aligned copying nRMSE <= 0.25": intact_nrmse <= 0.25,
        "Broken identities retain copying nRMSE <= 0.25": broken_nrmse <= 0.25,
        "Identity change alters nRMSE by <= 0.10": abs(intact_nrmse - broken_nrmse) <= 0.10,
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=17)

# %% [markdown]
# </details>
