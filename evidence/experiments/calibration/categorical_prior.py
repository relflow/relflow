# %% [markdown]
# ---
# title: Can it learn a 90% probability without a useful input?
# categories:
# - Calibration
# evidence-id: P060
# description: Generate yes answers 90% of the time, independently of the input. The right probability is therefore
#   90%.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: True yes probability
#   metric:
#   - prior
#   format: percent
# - label: Predicted, known inputs
#   metric:
#   - capacity_2_known
#   - mean_probability
#   format: percent
# - label: Predicted, new inputs
#   metric:
#   - capacity_2_novel
#   - mean_probability
#   format: percent
# evidence-group: Calibration
# evidence-result:
#   accelerator: cpu
#   checks:
#     'capacity_128: evaluation preserves vocabulary and counts': true
#     'capacity_128_known: excess expected NLL below 0.02': true
#     'capacity_128_known: probability RMSE below 0.06': true
#     'capacity_128_novel: excess expected NLL below 0.02': true
#     'capacity_128_novel: probability RMSE below 0.06': true
#     'capacity_2: evaluation preserves vocabulary and counts': true
#     'capacity_2_known: excess expected NLL below 0.02': true
#     'capacity_2_known: probability RMSE below 0.06': true
#     'capacity_2_novel: excess expected NLL below 0.02': true
#     'capacity_2_novel: probability RMSE below 0.06': true
#   duration_seconds: 24.585
#   metrics:
#     capacity_128_known:
#       excess_nll: 0.003848288849967796
#       mean_probability: 0.9214600841660285
#       prior_rmse: 0.023813673134784153
#     capacity_128_novel:
#       excess_nll: 0.0008281659894981419
#       mean_probability: 0.9117669463157654
#       prior_rmse: 0.01176694631576536
#     capacity_2_known:
#       excess_nll: 0.0020148409127090927
#       mean_probability: 0.9179295238791383
#       prior_rmse: 0.017960789788258682
#     capacity_2_novel:
#       excess_nll: 0.001916919878346579
#       mean_probability: 0.917550802230835
#       prior_rmse: 0.017550802230834935
#     oracle_nll: 0.3250829733914482
#     prior: 0.9
#     weighted_optimum: 0.75
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 563840f2eabed8b426e608be2bfe733d903104b23e1802c9f71596d5e4d3d4e4
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 52b1c5d38c3d
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
#     script_sha256: 1a26bbe39041c8799e14e279d3c3b12bc7b60ad26fd9514c1657ab3c2a547c5b
#     source_sha256: 6304e0b2b7ac4f7ce35dbc63b121c34d2655f6947e897a20b0d8d52da1a59de9
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 7601
#   started_at: '2026-11-01T01:47:54.438655+00:00'
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
# x: 0.0
# entity: id-12
# label: yes
# ```
#
# yes or no is a sampled hidden answer. A single answer does not determine the true probability.
#
# ```{typst}
# //| label: fig-evidence-categorical-prior
# //| fig-cap: "Amber cards are hidden prediction targets. The answer is sampled; its population probability is 90%."
# //| fig-alt: "Model tree with record, x, entity, label. Amber cards are hidden prediction targets. The answer is sampled; its population probability is 90%."
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number", detail: "Constant"),
#   node("entity", type: "Category", detail: "Independent identity"),
#   node("label", kind: "target", type: "Category", detail: "90/10 prevalence"),
# )))
# ```
#
# ## Comparison
#
# Compare known and new input identities, using two different internal output allocations.
#
# ## Result
#
# {{< evidence P060 status >}}
#
# The latest model learns probabilities near the population rate. Accuracy alone would not reveal whether its confidence was appropriate; this does not establish general probability calibration.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P060 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P060 script >}}
#
# ### Training and evaluation
#
# The evidence forces internal output allocations of 2 and 128 as experimental
# controls; users do not configure storage. Both use the `xs` preset, identical
# 32,768-row training streams, and 400 updates. Validation has its own seed.
# Each held-out evaluation has 8,192
# rows, with known or disjoint input IDs. Compare written probabilities with
# the true 0.9 prior using RMSE and expected log loss, not sample accuracy.
# The constant 0.9 oracle and the analytic weighted-loss optimum are controls.
# Gates are provisional and declared before the acceptance seed panel.
#
# ### Remaining work
#
# A ten-seed calibration panel and feature-dependent imbalanced populations
# remain necessary. This two-label evidence does not establish rare-label recall
# or calibration of other tensorfield losses.
#
# ### Complete experiment code
#

# %%
"""P060: recover an imbalanced population prior with uninformative inputs."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from relflow.tensorfields.shared.resize import Resize
from reporting import report

PROOF_ID = "P060"
BUDGET = 400


def records(*, rows: int, seed: int, novel: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for entity, positive in zip(rng.integers(0, 4096, rows), rng.random(rows) < 0.9, strict=True):
        yield {"x": 0.0, "entity": f"{'new' if novel else 'id'}-{entity}", "label": "yes" if positive else "no"}


def build(size: int) -> rf.Model:
    model = rf.Model.xs(
        x=rf.Number,
        entity=rf.Category(),
        label=rf.Category(mask=True),
    )
    # Experimental storage control only: allocation is not a schema option.
    node = model.nodes["/label"]
    resize = Resize()
    resize.embedding(node.embedder.embeddings["content"], size)
    resize.linear(node.decoder.linears["content"], size)
    node.embedder.counters["content"].resize(size, resize)
    resize.attribute(node.embedder.vocab, "size", size)
    resize.commit()
    return model


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    oracle_nll = float(-0.9 * np.log(0.9) - 0.1 * np.log(0.1))
    metrics = {"prior": 0.9, "weighted_optimum": 0.75, "oracle_nll": oracle_nll}
    checks = {}
    for size in (2, 128):
        lit.seed_everything(seed, workers=True)
        model = build(size)
        data = rf.SyntheticDataModule(
            train=partial(records, rows=32768, seed=seed + 1),
            validate=partial(records, rows=1024, seed=seed + 2),
            seed=seed,
        )
        model.fit(
            data,
            batch_size=128,
            learning_rate=0.002,
            early_stopping=False,
            rollback=False,
            precision="32-true",
            strategy="auto",
            accelerator=accelerator,
            devices=1,
            max_epochs=-1,
            max_steps=BUDGET if steps is None else min(steps, BUDGET),
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True,
            num_sanity_val_steps=0,
        )
        model.eval()
        counts = rf.Category.counts(model, "/label")
        vocabulary = rf.Category.vocabulary(model, "/entity")
        for novel in (False, True):
            rows = list(records(rows=8192, seed=seed + 3, novel=novel))
            predictions = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
            values = [row["label"]["content"] for row in predictions]
            p = np.asarray(
                [item["probability"] if item["value"] == "yes" else 1 - item["probability"] for item in values]
            )
            p = p.clip(1e-7, 1 - 1e-7)
            key = f"capacity_{size}_{'novel' if novel else 'known'}"
            rmse = float(np.sqrt(np.mean((p - 0.9) ** 2)))
            excess = float(np.mean(-0.9 * np.log(p) - 0.1 * np.log1p(-p)) - oracle_nll)
            metrics[key] = {"mean_probability": float(p.mean()), "prior_rmse": rmse, "excess_nll": excess}
            checks[f"{key}: probability RMSE below 0.06"] = rmse < 0.06
            checks[f"{key}: excess expected NLL below 0.02"] = excess < 0.02
        checks[f"capacity_{size}: evaluation preserves vocabulary and counts"] = vocabulary == rf.Category.vocabulary(
            model, "/entity"
        ) and counts == rf.Category.counts(model, "/label")
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7601)

# %% [markdown]
# </details>
