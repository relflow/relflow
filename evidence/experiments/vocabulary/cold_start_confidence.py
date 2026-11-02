# %% [markdown]
# ---
# title: How confident should it be about a new identity?
# categories:
# - Vocabulary
# evidence-id: P061
# description: Predict a yes/no outcome using a numeric signal and an identity-specific effect. New identities remove
#   only the latter information.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Weak signal, new identity, 20% training rate
#   metric:
#   - '0.2'
#   - novel
#   - ambiguous_confidence
#   format: percent
# - label: Strong signal, new identity, 20% training rate
#   metric:
#   - '0.2'
#   - novel
#   - decisive_confidence
#   format: percent
# evidence-group: Vocabulary
# evidence-result:
#   accelerator: cpu
#   checks:
#     '0.01: evaluation preserves mapping and exposures': true
#     '0.01: unseen spelling is uninformative': true
#     '0.2: evaluation preserves mapping and exposures': true
#     '0.2: unseen spelling is uninformative': true
#     Ambiguous unknown IDs retain confidence below 0.70: true
#     Decisive context retains confidence above 0.90: true
#     Known-ID probability RMSE below 0.30: true
#     Unknown-ID predictions beat the constant baseline: true
#     Unknown-ID probability RMSE below 0.10: true
#   duration_seconds: 28.816
#   metrics:
#     '0.01':
#       admission_mean_drift: 0.0003071454702876508
#       known:
#         accuracy: 0.802978515625
#         ambiguous_confidence: 0.6822458485209202
#         brier: 0.13774539864425028
#         constant_rmse: 0.41307056095828854
#         decisive_confidence: 0.979890597188566
#         nll: 0.4295551679401508
#         oracle_rmse: 0.24325100463038368
#       novel:
#         accuracy: 0.7852783203125
#         ambiguous_confidence: 0.5714815263234233
#         brier: 0.14561742906462838
#         constant_rmse: 0.3257906780529331
#         decisive_confidence: 0.9823842794232616
#         nll: 0.4490004669477017
#         oracle_rmse: 0.06678319238509788
#       renaming_drift: 0.0
#     '0.2':
#       admission_mean_drift: 0.0002874272468034178
#       known:
#         accuracy: 0.797119140625
#         ambiguous_confidence: 0.6495810806708707
#         brier: 0.13915601835738278
#         constant_rmse: 0.41307056095828854
#         decisive_confidence: 0.9797623936131904
#         nll: 0.43151307221366303
#         oracle_rmse: 0.24652841853289564
#       novel:
#         accuracy: 0.77880859375
#         ambiguous_confidence: 0.5768502663919366
#         brier: 0.14515934309868395
#         constant_rmse: 0.3257906780529331
#         decisive_confidence: 0.9813166697057495
#         nll: 0.44575804827618704
#         oracle_rmse: 0.060065833648231204
#       renaming_drift: 0.0
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 59d0294b92debef0b8e862c7a4f55242c270e7bc6a6414348aa8c05771f9a1d0
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 3a920816008b
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
#     script_sha256: 35ff703a2a4dbfdd2706fa2435aee207069f3872bc05074c0f714b0e7faf0235
#     source_sha256: ed65ba0af4c14528ae138350d6cc189aa858ff0999cfa67c7033a80f8dee17f3
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 7611
#   started_at: '2026-11-01T01:48:06.600130+00:00'
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
# entity: id-42
# label: yes
# ```
#
# The outcome is hidden and randomly sampled. Scores compare predicted probabilities with the generator’s known probabilities.
#
# ```{typst}
# //| label: fig-evidence-cold-start-confidence
# //| fig-cap: "Amber cards are hidden prediction targets. The numeric signal remains available for unfamiliar identities."
# //| fig-alt: "Model tree with record, x, entity, label. Amber cards are hidden prediction targets. The numeric signal remains available for unfamiliar identities."
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number", detail: "Generalizable evidence"),
#   node("entity", type: "Category", detail: "Persistent identity"),
#   node("label", kind: "target", type: "Category", detail: "Known yes/no labels"),
# )))
# ```
#
# ## Comparison
#
# Compare training with 1% or 20% of input identities made unavailable. Score new identities separately where the numeric signal is weak or strong.
#
# ## Result
#
# {{< evidence P061 status >}}
#
# A new identity does not always imply a 50% prediction: strong remaining context can still justify confidence. Neither unavailability rate is claimed to be best for every dataset.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P061 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P061 script >}}
#
# ### Training and controls
#
# Use 32,768 training rows over 4,096 identities, automatically growing storage,
# and 600 updates. Independent validation has 1,024 rows. Evaluation has 8,192
# known-ID rows and 8,192 entity-disjoint rows, with identical visible x and
# known yes/no output labels. The latent sign is only available to the evaluator.
#
# The 20% arm must beat the constant baseline, retain decisive confidence above
# 0.9, and keep ambiguous new-ID confidence below 0.7. Record both arms without
# demanding that one stochastic run of stronger masking always beats 1%.
# A final admission-only probe measures the prediction change without learning;
# this is a diagnostic limitation, not a promise that admission is safe.
#
# ### Complete experiment code
#

# %%
"""P061: confidence with sparse known IDs, unknown IDs, and decisive context."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P061"
BUDGET = 600
ENTITIES = 4096


def records(*, rows: int, seed: int, novel: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    identities = rng.integers(0, ENTITIES, rows)
    effects = np.random.default_rng(918).choice([-2.5, 2.5], ENTITIES)
    values = rng.uniform(-3, 3, rows)
    probabilities = 1 / (1 + np.exp(-(2 * values + effects[identities])))
    labels = rng.random(rows) < probabilities
    for x, identity, label in zip(values, identities, labels, strict=True):
        yield {"x": float(x), "entity": f"{'new' if novel else 'id'}-{identity}", "label": "yes" if label else "no"}


def build(unavailable: float) -> rf.Model:
    model = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        x=rf.Number,
        entity=rf.Category(p_unavailable=unavailable),
        label=rf.Category(mask=True),
    )
    return model


def probability(model: rf.Model, rows: list[dict]) -> np.ndarray:
    predictions = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    content = [row["label"]["content"] for row in predictions]
    return np.asarray([item["probability"] if item["value"] == "yes" else 1 - item["probability"] for item in content])


def score(p: np.ndarray, rows: list[dict], novel: bool) -> dict[str, float]:
    x = np.asarray([row["x"] for row in rows])
    if novel:
        oracle = (1 / (1 + np.exp(-(2 * x - 2.5))) + 1 / (1 + np.exp(-(2 * x + 2.5)))) / 2
    else:
        effects = np.random.default_rng(918).choice([-2.5, 2.5], ENTITIES)
        identity = np.asarray([int(row["entity"].split("-")[1]) for row in rows])
        oracle = 1 / (1 + np.exp(-(2 * x + effects[identity])))
    y = np.asarray([row["label"] == "yes" for row in rows], dtype=float)
    clipped = p.clip(1e-7, 1 - 1e-7)
    confidence = np.maximum(p, 1 - p)
    return {
        "oracle_rmse": float(np.sqrt(np.mean((p - oracle) ** 2))),
        "constant_rmse": float(np.sqrt(np.mean((0.5 - oracle) ** 2))),
        "nll": float(np.mean(-y * np.log(clipped) - (1 - y) * np.log1p(-clipped))),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p > 0.5) == y)),
        "ambiguous_confidence": float(confidence[np.abs(x) < 0.5].mean()),
        "decisive_confidence": float(confidence[np.abs(x) > 2.5].mean()),
    }


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    metrics, checks = {}, {}
    for unavailable in (0.01, 0.2):
        lit.seed_everything(seed, workers=True)
        model = build(unavailable)
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
        vocabulary = rf.Category.vocabulary(model, "/entity")
        counts = rf.Category.counts(model, "/entity")
        arm = {}
        for novel in (False, True):
            rows = list(records(rows=8192, seed=seed + 3, novel=novel))
            arm["novel" if novel else "known"] = score(probability(model, rows), rows, novel)
        rows = list(records(rows=2048, seed=seed + 4, novel=True))
        before = probability(model, rows)
        renamed = [{**row, "entity": "another-unseen-identity"} for row in rows]
        arm["renaming_drift"] = float(np.max(np.abs(before - probability(model, renamed))))
        checks[f"{unavailable}: evaluation preserves mapping and exposures"] = vocabulary == rf.Category.vocabulary(
            model, "/entity"
        ) and counts == rf.Category.counts(model, "/entity")
        checks[f"{unavailable}: unseen spelling is uninformative"] = arm["renaming_drift"] < 1e-5
        model.encode(pa.Table.from_pylist(rows), strata="train")
        arm["admission_mean_drift"] = float(np.mean(np.abs(before - probability(model, rows))))
        metrics[str(unavailable)] = arm
        if unavailable == 0.2:
            checks.update(
                {
                    "Known-ID probability RMSE below 0.30": arm["known"]["oracle_rmse"] < 0.30,
                    "Unknown-ID probability RMSE below 0.10": arm["novel"]["oracle_rmse"] < 0.10,
                    "Unknown-ID predictions beat the constant baseline": arm["novel"]["oracle_rmse"]
                    < arm["novel"]["constant_rmse"],
                    "Ambiguous unknown IDs retain confidence below 0.70": arm["novel"]["ambiguous_confidence"] < 0.70,
                    "Decisive context retains confidence above 0.90": arm["novel"]["decisive_confidence"] > 0.90,
                }
            )
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7611)

# %% [markdown]
# </details>
