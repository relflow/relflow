# %% [markdown]
# ---
# title: Can it follow IDs through larger lists and a second lookup?
# categories:
# - Sibling entity transfer
# evidence-id: P075
# description: Retrieve a source value by ID with increasing numbers of distractors, then add an intermediate ID link
#   for a two-step lookup.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Direct lookup, four sources
#   metric:
#   - hops_1
#   - ladder
#   - '4'
#   - intact
#   - normalized_rmse
#   format: error
# - label: Two-step lookup, four sources
#   metric:
#   - hops_2
#   - ladder
#   - '4'
#   - intact
#   - normalized_rmse
#   format: error
# evidence-group: Relational
# evidence-result:
#   accelerator: cpu
#   checks:
#     '1 hop, 16 sources: beats source-mean RMSE by 20%': false
#     '1 hop, 16 sources: broken-key nRMSE at least 0.80': true
#     '1 hop, 16 sources: corruption increases nRMSE by at least 0.30': false
#     '1 hop, 16 sources: finite source-unit errors': true
#     '1 hop, 16 sources: intact nRMSE at most 0.50': false
#     '1 hop, 16 sources: permutation nRMSE delta at most 0.10': true
#     '1 hop, 16 sources: reordered nRMSE at most 0.50': false
#     '1 hop, 2 sources: beats source-mean RMSE by 20%': false
#     '1 hop, 2 sources: broken-key nRMSE at least 0.80': false
#     '1 hop, 2 sources: corruption increases nRMSE by at least 0.30': false
#     '1 hop, 2 sources: finite source-unit errors': true
#     '1 hop, 2 sources: intact nRMSE at most 0.50': false
#     '1 hop, 2 sources: permutation nRMSE delta at most 0.10': true
#     '1 hop, 2 sources: reordered nRMSE at most 0.50': false
#     '1 hop, 4 sources: beats source-mean RMSE by 20%': false
#     '1 hop, 4 sources: broken-key nRMSE at least 0.80': true
#     '1 hop, 4 sources: corruption increases nRMSE by at least 0.30': false
#     '1 hop, 4 sources: finite source-unit errors': true
#     '1 hop, 4 sources: intact nRMSE at most 0.50': false
#     '1 hop, 4 sources: permutation nRMSE delta at most 0.10': true
#     '1 hop, 4 sources: reordered nRMSE at most 0.50': false
#     '1 hop, 8 sources: beats source-mean RMSE by 20%': false
#     '1 hop, 8 sources: broken-key nRMSE at least 0.80': true
#     '1 hop, 8 sources: corruption increases nRMSE by at least 0.30': false
#     '1 hop, 8 sources: finite source-unit errors': true
#     '1 hop, 8 sources: intact nRMSE at most 0.50': false
#     '1 hop, 8 sources: permutation nRMSE delta at most 0.10': true
#     '1 hop, 8 sources: reordered nRMSE at most 0.50': false
#     '2 hop, 2 sources: beats source-mean RMSE by 20%': false
#     '2 hop, 2 sources: broken-key nRMSE at least 0.80': false
#     '2 hop, 2 sources: corruption increases nRMSE by at least 0.30': false
#     '2 hop, 2 sources: finite source-unit errors': true
#     '2 hop, 2 sources: intact nRMSE at most 0.50': false
#     '2 hop, 2 sources: permutation nRMSE delta at most 0.10': true
#     '2 hop, 2 sources: reordered nRMSE at most 0.50': false
#     '2 hop, 4 sources: beats source-mean RMSE by 20%': false
#     '2 hop, 4 sources: broken-key nRMSE at least 0.80': true
#     '2 hop, 4 sources: corruption increases nRMSE by at least 0.30': false
#     '2 hop, 4 sources: finite source-unit errors': true
#     '2 hop, 4 sources: intact nRMSE at most 0.50': false
#     '2 hop, 4 sources: permutation nRMSE delta at most 0.10': true
#     '2 hop, 4 sources: reordered nRMSE at most 0.50': false
#   duration_seconds: 142.204
#   metrics:
#     hops_1:
#       ladder:
#         '16':
#           broken:
#             baseline_rmse: 0.5801314527604737
#             normalized_rmse: 0.9705963126403169
#             rmse: 0.563073448895986
#             source_mean_rmse: 0.5614836112231819
#           corruption_gap: 5.980871734201365e-05
#           intact:
#             baseline_rmse: 0.5801314527604737
#             normalized_rmse: 0.9705365039229749
#             rmse: 0.5630387519779066
#             source_mean_rmse: 0.5614836112231819
#           permutation_nrmse_delta: 3.144390845000178e-05
#           permuted:
#             baseline_rmse: 0.5801314527604737
#             normalized_rmse: 0.9705050600145249
#             rmse: 0.563020510377617
#             source_mean_rmse: 0.5614836112231819
#         '2':
#           broken:
#             baseline_rmse: 0.5718617051430289
#             normalized_rmse: 0.7292861279448586
#             rmse: 0.417050808663704
#             source_mean_rmse: 0.4090235781310266
#           corruption_gap: -0.00033347555113139027
#           intact:
#             baseline_rmse: 0.5718617051430289
#             normalized_rmse: 0.72961960349599
#             rmse: 0.4172415105609975
#             source_mean_rmse: 0.4090235781310266
#           permutation_nrmse_delta: 0.00033979819589968763
#           permuted:
#             baseline_rmse: 0.5718617051430289
#             normalized_rmse: 0.7292798053000903
#             rmse: 0.41704719298528575
#             source_mean_rmse: 0.4090235781310266
#         '4':
#           broken:
#             baseline_rmse: 0.5809317566310644
#             normalized_rmse: 0.8700044881996241
#             rmse: 0.5054132356067178
#             source_mean_rmse: 0.4995581280248714
#           corruption_gap: -2.7389132018829088e-05
#           intact:
#             baseline_rmse: 0.5809317566310644
#             normalized_rmse: 0.8700318773316429
#             rmse: 0.5054291468232941
#             source_mean_rmse: 0.4995581280248714
#           permutation_nrmse_delta: 0.001254407015278236
#           permuted:
#             baseline_rmse: 0.5809317566310644
#             normalized_rmse: 0.8687774703163647
#             rmse: 0.5047004219523782
#             source_mean_rmse: 0.4995581280248714
#         '8':
#           broken:
#             baseline_rmse: 0.5737306328297297
#             normalized_rmse: 0.9356796904044065
#             rmse: 0.5368281009016457
#             source_mean_rmse: 0.5355304868001156
#           corruption_gap: 0.0002637142189219066
#           intact:
#             baseline_rmse: 0.5737306328297297
#             normalized_rmse: 0.9354159761854846
#             rmse: 0.5366767999759374
#             source_mean_rmse: 0.5355304868001156
#           permutation_nrmse_delta: 0.0016970246630527708
#           permuted:
#             baseline_rmse: 0.5737306328297297
#             normalized_rmse: 0.9371130008485373
#             rmse: 0.5376504350097984
#             source_mean_rmse: 0.5355304868001157
#       steps: 600
#       training_mean: 0.0003915084068559229
#     hops_2:
#       ladder:
#         '2':
#           broken:
#             baseline_rmse: 0.5779031565075955
#             normalized_rmse: 0.6902184379714403
#             rmse: 0.3988794139834374
#             source_mean_rmse: 0.3975426418411088
#           corruption_gap: -2.9955464253550446e-07
#           intact:
#             baseline_rmse: 0.5779031565075955
#             normalized_rmse: 0.6902187375260829
#             rmse: 0.3988795870970108
#             source_mean_rmse: 0.3975426418411088
#           permutation_nrmse_delta: 0.0003772206529644162
#           permuted:
#             baseline_rmse: 0.5779031565075954
#             normalized_rmse: 0.6905959581790473
#             rmse: 0.39909758410305873
#             source_mean_rmse: 0.3975426418411088
#         '4':
#           broken:
#             baseline_rmse: 0.5775924351979377
#             normalized_rmse: 0.8737613054833413
#             rmse: 0.5046779202158522
#             source_mean_rmse: 0.5025557590002646
#           corruption_gap: -7.129020028751043e-06
#           intact:
#             baseline_rmse: 0.5775924351979377
#             normalized_rmse: 0.87376843450337
#             rmse: 0.5046820378838912
#             source_mean_rmse: 0.5025557590002646
#           permutation_nrmse_delta: 0.0018377674448326609
#           permuted:
#             baseline_rmse: 0.5775924351979377
#             normalized_rmse: 0.8756062019482027
#             rmse: 0.5057435184576796
#             source_mean_rmse: 0.5025557590002646
#       steps: 600
#       training_mean: -0.0006106177625085432
#   mode: full
#   outcome: not_met
#   provenance:
#     code_sha256: 0d18acc72bfc694c7a0f05e07a60af122ead46d0d6d3ae74bb0f9209b0a5d2cc
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: ed80e14f1d53
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
#     script_sha256: d59a0c25c12f6b867e726e966e2b9ed78af0085aa4f04e4995c4cd262a83dbeb
#     source_sha256: 32b58e87234e1ea5b65b75522c7928185855769be2d3d6a60bc21d6f5ae52b74
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 7501
#   started_at: '2026-11-01T01:49:51.168298+00:00'
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
#   - {entity_id: S0, value: 0.7}
#   - {entity_id: S1, value: -0.2}
#   - {entity_id: S2, value: 0.4}
#   - {entity_id: S3, value: -0.9}
# target:
#   - {entity_id: S1, value: -0.2}
#   - {entity_id: S0, value: 0.7}
# ```
#
# Target values are hidden. This example shows a direct lookup; the two-step arm adds a separate links collection.
#
# ```{typst}
# //| label: fig-evidence-retrieval-capacity
# //| fig-cap: "Amber cards are hidden prediction targets. The links collection is used only for the two-step lookup."
# //| fig-alt: "Model tree with lookup, source, entity_id, value, links, entity_id, source_id, target, entity_id, value. Amber cards are hidden prediction targets. The links collection is used only for the two-step lookup."
# #tree(node("lookup", kind: "root", children: (
#   node("source", kind: "branch", repeated: true, children: (
#     node("entity_id", type: "Hash"),
#     node("value", type: "Number"),
#   )),
#   node("links", kind: "branch", repeated: true, detail: "Two-hop arm only", children: (
#     node("entity_id", type: "Hash"),
#     node("source_id", type: "Hash"),
#   )),
#   node("target", kind: "branch", repeated: true, children: (
#     node("entity_id", type: "Hash"),
#     node("value", kind: "target", type: "Number"),
#   )),
# )))
# ```
#
# ## Comparison
#
# Change lookup keys while keeping answers fixed. Compare with averaging source values, and reorder the records without changing matches.
#
# ## Result
#
# {{< evidence P075 status >}}
#
# Direct lookup works in one of three seeds; the two-step lookup fails in all three. The capability checks remain failed, rather than being redefined as an expected limitation.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P075 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P075 script >}}
#
# ### Process and split unit
#
# Each observation samples independent source values from `Uniform(-1, 1)`.
# Two randomly chosen sources supply hidden targets; remaining sources are
# distractors. Every identity is fresh, including across train, validation, and
# test namespaces. Source and target orders are independent. In the two-hop arm,
# a separately shuffled link collection maps query keys to source keys. The
# model must compose those two equality relationships to recover a payload.
# Every requested key has exactly one source and, where applicable, one link.
# Missing and duplicate keys are excluded, so no undocumented tie or null
# convention enters the task. No observation contains a generating index field.
#
# The paired broken-link control retains targets and all key marginals while
# rotating the links' source IDs. Its visible relation points to wrong answers.
#
# ### Model tree and training
#
# Train separate one-hop and two-hop models for 600 AdamW updates each, batch
# size 64, learning rate 0.003. The `xs` preset uses two attention layers here;
# root and every branch retain tokens. One-hop training balances source counts
# 2/4/8/16, and two-hop training balances 2/4. Thus the ladder tests capacity at
# observed lengths, not length extrapolation. Fixed source capacity 16 and two
# target coordinates prevent the parameter budget changing with test length.
# Each arm has 4,096 train and 512 independent validation rows. Every ladder
# condition has 2,048 fresh test rows; validation never selects a checkpoint.
#
# Gates are declared before training: each condition must achieve normalized
# RMSE at most 0.50, beat its per-record source-mean baseline by at least 20%,
# and have broken-key normalized RMSE at least 0.80 with a gap at least 0.30.
# Complete-record permutation must preserve the learning gate and change
# normalized RMSE by at most 0.10. This is a performance robustness gate, not
# exact permutation invariance; branch attention retains positional capacity.
# Each failed condition is reported independently, including two-hop failures.
#
# ### Remaining work
#
# The three-seed panel retained the original gates and 600 updates per arm.
# Diagnose the one-hop optimization instability and two-hop information route
# using training and validation data before designing a follow-up experiment.
# A single successful one-hop seed does not meet the suite's stability contract.
# Even stable success would not establish exact joins, unobserved collection
# lengths, missing-key behavior, duplicate aggregation, or collision immunity.
# Source-unit errors and every condition failure remain in the evidence.
#
# ### Complete experiment code
#

# %%
"""P075: test fresh-key retrieval with distractors and an intermediate relation."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P075"
BUDGET = 600
BATCH_SIZE = 64
TRAIN_ROWS = 4096
TEST_ROWS = 2048
CAPACITIES = (2, 4, 8, 16)


def records(
    *, rows: int, seed: int, namespace: str, hops: int, capacity: int | None = None, broken: bool = False
) -> Iterator[dict]:
    """Fresh unique keys with a matched key-rotation control and fixed targets."""
    rng = np.random.default_rng(seed)
    choices = CAPACITIES if hops == 1 else CAPACITIES[:2]
    sizes = np.resize(choices, rows) if capacity is None else np.full(rows, capacity)
    rng.shuffle(sizes)
    for row, size in enumerate(sizes):
        keys = [f"{namespace}-{row:06d}-s{index:02d}" for index in range(size)]
        values = rng.uniform(-1.0, 1.0, size)
        selected = rng.choice(size, size=2, replace=False)
        source = [{"entity_id": keys[index], "value": float(values[index])} for index in rng.permutation(size)]
        if hops == 1:
            target = [
                {"entity_id": keys[(index + int(broken)) % size], "value": float(values[index])} for index in selected
            ]
            yield {"source": source, "target": target}
        else:
            query_keys = [f"{namespace}-{row:06d}-q{index:02d}" for index in range(size)]
            links = [
                {"entity_id": query_keys[index], "source_id": keys[(index + int(broken)) % size]}
                for index in rng.permutation(size)
            ]
            target = [{"entity_id": query_keys[index], "value": float(values[index])} for index in selected]
            yield {"source": source, "links": links, "target": target}


def permute(rows: list[dict], seed: int) -> list[dict]:
    """Reorder complete records independently, preserving each visible relation."""
    rng = np.random.default_rng(seed)
    return [
        {name: [items[index] for index in rng.permutation(len(items))] for name, items in row.items()} for row in rows
    ]


def build(hops: int) -> rf.Model:
    fields = {
        "source": rf.Branch(
            length=16,
            reduction=None,
            n_layers=2,
            entity_id=rf.Hash(n_hashes=4, n_bands=8),
            value=rf.Number,
        ),
        "target": rf.Branch(
            length=2,
            reduction=None,
            n_layers=2,
            entity_id=rf.Hash(n_hashes=4, n_bands=8),
            value=rf.Number(mask=True, objective="mse"),
        ),
    }
    if hops == 2:
        fields["links"] = rf.Branch(
            length=16,
            reduction=None,
            n_layers=2,
            entity_id=rf.Hash(n_hashes=4, n_bands=8),
            source_id=rf.Hash(n_hashes=4, n_bands=8),
        )
    model = rf.Model.xs(n_layers=2, n_heads=4, reduction=None, fields=fields)
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{**row, "target": [{"entity_id": item["entity_id"]} for item in row["target"]]} for row in rows]
    output = model.predict(pa.Table.from_pylist(visible))["predictions"].to_pylist()
    return np.asarray([[item["value"]["content"] for item in row["target"]] for row in output])


def score(predicted: np.ndarray, rows: list[dict], training_mean: float) -> dict:
    actual = np.asarray([[item["value"] for item in row["target"]] for row in rows])
    source_mean = np.asarray([np.mean([item["value"] for item in row["source"]]) for row in rows])[:, None]
    baseline = float(np.sqrt(np.mean((actual - training_mean) ** 2)))
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    return {
        "rmse": rmse,
        "baseline_rmse": baseline,
        "source_mean_rmse": float(np.sqrt(np.mean((actual - source_mean) ** 2))),
        "normalized_rmse": rmse / baseline,
    }


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    metrics, checks = {}, {}
    for hops in (1, 2):
        lit.seed_everything(seed, workers=True)
        model = build(hops)
        data = rf.SyntheticDataModule(
            train=partial(records, rows=TRAIN_ROWS, seed=seed * 100 + 1, namespace="train", hops=hops),
            validate=partial(records, rows=512, seed=seed * 100 + 2, namespace="validate", hops=hops),
            seed=seed,
        )
        model.fit(
            data,
            batch_size=BATCH_SIZE,
            learning_rate=0.003,
            early_stopping=False,
            rollback=False,
            precision="32-true",
            strategy="auto",
            accelerator=accelerator,
            devices=1,
            max_steps=BUDGET if steps is None else min(steps, BUDGET),
            max_epochs=-1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True,
            num_sanity_val_steps=0,
        )
        trainer = model.trainer
        model.eval()
        train = records(rows=TRAIN_ROWS, seed=seed * 100 + 1, namespace="train", hops=hops)
        training_mean = float(np.mean([item["value"] for row in train for item in row["target"]]))
        metrics[f"hops_{hops}"] = {"steps": trainer.global_step, "training_mean": training_mean, "ladder": {}}
        for capacity in CAPACITIES if hops == 1 else CAPACITIES[:2]:
            options = {
                "rows": TEST_ROWS,
                "seed": seed * 100 + 10 + capacity,
                "namespace": f"test-{hops}-{capacity}",
                "hops": hops,
                "capacity": capacity,
            }
            test = list(records(**options))
            broken = list(records(**options, broken=True))
            permuted = permute(test, seed * 100 + 40 + capacity)
            intact = score(predict(model, test), test, training_mean)
            corrupted = score(predict(model, broken), broken, training_mean)
            reordered = score(predict(model, permuted), permuted, training_mean)
            gap = corrupted["normalized_rmse"] - intact["normalized_rmse"]
            delta = abs(reordered["normalized_rmse"] - intact["normalized_rmse"])
            metrics[f"hops_{hops}"]["ladder"][capacity] = {
                "intact": intact,
                "broken": corrupted,
                "permuted": reordered,
                "corruption_gap": gap,
                "permutation_nrmse_delta": delta,
            }
            label = f"{hops} hop, {capacity} sources"
            checks.update(
                {
                    f"{label}: finite source-unit errors": bool(
                        np.isfinite([*intact.values(), *corrupted.values(), *reordered.values()]).all()
                    ),
                    f"{label}: intact nRMSE at most 0.50": intact["normalized_rmse"] <= 0.50,
                    f"{label}: beats source-mean RMSE by 20%": intact["rmse"] <= 0.8 * intact["source_mean_rmse"],
                    f"{label}: broken-key nRMSE at least 0.80": corrupted["normalized_rmse"] >= 0.80,
                    f"{label}: corruption increases nRMSE by at least 0.30": gap >= 0.30,
                    f"{label}: reordered nRMSE at most 0.50": reordered["normalized_rmse"] <= 0.50,
                    f"{label}: permutation nRMSE delta at most 0.10": delta <= 0.10,
                }
            )
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7501)

# %% [markdown]
# </details>
