# %% [markdown]
# ---
# title: Does label encounter order limit vocabulary coverage?
# categories:
# - Vocabulary
# evidence-id: P057
# description: Train on common and rare labels, changing which labels appear first.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Common-label coverage, rare first
#   metric:
#   - arms
#   - rare_first
#   - common_coverage
#   format: percent
# - label: Common-label coverage, common first
#   metric:
#   - arms
#   - common_first
#   - common_coverage
#   format: percent
# evidence-group: Vocabulary
# evidence-result:
#   accelerator: cpu
#   checks:
#     'common_first: all four labels are admitted': true
#     'common_first: all labels have training exposures': true
#     'common_first: allocation holds all four labels': true
#     'common_first: common identities have distinct learned effects': true
#     'common_first: common-label coverage is complete': true
#     'common_first: common-label nRMSE below 0.15': true
#     'common_first: first label IDs are preserved': true
#     'common_first: prediction does not change admission': true
#     'common_first: storage grows automatically': true
#     'rare_first: all four labels are admitted': true
#     'rare_first: all labels have training exposures': true
#     'rare_first: allocation holds all four labels': true
#     'rare_first: common identities have distinct learned effects': true
#     'rare_first: common-label coverage is complete': true
#     'rare_first: common-label nRMSE below 0.15': true
#     'rare_first: first label IDs are preserved': true
#     'rare_first: prediction does not change admission': true
#     'rare_first: storage grows automatically': true
#   duration_seconds: 9.821
#   metrics:
#     arms:
#       common_first:
#         allocated_size: 4
#         common_coverage: 1.0
#         counts:
#           cold: 9586
#           hot: 9604
#           rare-left: 5
#           rare-right: 5
#         identity_swap_drift: 2.0175565112843827
#         initial_size: 1
#         nrmse: 0.030879157291732663
#         steps: 300
#         vocabulary:
#         - cold
#         - hot
#         - rare-left
#         - rare-right
#       rare_first:
#         allocated_size: 4
#         common_coverage: 1.0
#         counts:
#           cold: 9586
#           hot: 9604
#           rare-left: 5
#           rare-right: 5
#         identity_swap_drift: 1.9995516301333327
#         initial_size: 1
#         nrmse: 0.03019800013266001
#         steps: 300
#         vocabulary:
#         - rare-left
#         - rare-right
#         - hot
#         - cold
#     baseline_rmse: 0.9999224678348755
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 9ec854441801b075150fe6f709ffe5e65fffd29faae93b6fe98f015cc0353d89
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: a0c98d360fd9
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
#     script_sha256: 69f6994ed7b3e99b62111e13c1e3f8100adb0be87ab919ccccca0bd4e142aecf
#     source_sha256: 77902e5d147ecc5edd5a0e7989597e5d115f735480ec98e0341af73e57d1c663
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 5701
#   started_at: '2026-11-01T01:47:51.336955+00:00'
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
# code: rare-left
# target: -1.0
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-capacity-admission-order
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with record, code, target. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("record", kind: "root", children: (
#   node("code", type: "Category", body: [Growing vocabulary with stable IDs]),
#   node("target", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Compare rare-first and common-first training. Check that all observed labels are admitted and that common-label distinctions learn.
#
# ## Result
#
# {{< evidence P057 status >}}
#
# Storage grows and retains earlier labels in both orders. This single-consumer experiment does not test distributed encounter order.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P057 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P057 script >}}
#
# ### Data and model
#
# A training pass has 4,096 rows: one of each rare label and 4,094 balanced
# common labels. The target is the label's effect plus Gaussian noise with SD
# 0.02. For common-first, move the first cold and hot observations to the front
# without duplicating, deleting, or otherwise changing any record.
# Validation uses 512 independent observations. Test generation uses 2,050
# independent observations and scores only its 2,048 common-label rows.
#
# ### Training and controls
#
# Each arm uses the same seed and 300 AdamW updates at learning rate 0.002,
# with `shuffle=False`, `num_workers=0`, and one device. Every arm must admit
# all four labels, retain the first two IDs, have common-label coverage of one,
# and achieve nRMSE below 0.15. Swapping cold and hot must change predictions
# by more than 1.5 on average, demonstrating use of their distinct identities.
# Both arms must grow to hold all four labels.
#
# ### Scope of evidence
#
# Earlier fixed-capacity runs used different gates. Assess this revision using
# a full run whose source fingerprint matches the current script; earlier
# scores do not establish its autoscaling behavior.
#
#
#
# This checks one-device admission and common-label learning. The singleton
# rare labels establish admission and ID order, not reliable rare-label
# predictions. It does not establish worker/DDP ordering or optimizer-state
# preservation after growth later in training. Dedicated runtime tests cover
# those invariants separately, including two-rank DDP, persistent workers,
# gradient accumulation, and checkpoint resume.
#
# ### Complete experiment code
#

# %%
"""P057: vocabulary growth retains admission order and learnable distinctions."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P057"
BUDGET = 300
EFFECTS = {"rare-left": -1.0, "rare-right": 1.0, "cold": -1.0, "hot": 1.0}


def records(*, rows: int, seed: int, common_first: bool = False) -> Iterator[dict]:
    if rows < 4:
        raise ValueError("capacity proof requires at least four observations")
    rng = np.random.default_rng(seed)
    common = np.resize(np.asarray(["cold", "hot"]), rows - 2)
    rng.shuffle(common)
    labels = ["rare-left", "rare-right", *common.tolist()]
    noise = rng.normal(0.0, 0.02, rows)
    observations = [
        {"code": label, "target": float(EFFECTS[label] + deviation)}
        for label, deviation in zip(labels, noise, strict=True)
    ]
    order = list(range(rows))
    if common_first:
        front = [labels.index("cold"), labels.index("hot")]
        order = front + [index for index in order if index not in front]
    for index in order:
        yield observations[index]


def build() -> rf.Model:
    return rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        code=rf.Category(p_unavailable=0.0),
        target=rf.Number(mask=True, objective="mse"),
    )


def prediction(model: rf.Model, rows: list[dict]) -> np.ndarray:
    output = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    return np.asarray([row["target"]["content"] for row in output], dtype=np.float64)


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    test = [row for row in records(rows=2050, seed=seed + 3) if row["code"] in {"cold", "hot"}]
    train = [row for row in records(rows=4096, seed=seed + 1) if row["code"] in {"cold", "hot"}]
    actual = np.asarray([row["target"] for row in test])
    baseline = float(np.sqrt(np.mean((actual - np.mean([row["target"] for row in train])) ** 2)))
    metrics = {"baseline_rmse": baseline, "arms": {}}
    checks = {}
    for name, common_first in (("rare_first", False), ("common_first", True)):
        lit.seed_everything(seed, workers=True)
        model = build()
        initial_size = model.nodes["/code"].embedder.size
        data = rf.SyntheticDataModule(
            train=partial(records, rows=4096, seed=seed + 1, common_first=common_first),
            validate=partial(records, rows=512, seed=seed + 2),
            seed=seed,
            shuffle=False,
            num_workers=0,
        )
        model.fit(
            data,
            batch_size=64,
            learning_rate=0.002,
            early_stopping=False,
            rollback=False,
            precision="32-true",
            strategy="auto",
            accelerator=accelerator,
            devices=1,
            max_epochs=-1,
            max_steps=BUDGET if steps is None else min(steps, BUDGET),
            deterministic=True,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        trainer = model.trainer
        model.eval()
        vocabulary = rf.Category.vocabulary(model, "/code")
        predicted = prediction(model, test)
        renamed = [{**row, "code": "hot" if row["code"] == "cold" else "cold"} for row in test]
        measured = {
            "steps": trainer.global_step,
            "initial_size": initial_size,
            "allocated_size": model.nodes["/code"].embedder.size,
            "vocabulary": vocabulary,
            "counts": rf.Category.counts(model, "/code"),
            "common_coverage": float(np.mean([row["code"] in vocabulary for row in test])),
            "nrmse": float(np.sqrt(np.mean((actual - predicted) ** 2))) / baseline,
            "identity_swap_drift": float(np.mean(np.abs(predicted - prediction(model, renamed)))),
        }
        first = ("cold", "hot") if common_first else ("rare-left", "rare-right")
        checks[f"{name}: all four labels are admitted"] = len(vocabulary) == 4 and set(vocabulary) == set(EFFECTS)
        checks[f"{name}: first label IDs are preserved"] = vocabulary[:2] == first
        checks[f"{name}: allocation holds all four labels"] = measured["allocated_size"] == 4
        checks[f"{name}: storage grows automatically"] = measured["allocated_size"] > initial_size
        checks[f"{name}: all labels have training exposures"] = all(
            measured["counts"].get(label, 0) > 0 for label in EFFECTS
        )
        checks[f"{name}: common-label coverage is complete"] = measured["common_coverage"] == 1.0
        checks[f"{name}: common-label nRMSE below 0.15"] = measured["nrmse"] < 0.15
        checks[f"{name}: common identities have distinct learned effects"] = measured["identity_swap_drift"] > 1.5
        checks[f"{name}: prediction does not change admission"] = rf.Category.vocabulary(model, "/code") == vocabulary
        metrics["arms"][name] = measured
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=5701)

# %% [markdown]
# </details>
