# %% [markdown]
# ---
# title: Can each item choose which value to hide?
# categories:
# - Masking
# evidence-id: P052
# description: Use an item’s selected flag to hide its value and predict it from x.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Correct visible x
#   metric:
#   - nrmse
#   format: error
# - label: Shuffled visible x
#   metric:
#   - corrupted_context_nrmse
#   format: error
# evidence-group: Masking
# evidence-result:
#   accelerator: cpu
#   checks:
#     Corrupted context nRMSE above 0.85: true
#     Hidden values cannot affect predictions: true
#     Prediction flags match selectors including padding: true
#     Selected coordinates are absent from encoder input: true
#     Selected held-out nRMSE below 0.25: true
#     Training objectives match selectors including padding: true
#   duration_seconds: 14.371
#   metrics:
#     baseline_rmse: 1.1622731152476307
#     corrupted_context_nrmse: 1.3885373380978816
#     hidden_value_drift: 0.0
#     nrmse: 0.005475918801274933
#     rmse: 0.006364513204000888
#     selected_coordinates: 1792
#     steps: 400
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 5a15e0cf978147d8608eb12ba3466da3028a152707c63c664fbc886d0af43e36
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: c5a86684c970
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
#     script_sha256: 0c6a6179ceeea06b922479244d902b392a8d4aa503adfb8d210388a84d17d2b6
#     source_sha256: c8bd2012b78581cecd8de5d6325e4e2b69ece0f2b613073678266b93aa34c63e
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 5201
#   started_at: '2026-11-01T01:47:15.895348+00:00'
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
# items:
#   - {x: 0.4, value: 1.1, selected: true}
#   - {x: -0.2, value: -0.1, selected: false}
# ```
#
# value is hidden and scored only where selected is true; the other item remains visible.
#
# ```{typst}
# //| label: fig-evidence-query-reconstruction
# //| fig-cap: "Selector notes identify hidden values. selected chooses which item values are hidden."
# //| fig-alt: "Model tree with record, items, x, value. Selector notes identify hidden values. selected chooses which item values are hidden."
# #tree(node("record", kind: "root", children: (
#   node("items", kind: "branch", repeated: true, children: (
#     node("x", type: "Number"),
#     node("value", type: "Number", body: [Skip and reconstruct when `selected`]),
#   )),
# )))
# ```
#
# ## Comparison
#
# Score only selected real items. Change their hidden values to check they cannot leak into predictions, and shuffle visible x to remove its signal.
#
# ## Result
#
# {{< evidence P052 status >}}
#
# Per-item selection works, including padded positions. This tests aligned local reconstruction, rather than retrieval across unrelated lists.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P052 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P052 script >}}
#
# ### Data and model
#
# Each independent bag contains one to four items. Item-local `x` is uniform on
# `[-1, 1]`, and `value = 2*x + 0.3`. Independent Boolean selectors hide roughly
# half the values, with at least one selected per bag. Entire bags are split
# into 2,048 training, 256 validation, and 1,024 test observations.
#
# ### Training and controls
#
# The [xs preset](../../concepts/model-tree.qmd#choose-a-size) supplies the
# architecture defaults; the items branch keeps `reduction=None` to retain
# its coordinate slots.
#
# Fit for 400 AdamW updates at learning rate 0.002. The primary gate is selected
# coordinate nRMSE below 0.25, relative to the selected training-label mean.
# Replacing visible `x` with independent samples must push nRMSE above 0.85.
# Replacing only hidden `value` inputs must change predictions by less than
# 1e-5. Encoded training masks and output `inferred` flags must exactly match
# the selectors, including unselected and padded positions.
#
# ### Complete experiment code
#

# %%
"""P052: query-selected reconstruction over variable-length repeated records."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P052"
LENGTH = 4
BUDGET = 400


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        length = int(rng.integers(1, LENGTH + 1))
        values = rng.uniform(-1.0, 1.0, length)
        selected = rng.random(length) < 0.5
        selected[int(rng.integers(length))] = True
        yield {
            "items": [
                {"x": float(value), "value": float(2 * value + 0.3), "selected": bool(chosen)}
                for value, chosen in zip(values, selected, strict=True)
            ]
        }


def build() -> rf.Model:
    return rf.Model.xs(
        items=rf.Branch(
            length=LENGTH,
            reduction=None,
            x=rf.Number,
            value=rf.Number(objective="mse", mask=rf.Mask(query="selected", skip=True, reconstruct=True)),
        ),
    )


def prediction(model: rf.Model, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    output = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    # Evaluation compares real coordinates against fixed-width selector matrices.
    # Public output has no cap padding; absent coordinates receive no score.
    content = np.zeros((len(rows), LENGTH), dtype=np.float64)
    inferred = np.zeros((len(rows), LENGTH), dtype=bool)
    for index, row in enumerate(output):
        values = [item["value"] for item in row["items"]]
        content[index, : len(values)] = [value["content"] for value in values]
        inferred[index, : len(values)] = [value["inferred"] for value in values]
    return content, inferred


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    data = rf.SyntheticDataModule(
        train=partial(records, rows=2048, seed=seed + 1),
        validate=partial(records, rows=256, seed=seed + 2),
        seed=seed,
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
    train = list(records(rows=2048, seed=seed + 1))
    test = list(records(rows=1024, seed=seed + 3))
    expected = np.asarray(
        [[item["selected"] for item in row["items"]] + [False] * (LENGTH - len(row["items"])) for row in test]
    )
    actual = np.asarray([item["value"] for row in test for item in row["items"] if item["selected"]])
    mean = float(np.mean([item["value"] for row in train for item in row["items"] if item["selected"]]))
    baseline = float(np.sqrt(np.mean((actual - mean) ** 2)))
    content, inferred = prediction(model, test)
    rng = np.random.default_rng(seed + 4)
    corrupted = [{"items": [{**item, "x": float(rng.uniform(-1, 1))} for item in row["items"]]} for row in test]
    poisoned = [
        {"items": [{**item, "value": 1000.0 if item["selected"] else item["value"]} for item in row["items"]]}
        for row in test
    ]
    broken, _ = prediction(model, corrupted)
    replaced, _ = prediction(model, poisoned)
    # Audit labels on an independent model, leaving fitted normalizers frozen.
    field = build().encode(pa.Table.from_pylist(test), strata="train")["/items/value"]
    measured = float(np.sqrt(np.mean((content[expected] - actual) ** 2)))
    metrics = {
        "steps": trainer.global_step,
        "selected_coordinates": int(expected.sum()),
        "rmse": measured,
        "baseline_rmse": baseline,
        "nrmse": measured / baseline,
        "corrupted_context_nrmse": float(np.sqrt(np.mean((broken[expected] - actual) ** 2))) / baseline,
        "hidden_value_drift": float(np.max(np.abs(content[expected] - replaced[expected]))),
    }
    return metrics, {
        "Selected held-out nRMSE below 0.25": metrics["nrmse"] < 0.25,
        "Corrupted context nRMSE above 0.85": metrics["corrupted_context_nrmse"] > 0.85,
        "Hidden values cannot affect predictions": metrics["hidden_value_drift"] < 1e-5,
        "Prediction flags match selectors including padding": bool(np.array_equal(inferred, expected)),
        "Training objectives match selectors including padding": bool(
            np.array_equal(field.trainable.cpu().numpy().reshape(expected.shape), expected)
        ),
        "Selected coordinates are absent from encoder input": not bool(
            field.present.cpu().numpy().reshape(expected.shape)[expected].any()
        ),
    }


if __name__ == "__main__":
    report(PROOF_ID, run, seed=5201)

# %% [markdown]
# </details>
