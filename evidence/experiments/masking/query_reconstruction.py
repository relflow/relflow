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
#   duration_seconds: 14.357
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
#     evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     gpu: null
#     host: 204cf2992614
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
#     script_sha256: 4d02ebd746de5608dc7227a86d658de16be5adecd779edc9dd5067ada544ee07
#     source_sha256: a387d5ec6a0eaf1803b573f118dd082d57680b91451a4bbe20af2d598eb7e2d4
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   seed: 5201
#   started_at: '2026-11-02T04:19:45.474008+00:00'
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
