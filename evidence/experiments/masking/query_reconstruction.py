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
#   duration_seconds: 15.527
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
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: 79baa38e0e43
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
#     script_sha256: 2eec44e08874533cbceefcd4e2102c888a4cf2d1ce67c049b793842cc9d51570
#     source_sha256: 8d7f46657e9c8bc889e59173951d4470b9c47ac271543eed2109abaa77988eed
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 5201
#   started_at: '2026-11-02T17:53:17.580583+00:00'
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
