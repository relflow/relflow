# %% [markdown]
# ---
# title: Can hiding values teach a relationship in both directions?
# categories:
# - Masking
# evidence-id: P051
# description: Sometimes hide x or y during training, then ask the model to recover the missing value from the other.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Recover y from x
#   metric:
#   - directions
#   - y
#   - nrmse
#   format: error
# - label: Recover x from y
#   metric:
#   - directions
#   - x
#   - nrmse
#   format: error
# evidence-group: Masking
# evidence-result:
#   accelerator: cpu
#   checks:
#     'x: held-out nRMSE below 0.25': true
#     'x: hidden values cannot affect predictions': true
#     'x: no context nRMSE above 0.85': true
#     'x: ordinary prediction requests no reconstruction': true
#     'x: sampling changes across epochs': true
#     'x: sampling is partial': true
#     'x: sampling repeats within an epoch': true
#     'x: shuffled context nRMSE above 0.85': true
#     'y: held-out nRMSE below 0.25': true
#     'y: hidden values cannot affect predictions': true
#     'y: no context nRMSE above 0.85': true
#     'y: ordinary prediction requests no reconstruction': true
#     'y: sampling changes across epochs': true
#     'y: sampling is partial': true
#     'y: sampling repeats within an epoch': true
#     'y: shuffled context nRMSE above 0.85': true
#   duration_seconds: 11.914
#   metrics:
#     directions:
#       x:
#         baseline_rmse: 0.57715648701892
#         both_hidden_nrmse: 1.0062535600441909
#         hidden_value_drift: 0.0
#         nrmse: 0.0682944964543327
#         rmse: 0.03941661165630875
#         sampled_fraction: 0.3564453125
#         shuffled_nrmse: 1.3603068436605643
#       y:
#         baseline_rmse: 1.15431297403784
#         both_hidden_nrmse: 1.0059962177876465
#         hidden_value_drift: 0.0
#         nrmse: 0.061393759224793926
#         rmse: 0.07086761279813494
#         sampled_fraction: 0.3505859375
#         shuffled_nrmse: 1.4131583560791532
#     steps: 400
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 262db4ceaf6d159f69ca4a1fffd1b999295459d5e04cd74519b1b1450ad6476a
#     cuda: null
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: 35e736faff3e
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
#     script_sha256: f9fa548337474269f1599b3fac4b34b8283c777975e8d398684cc14932df0887
#     source_sha256: 1168b2b1b695f263eb39ffff992ed32b114e65bc505e44cb8119099595d96e6f
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 5101
#   started_at: '2026-11-02T17:53:10.469382+00:00'
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
# x: 0.4
# y: 1.1
# hide_x: false
# hide_y: true
# ```
#
# Values marked for reconstruction are hidden. During training, the selection is sampled separately for each field.
#
# ```{typst}
# //| label: fig-evidence-sampled-reconstruction
# //| fig-cap: "Selector notes identify hidden values. Each field is sometimes hidden during training."
# //| fig-alt: "Model tree with record, x, y. Selector notes identify hidden values. Each field is sometimes hidden during training."
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number", body: [35% sampled; query `hide_x`]),
#   node("y", type: "Number", body: [35% sampled; query `hide_y`]),
# )))
# ```
#
# ## Comparison
#
# Score both directions on new records. Shuffle the visible value, or hide both, to remove the needed information.
#
# ## Result
#
# {{< evidence P051 status >}}
#
# The model learns both directions. When both values are hidden, the original answer cannot be identified; this is a small relationship test, not general imputation evidence.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P051 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P051 script >}}
#
# ### Data and model
#
# Independent records draw `x ~ Uniform(-1, 1)` and set `y = 2*x + 0.3`.
# Train, validation, and test contain 4,096, 512, and 2,048 independent rows.
# Each field has a 35% sampled reconstruction policy and a Boolean query policy.
# Query flags are false during fitting; inference turns on exactly one flag.
# Neither flag is an embedded field. No separate supervised target is added.
#
# ### Training and controls
#
# The model uses the [xs preset](../../concepts/model-tree.qmd#choose-a-size).
#
# Fit for 400 AdamW updates at learning rate 0.002. Evaluate the final model;
# no checkpoint or gate is selected using the test set. Each direction must
# achieve nRMSE below 0.25 relative to its training-mean baseline. Shuffling the
# remaining input, or hiding both fields, must give nRMSE above 0.85. Poisoning
# only a hidden source value must leave predictions unchanged within 1e-5.
# Mask diagnostics also check reproducibility, epoch variation, and the absence
# of sampled reconstruction requests during ordinary prediction.
#
# ### Complete experiment code
#

# %%
"""P051: sampled cross-field reconstruction with deterministic inference controls."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P051"
BUDGET = 400


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for value in rng.uniform(-1.0, 1.0, rows):
        yield {"x": float(value), "y": float(2 * value + 0.3), "hide_x": False, "hide_y": False}


def build() -> rf.Model:
    return rf.Model.xs(
        fields={
            name: rf.Number(
                objective="mse",
                mask=(
                    rf.Mask(rate=0.35, reconstruct=True),
                    rf.Mask(query=f"hide_{name}", reconstruct=True),
                ),
            )
            for name in ("x", "y")
        },
    )


def prediction(model: rf.Model, rows: list[dict], target: str) -> np.ndarray:
    output = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    return np.asarray([row[target]["content"] for row in output], dtype=np.float64)


def error(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    data = rf.SyntheticDataModule(
        train=partial(records, rows=4096, seed=seed + 1),
        validate=partial(records, rows=512, seed=seed + 2),
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
    train = list(records(rows=4096, seed=seed + 1))
    test = list(records(rows=2048, seed=seed + 3))
    source = pa.Table.from_pylist(test)
    # Training-mode audits must not update the fitted model with held-out moments.
    probe = build()
    encoded = [probe.encode(source, strata="train", seed=seed, epoch=epoch) for epoch in (0, 0, 1)]
    ordinary = probe.encode(source, strata="predict")
    metrics = {"steps": trainer.global_step, "directions": {}}
    checks = {}
    order = np.random.default_rng(seed + 4).permutation(len(test))
    for target, context in (("y", "x"), ("x", "y")):
        selected = [{**row, f"hide_{target}": True} for row in test]
        actual = np.asarray([row[target] for row in test])
        baseline = error(actual, float(np.mean([row[target] for row in train])))
        predicted = prediction(model, selected, target)
        shuffled = [{**row, context: test[index][context]} for row, index in zip(selected, order, strict=True)]
        hidden = [{**row, f"hide_{context}": True} for row in selected]
        poisoned = [{**row, target: 1000.0 + index} for index, row in enumerate(selected)]
        measured = {
            "rmse": error(actual, predicted),
            "baseline_rmse": baseline,
            "nrmse": error(actual, predicted) / baseline,
            "shuffled_nrmse": error(actual, prediction(model, shuffled, target)) / baseline,
            "both_hidden_nrmse": error(actual, prediction(model, hidden, target)) / baseline,
            "hidden_value_drift": float(np.max(np.abs(predicted - prediction(model, poisoned, target)))),
        }
        masks = [batch[f"/{target}"].trainable for batch in encoded]
        measured["sampled_fraction"] = float(masks[0].float().mean())
        metrics["directions"][target] = measured
        checks[f"{target}: held-out nRMSE below 0.25"] = measured["nrmse"] < 0.25
        checks[f"{target}: shuffled context nRMSE above 0.85"] = measured["shuffled_nrmse"] > 0.85
        checks[f"{target}: no context nRMSE above 0.85"] = measured["both_hidden_nrmse"] > 0.85
        checks[f"{target}: hidden values cannot affect predictions"] = measured["hidden_value_drift"] < 1e-5
        checks[f"{target}: sampling is partial"] = 0.25 < measured["sampled_fraction"] < 0.45
        checks[f"{target}: sampling repeats within an epoch"] = bool((masks[0] == masks[1]).all())
        checks[f"{target}: sampling changes across epochs"] = bool((masks[0] != masks[2]).any())
        checks[f"{target}: ordinary prediction requests no reconstruction"] = not bool(
            ordinary[f"/{target}"].inferred.any()
        )
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=5101)

# %% [markdown]
# </details>
