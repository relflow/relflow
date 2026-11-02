# %% [markdown]
# ---
# title: Can it predict the next value without seeing the future?
# categories:
# - Masking
# evidence-id: P054
# description: Expose the beginning of a simple numeric sequence, hide its remaining events, and predict the next
#   value.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Next value after two events
#   metric:
#   - prefixes
#   - '2'
#   - nrmse
#   format: error
# - label: Short rollout
#   metric:
#   - rollout
#   - nrmse
#   format: error
# evidence-group: Masking
# evidence-result:
#   accelerator: cpu
#   checks:
#     Free-running suffix nRMSE below 0.45: true
#     Rollout beats persistence: true
#     'prefix 2: exactly the next coordinate is requested': true
#     'prefix 2: exactly the next coordinate is trained': true
#     'prefix 2: future cannot affect prediction': true
#     'prefix 2: next-value nRMSE below 0.35': true
#     'prefix 2: only prefix values are encoder inputs': true
#     'prefix 2: shuffled prefix nRMSE above 0.85': true
#     'prefix 3: exactly the next coordinate is requested': true
#     'prefix 3: exactly the next coordinate is trained': true
#     'prefix 3: future cannot affect prediction': true
#     'prefix 3: next-value nRMSE below 0.35': true
#     'prefix 3: only prefix values are encoder inputs': true
#     'prefix 3: shuffled prefix nRMSE above 0.85': true
#     'prefix 4: exactly the next coordinate is requested': true
#     'prefix 4: exactly the next coordinate is trained': true
#     'prefix 4: future cannot affect prediction': true
#     'prefix 4: next-value nRMSE below 0.35': true
#     'prefix 4: only prefix values are encoder inputs': true
#     'prefix 4: shuffled prefix nRMSE above 0.85': true
#   duration_seconds: 18.533
#   metrics:
#     prefixes:
#       '2':
#         baseline_rmse: 0.7402522604809386
#         future_value_drift: 0.0
#         nrmse: 0.015187897317285103
#         shuffled_nrmse: 1.4230535040194878
#       '3':
#         baseline_rmse: 0.9089646612603506
#         future_value_drift: 0.0
#         nrmse: 0.013635079629137441
#         shuffled_nrmse: 1.410104123904363
#       '4':
#         baseline_rmse: 1.1017247260767682
#         future_value_drift: 0.0
#         nrmse: 0.013846941693610488
#         shuffled_nrmse: 1.4013290531545113
#     rollout:
#       baseline_rmse: 0.9287962705595906
#       nrmse: 0.03521582218570307
#       persistence_rmse: 0.5050971341401193
#       rmse: 0.032708324310770705
#     steps: 800
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: f87c7f5b8f5e5aab9f0c8fb5194845a6cca1589ff85b5496e534a89e5311851b
#     cuda: null
#     evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     gpu: null
#     host: 8ae96f5dbb3a
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
#     script_sha256: d5e7c3f77f43b42d0eec1b482dc85945d5223b240854f8c594513f584d2538bc
#     source_sha256: bf9d25aee25d242ad37aef3992ede204314d726fca74fa6aa17b1e913e679d22
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   seed: 5401
#   started_at: '2026-11-02T04:20:00.622426+00:00'
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
# events:
#   - {value: 0.2, future: false, next: false}
#   - {value: 0.5, future: false, next: false}
#   - {value: 0.8, future: true, next: true}
#   - {value: 1.1, future: true, next: false}
#   - {value: 1.4, future: true, next: false}
# ```
#
# Events marked future are hidden. next selects the single value to predict.
#
# ```{typst}
# //| label: fig-evidence-prefix-generation
# //| fig-cap: "Selector notes identify hidden values. future hides events; next chooses the answer."
# //| fig-alt: "Model tree with record, events, value. Selector notes identify hidden values. future hides events; next chooses the answer."
# #tree(node("record", kind: "root", children: (
#   node("events", kind: "branch", repeated: true, body: [Skip when `future`; no reduction], children: (
#     node("value", type: "Number", body: [Reconstruct when `next`]),
#   )),
# )))
# ```
#
# ## Comparison
#
# Change hidden future values to check they cannot influence prediction. Then feed predicted values back for a short rollout.
#
# ## Result
#
# {{< evidence P054 status >}}
#
# Next-value prediction and the short rollout work on these sequences. This does not establish language generation or general forecasting.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P054 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P054 script >}}
#
# ### Data and model
#
# A whole sequence draws `offset ~ Uniform(-1, 1)` and
# `increment ~ Uniform(-0.4, 0.4)`, then `value[t] = offset + t*increment`.
# Neither latent parameter is exposed. Choose a prefix length uniformly from
# 2, 3, and 4. The branch skips every suffix item; a leaf query requests only
# the first hidden value. Whole sequences are independently generated for
# training (4,096), validation (512), and testing (1,024).
#
# ### Training and controls
#
# Fit for 800 AdamW updates at learning rate 0.002. Score one-step predictions
# separately for all three prefix lengths, requiring nRMSE below 0.35. A rollout
# from the first two true values must achieve suffix nRMSE below 0.45 and beat
# persistence (repeating the second value). Baselines use training means at each
# position, not held-out statistics. Permuting entire visible prefixes between
# records retains the original labels and must give nRMSE above 0.85.
#
# Poison all suffix values, including the selected target: drift must remain
# below 1e-5. Training objectives and inference flags must select exactly one
# coordinate, while encoder presence selects exactly the prefix. The supplied
# correct suffix is used only as an evaluation label, never in the rollout.
#
# ### Complete experiment code
#

# %%
"""P054: leakage-free prefix conditioning and free-running arithmetic-sequence rollout."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P054"
LENGTH = 5
BUDGET = 800


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        offset = float(rng.uniform(-1.0, 1.0))
        increment = float(rng.uniform(-0.4, 0.4))
        cut = int(rng.integers(2, LENGTH))
        yield {
            "events": [
                {"value": offset + index * increment, "future": index >= cut, "next": index == cut}
                for index in range(LENGTH)
            ]
        }


def build() -> rf.Model:
    return rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        events=rf.Branch(
            length=LENGTH,
            reduction=None,
            mask=rf.Mask(query="future", skip=True, dropout=False),
            value=rf.Number(objective="mse", mask=rf.Mask(query="next", reconstruct=True)),
        ),
    )


def prediction(model: rf.Model, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    output = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    content = np.asarray(
        [[item["value"]["content"] for item in row["events"]] for row in output], dtype=np.float64
    ).reshape(len(rows), LENGTH)
    inferred = np.asarray(
        [[item["value"]["inferred"] for item in row["events"]] for row in output], dtype=bool
    ).reshape(len(rows), LENGTH)
    return content, inferred


def rollout(model: rf.Model, rows: list[dict]) -> np.ndarray:
    """Expose generated values only; never teacher-force the held-out suffix."""
    values = np.zeros((len(rows), LENGTH), dtype=np.float64)
    values[:, :2] = [[event["value"] for event in row["events"][:2]] for row in rows]
    for cut in range(2, LENGTH):
        inputs = [
            {
                "events": [
                    {"value": float(value), "future": index >= cut, "next": index == cut}
                    for index, value in enumerate(sequence)
                ]
            }
            for sequence in values
        ]
        content, _ = prediction(model, inputs)
        values[:, cut] = content[:, cut]
    return values


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
    test = list(records(rows=1024, seed=seed + 3))
    actual = np.asarray([[event["value"] for event in row["events"]] for row in test])
    mean = np.asarray([[event["value"] for event in row["events"]] for row in train]).mean(axis=0)
    order = np.random.default_rng(seed + 4).permutation(len(test))
    metrics = {"steps": trainer.global_step, "prefixes": {}}
    checks = {}
    # A separate model audits target selection without fitting held-out moments.
    probe = build()
    for cut in range(2, LENGTH):
        selected = [
            {
                "events": [
                    {**event, "future": index >= cut, "next": index == cut} for index, event in enumerate(row["events"])
                ]
            }
            for row in test
        ]
        content, inferred = prediction(model, selected)
        poisoned = [
            {
                "events": [
                    {**event, "value": 1000.0 + index} if index >= cut else event
                    for index, event in enumerate(row["events"])
                ]
            }
            for row in selected
        ]
        shuffled = [
            {
                "events": [
                    {**event, "value": float(actual[other, index])} if index < cut else event
                    for index, event in enumerate(row["events"])
                ]
            }
            for row, other in zip(selected, order, strict=True)
        ]
        replaced, _ = prediction(model, poisoned)
        broken, _ = prediction(model, shuffled)
        baseline = float(np.sqrt(np.mean((actual[:, cut] - mean[cut]) ** 2)))
        measured = {
            "nrmse": float(np.sqrt(np.mean((content[:, cut] - actual[:, cut]) ** 2))) / baseline,
            "baseline_rmse": baseline,
            "shuffled_nrmse": float(np.sqrt(np.mean((broken[:, cut] - actual[:, cut]) ** 2))) / baseline,
            "future_value_drift": float(np.max(np.abs(content[:, cut] - replaced[:, cut]))),
        }
        metrics["prefixes"][cut] = measured
        checks[f"prefix {cut}: next-value nRMSE below 0.35"] = measured["nrmse"] < 0.35
        checks[f"prefix {cut}: shuffled prefix nRMSE above 0.85"] = measured["shuffled_nrmse"] > 0.85
        checks[f"prefix {cut}: future cannot affect prediction"] = measured["future_value_drift"] < 1e-5
        field = probe.encode(pa.Table.from_pylist(selected), strata="train")["/events/value"]
        expected = np.broadcast_to(np.arange(LENGTH) == cut, inferred.shape)
        visible = np.broadcast_to(np.arange(LENGTH) < cut, inferred.shape)
        checks[f"prefix {cut}: exactly the next coordinate is requested"] = bool(np.array_equal(inferred, expected))
        checks[f"prefix {cut}: exactly the next coordinate is trained"] = bool(
            np.array_equal(field.trainable.cpu().numpy().reshape(expected.shape), expected)
        )
        checks[f"prefix {cut}: only prefix values are encoder inputs"] = bool(
            np.array_equal(field.present.cpu().numpy().reshape(visible.shape), visible)
        )
    generated = rollout(model, test)
    baseline = float(np.sqrt(np.mean((actual[:, 2:] - mean[2:]) ** 2)))
    rmse = float(np.sqrt(np.mean((generated[:, 2:] - actual[:, 2:]) ** 2)))
    persistence = float(np.sqrt(np.mean((actual[:, 1:2] - actual[:, 2:]) ** 2)))
    metrics["rollout"] = {
        "rmse": rmse,
        "baseline_rmse": baseline,
        "nrmse": rmse / baseline,
        "persistence_rmse": persistence,
    }
    checks["Free-running suffix nRMSE below 0.45"] = metrics["rollout"]["nrmse"] < 0.45
    checks["Rollout beats persistence"] = rmse < persistence
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=5401)

# %% [markdown]
# </details>
