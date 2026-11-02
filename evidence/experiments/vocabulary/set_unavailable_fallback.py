# %% [markdown]
# ---
# title: Can it distinguish an unknown set from an empty set?
# categories:
# - Vocabulary
# evidence-id: P064
# description: Predict whether a set is nonempty, including sets whose member names were absent from training.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Unknown-input error, fallback trained
#   metric:
#   - augmented_unknown
#   format: number
# - label: Unknown-input error, fallback untrained
#   metric:
#   - control_unknown
#   format: number
# evidence-group: Vocabulary
# evidence-result:
#   accelerator: cpu
#   checks:
#     Trained fallback improves unknown-input RMSE by more than 0.40: true
#     'augmented: unknown evaluation leaves vocabulary frozen': true
#     'augmented: unknown spelling and duplicates are invariant': true
#     'augmented_known: RMSE below 0.10': true
#     'augmented_unknown: RMSE below 0.10': true
#     'control: unknown evaluation leaves vocabulary frozen': true
#     'control: unknown spelling and duplicates are invariant': true
#     'control_known: RMSE below 0.10': true
#   duration_seconds: 14.865
#   metrics:
#     augmented_duplicate_and_identity_drift: 0.0
#     augmented_known: 0.010940806673487501
#     augmented_unknown: 0.010488261216694645
#     constant_rmse: 0.5
#     control_duplicate_and_identity_drift: 0.0
#     control_known: 0.004766920889184437
#     control_unknown: 0.6953890646843696
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 3a72da78b8052d12b4bdb7965d0b9a55a0d8ea21293310e9974d6502f2875aab
#     cuda: null
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: c70f1bcb3650
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
#     script_sha256: 2acd5cf578485bd11b0f10b027791e5b3523699a16c1c55ec021e4381b61e5ca
#     source_sha256: c9cbeae90a7b3fb6238b7868832fea22d29b32951ac8a6a546aedf7051f0eea7
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 7641
#   started_at: '2026-11-02T17:54:17.640160+00:00'
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
# tags: [tag-12]
# nonempty: 1.0
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-set-unavailable-fallback
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with record, tags, nonempty. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("record", kind: "root", children: (
#   node("tags", type: "Set", detail: "Known or unavailable membership"),
#   node("nonempty", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Compare training with input members sometimes made unavailable against training that never exposes this condition. Repeat unknown members to check duplicates.
#
# ## Result
#
# {{< evidence P064 status >}}
#
# The trained fallback distinguishes unknown nonempty input from empty input. It preserves the presence of members, rather than learning their individual meanings.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P064 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P064 script >}}
#
# ### Training and controls
#
# Each arm trains for 400 AdamW updates on 8,192 rows, with 512 independent
# validation rows. Evaluate 2,048 known and disjoint-label rows. The constant
# 0.5 predictor has RMSE 0.5. Both arms must learn familiar sets (RMSE < 0.10).
# Only the augmented arm must transfer to unknown sets (RMSE < 0.10); its
# gain over the untrained-fallback control must exceed 0.40 RMSE. Duplicate
# and unknown-spelling changes must leave predictions unchanged.
#
# ### Complete experiment code
#

# %%
"""P064: input corruption teaches an unavailable Set representation."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P064"
BUDGET = 400


def records(*, rows: int, seed: int, novel: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for label, nonempty in zip(rng.integers(32, size=rows), rng.random(rows) < 0.5, strict=True):
        yield {"tags": [f"{'novel' if novel else 'tag'}-{label}"] if nonempty else [], "nonempty": float(nonempty)}


def build(p_unavailable: float) -> rf.Model:
    model = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        tags=rf.Set(p_unavailable=p_unavailable),
        nonempty=rf.Number(mask=True, objective="mse"),
    )
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [row["nonempty"]["content"] for row in model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()]
    )


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    metrics, checks = {"constant_rmse": 0.5}, {}
    for probability in (0.0, 0.5):
        lit.seed_everything(seed, workers=True)
        model = build(probability)
        data = rf.SyntheticDataModule(
            train=partial(records, rows=8192, seed=seed + 1),
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
        vocabulary = rf.Set.vocabulary(model, "/tags")
        arm = "augmented" if probability else "control"
        for novel in (False, True):
            rows = list(records(rows=2048, seed=seed + 3, novel=novel))
            expected = np.asarray([row["nonempty"] for row in rows])
            values = predict(model, rows)
            key = f"{arm}_{'unknown' if novel else 'known'}"
            metrics[key] = float(np.sqrt(np.mean((values - expected) ** 2)))
            if not novel or probability:
                checks[f"{key}: RMSE below 0.10"] = metrics[key] < 0.10
            if novel:
                changed = [{**row, "tags": ["another", "another"] if row["tags"] else []} for row in rows]
                drift = float(np.max(np.abs(values - predict(model, changed))))
                metrics[f"{arm}_duplicate_and_identity_drift"] = drift
                checks[f"{arm}: unknown spelling and duplicates are invariant"] = drift < 1e-5
        checks[f"{arm}: unknown evaluation leaves vocabulary frozen"] = rf.Set.vocabulary(model, "/tags") == vocabulary
    checks["Trained fallback improves unknown-input RMSE by more than 0.40"] = (
        metrics["control_unknown"] - metrics["augmented_unknown"] > 0.40
    )
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7641)

# %% [markdown]
# </details>
