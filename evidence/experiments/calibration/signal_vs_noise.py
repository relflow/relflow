# %% [markdown]
# ---
# title: Can it learn a signal without inventing one?
# categories:
# - Calibration
# evidence-id: P017
# description: Predict a yes/no answer from a visible yes/no input.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Matching answers
#   metric:
#   - signal_auc
#   format: auc
# - label: Unrelated answers
#   metric:
#   - noise_auc
#   format: auc
# evidence-group: Calibration
# evidence-result:
#   accelerator: cpu
#   checks:
#     Independent noise AUC remains between 0.42 and 0.58: true
#     Signal AUC is at least 0.98: true
#     Signal exceeds noise by at least 0.40 AUC: true
#   duration_seconds: 11.086
#   metrics:
#     auc_gap: 0.5231304168701172
#     noise_auc: 0.4768695831298828
#     signal_auc: 1.0
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 919ff96a0ddf31c8ca1f70be0d4faf981f8826d2d853a3889eb82bde953b6de9
#     cuda: null
#     evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     gpu: null
#     host: be277a9ef9bf
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
#     script_sha256: 82fd9d4ff27752465b60392bd37f8ec659ec5cd9362084fb25a5efcdd58cefc0
#     source_sha256: 3c1827c411d86bccf70b9882ab67f15be5d1af993fedcfe41313af9f1fbae834
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   seed: 7
#   started_at: '2026-11-02T04:08:06.988716+00:00'
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
# x: -0.7
# segment: segment-3
# tags:
#   - round
#   - hot
# leak: true
# target: true
# ```
#
# leak is the visible yes/no signal; the other inputs are unrelated. target is hidden.
#
# ```{typst}
# //| label: fig-evidence-calibration-signal-vs-noise
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with calibration, x, segment, tags, leak, target. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("calibration", kind: "root", children: (
#   node("x", type: "Number"),
#   node("segment", type: "Category"),
#   node("tags", type: "Set"),
#   node("leak", type: "Boolean", width: 150pt,),
#   node("target", kind: "target", type: "Boolean",),
# )))
# ```
#
# ## Comparison
#
# Train one model where input and answer match, and another where they are unrelated. Both receive the same training budget.
#
# ## Result
#
# {{< evidence P017 status >}}
#
# The real signal is learned; unrelated answers remain near chance. This basic check makes harder learning results easier to interpret.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P017 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P017 script >}}
#
# ### How it works
#
# The positive model can learn to copy the visible Boolean through the shared
# context. A separately trained control sees the same schema, but `leak` is
# independent of the balanced target. That model should have no reliable way
# to rank positive examples above negative ones.
#
# Each model trains on 1,024 rows, validates on 512, and tests on 2,048. The
# three splits use independent random streams. Both runs use the `xs` preset
# and 12 deterministic epochs, so a difference in the information available
# explains the contrast.
#
# ### Remaining work
#
# Run three paired core seeds and a calibration panel of at least ten seeds.
# Record the resulting distributions before freezing the gates, and reconcile
# the implemented 2,048 test rows with the evidence plan's 4,096-row requirement.
#
# ### Complete experiment code
#

# %%
"""P017: distinguish an obvious Boolean signal from independent noise.

Only the relationship between ``leak`` and ``target`` changes. Both models use
independent train, validation, and test streams and the same training budget.
"""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P017"


def records(*, rows: int, seed: int, signal: bool) -> Iterator[dict]:
    """Yield balanced labels with matching or independent Boolean inputs."""
    rng = np.random.default_rng(seed)
    target = np.tile(np.array([False, True]), (rows + 1) // 2)[:rows]
    rng.shuffle(target)
    independent = rng.integers(0, 2, size=rows).astype(bool)
    tags = np.array(["red", "blue", "round", "square", "hot", "cold"])
    tag_rows = [rng.choice(tags, size=int(rng.integers(0, 4)), replace=False).tolist() for _ in range(rows)]
    values = rng.normal(size=rows)
    segments = rng.integers(0, 4, size=rows)
    for index in range(rows):
        yield {
            "x": float(values[index]),
            "segment": f"segment-{segments[index]}",
            "tags": tag_rows[index],
            "leak": bool(target[index] if signal else independent[index]),
            "target": bool(target[index]),
        }


def fit(*, signal: bool, seed: int, steps: int | None, accelerator: str) -> float:
    """Fit one relationship and measure its held-out Boolean AUC."""
    lit.seed_everything(seed, workers=True)
    model = rf.Model.xs(
        x=rf.Number,
        segment=rf.Category(p_unavailable=0.0),
        tags=rf.Set(p_unavailable=0.0),
        leak=rf.Boolean,
        target=rf.Boolean(mask=True),
    )
    data = rf.SyntheticDataModule(
        train=partial(records, rows=1024, seed=seed + 1, signal=signal),
        validate=partial(records, rows=512, seed=seed + 2, signal=signal),
        test=partial(records, rows=2048, seed=seed + 3, signal=signal),
        seed=seed,
    )
    model.fit(
        data,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.005),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
        accelerator=accelerator,
        max_epochs=12,
        max_steps=steps if steps is not None else -1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer = model.trainer
    metrics = trainer.test(model=model, datamodule=data, verbose=False)[0]
    return float(metrics[".target/test.auc.content"])


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    signal_auc = fit(signal=True, seed=seed, steps=steps, accelerator=accelerator)
    noise_auc = fit(signal=False, seed=seed, steps=steps, accelerator=accelerator)
    gap = signal_auc - noise_auc
    return {"signal_auc": signal_auc, "noise_auc": noise_auc, "auc_gap": gap}, {
        "Signal AUC is at least 0.98": signal_auc >= 0.98,
        "Independent noise AUC remains between 0.42 and 0.58": 0.42 <= noise_auc <= 0.58,
        "Signal exceeds noise by at least 0.40 AUC": gap >= 0.40,
    }


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7)

# %% [markdown]
# </details>
