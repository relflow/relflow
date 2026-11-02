# %% [markdown]
# ---
# title: Does reconstruction accuracy include unfamiliar answers?
# categories:
# - Vocabulary
# evidence-id: P062
# description: Hide a categorical answer and reconstruct it from numeric context, while varying how many test answers
#   belong to the learned vocabulary.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Known-answer accuracy, low coverage
#   metric:
#   - known_102
#   - accuracy.content
#   format: percent
# - label: Answers covered
#   metric:
#   - known_102
#   - coverage.content
#   format: percent
# - label: All-answer accuracy
#   metric:
#   - known_102
#   - accuracy.all
#   format: percent
# evidence-group: Vocabulary
# evidence-result:
#   accelerator: cpu
#   checks:
#     '0: coverage matches selected targets': true
#     '0: exact known and unavailable target counts': true
#     '1024: all-target accuracy includes unavailable errors': true
#     '1024: coverage matches selected targets': true
#     '1024: exact known and unavailable target counts': true
#     '1024: known accuracy exceeds 0.95': true
#     '102: all-target accuracy includes unavailable errors': true
#     '102: coverage matches selected targets': true
#     '102: exact known and unavailable target counts': true
#     '102: known accuracy exceeds 0.95': true
#     '512: all-target accuracy includes unavailable errors': true
#     '512: coverage matches selected targets': true
#     '512: exact known and unavailable target counts': true
#     '512: known accuracy exceeds 0.95': true
#     All-OOV all-target accuracy is zero: true
#     All-OOV conditional scores are undefined: true
#     Rebatching preserves counts and scores: true
#     Validation leaves mapping and exposures frozen: true
#   duration_seconds: 9.53
#   metrics:
#     batch_partition_drift: 0.0
#     known_0:
#       accuracy.all: 0.0
#       accuracy.content: null
#       accuracy.state: 1.0
#       coverage.content: 0.0
#       loss.content: null
#       loss.state: 7.992463906703051e-06
#       nll.content: null
#       targets.known: 0.0
#       targets.unavailable: 1024.0
#       vocabulary.size: 8.0
#     known_102:
#       accuracy.all: 0.099609375
#       accuracy.content: 1.0
#       accuracy.state: 1.0
#       coverage.content: 0.099609375
#       loss.content: 3.7689285818487406e-05
#       loss.state: 7.992463906703051e-06
#       nll.content: 3.7689285818487406e-05
#       targets.known: 102.0
#       targets.unavailable: 922.0
#       vocabulary.size: 8.0
#     known_1024:
#       accuracy.all: 1.0
#       accuracy.content: 1.0
#       accuracy.state: 1.0
#       coverage.content: 1.0
#       loss.content: 3.8553673221031204e-05
#       loss.state: 7.992463906703051e-06
#       nll.content: 3.8553673221031204e-05
#       targets.known: 1024.0
#       targets.unavailable: 0.0
#       vocabulary.size: 8.0
#     known_512:
#       accuracy.all: 0.5
#       accuracy.content: 1.0
#       accuracy.state: 1.0
#       coverage.content: 0.5
#       loss.content: 3.843376543954946e-05
#       loss.state: 7.992463906703051e-06
#       nll.content: 3.843376543954946e-05
#       targets.known: 512.0
#       targets.unavailable: 512.0
#       vocabulary.size: 8.0
#     reported_content_loss_drift: 0.0
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 70b400280ee63cdbb14fb2bf2246b271f71cb8a2e8ef315f2b9a3ef5f3e9dbd7
#     cuda: null
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: d90b3f5c4781
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
#     script_sha256: 3110a165141797a8c4e62fa0d035cca715d6446e8cee22b10affd9ba4913f14a
#     source_sha256: 341cde803bea1513d598fe17faac0060170f81423022789a38a06ad7d20e01a6
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 7621
#   started_at: '2026-11-02T17:54:09.556925+00:00'
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
# x: 2.0
# code: class-2
# hide: true
# ```
#
# code is hidden wherever hide is true. Unknown code names cannot be emitted by the learned output vocabulary.
#
# ```{typst}
# //| label: fig-evidence-pretraining-coverage
# //| fig-cap: "Selector notes identify hidden values. hide chooses which code answers are hidden."
# //| fig-alt: "Model tree with record, x, code. Selector notes identify hidden values. hide chooses which code answers are hidden."
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number", detail: "Class coordinate"),
#   node("code", type: "Category", detail: "Reconstruct where hide is true"),
# )))
# ```
#
# ## Comparison
#
# Compare known-answer accuracy, vocabulary coverage, and accuracy over all selected answers.
#
# ## Result
#
# {{< evidence P062 status >}}
#
# Known-only accuracy stays high even as coverage falls. Excluding unfamiliar answers from the content score does not mean they were recovered.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P062 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P062 script >}}
#
# ### Training and controls
#
# Train for 400 updates on 8,192 rows; storage grows to admit all eight labels.
# Evaluation selects exactly 1,024 of 2,048 independent records. Compare the
# original set with copies retaining 50%, 10%, or 0% of selected target names.
# All copies preserve the exact visible inputs and query-selected masks.
#
# Repeat the 10%-coverage case with unknown targets grouped into separate
# batches. Count-weighted epoch metrics must agree despite all-OOV batches.
# Scores with no known targets must be undefined, not zero reconstruction
# error. Naive accuracy over all selected valued targets counts OOV as wrong.
#
# ### Complete experiment code
#

# %%
"""P062: report stable reconstruction quality alongside pretraining coverage."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import relflow as rf
from reporting import report

PROOF_ID = "P062"
BUDGET = 400
LABELS = tuple(f"class-{index}" for index in range(8))


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for index, label in enumerate(rng.integers(0, len(LABELS), rows)):
        yield {"x": float(label), "code": LABELS[label], "hide": index % 2 == 0}


def build() -> rf.Model:
    model = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        x=rf.Number,
        code=rf.Category(mask=rf.Mask(query="hide", reconstruct=True)),
    )
    return model


def evaluate(trainer: lit.Trainer, model: rf.Model, rows: list[dict]) -> dict[str, float]:
    data = rf.SyntheticDataModule(validate=lambda: iter(rows), seed=0)
    result = trainer.validate(model, datamodule=data, verbose=False)[0]
    prefix = ".code/validate."
    return {name.removeprefix(prefix): float(value) for name, value in result.items() if name.startswith(prefix)}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    data = rf.SyntheticDataModule(
        train=partial(records, rows=8192, seed=seed + 1),
        validate=partial(records, rows=1024, seed=seed + 2),
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
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer
    original = list(records(rows=2048, seed=seed + 3))
    vocabulary = rf.Category.vocabulary(model, "/code")
    counts = rf.Category.counts(model, "/code")
    metrics, checks = {}, {}
    for keep in (1024, 512, 102, 0):
        selected = 0
        rows = []
        for row in original:
            unavailable = row["hide"] and selected >= keep
            rows.append({**row, "code": f"novel-{row['code']}" if unavailable else row["code"]})
            selected += int(row["hide"])
        measured = evaluate(trainer, model, rows)
        # Keep undefined quantities explicit without nonstandard JSON NaNs.
        metrics[f"known_{keep}"] = {name: value if np.isfinite(value) else None for name, value in measured.items()}
        checks[f"{keep}: exact known and unavailable target counts"] = (
            measured.get("targets.known") == keep and measured.get("targets.unavailable") == 1024 - keep
        )
        checks[f"{keep}: coverage matches selected targets"] = (
            abs(measured.get("coverage.content", -1) - keep / 1024) < 1e-7
        )
        if keep:
            checks[f"{keep}: known accuracy exceeds 0.95"] = measured.get("accuracy.content", -1) > 0.95
            checks[f"{keep}: all-target accuracy includes unavailable errors"] = (
                abs(measured.get("accuracy.all", -1) - measured.get("accuracy.content", 0) * keep / 1024) < 1e-7
            )
        else:
            checks["All-OOV conditional scores are undefined"] = all(
                name in measured and np.isnan(measured[name])
                for name in ("accuracy.content", "nll.content", "loss.content")
            )
            checks["All-OOV all-target accuracy is zero"] = measured.get("accuracy.all") == 0
        if keep == 102:
            rng = np.random.default_rng(seed + 4)
            permuted = [rows[index] for index in rng.permutation(len(rows))]
            regrouped = evaluate(trainer, model, permuted)
            names = (
                "targets.known",
                "targets.unavailable",
                "coverage.content",
                "accuracy.content",
                "accuracy.all",
                "nll.content",
                "loss.content",
            )
            drift = (
                max(abs(measured[name] - regrouped[name]) for name in names)
                if all(name in measured and name in regrouped for name in names)
                else None
            )
            metrics["batch_partition_drift"] = drift
            metrics["reported_content_loss_drift"] = abs(measured["loss.content"] - regrouped["loss.content"])
            checks["Rebatching preserves counts and scores"] = drift is not None and drift < 1e-5
    checks["Validation leaves mapping and exposures frozen"] = (
        rf.Category.vocabulary(model, "/code") == vocabulary and rf.Category.counts(model, "/code") == counts
    )
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7621)

# %% [markdown]
# </details>
