# %% [markdown]
# ---
# title: Can Cluster reconstruction hide missing coverage too?
# categories:
# - Vocabulary
# evidence-id: P065
# description: Reconstruct a Cluster answer from a categorical cue while reducing the share of test identities seen
#   in training.
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
#   duration_seconds: 16.167
#   metrics:
#     batch_partition_drift: 0.0
#     known_0:
#       accuracy.all: 0.0
#       accuracy.content: null
#       accuracy.state: 1.0
#       coverage.content: 0.0
#       loss.content: null
#       loss.state: 2.0099909306736663e-05
#       nll.content: null
#       targets.known: 0.0
#       targets.unavailable: 1024.0
#       vocabulary.size: 8.0
#     known_102:
#       accuracy.all: 0.099609375
#       accuracy.content: 1.0
#       accuracy.state: 1.0
#       coverage.content: 0.099609375
#       loss.content: 0.3977391719818115
#       loss.state: 2.0099909306736663e-05
#       nll.content: 0.3977391719818115
#       targets.known: 102.0
#       targets.unavailable: 922.0
#       vocabulary.size: 8.0
#     known_1024:
#       accuracy.all: 1.0
#       accuracy.content: 1.0
#       accuracy.state: 1.0
#       coverage.content: 1.0
#       loss.content: 0.3953397274017334
#       loss.state: 2.0099909306736663e-05
#       nll.content: 0.3953397274017334
#       targets.known: 1024.0
#       targets.unavailable: 0.0
#       vocabulary.size: 8.0
#     known_512:
#       accuracy.all: 0.5
#       accuracy.content: 1.0
#       accuracy.state: 1.0
#       coverage.content: 0.5
#       loss.content: 0.3977148234844208
#       loss.state: 2.0099909306736663e-05
#       nll.content: 0.3977148234844208
#       targets.known: 512.0
#       targets.unavailable: 512.0
#       vocabulary.size: 8.0
#     reported_content_loss_drift: 0.0
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 51db7839f859a0c107f3a00cd2cb039e6b481033110353215488e1646c39395a
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 22e305700dc9
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
#     script_sha256: 2190897d11465810aaef5d299b8e019462037ec36485121201a0c592ab92c25a
#     source_sha256: 29af68700b23befb1207f70abe643b06713f3c1ba69c95946b7ca050061c0c69
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 7651
#   started_at: '2026-11-01T01:48:24.056343+00:00'
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
# x: cue-2
# code: class-2
# hide: true
# ```
#
# code is hidden wherever hide is true. Unfamiliar answer identities are counted in coverage separately.
#
# ```{typst}
# //| label: fig-evidence-cluster-pretraining-coverage
# //| fig-cap: "Selector notes identify hidden values. hide chooses which code answers are hidden."
# //| fig-alt: "Model tree with record, x, code. Selector notes identify hidden values. hide chooses which code answers are hidden."
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Category", detail: "Familiar class cue"),
#   node("code", type: "Cluster", detail: "Reconstruct where hide is true"),
# )))
# ```
#
# ## Comparison
#
# Measure known-answer accuracy and all-answer accuracy separately from vocabulary coverage.
#
# ## Result
#
# {{< evidence P065 status >}}
#
# Coverage and recovery are different questions. The latest run also misses the required known-answer accuracy, so its reconstruction result remains incomplete.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P065 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P065 script >}}
#
# ### Training and controls
#
# Train for 800 updates on 8,192 rows; storage grows to admit all eight labels.
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
"""P065: report stable reconstruction quality alongside pretraining coverage."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import relflow as rf
from reporting import report

PROOF_ID = "P065"
BUDGET = 800
LABELS = tuple(f"class-{index}" for index in range(8))


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for index, label in enumerate(rng.integers(0, len(LABELS), rows)):
        yield {"x": f"cue-{label}", "code": LABELS[label], "hide": index % 2 == 0}


def build() -> rf.Model:
    model = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        x=rf.Category(p_unavailable=0.0),
        code=rf.Cluster(bounds=8, p_unavailable=1.0, mask=rf.Mask(query="hide", reconstruct=True)),
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
    vocabulary = rf.Cluster.vocabulary(model, "/code")
    counts = tuple(model.nodes["/code"].embedder.counters["content"].counts.tolist())
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
        rf.Cluster.vocabulary(model, "/code") == vocabulary
        and tuple(model.nodes["/code"].embedder.counters["content"].counts.tolist()) == counts
    )
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7651)

# %% [markdown]
# </details>
