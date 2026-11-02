# %% [markdown]
# ---
# title: Can it learn separate probabilities for Set members?
# categories:
# - Vocabulary
# evidence-id: P063
# description: With no useful context, red appears in 90% of answer sets and blue in 20%. Predict each member’s probability.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Predicted red membership
#   metric:
#   - capacity_2
#   - probabilities
#   - red
#   format: percent
# - label: Predicted blue membership
#   metric:
#   - capacity_2
#   - probabilities
#   - blue
#   format: percent
# evidence-group: Vocabulary
# evidence-result:
#   accelerator: cpu
#   checks:
#     '2: evaluation never admits unknown labels': true
#     '2: excess expected NLL below 0.02': true
#     '2: member and complete-set coverage are one half': true
#     '2: member counts include partial and all-OOV targets': true
#     '2: probabilities recover prevalence within 0.06': true
#     '2: rebatching preserves epoch scores': true
#     '512: evaluation never admits unknown labels': true
#     '512: excess expected NLL below 0.02': true
#     '512: member and complete-set coverage are one half': true
#     '512: member counts include partial and all-OOV targets': true
#     '512: probabilities recover prevalence within 0.06': true
#     '512: rebatching preserves epoch scores': true
#   duration_seconds: 15.855
#   metrics:
#     capacity_2:
#       excess_nll: 0.004481678124703459
#       probabilities:
#         blue: 0.2256091833114624
#         red: 0.860715925693512
#       steps: 400
#     capacity_512:
#       excess_nll: 0.005716963929016605
#       probabilities:
#         blue: 0.23143301904201508
#         red: 0.8562422394752502
#       steps: 400
#     coverage_2:
#       accuracy.content: 0.5
#       accuracy.set: 0.0
#       accuracy.state: 0.800000011920929
#       coverage.content: 0.5
#       coverage.set: 0.5
#       f1.all: 0.25
#       f1.content: 0.3333333432674408
#       loss.content: 1.039962649345398
#       loss.state: 3.055330991744995
#       precision.content: 0.25
#       recall.all: 0.25
#       recall.content: 0.5
#       sets.complete: 256.0
#       sets.valued: 512.0
#       targets.known: 256.0
#       targets.unavailable: 256.0
#       vocabulary.size: 2.0
#     coverage_512:
#       accuracy.content: 0.5
#       accuracy.set: 0.0
#       accuracy.state: 0.800000011920929
#       coverage.content: 0.5
#       coverage.set: 0.5
#       f1.all: 0.25
#       f1.content: 0.3333333432674408
#       loss.content: 1.0284032821655273
#       loss.state: 3.063979148864746
#       precision.content: 0.25
#       recall.all: 0.25
#       recall.content: 0.5
#       sets.complete: 256.0
#       sets.valued: 512.0
#       targets.known: 256.0
#       targets.unavailable: 256.0
#       vocabulary.size: 2.0
#     oracle_nll: 0.41274269846481804
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: c7a14d38b146ca8fbd3ce5a29d68f5aad89b8cc6d45138625a16d0d0c5089a20
#     cuda: null
#     evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     gpu: null
#     host: 0256c7c6ffe5
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
#     script_sha256: 4cfeddca8e0d7bbc84a8f608ea644501296ea80f76e9f0c2c867befab8ddb457
#     source_sha256: b02b5e5ab04cf973a828fed1d27b57319a0ca6e2c1cb8d4dec68963493462c89
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   seed: 7631
#   started_at: '2026-11-02T04:20:43.777626+00:00'
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
# x: 0.0
# labels: [red, blue]
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-set-reconstruction-coverage
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with record, x, labels. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number", detail: "Constant"),
#   node("labels", kind: "target", type: "Set", detail: "Independent memberships"),
# )))
# ```
#
# ## Comparison
#
# Compare internal output allocations and test sets that also contain unfamiliar members. Known positive and negative annotations remain usable.
#
# ## Result
#
# {{< evidence P063 status >}}
#
# The member probabilities recover their population rates. Unknown members still reduce coverage and cannot be emitted; annotations here are complete sets, not partially supplied labels.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P063 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P063 script >}}
#
# ### Training and controls
#
# Each arm sees 32,768 training rows and 1,024 independent validation rows.
# Evaluate written probabilities and expected Bernoulli log loss against the
# oracle. Then replace validation with equal numbers of partial-OOV, all-OOV,
# empty, known-only and null sets. Rebatch them in a different order. Member
# coverage must be 1/2, complete-set coverage 1/2, and all epochs must count
# exactly 256 known and 256 unknown positive memberships. Unused capacity
# must not change the meaning of the reported membership loss.
#
# ### Complete experiment code
#

# %%
"""P063: membership probability and coverage have explicit support."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from relflow.tensorfields.shared.resize import Resize
from reporting import report

PROOF_ID = "P063"
BUDGET = 400


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for red, blue in rng.random((rows, 2)) < np.asarray([0.9, 0.2]):
        yield {"x": 0.0, "labels": [name for name, present in (("red", red), ("blue", blue)) if present]}


def build(size: int) -> rf.Model:
    model = rf.Model(
        d_model=16,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        x=rf.Number,
        labels=rf.Set(p_unavailable=1.0, mask=True),
    )
    # Experimental storage control only: allocation is not a schema option.
    node = model.nodes["/labels"]
    resize = Resize()
    resize.embedding(node.embedder.embeddings["content"], size)
    resize.linear(node.decoder.linears["content"], size)
    node.embedder.counters["content"].resize(size, resize)
    resize.attribute(node.embedder.vocab, "size", size)
    resize.commit()
    return model


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    metrics, checks = {}, {}
    prior = np.asarray([0.9, 0.2])
    oracle = float(np.mean(-prior * np.log(prior) - (1 - prior) * np.log1p(-prior)))
    metrics["oracle_nll"] = oracle
    for size in (2, 512):
        lit.seed_everything(seed, workers=True)
        model = build(size)
        data = rf.SyntheticDataModule(
            train=partial(records, rows=32768, seed=seed + 1),
            validate=partial(records, rows=1024, seed=seed + 2),
            seed=seed,
        )
        model.fit(
            data,
            batch_size=128,
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
        vocabulary = rf.Set.vocabulary(model, "/labels")
        written = model.predict(pa.table({"x": [0.0]}))["predictions"].to_pylist()[0]["labels"]["content"]
        probabilities = {item["value"]: item["probability"] for item in written}
        p = np.asarray([probabilities[name] for name in ("red", "blue")]).clip(1e-7, 1 - 1e-7)
        excess = float(np.mean(-prior * np.log(p) - (1 - prior) * np.log1p(-p)) - oracle)
        metrics[f"capacity_{size}"] = {
            "probabilities": probabilities,
            "excess_nll": excess,
            "steps": trainer.global_step,
        }
        checks[f"{size}: probabilities recover prevalence within 0.06"] = bool(np.max(np.abs(p - prior)) < 0.06)
        checks[f"{size}: excess expected NLL below 0.02"] = excess < 0.02
        rows = [
            {"x": 0.0, "labels": labels}
            for labels in (["red", "novel"], ["novel"], [], ["blue"], None)
            for _ in range(128)
        ]
        results = []
        for order in (np.arange(len(rows)), np.random.default_rng(seed + 3).permutation(len(rows))):
            table = pa.Table.from_pylist([rows[index] for index in order])
            measured = trainer.validate(
                model, datamodule=rf.ArrowDataModule(validate=table, num_workers=0), verbose=False
            )[0]
            prefix = ".labels/validate."
            results.append(
                {key.removeprefix(prefix): float(value) for key, value in measured.items() if key.startswith(prefix)}
            )
        first, second = results
        metrics[f"coverage_{size}"] = {key: value if np.isfinite(value) else None for key, value in first.items()}
        checks[f"{size}: member counts include partial and all-OOV targets"] = (
            first.get("targets.known") == 256 and first.get("targets.unavailable") == 256
        )
        checks[f"{size}: member and complete-set coverage are one half"] = (
            first.get("coverage.content") == 0.5 and first.get("coverage.set") == 0.5
        )
        names = (
            "targets.known",
            "targets.unavailable",
            "coverage.content",
            "coverage.set",
            "loss.content",
            "accuracy.set",
            "f1.all",
        )
        checks[f"{size}: rebatching preserves epoch scores"] = all(
            name in first and name in second and abs(first[name] - second[name]) < 1e-5 for name in names
        )
        checks[f"{size}: evaluation never admits unknown labels"] = rf.Set.vocabulary(model, "/labels") == vocabulary
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7631)

# %% [markdown]
# </details>
