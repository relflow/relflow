# %% [markdown]
# ---
# title: Can admitting a new input change predictions before learning?
# categories:
# - Vocabulary
# evidence-id: P066
# description: Introduce a new cohort of input identities while keeping output labels and hidden-answer selection
#   fixed.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Largest probability change, second cohort
#   metric:
#   - '1'
#   - admission_max_drift
#   format: number
# - label: Average probability change, second cohort
#   metric:
#   - '1'
#   - admission_mean_drift
#   format: number
# evidence-group: Vocabulary
# evidence-result:
#   accelerator: cpu
#   checks:
#     '0: learned predictions beat constant baseline': true
#     '0: learned probability RMSE below 0.16': true
#     '0: shuffled identity worsens RMSE by more than 0.08': true
#     '1: admission actually adds training identities': true
#     '1: admission preserves predictions': true
#     '1: admission preserves reconstruction NLL': true
#     '1: cold evaluation preserves vocabulary': true
#     '1: cold marginal RMSE below 0.15': false
#     '1: learned predictions beat constant baseline': true
#     '1: learned probability RMSE below 0.16': true
#     '1: output coverage remains complete': true
#     '1: shuffled identity worsens RMSE by more than 0.08': true
#     '2: admission actually adds training identities': true
#     '2: admission preserves predictions': true
#     '2: admission preserves reconstruction NLL': true
#     '2: cold evaluation preserves vocabulary': true
#     '2: cold marginal RMSE below 0.15': true
#     '2: learned predictions beat constant baseline': true
#     '2: learned probability RMSE below 0.16': true
#     '2: output coverage remains complete': true
#     '2: shuffled identity worsens RMSE by more than 0.08': true
#   duration_seconds: 33.73
#   metrics:
#     '0':
#       learned:
#         brier: 0.13016813653281054
#         constant_rmse: 0.35653856748344714
#         nll: 0.39009981143772576
#         oracle_rmse: 0.09881737134570236
#       shuffled:
#         brier: 0.2761187454615636
#         constant_rmse: 0.35653856748344714
#         nll: 0.9079471927960872
#         oracle_rmse: 0.40221465714233645
#       steps: 400
#     '1':
#       admission_max_drift: 0.0
#       admission_mean_drift: 0.0
#       admitted:
#         brier: 0.25025677901059445
#         constant_rmse: 0.1832768630837761
#         nll: 0.7102360496200697
#         oracle_rmse: 0.16533115814184496
#       after_validation:
#         accuracy.all: 0.68603515625
#         accuracy.content: 0.68603515625
#         accuracy.state: 1.0
#         coverage.content: 1.0
#         loss.content: 0.710236132144928
#         loss.state: 0.0
#         nll.content: 0.710236132144928
#         targets.known: 4096.0
#         targets.unavailable: 0.0
#         vocabulary.size: 2.0
#       before_validation:
#         accuracy.all: 0.68603515625
#         accuracy.content: 0.68603515625
#         accuracy.state: 1.0
#         coverage.content: 1.0
#         loss.content: 0.710236132144928
#         loss.state: 0.0
#         nll.content: 0.710236132144928
#         targets.known: 4096.0
#         targets.unavailable: 0.0
#         vocabulary.size: 2.0
#       cold:
#         brier: 0.25025677901059445
#         constant_rmse: 0.1832768630837761
#         nll: 0.7102360496200697
#         oracle_rmse: 0.16533115814184496
#       learned:
#         brier: 0.12703330772120547
#         constant_rmse: 0.3581464679001816
#         nll: 0.3728922615371408
#         oracle_rmse: 0.08041273503482016
#       shuffled:
#         brier: 0.30907278196613025
#         constant_rmse: 0.3581464679001816
#         nll: 1.185680018041411
#         oracle_rmse: 0.43103875096176353
#       steps: 400
#     '2':
#       admission_max_drift: 0.0
#       admission_mean_drift: 0.0
#       admitted:
#         brier: 0.23091117008085618
#         constant_rmse: 0.1832768630837761
#         nll: 0.6600815420009657
#         oracle_rmse: 0.08695420407151533
#       after_validation:
#         accuracy.all: 0.6708984375
#         accuracy.content: 0.6708984375
#         accuracy.state: 1.0
#         coverage.content: 1.0
#         loss.content: 0.6600815653800964
#         loss.state: 0.0
#         nll.content: 0.6600815653800964
#         targets.known: 4096.0
#         targets.unavailable: 0.0
#         vocabulary.size: 2.0
#       before_validation:
#         accuracy.all: 0.6708984375
#         accuracy.content: 0.6708984375
#         accuracy.state: 1.0
#         coverage.content: 1.0
#         loss.content: 0.6600815653800964
#         loss.state: 0.0
#         nll.content: 0.6600815653800964
#         targets.known: 4096.0
#         targets.unavailable: 0.0
#         vocabulary.size: 2.0
#       cold:
#         brier: 0.23091117008085618
#         constant_rmse: 0.1832768630837761
#         nll: 0.6600815420009657
#         oracle_rmse: 0.08695420407151533
#       learned:
#         brier: 0.13000353869416703
#         constant_rmse: 0.3569987065711852
#         nll: 0.3891625634822828
#         oracle_rmse: 0.09318771445617331
#       shuffled:
#         brier: 0.31360807871628105
#         constant_rmse: 0.3569987065711852
#         nll: 1.2603889988298727
#         oracle_rmse: 0.42845138798358834
#       steps: 400
#   mode: full
#   outcome: not_met
#   provenance:
#     code_sha256: 275b84fb0628596489ac62f3ba8790a495aef7da3ed756c942b5ba61a2f04e34
#     cuda: null
#     evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     gpu: null
#     host: 9de9006c9438
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
#     script_sha256: 3ee44e4d2b83d10b012d8438f5eb8d1b33928e4c3aebab11efb55b973967ed2c
#     source_sha256: 638ca829313d9a384b2c7be114f11ee325615ba5ba7cc31ae7b8a0d6c26028a5
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   seed: 7801
#   started_at: '2026-11-02T04:21:04.581221+00:00'
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
# entity: cohort-0-id-42
# context: high
# label: yes
# hide: true
# ```
#
# label is hidden where hide is true. Each new cohort has unrelated identity effects.
#
# ```{typst}
# //| label: fig-evidence-rolling-input-admission
# //| fig-cap: "Amber cards are hidden prediction targets. hide chooses which label answers are hidden."
# //| fig-alt: "Model tree with record, entity, context, label. Amber cards are hidden prediction targets. hide chooses which label answers are hidden."
# #tree(node("record", kind: "root", children: (
#   node("entity", type: "Category", detail: "Growing input vocabulary"),
#   node("context", type: "Category", detail: "Fixed low/high evidence"),
#   node("label", kind: "target", type: "Category", detail: "Query-selected reconstruction"),
# )))
# ```
#
# ## Comparison
#
# Predict the same held-out rows just before and just after admitting independent training records, without an optimizer update.
#
# ## Result
#
# {{< evidence P066 status >}}
#
# The selected run records zero prediction drift and unchanged reconstruction NLL when new input identities are admitted, before an optimizer update. Its overall result remains unmet: the second cohort's cold marginal RMSE is 0.1653 against the declared 0.15 limit. Preserving predictions at admission does not by itself establish cold-input prediction quality. The measurements and library revision remain those recorded above.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P066 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P066 script >}}
#
# ### Training and controls
#
# Each stage has 8,192 training rows, 1,024 entity-disjoint validation rows,
# 4,096 independent held-out rows, and 400 AdamW updates. Optimizer state is
# restarted between fits; this tests staged onboarding, not uninterrupted
# optimizer/worker scheduling. A separate worker test checks admission itself.
# No admission uses validation or test observations. The latent identity effect
# is available only to the evaluator, never as a model field.
#
# Subsequent cohorts must have admission drift below 1e-6 and identical
# count-normalized reconstruction NLL. After learning, probability RMSE must
# beat the constant baseline and remain below 0.16; shuffling identities must
# worsen it by at least 0.08. Cold marginal RMSE must remain below 0.15.
# These quality gates are provisional; the admission invariant is structural.
# The same generator and gates run unchanged against the pre-fix baseline.
#
# ### Complete experiment code
#

# %%
"""P066: admission stability, cold probabilities, and subsequent identity learning."""

from collections.abc import Iterator

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P066"
BUDGET = 400
IDENTITIES = 256


def records(*, rows: int, seed: int, cohort: int = 0) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    identities = rng.integers(IDENTITIES, size=rows)
    effects = np.random.default_rng(918 + cohort).choice([-2.5, 2.5], IDENTITIES)
    context = rng.choice([-2.0, 2.0], rows)
    probabilities = 1 / (1 + np.exp(-(context + effects[identities])))
    outcomes = rng.random(rows) < probabilities
    for i, (identity, x, outcome) in enumerate(zip(identities, context, outcomes, strict=True)):
        yield {
            "entity": f"cohort-{cohort}-id-{identity}",
            "context": "low" if x < 0 else "high",
            "label": "yes" if outcome else "no",
            "hide": i % 4 != 0,
        }


def build() -> rf.Model:
    model = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        entity=rf.Category(p_unavailable=0.2),
        context=rf.Category(p_unavailable=0.0),
        label=rf.Category(p_unavailable=0.0, mask=rf.Mask(query="hide", reconstruct=True)),
    )
    return model


def probability(model: rf.Model, rows: list[dict]) -> np.ndarray:
    predictions = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    content = [row["label"]["content"] for row in predictions]
    return np.asarray([item["probability"] if item["value"] == "yes" else 1 - item["probability"] for item in content])


def score(p: np.ndarray, rows: list[dict], cohort: int, *, cold: bool = False) -> dict[str, float]:
    x = np.asarray([-2.0 if row["context"] == "low" else 2.0 for row in rows])
    if cold:
        oracle = (1 / (1 + np.exp(-(x - 2.5))) + 1 / (1 + np.exp(-(x + 2.5)))) / 2
    else:
        effects = np.random.default_rng(918 + cohort).choice([-2.5, 2.5], IDENTITIES)
        identities = [int(row["entity"].split("-")[-1]) for row in rows]
        oracle = 1 / (1 + np.exp(-(x + effects[identities])))
    y = np.asarray([row["label"] == "yes" for row in rows], dtype=float)
    clipped = p.clip(1e-7, 1 - 1e-7)
    return {
        "oracle_rmse": float(np.sqrt(np.mean((p - oracle) ** 2))),
        "constant_rmse": float(np.sqrt(np.mean((0.5 - oracle) ** 2))),
        "nll": float(np.mean(-y * np.log(clipped) - (1 - y) * np.log1p(-clipped))),
        "brier": float(np.mean((p - y) ** 2)),
    }


def evaluate(trainer: lit.Trainer, model: rf.Model, rows: list[dict]) -> dict[str, float]:
    data = rf.SyntheticDataModule(validate=lambda: iter(rows), seed=0)
    result = trainer.validate(model, datamodule=data, verbose=False)[0]
    prefix = ".label/validate."
    return {name.removeprefix(prefix): float(value) for name, value in result.items() if name.startswith(prefix)}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    metrics, checks = {}, {}
    trainer = None
    for cohort in range(3):
        rows = [{**row, "hide": True} for row in records(rows=4096, seed=seed + 100 + cohort, cohort=cohort)]
        training = list(records(rows=8192, seed=seed + 1 + cohort, cohort=cohort))
        arm = {}
        if trainer is not None:
            vocabulary = rf.Category.vocabulary(model, "/entity")
            before = probability(model, rows)
            before_metrics = evaluate(trainer, model, rows)
            checks[f"{cohort}: cold evaluation preserves vocabulary"] = (
                rf.Category.vocabulary(model, "/entity") == vocabulary
            )
            arm["cold"] = score(before, rows, cohort, cold=True)
            model.encode(pa.Table.from_pylist(training), strata="train")
            after = probability(model, rows)
            after_metrics = evaluate(trainer, model, rows)
            arm["admission_max_drift"] = float(np.max(np.abs(after - before)))
            arm["admission_mean_drift"] = float(np.mean(np.abs(after - before)))
            arm["admitted"] = score(after, rows, cohort, cold=True)
            arm["before_validation"] = before_metrics
            arm["after_validation"] = after_metrics
            checks.update(
                {
                    f"{cohort}: admission actually adds training identities": len(
                        rf.Category.vocabulary(model, "/entity")
                    )
                    == (cohort + 1) * IDENTITIES,
                    f"{cohort}: admission preserves predictions": arm["admission_max_drift"] < 1e-6,
                    f"{cohort}: admission preserves reconstruction NLL": abs(
                        before_metrics["nll.content"] - after_metrics["nll.content"]
                    )
                    < 1e-6,
                    f"{cohort}: output coverage remains complete": before_metrics["coverage.content"]
                    == after_metrics["coverage.content"]
                    == 1.0,
                    f"{cohort}: cold marginal RMSE below 0.15": arm["cold"]["oracle_rmse"] < 0.15,
                }
            )
        validation = list(records(rows=1024, seed=seed + 20 + cohort, cohort=cohort + 1))
        data = rf.SyntheticDataModule(train=lambda: iter(training), validate=lambda: iter(validation), seed=seed)
        # Lightning does not reset eval mode when fitting the same model again.
        model.train()
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
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True,
            num_sanity_val_steps=0,
        )
        trainer = model.trainer
        model.eval()
        arm["steps"] = trainer.global_step
        arm["learned"] = score(probability(model, rows), rows, cohort)
        permutation = np.random.default_rng(seed + 500).permutation(len(rows))
        shuffled = [{**row, "entity": rows[index]["entity"]} for row, index in zip(rows, permutation, strict=True)]
        arm["shuffled"] = score(probability(model, shuffled), rows, cohort)
        checks.update(
            {
                f"{cohort}: learned probability RMSE below 0.16": arm["learned"]["oracle_rmse"] < 0.16,
                f"{cohort}: learned predictions beat constant baseline": arm["learned"]["oracle_rmse"]
                < arm["learned"]["constant_rmse"],
                f"{cohort}: shuffled identity worsens RMSE by more than 0.08": arm["shuffled"]["oracle_rmse"]
                - arm["learned"]["oracle_rmse"]
                > 0.08,
            }
        )
        metrics[str(cohort)] = arm
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7801)

# %% [markdown]
# </details>
