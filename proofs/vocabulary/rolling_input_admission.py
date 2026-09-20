# %% [markdown]
# ---
# title: Category admission during staged pretraining
# categories: [Vocabulary]
# proof-id: P066
# description: New input identities should inherit the unavailable fallback before learning, without losing their ability to specialize.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P066 status >}}
#
# ## Insights
#
# Allocating an identity slot is not evidence about that identity. The current
# Category nevertheless activates a random vector when a previously unknown
# input receives a slot. This can change predictions without an optimizer step.
# Starting input content at zero removes that jump, but tested zero-content
# and factorized candidates regress identity matching or calibration elsewhere.
# A passing admission proof alone is not sufficient to adopt such a change.
#
# This proof reconstructs a query-masked categorical field across three
# training cohorts. Each cohort contains 256 unrelated identities in a
# 16,384-slot input vocabulary. In-training validation uses the *next*, wholly
# unseen cohort. Before learning each subsequent cohort, compare exactly the
# same held-out rows before and after admitting independent training records.
# The output vocabulary and reconstruction masks stay fixed.
#
# ```yaml
# entity: cohort-0-id-42
# context: high
# label: yes
# hide: true
# ```
#
# The identity has a persistent hidden effect on the probability of yes.
#
# ```yaml
# entity: cohort-1-id-42
# context: high
# label: no
# hide: true
# ```
#
# A new cohort has unrelated identity effects; the spelling gives no clue.
#
# ```yaml
# entity: cohort-1-id-42
# context: low
# label: no
# hide: false
# ```
#
# One quarter of training coordinates remain visible. Held-out scoring hides
# every label, so the answer cannot leak into the input.
#
# ```{typst}
# //| label: fig-proof-rolling-input-admission
# //| fig-cap: "Input identity changes from unavailable to admitted to learned; output labels remain fixed."
# //| fig-alt: "A root contains Category entity, Category context, and query-masked Category label."
# #tree(node("record", kind: "root", children: (
#   node("entity", type: "Category", detail: "Growing input vocabulary"),
#   node("context", type: "Category", detail: "Fixed low/high evidence"),
#   node("label", kind: "target", type: "Category", detail: "Query-selected reconstruction"),
# )))
# ```

# %%
"""P066: admission stability, cold probabilities, and subsequent identity learning."""

from collections.abc import Iterator

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
from reporting import report

import relflow as rf

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
        batch_size=128,
        dropout=0.0,
        entity=rf.Category(p_unavailable=0.2),
        context=rf.Category(p_unavailable=0.0),
        label=rf.Category(p_unavailable=0.0, mask=rf.Mask(query="hide", reconstruct=True)),
    )
    model.optimizer = rf.adamw(learning_rate=0.002)
    return model


def probability(model: rf.Model, rows: list[dict]) -> np.ndarray:
    predictions = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    content = [row["record/label"]["content"] for row in predictions]
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
    data = rf.SyntheticDataModule(model=model, validate=lambda: iter(rows), seed=0)
    result = trainer.validate(model, datamodule=data, verbose=False)[0]
    prefix = "record.label/validate."
    return {name.removeprefix(prefix): float(value) for name, value in result.items() if name.startswith(prefix)}


# %% [markdown]
# ## Training and controls
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


# %%
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
            vocabulary = rf.Category.vocabulary(model, "record/entity")
            before = probability(model, rows)
            before_metrics = evaluate(trainer, model, rows)
            checks[f"{cohort}: cold evaluation preserves vocabulary"] = (
                rf.Category.vocabulary(model, "record/entity") == vocabulary
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
                        rf.Category.vocabulary(model, "record/entity")
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
        trainer = lit.Trainer(
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
        validation = list(records(rows=1024, seed=seed + 20 + cohort, cohort=cohort + 1))
        data = rf.SyntheticDataModule(
            model=model, train=lambda: iter(training), validate=lambda: iter(validation), seed=seed
        )
        # Lightning does not reset eval mode when fitting the same model again.
        model.train()
        trainer.fit(model, datamodule=data)
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


# %% [markdown]
# ## Evidence and limits
#
# {{< proof P066 evidence >}}
#
# This isolates input Category admission with fixed yes/no output support.
# Growing output vocabularies, running numerical normalization, sampled masks,
# population shift, and subsequent optimizer steps can still change validation
# scores. The initialization candidates were rejected; the current model still
# fails admission stability. No universal calibration claim, Set/Cluster fix,
# or rare-ID generalization follows. See P036 and P060 for the identity-matching
# and prior-calibration regressions that blocked the tested alternatives.
#
# ## Reproduce
#
# {{< proof P066 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=7801)
