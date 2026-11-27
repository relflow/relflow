# %% [markdown]
# ---
# title: Can admitting a new input change predictions before learning?
# description: Introduce a new cohort of input identities while keeping output labels and hidden-answer selection
#   fixed.
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
# evidence-review:
#   question: Can admitting a new input change predictions before learning?
#   modeling: Train on one cohort of identities, then admit a new cohort from training data without an optimizer update.
#     Predict the same held-out rows immediately before and after admission. Measure the largest change in predicted
#     probability for each later cohort.
#   comparison:
#     metric: Prediction drift
#     kind: Recorded conditions
#     conditions:
#     - label: Second cohort · largest change
#       path:
#       - '1'
#       - admission_max_drift
#       format: number
#     - label: Third cohort · largest change
#       path:
#       - '2'
#       - admission_max_drift
#       format: number
#     definition: Absolute change in predictions after an operation, on the scale named in each condition. Zero means
#       no change.
#     direction: Task-specific interpretation
#   note: Unchanged predictions test the admission invariant only. Separate learning-quality gates must also hold
#     after continued fitting.
#   criteria:
#     prefixes:
#       '0': Initial cohort
#       '1': Second cohort
#       '2': Third cohort
# evidence-plot:
#   target: target
#   focus: What changes for existing and newly admitted identity cohorts?
#   comparison: Follow the same named cohorts before and after each admission phase.
#   limit: Do not mix a changed cohort population with a change in its predictions.
#   renderer: plot
# evidence-cost:
#   baseline: 65
#   candidate: 65
# ---
#
# Adding an input identity to a vocabulary and learning that identity's behavior are separate
# operations. This experiment follows three cohorts whose identity-specific effects are unrelated. The
# same model encounters each cohort in stages, allowing a direct test of whether admission alone
# changes its predictions before the next fitting phase begins.
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
#
# The illustrated row supplies an identity and the shared context `high`; the yes/no answer is sampled
# from their combined effect. The hide flag selects which training labels are reconstructed and is not
# a predictive schema field. A new cohort preserves the familiar low/high context and yes/no answer
# names, but its individual identity effects must be learned afresh.
#
# ## Comparison
#
# Predict the same held-out rows just before and just after admitting independent training records,
# without an optimizer update.
#
# For later cohorts, the same held-out rows are predicted immediately before and after independent
# training records admit their identities. No optimizer step occurs between those predictions. This
# isolates admission drift from learning. After fitting, shuffling identities across the held-out rows
# keeps context and outcome frequencies while breaking the personal effect the model should now use.
#
# ## Result
#
# The selected run records zero prediction drift and unchanged reconstruction NLL when new input
# identities are admitted, before an optimizer update. Its overall result remains unmet: the second
# cohort's cold marginal RMSE is 0.1653 against the declared 0.15 limit. Preserving predictions at
# admission does not by itself establish cold-input prediction quality. The measurements and library
# revision remain those recorded above.
#
# Zero admission drift is a storage and execution invariant, not a quality score. The cold probability
# error instead compares predictions with the best mixture over still-unknown identity effects. The
# learned error uses each identity's generating effect. Read both: stable admission can coexist with
# an unmet cold-quality criterion, and these staged fits do not test uninterrupted optimizer-state
# continuation.
#
# ## Method
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

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.10",
#     "lightning",
#     "numpy",
#     "polars",
#     "pyarrow",
#     "pyyaml>=6",
#     "relflow>=0.6.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "training"
# types = ["Category"]
# conditions = ["unseen-identities", "unseen-values", "changed-associations", "training-updates"]
# ///

"""P066: admission stability, cold probabilities, and subsequent identity learning."""

import sys
from collections.abc import Iterator
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
from reporting import experiment, report

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


class PredictionModel(rf.Model):
    entity = rf.Category(p_unavailable=0.2)
    context = rf.Category(p_unavailable=0.0)
    label = rf.Category(p_unavailable=0.0, task=rf.reconstruct(query="hide"))


def build() -> rf.Model:
    model = PredictionModel(d_model=32, n_layers=1, n_heads=4, dropout=0.0)
    return model


def probability(model: rf.Model, rows: list[dict]) -> np.ndarray:
    predictions = pl.DataFrame(model.predict(pa.Table.from_pylist(rows)))["predictions"].to_list()
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


@experiment("P066", seed=7801)
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


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    from evidence.diagnostics.data import human
    from evidence.diagnostics.plots import COLORS, MARKERS, axes_pair, dots

    m = bundle.record["metrics"]
    fig, axes = axes_pair()
    keys = sorted(m, key=int)
    for index, stage in enumerate(["cold", "admitted", "learned", "shuffled"]):
        present = [key for key in keys if stage in m[key]]
        axes[0].plot(
            [int(key) for key in present],
            [m[key][stage]["oracle_rmse"] for key in present],
            marker=MARKERS[index],
            color=COLORS[index],
            label=human(stage),
        )
    axes[0].set(
        xticks=list(map(int, keys)),
        xlabel="New identity cohort",
        ylabel="RMSE against its conditional oracle",
        title="Availability and learning are separate",
    )
    axes[0].legend(fontsize=8)
    present = [key for key in keys if "admission_max_drift" in m[key]]
    dots(
        axes[1],
        [f"Cohort {key}" for key in present],
        {"Maximum admission drift": [m[key]["admission_max_drift"] for key in present]},
        xlabel="Largest paired output change at admission",
    )
    return (
        fig,
        "Only measured stages are drawn for each cohort. The admission drift compares the same observations immediately before and after admitting their identities.",
        {},
    )


if __name__ == "__main__":
    report(run)
