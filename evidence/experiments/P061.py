# %% [markdown]
# ---
# title: How confident should it be about a new identity?
# categories:
# - Vocabulary
# description: Predict a yes/no outcome using a numeric signal and an identity-specific effect. New identities remove
#   only the latter information.
# evidence-readout:
# - label: Weak signal, new identity, 20% training rate
#   metric:
#   - '0.2'
#   - novel
#   - ambiguous_confidence
#   format: percent
# - label: Strong signal, new identity, 20% training rate
#   metric:
#   - '0.2'
#   - novel
#   - decisive_confidence
#   format: percent
# evidence-group: Vocabulary
# evidence-diagram:
#   caption: Amber cards are hidden prediction targets. The numeric signal remains available for unfamiliar identities.
#   alt: Model tree with record, x, entity, label. Amber cards are hidden prediction targets. The numeric signal remains
#     available for unfamiliar identities.
# evidence-review:
#   question: How confident should relflow be about a new identity?
#   modeling: Generate outcomes from a numeric signal plus an identity-specific effect. New identities remove only
#     the second source of information. Train with some input identities made unavailable, then compare confidence
#     on unseen IDs when the remaining numeric evidence is weak or decisive.
#   comparison:
#     kind: Recorded conditions
#     metric: Probability
#     conditions:
#     - label: Weak signal, new identity, 20% training rate
#       path:
#       - '0.2'
#       - novel
#       - ambiguous_confidence
#       format: percent
#     - label: Strong signal, new identity, 20% training rate
#       path:
#       - '0.2'
#       - novel
#       - decisive_confidence
#       format: percent
#     definition: Predicted probability of the named outcome or membership. The task determines the appropriate value;
#       higher is not always better.
#     direction: Task-specific interpretation
#     domain:
#     - 0
#     - 1
#   note: These conditions should have different confidence levels; higher confidence is not always better. Both bars
#     use the 20% input-unavailability training condition.
#   criteria:
#     prefixes:
#       '0.01': 1% input unavailability
#       '0.2': 20% input unavailability
# evidence-plot:
#   target: target
#   focus: What confidence is justified for an unseen identity?
#   comparison: Compare known and unavailable cohorts under each declared prior condition.
#   limit: Probability error, proper loss, and classification accuracy answer different questions.
#   renderer: plot
# evidence-cost:
#   baseline: 49
#   candidate: 49
# ---
#
# A new identity removes one source of information, but the numeric input can still be informative.
# Here the yes probability depends on both `x` and a persistent identity effect. The experiment asks
# whether predictions appropriately lose certainty about that individual effect while retaining
# certainty justified by a sufficiently strong numeric signal.
#
# ## Example
#
# ```yaml
# x: 0.0
# entity: id-42
# label: yes
# ```
#
# The outcome is hidden and randomly sampled. Scores compare predicted probabilities with the generator’s known probabilities.
#
# ```typst
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number", detail: "Generalizable evidence"),
#   node("entity", type: "Category", detail: "Persistent identity"),
#   node("label", kind: "target", type: "Category", detail: "Known yes/no labels"),
# )))
# ```
#
# At the illustrative `x = 0`, the two possible identity effects favor opposite answers; without
# knowing which effect belongs to a new identity, their average yes probability is 0.5. At a
# sufficiently large positive `x`, both effects favor yes. A confident prediction can then be
# reasonable even for an unfamiliar name. The shown yes label is sampled, not guaranteed.
#
# ## Comparison
#
# Compare training with 1% or 20% of input identities made unavailable. Score new identities separately where the numeric signal is weak or strong.
#
# Both training arms sometimes make identity content unavailable, at rates of 1% and 20%. This gives
# practice forecasting from the remaining context without changing the known yes/no output labels.
# Evaluation separately measures familiar and disjoint identity names. Renaming one unknown identity
# to another checks that its spelling does not supply an unlearned personal effect.
#
# ## Result
#
# A new identity does not always imply a 50% prediction: strong remaining context can still justify confidence. Neither unavailability rate is claimed to be best for every dataset.
#
# The confidence readouts use the probability of the more likely answer, so confident no and confident
# yes count alike. They distinguish weak-signal rows from strong-signal rows. Probability error is
# also compared with an evaluator-only oracle. These checks concern this mixture of identity effects
# and signals; they do not imply that stronger unavailability augmentation is universally better.
#
# ## Method
#
# ### Training and controls
#
# Use 32,768 training rows over 4,096 identities, automatically growing storage,
# and 600 updates. Independent validation has 1,024 rows. Evaluation has 8,192
# known-ID rows and 8,192 entity-disjoint rows, with identical visible x and
# known yes/no output labels. The latent sign is only available to the evaluator.
#
# The 20% arm must beat the constant baseline, retain decisive confidence above
# 0.9, and keep ambiguous new-ID confidence below 0.7. Record both arms without
# demanding that one stochastic run of stronger masking always beats 1%.
# A final admission-only probe measures the prediction change without learning;
# this is a diagnostic limitation, not a promise that admission is safe.
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
#     "relflow>=0.5.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "reliability"
# types = ["Category"]
# conditions = ["unseen-identities", "unseen-values", "noise"]
# ///

"""P061: confidence with sparse known IDs, unknown IDs, and decisive context."""

import sys
from collections.abc import Iterator
from functools import partial
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
from reporting import experiment, report

BUDGET = 600
ENTITIES = 4096


def records(*, rows: int, seed: int, novel: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    identities = rng.integers(0, ENTITIES, rows)
    effects = np.random.default_rng(918).choice([-2.5, 2.5], ENTITIES)
    values = rng.uniform(-3, 3, rows)
    probabilities = 1 / (1 + np.exp(-(2 * values + effects[identities])))
    labels = rng.random(rows) < probabilities
    for x, identity, label in zip(values, identities, labels, strict=True):
        yield {"x": float(x), "entity": f"{'new' if novel else 'id'}-{identity}", "label": "yes" if label else "no"}


def build(unavailable: float) -> rf.Model:
    class PredictionModel(rf.Model):
        x = rf.Number
        entity = rf.Category(p_unavailable=unavailable)
        label = rf.Category(mask=True)

    model = PredictionModel(d_model=32, n_layers=1, n_heads=4, dropout=0.0)
    return model


def probability(model: rf.Model, rows: list[dict]) -> np.ndarray:
    predictions = pl.DataFrame(model.predict(pa.Table.from_pylist(rows)))["predictions"].to_list()
    content = [row["label"]["content"] for row in predictions]
    return np.asarray([item["probability"] if item["value"] == "yes" else 1 - item["probability"] for item in content])


def score(p: np.ndarray, rows: list[dict], novel: bool) -> dict[str, float]:
    x = np.asarray([row["x"] for row in rows])
    if novel:
        oracle = (1 / (1 + np.exp(-(2 * x - 2.5))) + 1 / (1 + np.exp(-(2 * x + 2.5)))) / 2
    else:
        effects = np.random.default_rng(918).choice([-2.5, 2.5], ENTITIES)
        identity = np.asarray([int(row["entity"].split("-")[1]) for row in rows])
        oracle = 1 / (1 + np.exp(-(2 * x + effects[identity])))
    y = np.asarray([row["label"] == "yes" for row in rows], dtype=float)
    clipped = p.clip(1e-7, 1 - 1e-7)
    confidence = np.maximum(p, 1 - p)
    return {
        "oracle_rmse": float(np.sqrt(np.mean((p - oracle) ** 2))),
        "constant_rmse": float(np.sqrt(np.mean((0.5 - oracle) ** 2))),
        "nll": float(np.mean(-y * np.log(clipped) - (1 - y) * np.log1p(-clipped))),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p > 0.5) == y)),
        "ambiguous_confidence": float(confidence[np.abs(x) < 0.5].mean()),
        "decisive_confidence": float(confidence[np.abs(x) > 2.5].mean()),
    }


@experiment("P061", seed=7611)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    metrics, checks = {}, {}
    for unavailable in (0.01, 0.2):
        lit.seed_everything(seed, workers=True)
        model = build(unavailable)
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
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True,
            num_sanity_val_steps=0,
        )
        model.eval()
        vocabulary = rf.Category.vocabulary(model, "/entity")
        counts = rf.Category.counts(model, "/entity")
        arm = {}
        for novel in (False, True):
            rows = list(records(rows=8192, seed=seed + 3, novel=novel))
            arm["novel" if novel else "known"] = score(probability(model, rows), rows, novel)
        rows = list(records(rows=2048, seed=seed + 4, novel=True))
        before = probability(model, rows)
        renamed = [{**row, "entity": "another-unseen-identity"} for row in rows]
        arm["renaming_drift"] = float(np.max(np.abs(before - probability(model, renamed))))
        checks[f"{unavailable}: evaluation preserves mapping and exposures"] = vocabulary == rf.Category.vocabulary(
            model, "/entity"
        ) and counts == rf.Category.counts(model, "/entity")
        checks[f"{unavailable}: unseen spelling is uninformative"] = arm["renaming_drift"] < 1e-5
        model.encode(pa.Table.from_pylist(rows), strata="train")
        arm["admission_mean_drift"] = float(np.mean(np.abs(before - probability(model, rows))))
        metrics[str(unavailable)] = arm
        if unavailable == 0.2:
            checks.update(
                {
                    "Known-ID probability RMSE below 0.30": arm["known"]["oracle_rmse"] < 0.30,
                    "Unknown-ID probability RMSE below 0.10": arm["novel"]["oracle_rmse"] < 0.10,
                    "Unknown-ID predictions beat the constant baseline": arm["novel"]["oracle_rmse"]
                    < arm["novel"]["constant_rmse"],
                    "Ambiguous unknown IDs retain confidence below 0.70": arm["novel"]["ambiguous_confidence"] < 0.70,
                    "Decisive context retains confidence above 0.90": arm["novel"]["decisive_confidence"] > 0.90,
                }
            )
    return metrics, checks


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    import matplotlib.pyplot as plt
    import numpy as np

    from evidence.diagnostics.plots import axes_pair, caption, dots, finite, summary

    fig, axes = axes_pair()
    candidates = []
    for frame in bundle.frames:
        if frame["function"] != "score":
            continue
        values = frame["values"]
        p = bundle.value(values.get("p", values.get("predicted")))
        oracle = bundle.value(values.get("oracle", values.get("expected")))
        if p is None or oracle is None or isinstance(p, dict) or isinstance(oracle, dict):
            continue
        p, oracle = (np.asarray(p), np.asarray(oracle))
        if p.shape == oracle.shape and p.ndim == 1 and (len(p) > 1):
            candidates.append((frame, finite(p), finite(oracle)))
    if not candidates:
        plt.close(fig)
        return summary(bundle, spec)
    frame, p, oracle = candidates[0]
    density = axes[0].hexbin(
        oracle, p, gridsize=25, extent=(0, 1, 0, 1), mincnt=1, bins="log", cmap="viridis", rasterized=True
    )
    fig.colorbar(density, ax=axes[0], fraction=0.035, label="Records (log count)")
    axes[0].plot([0, 1], [0, 1], ls="--", color="gray")
    axes[0].set(
        title=caption(bundle.label(frame)),
        xlabel="Generator's conditional probability",
        ylabel="Model positive probability",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].set_aspect("equal")
    selected = candidates[:6]
    labels = [bundle.label(frame) for frame, _, _ in selected]
    if len(selected) != 4:
        raise ValueError("Cold-start calibration requires both identity groups at both training rates")
    labels = [
        f"{rate:.0%} unavailable training inputs · {group} identities"
        for rate in [0.01, 0.2]
        for group in ["known", "unseen"]
    ]
    axes[0].set_title(caption(labels[0]))
    errors = [float(np.sqrt(np.mean((p - oracle) ** 2))) for _, p, oracle in selected]
    dots(axes[1], labels, {"Probability RMSE": errors}, xlabel="RMSE against the known conditional probability")
    for frame, p, oracle in selected:
        result = bundle.value(frame["result"])
        key = "oracle_rmse" if "oracle_rmse" in result else "probability_rmse"
        if key in result:
            np.testing.assert_allclose(np.sqrt(np.mean((p - oracle) ** 2)), result[key], rtol=1e-10, atol=1e-12)
    return (
        fig,
        "The diagonal compares probabilities with an evaluator oracle, not with a single random label. Probability RMSE is recomputed from captured arrays and checked against the proof's own score.",
        {},
    )


if __name__ == "__main__":
    report(run)
