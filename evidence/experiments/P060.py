# %% [markdown]
# ---
# title: Can it learn a 90% probability without a useful input?
# description: Generate yes answers 90% of the time, independently of the input. The right probability is therefore
#   90%.
# evidence-readout:
# - label: True yes probability
#   metric:
#   - prior
#   format: percent
# - label: Predicted, known inputs
#   metric:
#   - capacity_2_known
#   - mean_probability
#   format: percent
# - label: Predicted, new inputs
#   metric:
#   - capacity_2_novel
#   - mean_probability
#   format: percent
# evidence-group: Calibration
# evidence-review:
#   question: Can relflow learn a 90% probability without a useful input?
#   modeling: Generate yes/no answers independently of the input, with yes occurring 90% of the time. Train the model
#     to predict the answer distribution. Compare its average predicted yes probability on familiar and unseen input
#     IDs with the generator’s recorded probability.
#   comparison:
#     kind: Recorded conditions
#     metric: Probability
#     conditions:
#     - label: True yes probability
#       path:
#       - prior
#       format: percent
#     - label: Predicted, known inputs
#       path:
#       - capacity_2_known
#       - mean_probability
#       format: percent
#     - label: Predicted, new inputs
#       path:
#       - capacity_2_novel
#       - mean_probability
#       format: percent
#     definition: Predicted probability of the named outcome or membership. The task determines the appropriate value;
#       higher is not always better.
#     direction: Task-specific interpretation
#     domain:
#     - 0
#     - 1
#   note: This chart shows probabilities, not accuracy. The full criteria also measure per-record probability error
#     and expected log loss.
#   criteria:
#     prefixes:
#       capacity_2: Two-row output allocation
#       capacity_128: 128-row output allocation
#       capacity_2_known: Two-row allocation · familiar inputs
#       capacity_2_novel: Two-row allocation · unseen inputs
#       capacity_128_known: 128-row allocation · familiar inputs
#       capacity_128_novel: 128-row allocation · unseen inputs
# evidence-plot:
#   target: label
#   focus: Does unused capacity change a learned class prior?
#   comparison: Compare known and novel-label probabilities at both declared capacities with the generating prior.
#   limit: A prior-probability comparison is not an empirical reliability curve.
#   renderer: plot
# evidence-cost:
#   baseline: 36
#   candidate: 36
# ---
#
# This task has a useful probability but no useful feature for separating individual outcomes. Every
# row has the same numeric input, and identity is independent of the sampled yes/no answer. The
# appropriate prediction is therefore the same 90% yes probability across observations, including
# observations carrying identities that were never encountered in training.
#
# ## Example
#
# ```yaml
# x: 0.0
# entity: id-12
# label: yes
# ```
#
# yes or no is a sampled hidden answer. A single answer does not determine the true probability.
#
#
# The illustrated `yes` is one possible draw, not a deterministic answer implied by `x = 0` or
# `id-12`. In an illustrative group of many such records, about nine in ten outcomes would be yes.
# Always choosing yes can consequently achieve high classification accuracy while a predicted
# probability of 100% would be unjustified and heavily penalized whenever no occurs.
#
# ## Comparison
#
# Compare known and new input identities, using two different internal output allocations.
#
# The two storage-allocation arms keep the data distribution and training setup fixed. They test
# whether extra unused output space changes the probabilities assigned to the populated labels.
# Known-identity and novel-identity evaluations retain the same uninformative numeric context; the
# unknown names should not create a different population prior.
#
# ## Result
#
# The latest model learns probabilities near the population rate. Accuracy alone would not reveal whether its confidence was appropriate; this does not establish general probability calibration.
#
# Probability RMSE compares each written yes probability with 0.9. Excess expected negative log
# likelihood compares the same probabilities with the ideal 90/10 forecast, accounting for both
# possible outcomes rather than rewarding a lucky sampled label. Agreement on this constant-prior task
# is a narrow calibration check; it does not establish calibration when probabilities depend on
# features or rare outcomes.
#
# ## Method
#
# ### Training and evaluation
#
# The evidence forces internal output allocations of 2 and 128 as experimental
# controls; users do not configure storage. Both use the frozen Evidence control policy, identical
# 32,768-row training streams, and 400 updates. Validation has its own seed.
# Each held-out evaluation has 8,192
# rows, with known or disjoint input IDs. Compare written probabilities with
# the true 0.9 prior using RMSE and expected log loss, not sample accuracy.
# The constant 0.9 oracle and the analytic weighted-loss optimum are controls.
# Gates are provisional and declared before the acceptance seed panel.
#
# ### Remaining work
#
# A ten-seed calibration panel and feature-dependent imbalanced populations
# remain necessary. This two-label evidence does not establish rare-label recall
# or calibration of other tensorfield losses.
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
# capability = "reliability"
# types = ["Category"]
# conditions = ["unseen-identities", "limited-information"]
# ///

"""P060: recover an imbalanced population prior with uninformative inputs."""

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
from policies import CONTROL
from relflow.tensorfields.shared.resize import Resize
from reporting import experiment, report

BUDGET = 400


def records(*, rows: int, seed: int, novel: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for entity, positive in zip(rng.integers(0, 4096, rows), rng.random(rows) < 0.9, strict=True):
        yield {"x": 0.0, "entity": f"{'new' if novel else 'id'}-{entity}", "label": "yes" if positive else "no"}


class Model(rf.Model):
    x = rf.Number
    entity = rf.Category()
    label = rf.Category(predict=True)


def build(size: int) -> rf.Model:

    model = Model(preset=CONTROL.name)
    # Experimental storage control only: allocation is not a schema option.
    node = model.nodes["/label"]
    resize = Resize()
    resize.embedding(node.embedder.embeddings["content"], size)
    resize.linear(node.decoder.linears["content"], size)
    node.embedder.counters["content"].resize(size, resize)
    resize.attribute(node.embedder.vocab, "size", size)
    resize.commit()
    return model


@experiment("P060", seed=7601)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    oracle_nll = float(-0.9 * np.log(0.9) - 0.1 * np.log(0.1))
    metrics = {"prior": 0.9, "weighted_optimum": 0.75, "oracle_nll": oracle_nll}
    checks = {}
    for size in (2, 128):
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
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True,
            num_sanity_val_steps=0,
        )
        model.eval()
        counts = rf.Category.counts(model, "/label")
        vocabulary = rf.Category.vocabulary(model, "/entity")
        for novel in (False, True):
            rows = list(records(rows=8192, seed=seed + 3, novel=novel))
            predictions = pl.DataFrame(model.predict(pa.Table.from_pylist(rows)))["predictions"].to_list()
            values = [row["label"]["content"] for row in predictions]
            p = np.asarray(
                [item["probability"] if item["value"] == "yes" else 1 - item["probability"] for item in values]
            )
            p = p.clip(1e-7, 1 - 1e-7)
            key = f"capacity_{size}_{'novel' if novel else 'known'}"
            rmse = float(np.sqrt(np.mean((p - 0.9) ** 2)))
            excess = float(np.mean(-0.9 * np.log(p) - 0.1 * np.log1p(-p)) - oracle_nll)
            metrics[key] = {"mean_probability": float(p.mean()), "prior_rmse": rmse, "excess_nll": excess}
            checks[f"{key}: probability RMSE below 0.06"] = rmse < 0.06
            checks[f"{key}: excess expected NLL below 0.02"] = excess < 0.02
        checks[f"capacity_{size}: evaluation preserves vocabulary and counts"] = vocabulary == rf.Category.vocabulary(
            model, "/entity"
        ) and counts == rf.Category.counts(model, "/label")
    return metrics, checks


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    from evidence.diagnostics.plots import axes_pair, dots

    m = bundle.record["metrics"]
    fig, axes = axes_pair()
    labels = ["Capacity 2, known", "Capacity 2, novel", "Capacity 128, known", "Capacity 128, novel"]
    keys = ["capacity_2_known", "capacity_2_novel", "capacity_128_known", "capacity_128_novel"]
    dots(
        axes[0],
        labels,
        {"Mean predicted probability": [m[key]["mean_probability"] for key in keys]},
        xlabel="Probability of yes",
        reference=(m["prior"], "Generating prior"),
    )
    axes[0].set_xlim(0, 1.02)
    dots(
        axes[1],
        labels,
        {"Excess NLL": [m[key]["excess_nll"] for key in keys]},
        xlabel="NLL minus the generating-prior NLL",
    )
    return (
        fig,
        "The reference is the generator's fixed prior. Unused vocabulary capacity and an unavailable label are separate conditions; these mean readouts are not a reliability curve.",
        {},
    )


if __name__ == "__main__":
    report(run)
