# %% [markdown]
# ---
# title: Can it learn a 90% probability without a useful input?
# categories:
# - Calibration
# description: Generate yes answers 90% of the time, independently of the input. The right probability is therefore
#   90%.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
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
# ```{typst}
# //| label: fig-evidence-categorical-prior
# //| fig-cap: "Amber cards are hidden prediction targets. The answer is sampled; its population probability is 90%."
# //| fig-alt: "Model tree with record, x, entity, label. Amber cards are hidden prediction targets. The answer is sampled; its population probability is 90%."
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number", detail: "Constant"),
#   node("entity", type: "Category", detail: "Independent identity"),
#   node("label", kind: "target", type: "Category", detail: "90/10 prevalence"),
# )))
# ```
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
# {{< evidence P060 status >}}
#
# The latest model learns probabilities near the population rate. Accuracy alone would not reveal whether its confidence was appropriate; this does not establish general probability calibration.
#
# Probability RMSE compares each written yes probability with 0.9. Excess expected negative log
# likelihood compares the same probabilities with the ideal 90/10 forecast, accounting for both
# possible outcomes rather than rewarding a lucky sampled label. Agreement on this constant-prior task
# is a narrow calibration check; it does not establish calibration when probabilities depend on
# features or rare outcomes.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P060 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P060 script >}}
#
# ### Training and evaluation
#
# The evidence forces internal output allocations of 2 and 128 as experimental
# controls; users do not configure storage. Both use the `xs` preset, identical
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
# ### Complete experiment code
#

# %%
"""P060: recover an imbalanced population prior with uninformative inputs."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
from relflow.presets.base import resolve
from relflow.tensorfields.shared.resize import Resize
from reporting import experiment, report

BUDGET = 400


def records(*, rows: int, seed: int, novel: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for entity, positive in zip(rng.integers(0, 4096, rows), rng.random(rows) < 0.9, strict=True):
        yield {"x": 0.0, "entity": f"{'new' if novel else 'id'}-{entity}", "label": "yes" if positive else "no"}


def build(size: int) -> rf.Model:
    # Freeze the experimental architecture independently of SDK preset changes.
    control_preset = rf.presets.Preset(
        name="xs",
        d_model=64,
        root=rf.presets.BranchDefaults(n_layers=1, n_heads=2, reduction=rf.Attention()),
        branch=rf.presets.BranchDefaults(n_layers=1, n_heads=2, reduction=rf.Attention()),
        leaf=rf.presets.LeafDefaults(n_heads=2),
    )
    model = rf.Model(
        schema=resolve(
            control_preset,
            x=rf.Number,
            entity=rf.Category(),
            label=rf.Category(mask=True),
        )
    )
    model.preset = control_preset
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


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
