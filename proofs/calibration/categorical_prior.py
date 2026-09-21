# %% [markdown]
# ---
# title: Categorical Prior Probabilities
# categories: [Calibration]
# proof-id: P060
# description: Uninformative inputs should recover class prevalence rather than a class-weighted decision distribution.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P060 status >}}
#
# ## Insights
#
# Accuracy alone cannot tell a 90%-confident majority prediction from a
# 75%-confident one. This controlled task has no useful input signal, so its
# probability target is known exactly. Inverse-square-root frequency weighting
# would instead favor a 75/25 distribution at a 90/10 population prevalence.
#
# This tests a probability objective, not universal calibration or OOD detection.
# Unused output capacity must not redefine the probabilities written over the
# two populated labels. Unknown input identities do not change the population.

# %%
"""P060: recover an imbalanced population prior with uninformative inputs."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
from reporting import report

import relflow as rf
from relflow.helpers.resize import Resize

PROOF_ID = "P060"
BUDGET = 400

# %% [markdown]
# ## Examples
#
# ```yaml
# x: 0.0
# entity: id-12
# label: yes
# ```
#
# ```yaml
# x: 0.0
# entity: id-12
# label: no
# ```
#
# Both answers occur for the same visible context; yes occurs with probability
# 0.9. Entity IDs are independent distractors, not semantic descriptions.
#
# ```yaml
# x: 0.0
# entity: new-12
# label: yes
# ```
#
# Entity-disjoint evaluation uses the same prior. The output labels remain
# familiar; this is not an unknown-output detection task.
#
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-categorical-prior
# //| fig-cap: "Constant numerical context and an independent identity cannot identify an individual outcome."
# //| fig-alt: "A root contains constant Number x, independent Category entity, and hidden Category label."
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number", detail: "Constant"),
#   node("entity", type: "Category", detail: "Independent identity"),
#   node("label", kind: "target", type: "Category", detail: "90/10 prevalence"),
# )))
# ```


# %%
def records(*, rows: int, seed: int, novel: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for entity, positive in zip(rng.integers(0, 4096, rows), rng.random(rows) < 0.9, strict=True):
        yield {"x": 0.0, "entity": f"{'new' if novel else 'id'}-{entity}", "label": "yes" if positive else "no"}


def build(size: int) -> rf.Model:
    model = rf.Model.xs(
        batch_size=128,
        x=rf.Number,
        entity=rf.Category(),
        label=rf.Category(mask=True),
    )
    # Experimental storage control only: allocation is not a schema option.
    node = model.nodes["/label"]
    resize = Resize()
    resize.embedding(node.embedder.embeddings["content"], size)
    resize.linear(node.decoder.linears["content"], size)
    node.embedder.counters["content"].resize(size, resize)
    resize.attribute(node.embedder.vocab, "size", size)
    resize.commit()
    model.optimizer = rf.adamw(learning_rate=0.002)
    return model


# %% [markdown]
# ## Training and evaluation
#
# The proof forces internal output allocations of 2 and 128 as experimental
# controls; users do not configure storage. Both use the `xs` preset, identical
# 32,768-row training streams, and 400 updates. Validation has its own seed.
# Each held-out evaluation has 8,192
# rows, with known or disjoint input IDs. Compare written probabilities with
# the true 0.9 prior using RMSE and expected log loss, not sample accuracy.
# The constant 0.9 oracle and the analytic weighted-loss optimum are controls.
# Gates are provisional and declared before the acceptance seed panel.


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    oracle_nll = float(-0.9 * np.log(0.9) - 0.1 * np.log(0.1))
    metrics = {"prior": 0.9, "weighted_optimum": 0.75, "oracle_nll": oracle_nll}
    checks = {}
    for size in (2, 128):
        lit.seed_everything(seed, workers=True)
        model = build(size)
        data = rf.SyntheticDataModule(
            model=model,
            train=partial(records, rows=32768, seed=seed + 1),
            validate=partial(records, rows=1024, seed=seed + 2),
            seed=seed,
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
        trainer.fit(model, datamodule=data)
        model.eval()
        counts = rf.Category.counts(model, "/label")
        vocabulary = rf.Category.vocabulary(model, "/entity")
        for novel in (False, True):
            rows = list(records(rows=8192, seed=seed + 3, novel=novel))
            predictions = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
            values = [row["/label"]["content"] for row in predictions]
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


# %% [markdown]
# ## Evidence
#
# {{< proof P060 evidence >}}
#
# ## Remaining work
#
# A ten-seed calibration panel and feature-dependent imbalanced populations
# remain necessary. This two-label proof does not establish rare-label recall
# or calibration of other tensorfield losses.
#
# ## Reproduce
#
# {{< proof P060 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=7601)
