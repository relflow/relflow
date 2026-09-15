# %% [markdown]
# ---
# title: Transfer Values by an Unseen Identity
# categories:
# - Sibling entity transfer
# proof-id: P037
# description: Route random source values to sibling targets using fresh Hash identities.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# Fresh identifiers link two independently ordered branches. The model must
# retrieve each target's matching source value without relying on a persistent
# vocabulary, a population average, or shared position.
#
# {{< proof P037 status >}}
#
# ## Insights
#
# **Fresh identities can connect source values to targets in another branch
# without a learned identity vocabulary.** Shared Hash encoding preserves
# within-batch equality. Coordinate-local encoding binds each source key to its
# value, and visible target keys can condition repeated queries over source
# context carried through the root.
#
# Independent branch orders remove reliable positional copying. Rotating only
# target keys while retaining target values breaks accuracy, showing that the
# association matters. Retaining tokens also outperforms the tested compressed
# route, unlike the small persistent-Category control. This is evidence for two
# entities under one protocol, not an exact join or unlimited memory. Missing
# sources, duplicate identities, and collisions require explicit conventions
# and further tests.
#
# ## Setup

# %%
"""Transfer values between sibling branches using fresh Hash identities.

Every split uses unseen names and independent source/target order. Compare
compressed and preserved routes, then rotate query keys while retaining targets
to test whether improved transfer really depends on identity.

Run this file with --help for seed, training-budget, and reporting options.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P037"
PAIR_COUNT = 2

# %% [markdown]
# ## Examples
#
# ### Match identities that appear in both branches
#
# ```yaml
# source:
#   - {entity_id: row-42-K1, value: 0.7}
#   - {entity_id: row-42-K2, value: -0.2}
# target:
#   - {entity_id: row-42-K2, value: -0.2}
#   - {entity_id: row-42-K1, value: 0.7}
# ```
#
# Target values are supervision hidden by `mask=True`. Every row has new
# identities, and train, validation, and test use disjoint key namespaces.
# Target identities and source records remain visible.
#
# ### A new row brings unseen keys
#
# ```yaml
# source:
#   - {entity_id: row-99-P, value: 0.1}
#   - {entity_id: row-99-Q, value: -0.9}
# target:
#   - {entity_id: row-99-Q, value: -0.9}
#   - {entity_id: row-99-P, value: 0.1}
# ```
#
# The identifiers and source values are new, but the equality relationship is
# the same. Correct hidden targets follow their matching source values without
# requiring either identifier to belong to a persistent learned vocabulary.
#
# ### Keep target values but rotate their keys
#
# ```yaml
# source:
#   - {entity_id: row-42-K1, value: 0.7}
#   - {entity_id: row-42-K2, value: -0.2}
# target:
#   - {entity_id: row-42-K1, value: -0.2}
#   - {entity_id: row-42-K2, value: 0.7}
# ```
#
# This intervention starts from the first example. The target labels deliberately
# retain the original values even though the keys now select the other source.
# A rise in error against those retained labels tests dependence on identity.
#
# ## Synthetic data and controls


# %%
def records(*, rows: int, seed: int, namespace: str, broken_identity: bool = False) -> Iterator[dict]:
    """Generate observation-local key/value associations in sibling branches."""
    rng = np.random.default_rng(seed)
    for row in range(rows):
        keys = [f"{namespace}-{row:06d}-{pair:02d}" for pair in range(PAIR_COUNT)]
        values = rng.uniform(-1.0, 1.0, size=PAIR_COUNT)
        query_keys = list(keys)
        if broken_identity:
            query_keys = query_keys[1:] + query_keys[:1]
        source_order = rng.permutation(PAIR_COUNT)
        query_order = rng.permutation(PAIR_COUNT)
        source = [{"entity_id": keys[index], "value": float(values[index])} for index in source_order]
        target = [{"entity_id": query_keys[index], "value": float(values[index])} for index in query_order]
        yield {"source": source, "target": target}


def normalized_rmse(model: rf.Model, records: list[dict]) -> float:
    actual = np.asarray([item["value"] for row in records for item in row["target"]])
    output = model.predict(records).to_pylist()
    predicted = np.asarray(
        [coordinate["content"] for row in output for coordinate in row["predictions"]["association/target/value"]]
    )
    return float(np.sqrt(np.mean(np.square(actual - predicted)) / np.mean(np.square(actual))))


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-hash-identity
# //| fig-cap: "Two separately trained models share this schema. Retained and compressed labels describe alternative reductions at the root and both branches; target values are masked."
# //| fig-alt: "Association has two source and two target records using Hash identities. Source values are visible and target values are masked. At the root and each branch, the retained model keeps all tokens and the compressed model uses one attention summary."
# #tree(node("association", kind: "root", width: 150pt, body: [
#   - *Retained:* Keep all tokens
#   - *Compressed:* Attention · 1 summary
# ], children: (
#   node("source", kind: "branch", repeated: true, width: 150pt, body: [
#     - *Capacity:* 2 items
#     - *Retained:* Keep all tokens
#     - *Compressed:* Attention · 1 summary
#   ], children: (
#     node("entity_id", type: "Hash"),
#     node("value", type: "Number"),
#   )),
#   node("target", kind: "branch", repeated: true, width: 150pt, body: [
#     - *Capacity:* 2 items
#     - *Retained:* Keep all tokens
#     - *Compressed:* Attention · 1 summary
#   ], children: (
#     node("entity_id", type: "Hash"),
#     node("value", kind: "target", type: "Number", width: 150pt, body: [
#       - *Input:* always hidden
#     ]),
#   )),
# )))
# ```
#
# The retained route uses `reduction=None` at both branches and root. A matched
# control replaces those reductions with `rf.Attention()`.
#
# ## How it works
#
# Hash preserves equality within an encoded batch without a learned vocabulary.
# Coordinate-local mixing binds source identity to value, and retained source
# context remains available to queries conditioned on visible target identities.
# The key-rotation control leaves target values and source records unchanged
# while breaking the intended match.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    test = list(records(rows=1024, seed=seed + 3, namespace="test"))
    broken = list(records(rows=1024, seed=seed + 3, namespace="test", broken_identity=True))
    metrics = {}
    # Fit both routes from the same seed and data, changing only reduction.
    for route, reduction in (("compressed", rf.Attention()), ("preserved", None)):
        lit.seed_everything(seed, workers=True)
        model = rf.Model(
            name="association",
            d_model=64,
            n_layers=2,
            n_heads=4,
            reduction=reduction,
            batch_size=128,
            optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
            source=rf.Branch(
                length=PAIR_COUNT,
                n_layers=2,
                reduction=reduction,
                entity_id=rf.Hash(n_hashes=4, n_bands=8),
                value=rf.Number,
            ),
            target=rf.Branch(
                length=PAIR_COUNT,
                n_layers=2,
                reduction=reduction,
                entity_id=rf.Hash(n_hashes=4, n_bands=8),
                value=rf.Number(mask=True, objective="mse"),
            ),
        )
        data = rf.SyntheticDataModule(
            model=model,
            train=partial(records, rows=2048, seed=seed + 1, namespace="train"),
            validate=partial(records, rows=512, seed=seed + 2, namespace="validate"),
            seed=seed,
        )
        trainer = lit.Trainer(
            accelerator=accelerator,
            max_steps=600 if steps is None else min(steps, 600),
            max_epochs=-1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            deterministic=True,
            check_val_every_n_epoch=5,
        )
        trainer.fit(model, datamodule=data)
        metrics[f"{route}_nrmse"] = normalized_rmse(model, test)
        metrics[f"{route}_broken_nrmse"] = normalized_rmse(model, broken)
    compressed = metrics["compressed_nrmse"]
    intact = metrics["preserved_nrmse"]
    broken_error = metrics["preserved_broken_nrmse"]
    checks = {
        "Finite errors": bool(np.isfinite(list(metrics.values())).all()),
        "Preserved transfer calibration nRMSE <= 1.10": intact <= 1.10,
        "Preserved transfer nRMSE <= 0.25": intact <= 0.25,
        "Broken identities nRMSE >= 0.80": broken_error >= 0.80,
        "Identity corruption increases nRMSE by >= 0.50": broken_error - intact >= 0.50,
        "Preserving Hash tokens improves nRMSE by >= 0.15": intact <= compressed - 0.15,
    }
    return metrics, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P037 evidence >}}
#
# ## Remaining work
#
# Repeat three core seeds and ten calibration seeds, then sweep 2, 4, 8, and 16
# entities. Add explicit permutation/equivariance tests, absent sources, duplicate
# keys, padding, and Hash collisions. The [Category control](category-identity.html)
# uses recurring identities and succeeds under the tested compression. Exact
# production retrieval can use a preprocessing join.
#
# ## Reproduce
#
# {{< proof P037 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=17)
