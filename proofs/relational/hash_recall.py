# %% [markdown]
# ---
# title: Recall by an Unseen Key
# categories:
# - Associative recall
# proof-id: P025
# description: Retrieve a visible source value for a shuffled query sharing an unseen
#   Hash identity.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# A query must recover the source value with its matching identity from one
# shuffled memory branch. Every row has fresh values and identifiers; fixed
# positions and persistent vocabulary memorization cannot determine the answer.
#
# {{< proof P025 status >}}
#
# ## Insights
#
# **The model recalls values for unseen keys, and breaking the key relationship
# removes that skill.** Fresh values and independently shuffled records prevent
# fixed positions or remembered answers from reliably solving the task. Hash
# preserves equality without a learned vocabulary; the Boolean query mask hides
# only query values, leaving source values visible.
#
# The decoder can use both retained ancestor context and the value field's own
# visible source parcel. Aligned identity and role fields condition each repeated
# query. Rotating query keys while retaining original targets tests whether that
# relationship matters. This supports learned recall for two pairs, not a
# deterministic lookup guarantee. Larger memories, absent or duplicate keys,
# and collisions remain untested.
#
# ## Setup

# %%
"""Retrieve unseen keys from source/query records shuffled into one branch.

Only query values are masked. Independent order and unseen identities prevent
positional copying and vocabulary memorization. Rotating query keys while
retaining targets tests whether successful recall actually uses identity.

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

PROOF_ID = "P025"
PAIR_COUNT = 2

# %% [markdown]
# ## Examples
#
# ### Shuffled queries find their sources
#
# ```yaml
# memory:
#   - {role: source, is_query: false, entity_id: K1, value: 0.4}
#   - {role: query, is_query: true, entity_id: K2, value: -0.8}
#   - {role: source, is_query: false, entity_id: K2, value: -0.8}
#   - {role: query, is_query: true, entity_id: K1, value: 0.4}
# ```
#
# Query values are supervision hidden by `is_query`. Source values remain
# visible, and train, validation, and test identities use disjoint namespaces.
#
# ### New values change the answers
#
# ```yaml
# memory:
#   - {role: query, is_query: true, entity_id: K2, value: -0.3}
#   - {role: source, is_query: false, entity_id: K1, value: 0.9}
#   - {role: query, is_query: true, entity_id: K1, value: 0.9}
#   - {role: source, is_query: false, entity_id: K2, value: -0.3}
# ```
#
# Queries are interleaved in another order, and source values have changed.
# The correct hidden targets follow the matching source values, not a remembered
# number for K1 or K2.
#
# ### Rotate only query identities
#
# ```yaml
# memory:
#   - {role: source, is_query: false, entity_id: K1, value: 0.4}
#   - {role: query, is_query: true, entity_id: K1, value: -0.8}
#   - {role: source, is_query: false, entity_id: K2, value: -0.8}
#   - {role: query, is_query: true, entity_id: K2, value: 0.4}
# ```
#
# The control keeps the first record's query targets but swaps the query keys.
# Those retained targets are deliberately no longer the values of their visible
# keys. A model using identity should lose accuracy against them.
#
# ## Synthetic data and controls


# %%
def records(*, rows: int, seed: int, namespace: str, broken_identity: bool = False) -> Iterator[dict]:
    """Generate unseen, observation-local key/value associations."""
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
        memory = [{**item, "role": "source", "is_query": False} for item in source]
        memory.extend(({**item, "role": "query", "is_query": True} for item in target))
        rng.shuffle(memory)
        yield {"memory": memory}


def normalized_rmse(model: rf.Model, records: list[dict]) -> float:
    """Score only query coordinates; source values remain visible to the model."""
    output = model.predict(records).to_pylist()
    actual, predicted = [], []
    for row, result in zip(records, output, strict=True):
        for item, coordinate in zip(row["memory"], result["predictions"]["association/memory/value"], strict=True):
            if item["is_query"]:
                actual.append(item["value"])
                predicted.append(coordinate["content"])
    actual = np.asarray(actual)
    return float(np.sqrt(np.mean(np.square(np.asarray(predicted) - actual)) / np.mean(np.square(actual))))


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-hash-recall
# //| fig-cap: "The root and shuffled memory branch keep all tokens. The Boolean mask hides query values while leaving source values visible."
# //| fig-alt: "Association contains four source/query memory records with Hash identity, Category role, Boolean is-query, and Number value. The value mask reads is_query, preserving source values and hiding query targets. Root and branch keep all tokens."
# #tree(node("association", kind: "root", width: 150pt, body: [
#   - *Reduction:* Keep all tokens
# ], children: (
#   node("memory", kind: "branch", repeated: true, width: 150pt, body: [
#     - *Capacity:* 4 items
#     - *Reduction:* Keep all tokens
#   ], children: (
#     node("entity_id", type: "Hash"),
#     node("role", type: "Category"),
#     node("is_query", type: "Boolean"),
#     node("value", type: "Number", width: 150pt, body: [
#       - *Mask query:* `is_query`
#       - *Sources:* Visible input
#       - *Queries:* Hidden targets
#     ]),
#   )),
# )))
# ```
#
# Both branch and root use `reduction=None`. The value field uses
# `rf.Mask(query="is_query", dropout=False, reconstruct=True)`.
#
# ## How it works
#
# Coordinate-local encoding binds a source identity to its value. Visible fields
# at each query coordinate condition its decoder query over retained memory.
# Source and query orders are independently shuffled. Rotating only query keys
# while retaining target values should destroy the original association.
# The [aligned control](aligned-position-control.html) shows what happens when
# position alone supplies an alternative answer path.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        name="association",
        d_model=64,
        n_layers=2,
        n_heads=4,
        reduction=None,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        memory=rf.Branch(
            length=2 * PAIR_COUNT,
            n_layers=2,
            reduction=None,
            entity_id=rf.Hash(n_hashes=4, n_bands=8),
            role=rf.Category(size=2, p_unavailable=0.0),
            is_query=rf.Boolean,
            value=rf.Number(mask=rf.Mask(query="is_query", dropout=False, reconstruct=True), objective="mse"),
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
        max_steps=800 if steps is None else min(steps, 800),
        max_epochs=-1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        check_val_every_n_epoch=5,
    )
    trainer.fit(model, datamodule=data)
    test = list(records(rows=1024, seed=seed + 3, namespace="test"))
    broken = list(records(rows=1024, seed=seed + 3, namespace="test", broken_identity=True))
    intact_nrmse = normalized_rmse(model, test)
    broken_nrmse = normalized_rmse(model, broken)
    metrics = {
        "intact_nrmse": intact_nrmse,
        "broken_nrmse": broken_nrmse,
        "corruption_gap": broken_nrmse - intact_nrmse,
    }
    checks = {
        "Finite errors": bool(np.isfinite([intact_nrmse, broken_nrmse]).all()),
        "Recall calibration nRMSE <= 0.90": intact_nrmse <= 0.90,
        "Identity recall nRMSE <= 0.25": intact_nrmse <= 0.25,
        "Broken identities nRMSE >= 0.80": broken_nrmse >= 0.80,
        "Corruption increases nRMSE by >= 0.50": broken_nrmse - intact_nrmse >= 0.50,
    }
    return metrics, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P025 evidence >}}
#
# ## Remaining work
#
# Calibrate across three core seeds and ten lightweight seeds. Test complete
# record permutations, simultaneous key renaming, larger memories, absent or
# duplicate keys, padding, and Hash collisions. Use an explicit application
# lookup when the mapping must be exact.
#
# ## Reproduce
#
# {{< proof P025 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=17)
