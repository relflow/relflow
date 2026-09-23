# %% [markdown]
# ---
# title: Copying by Position
# categories:
# - Associative recall
# proof-id: P024
# description: An aligned copying control shows why correct answers alone do not demonstrate
#   identity lookup.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# Matching source and query positions lets a model copy values without using
# identity. This deliberately easy control checks that positional copying can
# survive even when query keys become wrong.
#
# {{< proof P024 status >}}
#
# ## Insights
#
# **A model can copy the right value using position while ignoring matching identities.** Aligned source and
# query positions give the
# model another way to recover every target. Rotating query keys while retaining
# target positions leaves accuracy high, so this control does not establish
# identity-based recall.
#
# Repeated decoder queries retain positional information. Although branch and
# root reductions are compressed, decoder memory also includes the target
# value field's own parcel, which still contains visible source values. Success
# therefore does not show that one summary preserves all source information.
# Keep this control beside shuffled-key recall: their different responses to
# broken keys distinguish copying by position from using identity.
#
# ## Setup

# %%
"""Show why aligned copying is insufficient evidence of identity lookup.

The source and query halves have matching positions. A compressed model can
copy values by position even after query keys are changed. Successful copying
on both versions is the expected control behavior.

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

PROOF_ID = "P024"
PAIR_COUNT = 2

# %% [markdown]
# ## Examples
#
# ### Aligned source and query halves
#
# ```yaml
# memory:
#   - {role: source, is_query: false, entity_id: K1, value: 0.4}
#   - {role: source, is_query: false, entity_id: K2, value: -0.8}
#   - {role: query, is_query: true, entity_id: K1, value: 0.4}
#   - {role: query, is_query: true, entity_id: K2, value: -0.8}
# ```
#
# The final two values are supervision hidden by the Boolean mask. The query
# half uses the same relative order as the source half.
#
# ### Reorder both halves together
#
# ```yaml
# memory:
#   - {role: source, is_query: false, entity_id: K2, value: -0.8}
#   - {role: source, is_query: false, entity_id: K1, value: 0.4}
#   - {role: query, is_query: true, entity_id: K2, value: -0.8}
#   - {role: query, is_query: true, entity_id: K1, value: 0.4}
# ```
#
# Both halves now put K2 first. The correct query targets reverse with them, but
# copying by relative position still solves the example without comparing keys.
#
# ### Change keys without changing target positions
#
# ```yaml
# memory:
#   - {role: source, is_query: false, entity_id: K1, value: 0.4}
#   - {role: source, is_query: false, entity_id: K2, value: -0.8}
#   - {role: query, is_query: true, entity_id: K2, value: 0.4}
#   - {role: query, is_query: true, entity_id: K1, value: -0.8}
# ```
#
# This control retains the first example's query values while rotating query
# keys. The targets intentionally disagree with identity lookup. A positional
# copier still matches them; the test requires that accuracy remain high.
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
        query_order = source_order
        source = [{"entity_id": keys[index], "value": float(values[index])} for index in source_order]
        target = [{"entity_id": query_keys[index], "value": float(values[index])} for index in query_order]
        memory = [{**item, "role": "source", "is_query": False} for item in source]
        memory.extend(({**item, "role": "query", "is_query": True} for item in target))
        yield {"memory": memory}


def normalized_rmse(model: rf.Model, records: list[dict]) -> float:
    """Score only query coordinates; source values remain visible to the model."""
    output = model.predict(records).to_pylist()
    actual, predicted = [], []
    for row, result in zip(records, output, strict=True):
        for item, coordinate in zip(row["memory"], result["predictions"]["/memory/value"], strict=True):
            if item["is_query"]:
                actual.append(item["value"])
                predicted.append(coordinate["content"])
    actual = np.asarray(actual)
    return float(np.sqrt(np.mean(np.square(np.asarray(predicted) - actual)) / np.mean(np.square(actual))))


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-aligned-position-control
# //| fig-cap: "Aligned source and query records use attention summaries; visible source-value memory also reaches the decoder. Only query values are masked."
# //| fig-alt: "Association contains four source/query memory records with Hash identity, Category role, Boolean is-query, and Number value. The value mask reads is_query, preserving source values and hiding query targets. Root and branch each use one attention summary."
# #tree(node("association", kind: "root", width: 150pt, body: [
#   - *Reduction:* `Attention`
#   - *Learned summaries:* 1
# ], children: (
#   node("memory", kind: "branch", repeated: true, width: 150pt, body: [
#     - *Capacity:* 4 items
#     - *Reduction:* `Attention`
#     - *Learned summaries:* 1
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
# The branch and root use `rf.Attention()`. The value field uses
# `rf.Mask(query="is_query", dropout=False, reconstruct=True)`; source values
# remain input while query values supply reconstruction targets.
#
# ## How it works
#
# The decoder has positional information, so it can learn to copy the first
# source to the first query and the second source to the second query. Rotating
# query identities leaves these target positions unchanged. A model following
# position should therefore remain accurate after this intervention.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=64,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(),
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        memory=rf.Branch(
            length=2 * PAIR_COUNT,
            n_layers=2,
            reduction=rf.Attention(),
            entity_id=rf.Hash(n_hashes=4, n_bands=8),
            role=rf.Category(p_unavailable=0.0),
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
        "Aligned copying nRMSE <= 0.25": intact_nrmse <= 0.25,
        "Broken identities retain copying nRMSE <= 0.25": broken_nrmse <= 0.25,
        "Identity change alters nRMSE by <= 0.10": abs(intact_nrmse - broken_nrmse) <= 0.10,
    }
    return metrics, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P024 evidence >}}
#
# ## Remaining work
#
# Repeat seed calibration and retain this control alongside identity proofs.
# Its successful predictions cannot establish unseen-key lookup: the required
# information is already available through a fixed position relationship.
#
# ## Reproduce
#
# {{< proof P024 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=17)
