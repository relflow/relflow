# %% [markdown]
# ---
# title: Is it copying by position instead of by ID?
# description: Place source values and questions in matching positions, then ask the model to copy the values.
# evidence-readout:
# - label: Matching IDs
#   metric:
#   - intact_nrmse
#   format: error
# - label: Changed IDs
#   metric:
#   - broken_nrmse
#   format: error
# evidence-group: Relational
# evidence-review:
#   question: Can matching positions make an apparent ID lookup misleading?
#   modeling: Place source values and their queries in corresponding positions, with matching IDs. Hide the copied
#     values. Then rotate query IDs without moving the queries or changing their expected answers. Compare error before
#     and after the ID change.
#   comparison:
#     kind: Controlled comparison
#     metric: Normalized RMSE
#     conditions:
#     - label: Changed IDs
#       path:
#       - broken_nrmse
#       format: number
#     - label: Matching IDs
#       path:
#       - intact_nrmse
#       format: number
#     definition: Root mean squared error divided by the reference scale defined in this experiment. Zero is exact;
#       one equals that reference scale.
#     direction: Lower error is better
#   note: The checks intentionally test a positional shortcut. Remaining accurate after IDs change is not evidence
#     of an ID-based lookup.
# evidence-plot:
#   mode: regression
#   target: memory/value
#   focus: Can aligned positions explain apparent identity retrieval?
#   comparison: Compare intact and identity-rotated queries while positions remain aligned.
#   limit: Position copying can solve this control; it does not establish use of identity keys.
# evidence-cost:
#   baseline: 100
#   candidate: 99
# ---
#
# This is a deliberately misleading lookup setup: the right value can be copied without using an ID
# at all. Two visible source entries are followed by two query entries in the same relative order.
# Only query values are masked and scored, so the source values remain available as evidence.
#
# ## Example
#
# ```yaml
# memory:
#   - {role: source, is_query: false, entity_id: K1, value: 0.4}
#   - {role: source, is_query: false, entity_id: K2, value: -0.8}
#   - {role: query, is_query: true, entity_id: K1, value: 0.4}
#   - {role: query, is_query: true, entity_id: K2, value: -0.8}
# ```
#
# Source values are visible. Query values are the hidden answers, selected by is_query.
#
# In the illustration, the first query asks for K1 and should return 0.4; the second asks for K2 and
# should return −0.8. But the same answers can be obtained by copying the first source into the
# first query and the second source into the second query. Correct predictions alone cannot tell
# those strategies apart.
#
# ## Comparison
#
# Change question IDs while leaving their positions and answers fixed. If accuracy survives,
# matching IDs were unnecessary.
#
# The damaged version exchanges the query IDs while retaining source order, query positions, and the
# original answers. Genuine lookup using the changed IDs would follow different source values and
# therefore worsen the old-target score. Positional copying can keep its low error. This control
# intentionally regards accurate predictions after changing IDs as evidence that identity was
# unnecessary for the task.
#
# ## Result
#
# Accuracy survives the changed IDs. Success on aligned records can therefore hide a positional
# shortcut.
#
# These errors are normalized against always predicting zero for query values. Small errors on both
# versions, with little gap, demonstrate that the apparent lookup task admits a shortcut. A passing
# result here is not a positive claim about associative recall; the independently shuffled lookup
# experiment removes this fixed source-to-query position relationship.
#
# ## Method
#
# ### How it works
#
# The decoder has positional information, so it can learn to copy the first
# source to the first query and the second source to the second query. Rotating
# query identities leaves these target positions unchanged. A model following
# position should therefore remain accurate after this intervention.
#
# ### Remaining work
#
# Repeat seed calibration and retain this control alongside identity evidences.
# Its successful predictions cannot establish unseen-key lookup: the required
# information is already available through a fixed position relationship.
#

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "lightning",
#     "numpy",
#     "polars",
#     "pyyaml>=6",
#     "relflow>=0.5.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "structure"
# types = []
# conditions = ["changed-associations"]
# ///

"""Show why aligned copying is insufficient evidence of identity lookup.

The source and query halves have matching positions. A compressed model can
copy values by position even after query keys are changed. Successful copying
on both versions is the expected control behavior.

Run this file with --help for seed, training-budget, and reporting options.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from functools import partial
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
import torch
from reporting import experiment, report
from visibility import reconstruct

PAIR_COUNT = 2


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
        memory.extend({**item, "role": "query", "is_query": True} for item in target)
        yield {"memory": memory}


def normalized_rmse(model: rf.Model, records: list[dict]) -> float:
    """Score only query coordinates; source values remain visible to the model."""
    output = pl.DataFrame(model.predict(records)).to_dicts()
    actual, predicted = [], []
    for row, result in zip(records, output, strict=True):
        for item, prediction in zip(row["memory"], result["predictions"]["memory"], strict=True):
            if item["is_query"]:
                actual.append(item["value"])
                predicted.append(prediction["value"]["content"])
    actual = np.asarray(actual)
    return float(np.sqrt(np.mean(np.square(np.asarray(predicted) - actual)) / np.mean(np.square(actual))))


class MemoryRecord(rf.Branch):
    entity_id = rf.Hash(n_hashes=4, n_bands=8)
    role = rf.Category(p_unavailable=0.0)
    is_query = rf.Boolean
    value = rf.Number(**reconstruct("is_query"), objective="mse")


class PredictionModel(rf.Model):
    memory = MemoryRecord(length=2 * PAIR_COUNT, n_layers=2, reduction=rf.Attention())


@experiment("P024", seed=17)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = PredictionModel(d_model=64, n_layers=2, n_heads=4, reduction=rf.Attention())
    data = rf.SyntheticDataModule(
        train=partial(records, rows=2048, seed=seed + 1, namespace="train"),
        validate=partial(records, rows=512, seed=seed + 2, namespace="validate"),
        seed=seed,
    )
    model.fit(
        data,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.003),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
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


if __name__ == "__main__":
    report(run)
