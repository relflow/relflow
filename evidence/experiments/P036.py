# %% [markdown]
# ---
# title: Can familiar IDs connect two lists?
# description: Use matching category IDs to transfer changing values from a source list to a separately ordered target
#   list.
# evidence-readout:
# - label: Compressed route
#   metric:
#   - compressed_nrmse
#   format: error
# - label: Items retained
#   metric:
#   - preserved_nrmse
#   format: error
# - label: Changed IDs, items retained
#   metric:
#   - preserved_broken_nrmse
#   format: error
# evidence-group: Relational
# evidence-review:
#   question: Can familiar IDs connect two lists?
#   modeling: Give a source list of category IDs and changing values, then a separately ordered query list using familiar
#     IDs. Hide each requested source value. Compare correct query IDs with rotated IDs that retain their original
#     targets.
#   comparison:
#     metric: Normalized RMSE
#     kind: Controlled comparison
#     conditions:
#     - label: Query IDs changed
#       path:
#       - preserved_broken_nrmse
#       format: number
#     - label: Matching familiar IDs
#       path:
#       - preserved_nrmse
#       format: number
#     definition: Root mean squared error divided by the reference scale defined in this experiment. Zero is exact;
#       one equals that reference scale.
#     direction: Lower error is better
#   note: This comparison uses the route that retains source items. Compressed-route results remain below; it does
#     not test unseen Category labels.
# evidence-plot:
#   mode: regression
#   target: target/value
#   focus: Does compressing a Category route lose observation-local associations?
#   comparison: Compare compressed and preserved routes, each with intact and broken identities.
#   limit: The persistent Category vocabulary is small; numeric associations still vary by observation.
# evidence-cost:
#   baseline: 174
#   candidate: 174
# ---
#
# ## Example
#
# The source list supplies an ID and a changing value; each target item supplies an ID and asks for its
# hidden value. In the illustration, target `entity-B` must receive `-0.2` even though it comes first in the
# target list and second in the source list. Target `entity-A` must receive `0.7`. The two IDs are familiar
# vocabulary entries across splits, but their numeric values are drawn anew for each record.
#
# ```yaml
# source:
#   - {entity_id: entity-A, value: 0.7}
#   - {entity_id: entity-B, value: -0.2}
# target:
#   - {entity_id: entity-B, value: -0.2}
#   - {entity_id: entity-A, value: 0.7}
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ## Comparison
#
# Source and target order vary independently, so copying the first source value to the first target cannot
# solve the task. The identity intervention swaps the target IDs while keeping the original target values
# for scoring: the visible lookup now points to the wrong value. Both a compressed route and a route
# retaining item representations train on the same generated records and seed, isolating the representation
# choice under this small setup.
#
# Change target IDs while retaining original answers. Compare compressed summaries with a route that keeps encoded items.
#
# ## Result
#
# Each error is divided by the error of always predicting zero, the natural constant reference for these
# symmetric targets. Read the intact routes alongside the changed-ID control: good intact error alone would
# not establish that IDs drive the transfer. There are exactly two persistent category IDs and two items per
# side. The result does not test unfamiliar vocabulary, absent matches, duplicate keys, or the correctness
# guarantees of a database join.
#
# The selected run compares both routes with two familiar IDs. New IDs and larger vocabularies are separate tasks.
#
# ## Method
#
# ### How it works
#
# Source coordinates bind keys to values. Target identities condition repeated
# decoder queries that can attend to source context through the root. Rotating
# target keys while retaining the original supervised values breaks the correct
# matching and should remove the learned skill.
#
# ### Remaining work
#
# Repeat three core seeds and ten calibration seeds, then increase identities
# and test missing, duplicate, padded, or absent matches. This small persistent
# vocabulary result does not establish compressed retrieval for fresh IDs; the
# [Hash case](hash-identity.html) tests that distinction. Use an exact join in
# preprocessing when exact value transfer is required.
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
# types = ["Category"]
# conditions = ["changed-associations", "representations"]
# ///

"""Transfer values between sibling branches using familiar Category identities.

Source and target orders vary independently. Train compressed and preserved
routes on the same records, then rotate only query identities. This distinguishes
keyed transfer from position copying with two persistent vocabulary entries.

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
from visibility import TARGET

PAIR_COUNT = 2


def records(*, rows: int, seed: int, namespace: str, broken_identity: bool = False) -> Iterator[dict]:
    """Generate observation-local key/value associations in sibling branches."""
    rng = np.random.default_rng(seed)
    for row in range(rows):
        keys = [f"entity-{chr(ord('A') + pair)}" for pair in range(PAIR_COUNT)]
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
    output = pl.DataFrame(model.predict(records)).to_dicts()
    predicted = np.asarray([item["value"]["content"] for row in output for item in row["predictions"]["target"]])
    return float(np.sqrt(np.mean(np.square(actual - predicted)) / np.mean(np.square(actual))))


@experiment("P036", seed=17)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    test = list(records(rows=1024, seed=seed + 3, namespace="test"))
    broken = list(records(rows=1024, seed=seed + 3, namespace="test", broken_identity=True))
    metrics = {}
    # Fit both routes from the same seed and data, changing only reduction.
    for route, reduction in (("compressed", rf.Attention()), ("preserved", None)):
        lit.seed_everything(seed, workers=True)

        class SourceRecord(rf.Branch):
            entity_id = rf.Category(p_unavailable=0.0)
            value = rf.Number

        class TargetRecord(rf.Branch):
            entity_id = rf.Category(p_unavailable=0.0)
            value = rf.Number(**TARGET, objective="mse")

        class PredictionModel(rf.Model):
            source = SourceRecord(length=PAIR_COUNT, n_layers=2, reduction=reduction)
            target = TargetRecord(length=PAIR_COUNT, n_layers=2, reduction=reduction)

        model = PredictionModel(d_model=64, n_layers=2, n_heads=4, reduction=reduction)
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
            max_steps=600 if steps is None else min(steps, 600),
            max_epochs=-1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            deterministic=True,
            check_val_every_n_epoch=5,
        )
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
        "Compressed Category transfer nRMSE <= 0.25": compressed <= 0.25,
    }
    return metrics, checks


if __name__ == "__main__":
    report(run)
