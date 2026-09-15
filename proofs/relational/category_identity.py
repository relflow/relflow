# %% [markdown]
# ---
# title: Transfer Values by Category Identity
# categories:
# - Sibling entity transfer
# proof-id: P036
# description: Match targets to source values across sibling branches using persistent
#   Category identities.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# The model must carry each source value to a target with the same identity in
# another branch. Values change on every row and the two branch orders are
# independently shuffled, leaving the key as the useful link.
#
# {{< proof P036 status >}}
#
# ## Insights
#
# **The model transfers changing values by identity across independently ordered
# branches.** The two categories recur, but their source values are resampled
# for every row, so remembering one value per category cannot solve the task.
# Visible target identities can condition repeated decoder queries over source
# evidence routed through the root.
#
# Rotating target keys while retaining their original values removes accuracy,
# supporting keyed transfer rather than positional copying. Both retained and
# compressed routes work in this small persistent-vocabulary case. That does
# not establish compressed retrieval for unseen identities or larger category
# spaces. Treat it as evidence of a learned communication path; use an exact
# join when the matching rule and transferred value must be guaranteed.
#
# ## Setup

# %%
"""Transfer values between sibling branches using familiar Category identities.

Source and target orders vary independently. Train compressed and preserved
routes on the same records, then rotate only query identities. This distinguishes
keyed transfer from position copying with two persistent vocabulary entries.

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

PROOF_ID = "P036"
PAIR_COUNT = 2

# %% [markdown]
# ## Examples
#
# ### Match independently ordered branches
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
# Target values are supervision excluded from embedding by `mask=True`; source
# values remain visible. The same two Category identities recur across splits,
# but their randomly resampled values cannot be memorized by identity alone.
#
# ### The same identities carry new values
#
# ```yaml
# source:
#   - {entity_id: entity-A, value: -0.6}
#   - {entity_id: entity-B, value: 0.9}
# target:
#   - {entity_id: entity-B, value: 0.9}
#   - {entity_id: entity-A, value: -0.6}
# ```
#
# The categories recur, but the correct target values change with this row's
# source. Remembering a constant value for each category cannot solve the task.
#
# ### Break the target identities
#
# ```yaml
# source:
#   - {entity_id: entity-A, value: 0.7}
#   - {entity_id: entity-B, value: -0.2}
# target:
#   - {entity_id: entity-A, value: -0.2}
#   - {entity_id: entity-B, value: 0.7}
# ```
#
# The control rotates the first example's target keys while retaining its
# supervised values. Those retained targets now conflict with their source
# matches. Accurate keyed retrieval should lose accuracy against these labels.
#
# ## Synthetic data and controls


# %%
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
    output = model.predict(records).to_pylist()
    predicted = np.asarray(
        [coordinate["content"] for row in output for coordinate in row["predictions"]["association/target/value"]]
    )
    return float(np.sqrt(np.mean(np.square(actual - predicted)) / np.mean(np.square(actual))))


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-category-identity
# //| fig-cap: "Two separately trained models share this schema. Retained and compressed labels describe alternative reductions at the root and both branches; target values are masked."
# //| fig-alt: "Association has two source and two target records using Category identities. Source values are visible and target values are masked. At the root and each branch, the retained model keeps all tokens and the compressed model uses one attention summary."
# #tree(node("association", kind: "root", width: 150pt, body: [
#   - *Retained:* Keep all tokens
#   - *Compressed:* Attention · 1 summary
# ], children: (
#   node("source", kind: "branch", repeated: true, width: 150pt, body: [
#     - *Capacity:* 2 items
#     - *Retained:* Keep all tokens
#     - *Compressed:* Attention · 1 summary
#   ], children: (
#     node("entity_id", type: "Category"),
#     node("value", type: "Number"),
#   )),
#   node("target", kind: "branch", repeated: true, width: 150pt, body: [
#     - *Capacity:* 2 items
#     - *Retained:* Keep all tokens
#     - *Compressed:* Attention · 1 summary
#   ], children: (
#     node("entity_id", type: "Category"),
#     node("value", kind: "target", type: "Number", width: 150pt, body: [
#       - *Input:* always hidden
#     ]),
#   )),
# )))
# ```
#
# The retained route uses `reduction=None` at both branches and the root. The
# matched compressed route uses `rf.Attention()` at all three locations.
#
# ## How it works
#
# Source coordinates bind keys to values. Target identities condition repeated
# decoder queries that can attend to source context through the root. Rotating
# target keys while retaining the original supervised values breaks the correct
# matching and should remove the learned skill.
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
                entity_id=rf.Category(size=PAIR_COUNT, p_unavailable=0.0),
                value=rf.Number,
            ),
            target=rf.Branch(
                length=PAIR_COUNT,
                n_layers=2,
                reduction=reduction,
                entity_id=rf.Category(size=PAIR_COUNT, p_unavailable=0.0),
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
        "Compressed Category transfer nRMSE <= 0.25": compressed <= 0.25,
    }
    return metrics, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P036 evidence >}}
#
# ## Remaining work
#
# Repeat three core seeds and ten calibration seeds, then increase identities
# and test missing, duplicate, padded, or absent matches. This small persistent
# vocabulary result does not establish compressed retrieval for fresh IDs; the
# [Hash case](hash-identity.html) tests that distinction. Use an exact join in
# preprocessing when exact value transfer is required.
#
# ## Reproduce
#
# {{< proof P036 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=17)
