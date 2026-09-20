# %% [markdown]
# ---
# title: A Plain Cluster Input Stays Dormant
# categories:
# - Clustering
# proof-id: P019
# description: Supervision on a sibling field does not engage adaptive Cluster commitment
#   without a Cluster reconstruction objective.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# Adding a Cluster field as an input does not automatically train its adaptive
# commitment mechanism. This control keeps the repeated identities and stable
# labels, but removes reconstruction from the Cluster field.
#
# {{< proof P019 status >}}
#
# ## Insights
#
# **A prediction target beside a Cluster field does not activate that field’s adaptive grouping.** The
# identities repeat and carry stable labels, but the Cluster field is only an input.
#
# The expected result is unchanged committed count and zero adherence, as checked by this experiment. This
# does not mean every parameter or the label decoder is frozen: the checks concern these specific
# adaptive-state diagnostics. No label-prediction score is established here.
#
# When adaptive grouping is intended, add a reconstructing mask to the Cluster field before changing
# capacity or training longer. Then check predictive usefulness and assignment quality separately; merely
# activating commitment does not establish either.
#
# ## Setup

# %%
"""P019: show that a plain-input Cluster keeps its adaptive state dormant.

The supervised label objective does not invoke the Cluster reconstruction
loss. Repeated identities alone should leave commitment and adherence frozen.
"""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import torch
from reporting import report

import relflow as rf
from relflow.tensorfields.extensions.cluster import Embedder

PROOF_ID = "P019"

# %% [markdown]
# ## Examples
#
# All these identities recur in the generated training table. The label is
# hidden with `mask=True`, but the Cluster input has no reconstructing mask.
#
# ### Repeated identity with a stable label
#
# ```yaml
# merchant_id: c2-id7
# label: L2
# ```
#
# This record appears 20 times. Repetition alone does not invoke the Cluster
# reconstruction loss.
#
# ### Another identity with the same behavior
#
# ```yaml
# merchant_id: c2-id3
# label: L2
# ```
#
# The generator places this identity in the same group. Predicting the sibling
# label still does not engage adaptive Cluster commitment.
#
# ### An identity from another group
#
# ```yaml
# merchant_id: c4-id8
# label: L4
# ```
#
# The data contains a real contrast between groups; it is not a one-label
# process. Even so, the expected diagnostic is unchanged commitment and zero
# adherence. No label-prediction score or recovered assignment for these
# individual records is asserted.
#
# ## Synthetic data and controls


# %%
def records(seed: int) -> Iterator[dict]:
    """Repeat each of 50 identities twenty times with one stable regime label."""
    rows = [
        {"merchant_id": f"c{cluster}-id{identity}", "label": f"L{cluster}"}
        for cluster in range(5)
        for identity in range(10)
        for _ in range(20)
    ]
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed)).tolist()
    for index in order:
        yield rows[index]


class Trajectory(lit.Callback):
    """Inspect Cluster internals; these diagnostics do not establish partition accuracy."""

    def __init__(self, address: rf.Address) -> None:
        self.address = address
        self.rows: list[dict] = []

    def on_train_epoch_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        embedder = pl_module.nodes[self.address].embedder
        if not isinstance(embedder, Embedder):
            raise TypeError(f"{self.address} requires a Cluster embedder, got {type(embedder).__name__}")
        usage = embedder.usage_ema.detach()
        probabilities = usage / usage.sum().clamp_min(1e-12)
        bounded = probabilities.clamp_min(1e-12)
        entropy = -(bounded * bounded.log()).sum()
        self.rows.append(
            {
                "epoch": trainer.current_epoch,
                "n_committed": int(embedder.committed.sum().item()),
                "perplexity": float(torch.exp(entropy).item()),
                "adherence": float(embedder.adherence_ema.item()),
            }
        )


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-cluster-plain-input-is-dormant
# //| fig-cap: "Merchant ID has no reconstruction objective. Only the label is a hidden prediction target."
# //| fig-alt: "Event contains a plain Cluster merchant ID input with reconstruction disabled, and a Category label target always hidden from input."
# #tree(node("event", kind: "root", children: (
#   node("merchant_id", type: "Cluster", width: 150pt, body: [
#     - *Role:* plain input
#     - *Reconstruct:* disabled
#   ]),
#   node("label", kind: "target", type: "Category", body: [
#     - *Input:* always hidden
#   ]),
# )))
# ```
#
# ## How it works
#
# Cluster's reconstruction loss updates its adaptive usage and commitment state.
# Training only the sibling label objective does not invoke that loss. The
# absence of a Cluster reconstruction objective therefore leaves the measured
# commitment mechanism inactive even though the identity recurs.
#
# This model uses 1,000 observations: five label groups, ten identities per group,
# and 20 observations per identity. It trains for five deterministic epochs,
# reusing the same table for validation. A callback inspects the committed count
# and adherence at each epoch.
#
# To engage the mechanism, the paired
# [reconstructing-label experiment](reconstructing-category-labels.html) adds
# `rf.Mask(rate=0.5, reconstruct=True)` to the Cluster field.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        batch_size=64,
        merchant_id=rf.Cluster(n_clusters=(3, 15)),
        label=rf.Category(mask=True, p_unavailable=0.0),
    )
    model.optimizer = lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3)
    source = partial(records, seed=seed)
    data = rf.SyntheticDataModule(model=model, train=source, validate=source, seed=seed)
    trajectory = Trajectory(rf.Address("/", "merchant_id"))
    trainer = lit.Trainer(
        accelerator=accelerator,
        max_epochs=5,
        max_steps=steps if steps is not None else -1,
        callbacks=[trajectory],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model=model, datamodule=data)

    committed = [row["n_committed"] for row in trajectory.rows]
    adherence = [row["adherence"] for row in trajectory.rows]
    return {"trajectory": trajectory.rows, "committed": committed, "adherence": adherence}, {
        "Committed count remains constant": len(set(committed)) == 1,
        "Adherence remains exactly zero": all(value == 0.0 for value in adherence),
    }


# %% [markdown]
# ## Evidence
#
# {{< proof P019 evidence >}}
#
# ## Remaining work
#
# Repeat this negative control alongside the positive variants across three
# core seeds. The broader family still needs independent held-out observations,
# partition recovery, and unique-identity and no-regime controls. Its count
# thresholds also need at least ten calibration seeds.
#
# ## Reproduce
#
# {{< proof P019 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=0)
