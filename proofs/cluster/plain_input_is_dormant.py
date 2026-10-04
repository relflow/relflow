# %% [markdown]
# ---
# title: Does a Cluster input group itself automatically?
# categories:
# - Clustering
# proof-id: P019
# description: Use a Cluster field only as an input beside a hidden prediction target.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: Groups at the final epoch
#   metric:
#   - committed
#   - -1
#   format: number
# - label: Grouping activity at the final epoch
#   metric:
#   - adherence
#   - -1
#   format: number
# ---
#
# ## Example
#
# ```yaml
# merchant_id: c2-id7
# label: L2
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-proof-cluster-plain-input-is-dormant
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with event, merchant_id, label. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("event", kind: "root", children: (
#   node("merchant_id", type: "Cluster", width: 150pt,),
#   node("label", kind: "target", type: "Category",),
# )))
# ```
#
# ## Comparison
#
# Observe its adaptive grouping state before and after training, without asking the Cluster field to reconstruct itself.
#
# ## Result
#
# {{< proof P019 status >}}
#
# Its grouping state stays dormant in this setup. Add a reconstructing mask when adaptive grouping is intended; then test whether the groups are useful.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P019 evidence >}}
#
# ### Run this experiment
#
# {{< proof P019 script >}}
#
# ### How it works
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
# ### Remaining work
#
# Repeat this negative control alongside the positive variants across three
# core seeds. The broader family still needs independent held-out observations,
# partition recovery, and unique-identity and no-regime controls. Its count
# thresholds also need at least ten calibration seeds.
#
# ### Complete experiment code
#

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


if __name__ == "__main__":
    report(PROOF_ID, run, seed=0)

# %% [markdown]
# </details>
