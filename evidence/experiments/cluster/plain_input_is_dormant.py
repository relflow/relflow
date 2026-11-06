# %% [markdown]
# ---
# title: Does a Cluster input group itself automatically?
# categories:
# - Cluster
# description: Use a Cluster field only as an input beside a hidden prediction target.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
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
# evidence-group: Representation
# ---
#
# The name Cluster can suggest that adaptive grouping happens whenever identities recur. This
# negative control checks that expectation. The model uses merchant_id only as an input and predicts
# a hidden label; the identity field itself is never selected as a reconstruction objective.
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
# The illustrated identity c2-id7 always accompanies label L2 in the synthetic records. There are
# ten identities for each of five labels, and every identity repeats twenty times. The names are
# synthetic identifiers, not evidence about real merchants. Predicting L2 for this familiar identity
# and updating the adaptive grouping mechanism are separate behaviors.
#
# ```{typst}
# //| label: fig-evidence-cluster-plain-input-is-dormant
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
# Observe its adaptive grouping state before and after training, without asking the Cluster field to
# reconstruct itself.
#
# The callback watches committed count and adherence through five epochs of label training. With no
# identity reconstruction loss, the expected result is unchanged count and zero adherence. This is a
# mechanism check rather than a failed demand for useful groups. The reconstructing-label experiment
# adds the missing objective; it must then face its own quality checks.
#
# ## Result
#
# {{< evidence P019 status >}}
#
# Its grouping state stays dormant in this setup. Add a reconstructing mask when adaptive grouping
# is intended; then test whether the groups are useful.
#
#
# A passing status here means the intended inactivity was observed. It does not mean the model
# discovered groups or achieved any particular held-out label accuracy: those quantities are not the
# reported checks. Validation reuses the training observations, so this example also provides no
# evidence for prediction on new identities or independently sampled records.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P019 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P019 script >}}
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
import relflow as rf
import torch
from reporting import experiment, report


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
        if not isinstance(embedder, rf.TENSORFIELDS["cluster"].Embedder):
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


@experiment("P019", seed=0)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        merchant_id=rf.Cluster(bounds=(3, 15)),
        label=rf.Category(mask=True, p_unavailable=0.0),
    )
    source = partial(records, seed=seed)
    data = rf.SyntheticDataModule(train=source, validate=source, seed=seed)
    trajectory = Trajectory(rf.Address("merchant_id"))
    model.fit(
        data,
        batch_size=64,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.003),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
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

    committed = [row["n_committed"] for row in trajectory.rows]
    adherence = [row["adherence"] for row in trajectory.rows]
    return {"trajectory": trajectory.rows, "committed": committed, "adherence": adherence}, {
        "Committed count remains constant": len(set(committed)) == 1,
        "Adherence remains exactly zero": all(value == 0.0 for value in adherence),
    }


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
