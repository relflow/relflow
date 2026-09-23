# %% [markdown]
# ---
# title: Cluster Commitment with Stable Labels
# categories:
# - Clustering
# proof-id: P020
# description: Repeated identities and a reconstruction objective can move adaptive
#   cluster commitment toward the generating group count.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# Repeated merchant identities each have a stable label drawn from five groups.
# The model predicts that label while also reconstructing masked identity
# representations. The experiment checks how many clusters become committed
# and how broadly they are used.
#
# {{< proof P020 status >}}
#
# ## Insights
#
# **The right number of clusters does not mean the right identities were grouped together.** Repeated
# identities provide consistent label evidence, and the Cluster reconstruction loss updates usage and
# commitment.
#
# The recorded run ends near five groups. However, committed count is derived by rounding and clamping usage
# perplexity: these are coupled diagnostics, not independent confirmations of recovery. Incorrect groups can
# satisfy both gates.
#
# Training and validation reuse observations, with no held-out skill or assignment-agreement requirement.
# Treat this as evidence about mechanism activation. Demonstrating useful clustering still requires
# independent predictions, partition recovery, and controls without genuine generating groups.
#
# ## Setup

# %%
"""P020: track adaptive clusters across five repeated-identity labels.

Training and validation deliberately reuse observations. Count and usage
perplexity diagnose the mechanism; they do not prove partition recovery or
held-out prediction quality.
"""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import torch
from reporting import report

import relflow as rf
from relflow.tensorfields.extensions.cluster import Embedder

PROOF_ID = "P020"

# %% [markdown]
# ## Examples
#
# Each generating identity appears 20 times with a stable hidden `label`.
# The Cluster field is an input with a reconstructing mask; its displayed
# identity is not always visible during training.
#
# ### One identity in the first group
#
# ```yaml
# merchant_id: c0-id3
# label: L0
# ```
#
# Repeated observations give this identity consistent evidence for label `L0`.
#
# ### A peer identity in the same group
#
# ```yaml
# merchant_id: c0-id9
# label: L0
# ```
#
# The identifier differs, but the behavior matches. These two identities belong
# together in the generator's partition.
#
# ### A different group
#
# ```yaml
# merchant_id: c4-id8
# label: L4
# ```
#
# This identity belongs to a different generating group. The five groups have
# ten identities each, for 1,000 total observations. Their label relationship
# is ground truth; the present proof checks cluster count and usage, not whether
# these particular identities received the correct learned assignments.
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
# //| label: fig-proof-cluster-reconstructing-category-labels
# //| fig-cap: "Merchant ID uses a 50% training mask with reconstruction enabled. Its label is always hidden from model inputs."
# //| fig-alt: "Event contains a Cluster merchant ID with 50% training masking and reconstruction enabled, and a Category label target always hidden from input."
# #tree(node("event", kind: "root", children: (
#   node("merchant_id", type: "Cluster", width: 150pt, body: [
#     - *Mask:* 50% in training
#     - *Reconstruct:* enabled
#   ]),
#   node("label", kind: "target", type: "Category", body: [
#     - *Input:* always hidden
#   ]),
# )))
# ```
#
# ## How it works
#
# The merchant field uses
# `rf.Cluster(mask=rf.Mask(rate=0.5, reconstruct=True), ...)`. Its reconstruction
# loss updates adaptive usage and commitment state. Repetition lets evidence
# accumulate for an identity. `label=rf.Category(mask=True, ...)` supplies the
# supervised task without exposing its answer as an input.
#
# Assignments belong to stable identities. Row-varying sibling context does not
# directly condition the Cluster reconstruction query. The
# [plain-input control](plain-input-is-dormant.html) shows what happens when the
# Cluster reconstruction objective is absent.
#
# The model trains for 30 deterministic epochs. A callback inspects internal
# Cluster state at each epoch; the gates examine its final five entries.
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
        merchant_id=rf.Cluster(n_clusters=(3, 15), mask=rf.Mask(rate=0.5, reconstruct=True)),
        label=rf.Category(mask=True, p_unavailable=0.0),
    )
    model.optimizer = lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3)
    source = partial(records, seed=seed)
    data = rf.SyntheticDataModule(model=model, train=source, validate=source, seed=seed)
    trajectory = Trajectory(rf.Address("/", "merchant_id"))
    trainer = lit.Trainer(
        accelerator=accelerator,
        max_epochs=30,
        max_steps=steps if steps is not None else -1,
        callbacks=[trajectory],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model=model, datamodule=data)

    tail = trajectory.rows[-5:]
    committed = [row["n_committed"] for row in tail]
    perplexity = [row["perplexity"] for row in tail]
    return {
        "trajectory": trajectory.rows,
        "final_committed": committed,
        "final_perplexity": perplexity,
    }, {
        "Final committed counts remain between 4 and 7": all(4 <= value <= 7 for value in committed),
        "Final perplexity remains within 1.5 of five": all(abs(value - 5) <= 1.5 for value in perplexity),
        "Terminal commitment is above the lower bound": committed[-1] != 3,
        "Terminal commitment is below the upper bound": committed[-1] != 15,
    }


# %% [markdown]
# ## Evidence
#
# {{< proof P020 evidence >}}
#
# ## Remaining work
#
# Use independent observations for each split, add held-out downstream skill
# against a marginal baseline, and require adjusted Rand index (ARI) ≥ 0.80
# for partition recovery. Add unique-identity and no-regime controls. Replace
# internal-state inspection with a public diagnostic when available.
#
# The family requires three passing core seeds and at least ten calibration
# seeds before its terminal-window thresholds can be promoted.
#
# ## Reproduce
#
# {{< proof P020 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=0)
