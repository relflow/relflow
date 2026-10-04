# %% [markdown]
# ---
# title: Do stable labels produce the right groups?
# categories:
# - Clustering
# proof-id: P020
# description: Repeated identities carry stable labels. Ask the Cluster field to reconstruct identities
#   while predicting their labels.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: Groups at the final epoch
#   metric:
#   - final_committed
#   - -1
#   format: number
# ---
#
# ## Example
#
# ```yaml
# merchant_id: c0-id3
# label: L0
# ```
#
# The label is hidden; identities are sometimes hidden for reconstruction during training.
#
# ```{typst}
# //| label: fig-proof-cluster-reconstructing-category-labels
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
# Compare learned group count and usage with the five hidden label groups.
#
# ## Result
#
# {{< proof P020 status >}}
#
# The model reaches roughly five groups, but count and usage do not measure whether grouping is correct. Reused validation records also leave independent predictive quality untested.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P020 evidence >}}
#
# ### Run this experiment
#
# {{< proof P020 script >}}
#
# ### How it works
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
# ### Remaining work
#
# Use independent observations for each split, add held-out downstream skill
# against a marginal baseline, and require adjusted Rand index (ARI) ≥ 0.80
# for partition recovery. Add unique-identity and no-regime controls. Replace
# internal-state inspection with a public diagnostic when available.
#
# The family requires three passing core seeds and at least ten calibration
# seeds before its terminal-window thresholds can be promoted.
#
# ### Complete experiment code
#

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


if __name__ == "__main__":
    report(PROOF_ID, run, seed=0)

# %% [markdown]
# </details>
