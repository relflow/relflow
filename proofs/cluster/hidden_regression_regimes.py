# %% [markdown]
# ---
# title: Cluster Commitment with Hidden Regimes
# categories:
# - Clustering
# proof-id: P018
# description: Adaptive cluster state can settle near five generating regression regimes
#   without explicit group labels.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# Each repeated identity follows one of five hidden relationships between `x`
# and `y`. The model receives no group label. This experiment asks whether
# adaptive Cluster state settles near the generating number of regimes.
#
# {{< proof P018 status >}}
#
# ## Insights
#
# **Settling near five clusters does not yet show that the model discovered the five generating functions.**
# Repeated identities follow stable regimes while their numeric inputs vary; reconstruction engages adaptive
# Cluster state alongside the supervised numeric task.
#
# The recorded gates inspect terminal count and usage. Count is calculated from usage perplexity, so those
# diagnostics are coupled. Neither tells us whether identities share the correct regime or whether
# predictions work on independent observations.
#
# The current train and validation tables are identical. Evaluate held-out regression and assignment
# agreement before treating this as regime discovery. The existing result supports adaptive-state behavior,
# with the harder behavioral claim still open.
#
# ## Setup

# %%
"""P018: track adaptive clusters across five hidden regression regimes.

Training and validation deliberately reuse observations. Count and usage
perplexity diagnose the mechanism; they do not prove partition recovery or
held-out prediction quality.
"""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf
from relflow.tensorfields.extensions.cluster import Embedder

PROOF_ID = "P018"

# %% [markdown]
# ## Examples
#
# These are idealized values before observation noise, with illustrative regime
# assignments. The seeded generator assigns regimes randomly; these ID-to-regime
# choices are not reported measurements. `y` is always hidden from embedding.
#
# ### One identity at the start of its curve
#
# ```yaml
# id: id-0007
# x: 0.0
# y: 10.0
# ```
#
# An identity assigned to `sin(x) + 10` has a noiseless target of 10 here.
#
# ### The same identity at another input
#
# ```yaml
# id: id-0007
# x: 1.5707963267948966
# y: 11.0
# ```
#
# At approximately π/2, the same regime gives 11. Its target varies with `x`,
# but its generating group stays fixed.
#
# ### Another regime at the same input
#
# ```yaml
# id: id-0042
# x: 1.5707963267948966
# y: -2.0
# ```
#
# An identity assigned to the constant `-2` regime gives a different answer for
# the same `x`. Repeated identity evidence can distinguish these processes.
# The examples describe the generating functions, not verified learned
# assignments or held-out predictions.
#
# ## Synthetic data and controls


# %%
def records(seed: int) -> Iterator[dict]:
    """Repeat 100 identities thirty times across five hidden regression regimes."""
    rng = np.random.default_rng(seed)
    functions = (
        lambda x: np.sin(x) + 10.0,
        lambda x: np.cos(x) + 4.0,
        lambda x: np.zeros_like(x) - 2.0,
        lambda x: 0.5 * x - 8.0,
        lambda x: -0.3 * x - 14.0,
    )
    id_to_cluster = rng.integers(0, 5, size=100)
    identities = np.repeat(np.arange(100), 30)
    clusters = id_to_cluster[identities]
    x = rng.uniform(0.0, 10.0, size=3000) + rng.normal(0.0, 0.05, size=3000)
    y = np.empty(3000)
    for cluster, function in enumerate(functions):
        selected = clusters == cluster
        y[selected] = function(x[selected]) + rng.normal(0.0, 0.05, size=int(selected.sum()))
    for index in rng.permutation(3000):
        yield {"id": f"id-{identities[index]:04d}", "x": float(x[index]), "y": float(y[index])}


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
# //| label: fig-proof-cluster-hidden-regression-regimes
# //| fig-cap: "The ID uses a 50% training mask with reconstruction enabled. The supervised y target is always hidden from input."
# //| fig-alt: "Observation contains a Cluster ID input with 50% training masking and reconstruction enabled, a Number x input, and a Number y target always hidden from input."
# #tree(node("obs", kind: "root", children: (
#   node("id", type: "Cluster", width: 150pt, body: [
#     - *Mask:* 50% in training
#     - *Reconstruct:* enabled
#   ]),
#   node("x", type: "Number"),
#   node("y", kind: "target", type: "Number", body: [
#     - *Input:* always hidden
#   ]),
# )))
# ```
#
# ## How it works
#
# The generator supplies 100 identities with 30 observations each. Its five
# regimes are `sin(x) + 10`, `cos(x) + 4`, `-2`, `0.5x - 8`, and `-0.3x - 14`,
# with small additive noise. The repeated identity lets the model associate
# behavior across changing `x` values.
#
# The ID field uses `rf.Mask(rate=0.5, reconstruct=True)` to engage the Cluster
# reconstruction loss. The numeric objective adds pressure to explain `y`.
# Cluster assignments belong to identities; row-varying `x` does not directly
# condition the query that reconstructs the identity itself.
#
# The model trains for 30 deterministic epochs. Training and validation
# currently use the same 3,000 observations. An internal-state callback measures
# the final five epochs.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        batch_size=128,
        id=rf.Cluster(n_clusters=(3, 15), mask=rf.Mask(rate=0.5, reconstruct=True)),
        x=rf.Number,
        y=rf.Number(mask=True),
    )
    model.optimizer = lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3)
    source = partial(records, seed=seed + 42)
    data = rf.SyntheticDataModule(model=model, train=source, validate=source, seed=seed)
    trajectory = Trajectory(rf.Address("/", "id"))
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
# {{< proof P018 evidence >}}
#
# ## Remaining work
#
# Create independent train, validation, and test observations. Measure held-out
# regression against a marginal baseline and require permutation-invariant
# partition recovery with ARI ≥ 0.80. Add unique-identity and no-regime controls,
# and replace internal instrumentation with a public diagnostic when available.
#
# Repeat all behavioral and partition gates across three core seeds and at least
# ten calibration seeds before promoting this proof.
#
# ## Reproduce
#
# {{< proof P018 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=0)
