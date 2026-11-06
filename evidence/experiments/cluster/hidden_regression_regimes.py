# %% [markdown]
# ---
# title: Do five clusters mean five behaviors were found?
# categories:
# - Cluster
# description: Repeated identities follow five hidden numerical rules. Watch how the Cluster field groups them during
#   training.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Groups at the final epoch
#   metric:
#   - final_committed
#   - -1
#   format: number
# evidence-group: Representation
# ---
#
# One hundred recurring identities generate numeric observations under five different hidden rules.
# The model sees id and x and learns to predict y, while identity reconstruction activates adaptive
# clustering. This experiment watches whether the number and usage of clusters approach five; it
# does not directly score which identities should share a group.
#
# ## Example
#
# ```yaml
# id: id-0007
# x: 0.0
# y: -8.0
# ```
#
# The numeric answer is hidden. Identity reconstruction activates grouping; the generating rule is
# not a model input.
#
# For the default generator seed, id-0007 follows the rule 0.5 × x − 8. At x = 0 its underlying
# answer is −8, as illustrated before the generator adds small noise. Another identity may follow a
# sine curve or a different offset. Repeated observations let the model encounter each identity
# under changing x values.
#
# ```{typst}
# //| label: fig-evidence-cluster-hidden-regression-regimes
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with obs, id, x, y. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("obs", kind: "root", children: (
#   node("id", type: "Cluster", width: 150pt,),
#   node("x", type: "Number"),
#   node("y", kind: "target", type: "Number",),
# )))
# ```
#
# ## Comparison
#
# Compare the number and usage of learned groups with the five generating rules.
#
# A callback records committed cluster count and usage at each training epoch. The final five epochs
# are compared with a window around five, while checking that the count has not simply stopped at an
# allowed bound. Crucially, training and validation reuse the same observations. There is no
# independent prediction test, wrong-group control, or measured agreement with the generating
# partition here.
#
# ## Result
#
# {{< evidence P018 status >}}
#
# The selected run does not meet the count-and-usage gates. Those diagnostics do not measure correct
# membership; training and validation also reuse the same records, leaving independent prediction
# and grouping quality untested.
#
#
# Cluster count is simply how many groups the adaptive mechanism commits. Usage perplexity
# summarizes how broadly the available groups are used; it is not classification accuracy. The
# selected run misses the intended count-and-usage window. Even a count of exactly five would not
# show that identities following the same rule were assigned together.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P018 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P018 script >}}
#
# ### How it works
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
# ### Remaining work
#
# Create independent train, validation, and test observations. Measure held-out
# regression against a marginal baseline and require permutation-invariant
# partition recovery with ARI ≥ 0.80. Add unique-identity and no-regime controls,
# and replace internal instrumentation with a public diagnostic when available.
#
# Repeat all behavioral and partition gates across three core seeds and at least
# ten calibration seeds before promoting this evidence.
#
# ### Complete experiment code
#

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
import relflow as rf
import torch
from reporting import experiment, report


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


@experiment("P018", seed=0)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        id=rf.Cluster(bounds=(3, 15), mask=rf.Mask(rate=0.5, reconstruct=True)),
        x=rf.Number,
        y=rf.Number(mask=True),
    )
    source = partial(records, seed=seed + 42)
    data = rf.SyntheticDataModule(train=source, validate=source, seed=seed)
    trajectory = Trajectory(rf.Address("id"))
    model.fit(
        data,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.003),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
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
    report(run)

# %% [markdown]
# </details>
