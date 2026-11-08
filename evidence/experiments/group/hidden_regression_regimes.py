# %% [markdown]
# ---
# title: Do five groups mean five behaviors were found?
# categories:
# - Group
# description: Repeated identities follow five hidden numerical rules. Watch how the Group field assigns them during
#   training.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Assigned groups at the final epoch
#   metric:
#   - final_assigned_groups
#   - -1
#   format: number
# evidence-group: Representation
# ---
#
# One hundred recurring identities generate numeric observations under five different hidden rules.
# The model sees id and x and learns to predict y, while a masked Group objective trains the
# identity's state head. This experiment watches whether the number and usage of assigned groups
# approach five; it does not directly score which identities should share a group.
#
# ## Example
#
# ```yaml
# id: id-0007
# x: 0.0
# y: -8.0
# ```
#
# The numeric answer is hidden. Group assignments are learned from the numeric task; the
# generating rule is not a model input.
#
# For the default generator seed, id-0007 follows the rule 0.5 × x − 8. At x = 0 its underlying
# answer is −8, as illustrated before the generator adds small noise. Another identity may follow a
# sine curve or a different offset. Repeated observations let the model encounter each identity
# under changing x values.
#
# ```{typst}
# //| label: fig-proof-group-hidden-regression-regimes
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with obs, id, x, y. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("obs", kind: "root", children: (
#   node("id", type: "Group", width: 150pt,),
#   node("x", type: "Number"),
#   node("y", kind: "target", type: "Number",),
# )))
# ```
#
# ## Comparison
#
# Compare occupancy across the fixed group slots with the five generating rules. Occupancy does not establish that the correct identities were grouped.
#
# A callback records the assigned-group count and usage at each training epoch. The final five epochs
# are compared with a window around five, while checking that the count has not simply stopped at an
# allowed bound. Crucially, training and validation reuse the same observations. There is no
# independent prediction test, wrong-group control, or measured agreement with the generating
# partition here.
#
# ## Result
#
# {{< evidence P018 status >}}
#
# Archived measurements use the former Cluster contract and do not validate this Group-based
# source. A fresh Group run must establish whether assignments organize around the five rules;
# training and validation reuse the same records, so independent prediction and grouping quality
# remain untested.
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
# Group has 15 fixed assignment slots. `rf.Mask(rate=0.5, reconstruct=True)`
# trains its state head, not its group assignments; the numeric objective
# provides downstream pressure to explain `y`. Assignments belong to identities,
# while row-varying `x` does not directly condition the identity query.
#
# The model trains for 30 deterministic epochs. Training and validation
# currently use the same 3,000 observations. A public assignment snapshot
# measures occupied groups and assignment perplexity during the final five
# epochs.
#
# ### Remaining work
#
# Create independent train, validation, and test observations. Measure held-out
# regression against a marginal baseline and require permutation-invariant
# partition recovery with ARI ≥ 0.80. Add unique-identity and no-regime controls.
#
# Repeat all behavioral and partition gates across three core seeds and at least
# ten calibration seeds before promoting this proof.
#
# ### Complete experiment code
#

# %%
"""P018: track Group assignments across five hidden regression regimes.

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
    id_to_group = rng.integers(0, 5, size=100)
    identities = np.repeat(np.arange(100), 30)
    groups = id_to_group[identities]
    x = rng.uniform(0.0, 10.0, size=3000) + rng.normal(0.0, 0.05, size=3000)
    y = np.empty(3000)
    for group, function in enumerate(functions):
        selected = groups == group
        y[selected] = function(x[selected]) + rng.normal(0.0, 0.05, size=int(selected.sum()))
    for index in rng.permutation(3000):
        yield {"id": f"id-{identities[index]:04d}", "x": float(x[index]), "y": float(y[index])}


class Trajectory(lit.Callback):
    """Track public Group assignments without claiming partition accuracy."""

    def __init__(self, address: rf.Address) -> None:
        self.address = address
        self.rows: list[dict] = []

    def on_train_epoch_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        assignments = rf.Group.assignments(pl_module, self.address)
        group_ids = np.asarray([assignment["group"] for assignment in assignments.values()])
        usage = np.bincount(group_ids, minlength=15).astype(float)
        probabilities = usage / usage.sum()
        entropy = -np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0]))
        self.rows.append(
            {
                "epoch": trainer.current_epoch,
                "n_assigned_groups": int(np.count_nonzero(usage)),
                "perplexity": float(np.exp(entropy)),
            }
        )


@experiment("P018", seed=0)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        id=rf.Group(n_groups=15, mask=rf.Mask(rate=0.5, reconstruct=True)),
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
    assigned_groups = [row["n_assigned_groups"] for row in tail]
    perplexity = [row["perplexity"] for row in tail]
    return {
        "trajectory": trajectory.rows,
        "final_assigned_groups": assigned_groups,
        "final_perplexity": perplexity,
    }, {
        "Final assigned-group counts remain between 4 and 7": all(4 <= value <= 7 for value in assigned_groups),
        "Final perplexity remains within 1.5 of five": all(abs(value - 5) <= 1.5 for value in perplexity),
        "Terminal assignment count is above the lower bound": assigned_groups[-1] != 3,
        "Terminal assignment count is below the upper bound": assigned_groups[-1] != 15,
    }


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
