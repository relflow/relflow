# %% [markdown]
# ---
# title: Can a plain Group recover label groups?
# categories:
# - Group
# description: Use a Group field as input beside a hidden prediction target and require its assignments to recover
#   the label groups.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Label agreement at the final epoch
#   metric:
#   - adjusted_rand
#   - -1
#   format: number
# - label: Assigned groups at the final epoch
#   metric:
#   - assigned_groups
#   - -1
#   format: number
# evidence-group: Representation
# ---
#
# A Group field learns assignments from downstream objectives. This experiment checks whether that
# is enough to organize repeated identities. The model uses merchant_id only as an input and predicts
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
# //| label: fig-proof-group-plain-input-is-dormant
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with event, merchant_id, label. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("event", kind: "root", children: (
#   node("merchant_id", type: "Group", width: 150pt,),
#   node("label", kind: "target", type: "Category",),
# )))
# ```
#
# ## Comparison
#
# Compare public Group assignments with the stable labels when the field has no reconstruction mask.
#
# ## Result
#
# {{< evidence P019 status >}}
#
# Archived measurements use the former Cluster contract and do not validate this Group-based
# source. This run requires Group assignments to recover the stable labels.
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
# Group assignments are learned through downstream objectives, not by an
# adaptive commitment mechanism. This proof leaves off the Group reconstruction
# mask and measures whether the sibling label objective alone organizes the
# repeated identities.
#
# This model uses 1,000 observations: five label groups, ten identities per group,
# and 20 observations per identity. It trains for 1,000 deterministic epochs,
# reusing the same table for validation. A callback records the public assignment
# map and its adjusted Rand agreement with the known label groups.
#
# The paired [reconstructing-label experiment](reconstructing-category-labels.html)
# adds a mask to the Group field. That mask trains the state head; assignment
# learning still depends on the downstream label objective.
#
# ### Remaining work
#
# Repeat this positive recovery check alongside controls across three core seeds.
# The broader family still needs independent held-out observations, unique-
# identity and no-regime controls.
#
# ### Complete experiment code
#

# %%
"""P019: test whether a plain Group recovers repeated label groups.

The supervised label objective can update Group assignments without a
reconstructing mask. Require near-perfect agreement with the known labels.
"""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import experiment, report


def records(seed: int) -> Iterator[dict]:
    """Repeat each of 50 identities twenty times with one stable regime label."""
    rows = [
        {"merchant_id": f"c{group}-id{identity}", "label": f"L{group}"}
        for group in range(5)
        for identity in range(10)
        for _ in range(20)
    ]
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed)).tolist()
    for index in order:
        yield rows[index]


class Trajectory(lit.Callback):
    """Measure Group assignments learned from the supervised label target."""

    def __init__(self, address: rf.Address) -> None:
        self.address = address
        self.rows: list[dict] = []

    def on_train_epoch_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        assignments = rf.Group.assignments(pl_module, self.address)
        tokens = list(assignments)
        labels = np.asarray([int(token[1]) for token in tokens])
        groups = np.asarray([assignments[token]["group"] for token in tokens])
        self.rows.append(
            {
                "epoch": trainer.current_epoch,
                "adjusted_rand": adjusted_rand(labels, groups),
                "assigned_groups": int(np.unique(groups).size),
            }
        )


def adjusted_rand(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Measure label-group agreement without depending on assignment names."""
    _, left = np.unique(actual, return_inverse=True)
    _, right = np.unique(predicted, return_inverse=True)
    cells = np.zeros((left.max() + 1, right.max() + 1), dtype=np.int64)
    np.add.at(cells, (left, right), 1)
    joint = float(np.sum(cells * (cells - 1) / 2))
    row = cells.sum(axis=1)
    column = cells.sum(axis=0)
    left_pairs = float(np.sum(row * (row - 1) / 2))
    right_pairs = float(np.sum(column * (column - 1) / 2))
    all_pairs = len(actual) * (len(actual) - 1) / 2
    expected = left_pairs * right_pairs / all_pairs
    maximum = (left_pairs + right_pairs) / 2
    return 1.0 if maximum == expected else (joint - expected) / (maximum - expected)


@experiment("P019", seed=0)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        merchant_id=rf.Group(n_groups=5),
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
        max_epochs=1000,
        max_steps=steps if steps is not None else -1,
        callbacks=[trajectory],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )

    adjusted_rand_scores = [row["adjusted_rand"] for row in trajectory.rows]
    assigned_groups = [row["assigned_groups"] for row in trajectory.rows]
    return {
        "trajectory": trajectory.rows,
        "adjusted_rand": adjusted_rand_scores,
        "assigned_groups": assigned_groups,
    }, {
        "Plain Group assignments recover label groups at ARI 0.95 or better": all(
            value >= 0.95 for value in adjusted_rand_scores
        ),
        "More than one group is assigned": all(value > 1 for value in assigned_groups),
    }


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
