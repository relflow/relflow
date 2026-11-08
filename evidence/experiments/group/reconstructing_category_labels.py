# %% [markdown]
# ---
# title: Do stable labels produce the right groups?
# categories:
# - Group
# description: Repeated identities carry stable labels. Ask the Group field to assign identities while predicting
#   their labels.
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
# Repeated identities carry one of five stable labels. The model learns the hidden label and also
# reconstructs the Group state of masked identities. The experiment asks whether the assigned-group
# count and usage settle near five, rather than assuming that recurrence alone organizes the
# identities.
#
# ## Example
#
# ```yaml
# merchant_id: c0-id3
# label: L0
# ```
#
# The label is hidden; the merchant identity is sometimes masked during training.
#
# The illustrated c0-id3 identity always has label L0. Other identities such as c0-id8 share that
# label, while c1-id3 belongs to L1. These are complete synthetic identity tokens. The model is not
# given a separate group field, and the hidden label is a training target rather than a visible
# prediction input.
#
# ```{typst}
# //| label: fig-proof-group-reconstructing-category-labels
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
# Compare occupied assignment slots and their entropy with the five hidden label groups.
#
# Compared with the plain-input control, the important change is a reconstructing mask on
# merchant_id. A callback then follows commitment and usage through thirty epochs and checks the
# final five entries. Both training and validation reuse the same generated records. There is no
# independent test of predicted labels and no direct comparison between learned memberships and the
# five generating classes.
#
# ## Result
#
# {{< evidence P020 status >}}
#
# Archived measurements use the former Cluster contract and do not validate this Group-based
# source. The current fixed-slot assignments still need a full run; occupied slots and their
# entropy do not by themselves establish correct grouping. Reused validation records also leave
# independent predictive quality untested.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P020 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P020 script >}}
#
# ### How it works
#
# The merchant field uses `rf.Group(n_groups=15, mask=...)`, so it has a fixed
# set of assignment slots. Its reconstructing mask trains the state head, not
# the group assignments. The hidden category target supplies the downstream
# objective that can update assignment angles; repetition ties each assignment
# to a stable identity.
#
# Assignments belong to stable identities. Row-varying sibling context does not
# directly condition the Group identity query. The
# [plain-input control](plain-input-is-dormant.html) measures assignments without
# a reconstructing mask.
#
# The model trains for 30 deterministic epochs. A callback snapshots public
# assignments at each epoch; the gates examine occupancy and entropy over the
# final five entries.
#
# ### Remaining work
#
# Use independent observations for each split, add held-out downstream skill
# against a marginal baseline, and require adjusted Rand index (ARI) ≥ 0.80
# for partition recovery. Add unique-identity and no-regime controls.
#
# The family requires three passing core seeds and at least ten calibration
# seeds before its terminal-window thresholds can be promoted.
#
# ### Complete experiment code
#

# %%
"""P020: track Group assignments across five repeated-identity labels.

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


@experiment("P020", seed=0)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        merchant_id=rf.Group(n_groups=15, mask=rf.Mask(rate=0.5, reconstruct=True)),
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
