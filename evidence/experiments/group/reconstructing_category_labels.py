# %% [markdown]
# ---
# title: Can a masked Group recover label groups?
# categories:
# - Group
# description: Mask and reconstruct the Group state while a hidden prediction target trains assignments to recover
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
# - label: Held-out label accuracy
#   metric: [heldout_accuracy]
#   format: percent
# - label: Shuffled-identity accuracy
#   metric: [shuffled_accuracy]
#   format: percent
# evidence-group: Representation
# ---
#
# A Group field learns assignments from downstream objectives. This experiment checks whether that
# is enough to organize repeated identities. The model predicts a hidden label while masking the Group identity state on half of training rows.
# The state reconstruction objective and downstream label objective train together.
#
# ## Example
#
# ```yaml
# merchant_id: c2-id7
# context: 0.3
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
# //| label: fig-proof-group-reconstructing-category-labels
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with event, merchant_id, label. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("event", kind: "root", children: (
#   node("merchant_id", type: "Group", width: 150pt,),
#   node("context", type: "Number"),
#   node("label", kind: "target", type: "Category",),
# )))
# ```
#
# ## Comparison
#
# Compare final assignments and held-out label predictions with a test-time
# identity-shuffle control. Independent context values carry no label signal.
#
# ## Result
#
# {{< evidence P110 status >}}
#
# Archived measurements use the former Cluster contract and do not validate this Group-based
# source. This run requires Group assignments to recover the stable labels.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P110 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P110 script >}}
#
# ### How it works
#
# Group assignments are learned through downstream objectives, not by an
# adaptive commitment mechanism. This proof adds a Group reconstruction mask and measures whether the sibling
# label objective still organizes repeated identities while the state head trains.
#
# The fixed generator has five label groups and ten known identities per group.
# Training, validation and test use independently generated nuisance context
# values, while sharing identities. Training uses 20 records per identity and
# validation and test each use four. The model trains for 4,800 updates with
# batch size 64, AdamW at 0.003 and EMA 0.99. A callback records assignment
# agreement each epoch, but the ARI criterion applies to the final assignment.
# Require final ARI at least 0.95, held-out accuracy at least 0.90, and
# shuffled-identity accuracy at most 0.35. These criteria are hypotheses, not
# reused measurements from the old P019 proof.
#
# The [plain-input experiment](plain-input-is-dormant.html) leaves the Group
# state unmasked. Both experiments use independent nuisance contexts and assess
# known identities with an identity-shuffle control.
#
# ### Remaining work
#
# This proof tests new observations of known identities, not unseen identities.
# The default seed is 11001; fresh confirmation seeds 11002 and 11003 are
# declared before evaluation. They are separate experiments, not retries.
#
# ### Complete experiment code
#

# %%
"""P110: test whether a masked Group recovers repeated label groups.

A reconstructing mask trains the Group state head alongside the supervised
label objective. Require near-perfect agreement with the known labels.
"""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
from reporting import experiment, report

BUDGET = 4800


def records(seed: int, repeats: int = 20) -> Iterator[dict]:
    """Generate independent nuisance context for each repeated identity."""
    rng = np.random.default_rng(seed)
    rows = [
        {"merchant_id": f"c{group}-id{identity}", "context": float(rng.uniform(-1, 1)), "label": f"L{group}"}
        for group in range(5)
        for identity in range(10)
        for _ in range(repeats)
    ]
    for index in rng.permutation(len(rows)):
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


@experiment("P110", seed=11001)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        merchant_id=rf.Group(n_groups=5, mask=rf.Mask(rate=0.5, reconstruct=True), p_unavailable=0.0),
        context=rf.Number,
        label=rf.Category(mask=True, p_unavailable=0.0),
    )
    data = rf.SyntheticDataModule(
        train=partial(records, seed=seed * 100 + 1),
        validate=partial(records, seed=seed * 100 + 2, repeats=4),
        seed=seed,
    )
    trajectory = Trajectory(rf.Address("merchant_id"))
    model.fit(
        data,
        batch_size=64,
        learning_rate=0.003,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=BUDGET if steps is None else min(steps, BUDGET),
        callbacks=[rf.EMA(decay=0.99), trajectory],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    model.eval()
    state = rf.Group.assignments(model, rf.Address("merchant_id"))
    test = list(records(seed * 100 + 3, repeats=4))
    order = np.random.default_rng(seed * 100 + 4).permutation(len(test))
    shuffled = [{**row, "merchant_id": test[index]["merchant_id"]} for row, index in zip(test, order, strict=True)]

    def accuracy(rows):
        visible = pl.DataFrame([{key: row[key] for key in ("merchant_id", "context")} for row in rows])
        output = model.predict(visible)["predictions"].to_list()
        return float(
            np.mean([out["label"]["content"]["value"] == row["label"] for out, row in zip(output, rows, strict=True)])
        )

    heldout_accuracy = accuracy(test)
    shuffled_accuracy = accuracy(shuffled)
    scores = [row["adjusted_rand"] for row in trajectory.rows]
    assigned = [row["assigned_groups"] for row in trajectory.rows]
    final_ari = scores[-1]
    return {
        "trajectory": trajectory.rows,
        "adjusted_rand": scores,
        "assigned_groups": assigned,
        "final_ari": final_ari,
        "heldout_accuracy": heldout_accuracy,
        "shuffled_accuracy": shuffled_accuracy,
        "assigned_identities": len(state),
        "steps": model.trainer.global_step,
    }, {
        "Final assignments recover labels with ARI at least 0.95": final_ari >= 0.95,
        "Held-out known-identity accuracy is at least 0.90": heldout_accuracy >= 0.90,
        "Shuffled-identity accuracy is at most 0.35": shuffled_accuracy <= 0.35,
        "All identities are assigned and evaluation preserves assignments": len(state) == 50
        and state == rf.Group.assignments(model, rf.Address("merchant_id")),
    }


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
