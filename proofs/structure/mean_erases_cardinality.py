# %% [markdown]
# ---
# title: An Average Does Not Preserve Count
# categories:
# - Hierarchical statistics
# proof-id: P041
# description: Mean can preserve a scalar average while erasing the count of identical
#   repeated tokens when branch attention is disabled.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# One copy of a value and six copies have the same mean, but different totals.
# This paired case tests whether a compact branch summary preserves the
# information its parent actually needs.
#
# {{< proof P041 status >}}
#
# ## Insights
#
# **With branch attention disabled, averaging identical encoded values erases how many copies were
# present.** One amount and six copies yield the same branch summary, so a downstream decoder cannot recover
# their different totals from that summary.
#
# The recorded model learns the average but produces identical total predictions for the explicit
# one-versus-six pair. This is an expected information boundary, not simply insufficient training. Mean
# reduces encoded tokens rather than calculating a raw numerical average directly.
#
# Choose a summary that retains what the parent needs. The result specifically uses `attention=None`;
# preceding attention can change the available information. Exact business totals belong in preprocessing,
# while learned alternatives need their own behavioral checks.
#
# ## Setup

# %%
"""P041: learn an average while deliberately erasing repeated-value counts.

With branch attention disabled, Mean represents one and six identical inputs
the same way. A downstream sum therefore cannot identify their different totals.
"""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P041"

# %% [markdown]
# ## Examples
#
# Both answers are hidden Number targets. The branch uses `attention=None`
# and `rf.Mean()`, so repeated identical amounts have the same reduced summary.
#
# ### One valued item
#
# ```yaml
# transactions:
#   - amount: 1.25
# mean_amount: 1.25
# global_total: 1.25
# ```
#
# The average and total happen to be equal for a single item.
#
# ### Six copies of the same item
#
# ```yaml
# transactions:
#   - amount: 1.25
#   - amount: 1.25
#   - amount: 1.25
#   - amount: 1.25
#   - amount: 1.25
#   - amount: 1.25
# mean_amount: 1.25
# global_total: 7.5
# ```
#
# These are the different correct totals for the explicit one-versus-six
# comparison. The controlled Mean route produces identical total predictions;
# it does not recover both answers shown here.
#
# ### Another average with a different count
#
# ```yaml
# transactions:
#   - amount: 0.75
#   - amount: 0.75
#   - amount: 0.75
# mean_amount: 0.75
# global_total: 2.25
# ```
#
# Changing the common amount changes the average and its encoded summary.
# The held-out average task varies both this amount and the number of repeats,
# while the count distinction remains unavailable in the isolated Mean route.
#
# ## Synthetic data and controls


# %%
def records(*, rows: int, seed: int) -> Iterator[dict]:
    """Yield variable-length bags of equal values with mean and sum targets."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        count = int(rng.integers(1, 7))
        amount = float(rng.uniform(0.5, 1.5))
        yield {
            "transactions": [{"amount": amount} for _ in range(count)],
            "mean_amount": amount,
            "global_total": count * amount,
        }


def prediction(model: rf.Model, rows: list[dict], source: str, target: str) -> np.ndarray:
    """Predict one hidden scalar from its visible repeated context."""
    inputs = [{source: row[source]} for row in rows]
    output = model.predict(inputs).to_pylist()
    return np.asarray([row["predictions"][f"record/{target}"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-structure-mean-erases-cardinality
# //| fig-cap: "Transactions use Mean reduction with attention disabled and capacity for six items. Both numerical targets are hidden from input."
# //| fig-alt: "Record contains repeated transactions with capacity six, Mean reduction and attention set to None, a Number amount input, and Number mean amount and global total targets always hidden from input."
# #tree(node("record", kind: "root", children: (
#   node("transactions", kind: "branch", repeated: true, width: 150pt, body: [
#     - *Reduction:* Mean
#     - *Attention:* `None`
#     - *Capacity:* 6 items
#   ], children: (
#     node("amount", type: "Number"),
#   )),
#   node("mean_amount", kind: "target", type: "Number", body: [
#     - *Input:* always hidden
#   ]),
#   node("global_total", kind: "target", type: "Number", body: [
#     - *Input:* always hidden
#   ]),
# )))
# ```
#
# ## How it works
#
# The transaction branch uses `attention=None, reduction=rf.Mean()`. Its
# identical encoded amounts are averaged without preceding branch attention
# introducing count or position information. The parent sees the same summary
# for one and six repeats, so it cannot recover their different totals.
#
# The synthetic training process varies the number of repeated equal amounts
# from one to six and varies the shared amount across observations. After 300
# deterministic steps, evaluation checks the average on 512 held-out rows
# and compares predictions for the explicit one-versus-six pair. Training and
# validation contain 1,024 and 256 independently generated rows.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    data_seed = (seed - 2599) % (2**32)
    train = partial(records, rows=1024, seed=data_seed + 1)
    validate = partial(records, rows=256, seed=data_seed + 2)
    test = list(records(rows=512, seed=data_seed + 3))
    model = rf.Model(
        d_model=24,
        n_layers=1,
        n_heads=4,
        batch_size=64,
        transactions=rf.Branch(length=6, attention=None, reduction=rf.Mean(), amount=rf.Number),
        mean_amount=rf.Number(mask=True, objective="mse"),
        global_total=rf.Number(mask=True, objective="mse"),
    )
    model.optimizer = lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3)
    data = rf.SyntheticDataModule(model=model, train=train, validate=validate, seed=seed)
    trainer = lit.Trainer(
        accelerator=accelerator,
        max_epochs=-1,
        max_steps=steps if steps is not None else 300,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model=model, datamodule=data)

    test_mean = np.asarray([row["mean_amount"] for row in test])
    train_mean = np.mean([row["mean_amount"] for row in train()])
    baseline = rmse(test_mean, float(train_mean))
    mean_rmse = rmse(test_mean, prediction(model, test, "transactions", "mean_amount"))
    mean_nrmse = mean_rmse / baseline
    paired = [
        {"transactions": [{"amount": 1.25}]},
        {"transactions": [{"amount": 1.25} for _ in range(6)]},
    ]
    paired_total = prediction(model, paired, "transactions", "global_total")
    return {
        "steps": trainer.global_step,
        "mean_baseline_rmse": baseline,
        "mean_rmse": mean_rmse,
        "mean_nrmse": mean_nrmse,
        "paired_total_predictions": paired_total.tolist(),
    }, {
        "Mean normalized RMSE is below 0.25": mean_nrmse < 0.25,
        "One and six equal values produce the same total prediction": bool(
            np.allclose(paired_total[0], paired_total[1], atol=1e-6, rtol=0.0)
        ),
    }


# %% [markdown]
# ## Evidence
#
# {{< proof P041 evidence >}}
#
# ## Remaining work
#
# Broaden the value patterns and nested partitions. The family still needs a
# nested `global_total` task and a `largest_session_average` target, then three
# paired core seeds and at least ten calibration seeds for the learned gates.
#
# Use [preprocessing](../../guides/preprocessors.qmd) for exact business
# arithmetic. For learned hierarchy compression, the
# [nested Attention case](attention-preserves-nested-cardinality.html) tests a
# route that retains cardinality information.
#
# ## Reproduce
#
# {{< proof P041 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=2599)
