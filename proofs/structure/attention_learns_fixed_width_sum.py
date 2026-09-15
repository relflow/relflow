# %% [markdown]
# ---
# title: Learning a Fixed-Length Sum
# categories:
# - Hierarchical statistics
# proof-id: P039
# description: A learned Attention reduction can retain enough information to approximate
#   the sum of six visible values.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# Before testing a nested hierarchy, check whether the model can learn a sum
# at one level. Every observation here contains six independently drawn
# amounts, all of which contribute to the answer.
#
# {{< proof P039 status >}}
#
# ## Insights
#
# **The model can learn to approximately add six visible numbers.** All six independently drawn values are
# visible, making this a useful positive control before testing more complicated nested statistics.
#
# The accuracy check compares held-out error with a constant training-mean baseline. Meeting it demonstrates
# substantially lower error, but this experiment does not include a prediction-level permutation control.
# The task always contains six items.
#
# Do not extend this result to variable cardinality, arbitrary hierarchy depth, or exact arithmetic. The
# nested-cardinality case asks whether local structure survives successive reductions; this control only
# establishes that the model can learn a numerical sum at one level.
#
# ## Setup

# %%
"""P039: learn a fixed-width sum through Attention reduction.

All six independent values must contribute. Test RMSE is compared with the
constant training-mean predictor; learned sums remain approximations.
"""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P039"

# %% [markdown]
# ## Examples
#
# Each record contains exactly six visible amounts. `global_total` is their
# mathematical sum, supplied as a hidden Number target rather than an input.
#
# ### Mixed signs with a positive total
#
# ```yaml
# transactions:
#   - amount: 0.8
#   - amount: -0.4
#   - amount: 0.2
#   - amount: 0.5
#   - amount: -0.5
#   - amount: 0.9
# global_total: 1.5
# ```
#
# Positive and negative contributions must both reach the decoder.
#
# ### One contribution changes sign
#
# ```yaml
# transactions:
#   - amount: 0.8
#   - amount: -0.4
#   - amount: 0.2
#   - amount: 0.5
#   - amount: -0.5
#   - amount: -0.9
# global_total: -0.3
# ```
#
# Flipping the final amount changes the total by −1.8. This illustrates the
# sum rule; the executable proof scores independently generated records rather
# than asserting this particular paired intervention.
#
# ### Cancellation without empty input
#
# ```yaml
# transactions:
#   - amount: 0.8
#   - amount: -0.8
#   - amount: 0.3
#   - amount: -0.3
#   - amount: -0.6
#   - amount: 0.6
# global_total: 0.0
# ```
#
# A zero total can arise from six present, nonzero values. These exact answers
# illustrate the target function; the learned model is assessed by aggregate
# approximation error, not exact predictions on these examples.
#
# ## Synthetic data and controls


# %%
def records(*, rows: int, seed: int) -> Iterator[dict]:
    """Yield six transactions and their total."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        amounts = rng.uniform(-1.0, 1.0, size=6)
        yield {
            "transactions": [{"amount": float(amount)} for amount in amounts],
            "global_total": float(amounts.sum()),
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
# //| label: fig-proof-structure-attention-learns-fixed-width-sum
# //| fig-cap: "The transaction branch and root each reduce to one Attention output. Six visible values supply the hidden total target."
# //| fig-alt: "Record and its repeated transactions each use Attention reduction with one output; the branch has capacity for six Number amounts, and the Number global total target is always hidden from input."
# #tree(node("record", kind: "root", width: 150pt, body: [
#   - *Reduction:* Attention
#   - *Outputs:* 1
# ], children: (
#   node("transactions", kind: "branch", repeated: true, width: 150pt, body: [
#     - *Reduction:* Attention
#     - *Outputs:* 1
#     - *Capacity:* 6 items
#   ], children: (
#     node("amount", type: "Number"),
#   )),
#   node("global_total", kind: "target", type: "Number", body: [
#     - *Input:* always hidden
#   ]),
# )))
# ```
#
# ## How it works
#
# Both the transaction branch and the root use `rf.Attention` reduction.
# The model must transform the visible amounts into a summary from which a
# numerical decoder can approximate their sum. Fixed length makes this a
# simpler control than inferring totals through several variable-size groups.
#
# Training, validation, and test draw 512, 128, and 256 independent observations.
# Each amount is sampled uniformly from −1 to 1. After 400 deterministic
# steps, the model predicts totals from the transaction inputs alone.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    data_seed = (seed - 2600) % (2**32)
    train = partial(records, rows=512, seed=data_seed + 11)
    validate = partial(records, rows=128, seed=data_seed + 22)
    test = list(records(rows=256, seed=data_seed + 33))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        batch_size=64,
        transactions=rf.Branch(
            length=6,
            n_layers=2,
            reduction=rf.Attention(n_layers=2),
            amount=rf.Number,
        ),
        global_total=rf.Number(mask=True, objective="mse"),
    )
    model.optimizer = lambda module: torch.optim.Adam(module.parameters(), lr=1e-3)
    data = rf.SyntheticDataModule(model=model, train=train, validate=validate, seed=seed)
    trainer = lit.Trainer(
        accelerator=accelerator,
        max_epochs=-1,
        max_steps=steps if steps is not None else 400,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model=model, datamodule=data)

    train_target = np.asarray([row["global_total"] for row in train()])
    validate_rows = list(validate())
    validate_target = np.asarray([row["global_total"] for row in validate_rows])
    test_target = np.asarray([row["global_total"] for row in test])
    baseline = rmse(test_target, float(train_target.mean()))
    validate_rmse = rmse(validate_target, prediction(model, validate_rows, "transactions", "global_total"))
    test_rmse = rmse(test_target, prediction(model, test, "transactions", "global_total"))
    nrmse = test_rmse / baseline
    return {
        "steps": trainer.global_step,
        "baseline_rmse": baseline,
        "validate_rmse": validate_rmse,
        "test_rmse": test_rmse,
        "test_nrmse": nrmse,
        "target_std": float(test_target.std()),
    }, {"Test normalized RMSE is below 0.25": nrmse < 0.25}


# %% [markdown]
# ## Evidence
#
# {{< proof P039 evidence >}}
#
# ## Remaining work
#
# Add the nested, regrouping-invariant `global_total` case: the present control
# has only one collection level. The family also calls for a
# `largest_session_average` target, broader value patterns, multiple Attention
# output counts, and nested capacity ranges, followed by repeated-seed gates.
#
# The [nested-cardinality proof](attention-preserves-nested-cardinality.html)
# tests a statistic whose answer changes when those same amounts are regrouped.
# Use [preprocessing](../../guides/preprocessors.qmd) when a sum must be exact.
#
# ## Reproduce
#
# {{< proof P039 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=2600)
