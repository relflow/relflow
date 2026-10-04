# %% [markdown]
# ---
# title: Can it add six numbers?
# categories:
# - Hierarchical statistics
# proof-id: P039
# description: Predict the total of six independently chosen amounts.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: Total prediction
#   metric:
#   - test_nrmse
#   format: error
# ---
#
# ## Example
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
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-proof-structure-attention-learns-fixed-width-sum
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, transactions, amount, global_total. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("transactions", kind: "branch", repeated: true, width: 120pt, children: (
#     node("amount", type: "Number"),
#   )),
#   node("global_total", kind: "target", type: "Number",),
# )))
# ```
#
# ## Comparison
#
# Compare test error with a simple predictor that always returns the training-set average total.
#
# ## Result
#
# {{< proof P039 status >}}
#
# The model learns this small sum. This baseline does not test item reordering, varying lengths, deeper nesting, or exact addition.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P039 evidence >}}
#
# ### Run this experiment
#
# {{< proof P039 script >}}
#
# ### How it works
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
# ### Remaining work
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
# ### Complete experiment code
#

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
    return np.asarray([row["predictions"][f"/{target}"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


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


if __name__ == "__main__":
    report(PROOF_ID, run, seed=2600)

# %% [markdown]
# </details>
