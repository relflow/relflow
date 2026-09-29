# %% [markdown]
# ---
# title: When Mean loses contribution count
# categories:
# - Weighted aggregation
# proof-id: P013
# description: Averaging identical encoded contributions removes the multiplicity needed
#   to recover their sum.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P013 status >}}
#
# ## Insights
#
# **This Mean configuration discards the count required to recover a sum.** With item attention disabled,
# averaging identical encoded contributions produces the same summary for one copy or many. A larger decoder
# cannot recover a distinction that no longer reaches it.
#
# The model still learns the contribution average, showing that useful content survives. Its matching sum
# predictions are the intended limitation, even though the correct totals differ. To predict variable-length
# totals, preserve multiplicity through a suitable reduction or supply an explicit count. Mean remains
# useful when repeating the whole collection should leave the desired answer unchanged.
#
# ## Setup

# %%
"""Mean is a weighted-sum footgun when item cardinality varies.

Learn the raw average while a no-attention Mean discards multiplicity.

Run: uv run python proofs/run.py P013"""

from __future__ import annotations

from collections.abc import Iterator

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P013"
ITEMS = 6

# %% [markdown]
# ## Examples
#
# Targets use `mask=True`; the labels below are supervision hidden from the encoder.
#
# ### Six copies need a larger sum
#
# ```yaml
# items:
#   - {contribution: 0.75}
#   - {contribution: 0.75}
#   - {contribution: 0.75}
#   - {contribution: 0.75}
#   - {contribution: 0.75}
#   - {contribution: 0.75}
# mean_contribution: 0.75
# weighted_sum: 4.5
# ```
#
# The correct average is 0.75 and the correct sum is 4.5.
#
# ### One copy has the same Mean summary
#
# ```yaml
# items:
#   - {contribution: 0.75}
# mean_contribution: 0.75
# weighted_sum: 0.75
# ```
#
# This is the matched count-erasure control. Its encoded Mean summary matches
# the six-copy example, so the sum predictions agree even though their correct
# labels differ. The proof does not record those individual predicted values.
#
# ### Change the contribution itself
#
# ```yaml
# items:
#   - {contribution: -0.25}
#   - {contribution: -0.25}
#   - {contribution: -0.25}
#   - {contribution: -0.25}
#   - {contribution: -0.25}
#   - {contribution: -0.25}
# mean_contribution: -0.25
# weighted_sum: -1.5
# ```
#
# Changing the repeated value gives Mean different content to encode. The
# average target changes to −0.25, but multiplicity is still unavailable to the
# sum decoder.
#
# ## Synthetic data and controls


# %%
def repeated_contribution_records(*, rows: int, seed: int) -> Iterator[dict]:
    """Draw variable counts of repeated products for the Mean counterexample."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        count = int(rng.integers(1, ITEMS + 1))
        contribution = float(rng.uniform(-1.25, 1.25))
        yield {
            "items": [{"contribution": contribution}] * count,
            "mean_contribution": contribution,
            "weighted_sum": count * contribution,
        }


def prediction(model: rf.Model, observations: list[dict], target: str) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row[f"/{target}"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["mean_contribution"] for row in test], dtype=np.float64)
    baseline_rmse = rmse(
        actual, float(np.asarray([row["mean_contribution"] for row in train], dtype=np.float64).mean())
    )
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline_rmse, "nrmse": measured / baseline_rmse}


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-mean-erases-contribution-count
# //| fig-cap: "Item attention is off and Mean removes contribution count; later root Attention cannot recover that lost distinction."
# //| fig-alt: "Record contains repeated items with contribution inputs, and hidden mean contribution, weighted sum targets. Root reduction: Attention. Item reduction: Mean; capacity 6; branch attention Off."
# #tree(node("record", kind: "root", width: 150pt, body: [
#     - *Reduction:* Attention
#   ], children: (
#   node("items", kind: "branch", repeated: true, width: 155pt, body: [
#       - *Reduction:* Mean
#       - *Branch attention:* Off
#       - *Capacity:* 6 items
#     ], children: (
#     node("contribution", type: "Number"),
#   )),
#   node("mean_contribution", width: 150pt, kind: "target", type: "Number"),
#   node("weighted_sum", kind: "target", type: "Number"),
# )))
# ```
#
# ## How it works
#
# The item branch uses `attention=None` and `rf.Mean()`. Every item in a
# record repeats the same randomly drawn contribution. Averaging its identical
# encoded tokens produces the same representation for one repetition or six.
# No later decoder can reconstruct the removed count from that representation.
#
# The model should learn the contribution average. A separate probe compares sum
# predictions for one and six copies of 0.75: those predictions must agree,
# although their correct sums are 0.75 and 4.5. The passing assertion therefore
# demonstrates an information-loss boundary, not successful sum learning.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(repeated_contribution_records(rows=1024, seed=seed + 1))
    test = list(repeated_contribution_records(rows=512, seed=seed + 3))
    model = rf.Model(
        d_model=24,
        n_layers=1,
        n_heads=4,
        batch_size=64,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.003),
        items=rf.Branch(length=ITEMS, attention=None, reduction=rf.Mean(), contribution=rf.Number),
        mean_contribution=rf.Number(mask=True, objective="mse"),
        weighted_sum=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        model=model,
        train=lambda: repeated_contribution_records(rows=1024, seed=seed + 1),
        validate=lambda: repeated_contribution_records(rows=256, seed=seed + 2),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=300 if steps is None else min(steps, 300),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=datamodule)

    # Evaluate held-out answers and retain their original labels in corruption controls.
    mean_score = score(train=train, test=test, predicted=prediction(model, test, "mean_contribution"))
    paired = [{"items": [{"contribution": 0.75}]}, {"items": [{"contribution": 0.75}] * ITEMS}]
    paired_sum = prediction(model, paired, "weighted_sum")
    metrics = {"mean_score": mean_score, "paired_sum": paired_sum.tolist(), "steps": trainer.global_step}
    checks = {
        "Mean contribution nRMSE below 0.25": bool(mean_score["nrmse"] < 0.25),
        "Mean reduction erases contribution count": bool(
            np.allclose(paired_sum[0], paired_sum[1], atol=1e-06, rtol=0.0)
        ),
    }
    return metrics, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P013 evidence >}}
#
# ## Remaining work
#
# Check the numerical invariance across seeds. Count-preserving routes are
# covered separately by [Attention sums](attention-sum-unseen-lengths.html) and
# [visible count with Mean](visible-count-sum.html); neither changes what this
# no-attention Mean configuration discards.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ## Reproduce
#
# {{< proof P013 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=3309)
