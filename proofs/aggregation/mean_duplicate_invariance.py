# %% [markdown]
# ---
# title: A mean that survives bag duplication
# categories:
# - Cardinality generalization
# proof-id: P003
# description: Learn an arithmetic mean while preserving predictions when complete items
#   are reordered or duplicated.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P003 status >}}
#
# ## Insights
#
# **With item attention disabled, Mean preserves the answer when a whole bag is repeated.** It averages
# encoded Number tokens, so duplicating every item leaves the representation unchanged. The decoder must
# still learn how that representation corresponds to the arithmetic mean of the raw amounts.
#
# That invariance is useful for an average and destructive for a total. The same duplicated bag has twice
# the sum, which cannot be recovered from an unchanged summary without additional information. This proof
# checks learned mean accuracy together with permutation and duplication stability. It does not imply that
# every architecture containing Mean discards count; earlier item interaction can change what reaches the
# reducer.
#
# ## Setup

# %%
"""Mean is invariant here with item attention disabled.

Learn average and retain it exactly under two multiset interventions.

Run: uv run python proofs/run.py P003"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P003"
TRAIN_MAX = 6
CAPACITY = 12

# %% [markdown]
# ## Examples
#
# Targets use `mask=True`; the labels below are supervision hidden from the encoder.
#
# ### Learn the raw average
#
# ```yaml
# items:
#   - {amount: 0.2}
#   - {amount: 0.8}
# mean_amount: 0.5
# ```
#
# The desired average is 0.5; Mean itself averages encoded tokens rather than raw amounts.
#
# ### Change the content
#
# ```yaml
# items:
#   - {amount: -0.2}
#   - {amount: 0.8}
# mean_amount: 0.3
# ```
#
# Changing one amount changes the desired mean to 0.3.
#
# ### Repeat the whole bag
#
# ```yaml
# items:
#   - {amount: 0.2}
#   - {amount: 0.8}
#   - {amount: 0.2}
#   - {amount: 0.8}
# mean_amount: 0.5
# ```
#
# Duplicating the first bag keeps its average at 0.5. With item attention
# disabled, the encoded Mean summary and downstream prediction stay unchanged
# to the precision required by the proof.
#
# ## Synthetic data and controls


# %%
def random_records(*, rows: int, seed: int, minimum: int = 1, maximum: int = TRAIN_MAX) -> Iterator[dict]:
    """Draw variable-length numerical bags and their mean and sum."""
    if not 1 <= minimum <= maximum <= CAPACITY:
        raise ValueError(f"length range must satisfy 1 <= minimum <= maximum <= {CAPACITY}")
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        length = int(rng.integers(minimum, maximum + 1))
        values = rng.uniform(-1.0, 1.0, size=length)
        row: dict[str, object] = {
            "items": [{"amount": float(value)} for value in values],
            "mean_amount": float(values.mean()),
            "total": float(values.sum()),
        }
        yield row


def duplicate(observations: list[dict]) -> list[dict]:
    """Duplicate every complete bag and update its algebraic targets."""
    rows = deepcopy(observations)
    if any((len(row["items"]) * 2 > CAPACITY for row in rows)):
        raise ValueError(f"duplicated collection exceeds configured capacity {CAPACITY}")
    for row in rows:
        row["items"] = [*row["items"], *row["items"]]
        row["total"] *= 2.0
    return rows


def permute(observations: list[dict], *, seed: int) -> list[dict]:
    """Jointly permute complete items without changing any target."""
    rng = np.random.default_rng(seed)
    rows = deepcopy(observations)
    for row in rows:
        items = row["items"]
        row["items"] = [items[index] for index in rng.permutation(len(items))]
    return rows


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["/mean_amount"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["mean_amount"] for row in test], dtype=np.float64)
    baseline = rmse(actual, float(np.asarray([row["mean_amount"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline, "nrmse": measured / baseline}


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-mean-duplicate-invariance
# //| fig-cap: "With item attention off, Mean preserves the encoded average under complete-bag duplication."
# //| fig-alt: "Record contains repeated items with amount inputs, and hidden mean amount targets. Root reduction: Attention. Item reduction: Mean; capacity 12; branch attention Off."
# #tree(node("record", kind: "root", width: 150pt, body: [
#     - *Reduction:* Attention
#   ], children: (
#   node("items", kind: "branch", repeated: true, width: 155pt, body: [
#       - *Reduction:* Mean
#       - *Branch attention:* Off
#       - *Capacity:* 12 items
#     ], children: (
#     node("amount", type: "Number"),
#   )),
#   node("mean_amount", kind: "target", type: "Number"),
# )))
# ```
#
# ## How it works
#
# The item branch uses `attention=None` and `rf.Mean()`. It averages encoded
# Number tokens rather than directly averaging raw amounts. A learned downstream
# path must still map that representation to the arithmetic-mean target.
#
# Without preceding item attention, permuting tokens or duplicating the complete
# bag leaves their encoded average unchanged. The test checks both invariances
# at numerical precision. This is correct for the mean target; a sum would change
# from 1.0 to 2.0 when the illustrated bag is duplicated and could not be recovered
# from the same summary without additional count information.
#
# ## Training and evaluation


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(random_records(rows=768, seed=seed + 1))
    test = list(random_records(rows=384, seed=seed + 3, maximum=3))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        items=rf.Branch(length=CAPACITY, attention=None, n_layers=2, reduction=rf.Mean(), amount=rf.Number),
        mean_amount=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        model=model,
        train=lambda: random_records(rows=768, seed=seed + 1),
        validate=lambda: random_records(rows=192, seed=seed + 2),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=350 if steps is None else min(steps, 350),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=datamodule)

    # Evaluate held-out answers and retain their original labels in corruption controls.
    intact_prediction = prediction(model, test)
    measured = score(train=train, test=test, predicted=intact_prediction)
    permutation_prediction = prediction(model, permute(test, seed=seed + 4))
    duplication_prediction = prediction(model, duplicate(test))
    scale = float(np.std(np.asarray([row["mean_amount"] for row in test], dtype=np.float64)))
    permutation_drift = rmse(intact_prediction, permutation_prediction) / scale
    duplication_drift = rmse(intact_prediction, duplication_prediction) / scale
    metrics = {
        "measured": measured,
        "permutation_drift": permutation_drift,
        "duplication_drift": duplication_drift,
        "steps": trainer.global_step,
    }
    checks = {
        "Mean nRMSE below 0.25": bool(measured["nrmse"] < 0.25),
        "Permutation drift below 1e-6 target SD": bool(permutation_drift < 1e-06),
        "Duplication drift below 1e-6 target SD": bool(duplication_drift < 1e-06),
    }
    return metrics, checks


# %% [markdown]
# ## Evidence
#
# {{< proof P003 evidence >}}
#
# ## Remaining work
#
# Repeat across seeds and test missing values, empty collections, and nested
# placement. The invariance applies to this no-item-attention path; it does not
# imply that every model using Mean is insensitive to multiplicity.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ## Reproduce
#
# {{< proof P003 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=3600)
