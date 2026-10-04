# %% [markdown]
# ---
# title: Can percentile-based predictions return useful original values?
# categories:
# - Quantile
# proof-id: P067
# description: Represent amounts and costs by their positions in the training distribution, then convert
#   predicted costs back to their original units.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: Original amounts
#   metric:
#   - normalized_rmse
#   format: error
# - label: Shuffled amounts
#   metric:
#   - shuffled_normalized_rmse
#   format: error
# ---
#
# ## Example
#
# ```yaml
# amount: 100.0
# cost: 506.0
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-proof-quantile-rank-regression
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with order, amount, cost. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("order", kind: "root", children: (
#   node("amount", type: "Quantile", detail: "Skewed numeric input"),
#   node("cost", kind: "target", type: "Quantile", detail: "Original-unit prediction"),
# )))
# ```
#
# ## Comparison
#
# Score new records in source units and shuffle amounts while retaining their original costs. Check that evaluation does not change the learned distributions.
#
# ## Result
#
# {{< proof P067 status >}}
#
# The learned relationship remains useful after conversion back. Shuffling removes accuracy; this does not establish behavior beyond the learned numerical range.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P067 evidence >}}
#
# ### Run this experiment
#
# {{< proof P067 script >}}
#
# ### Synthetic process and observability
#
# Draw an independent latent `u ~ Uniform(0, 1)` for each row, then set
# `amount = 50 * (exp(3u) - 1)` and
# `cost = 200 + 3 * amount + Normal(0, 10)`. The positive amounts are skewed;
# the model sees only amount, never the generating `u`. Cost is always hidden
# from embedding. Its conditional mean and median coincide, and the known
# source-unit noise RMSE is 10. A random row split is appropriate because rows
# are independent and contain no persistent identities.
#
# ### Training and controls
#
# Train the `xs` preset for 400 AdamW updates, batch size 128 and learning rate
# 0.002, using 4,096 training rows and 1,024 independently seeded validation
# rows. Validation does not select a checkpoint: evaluation uses the fixed
# final update. Test 2,048 new rows once. Both inputs and targets use the
# default digest compression; the target objective is percentile MSE.
#
# The predeclared gate is original-unit RMSE divided by the
# training-mean baseline RMSE below 0.25. Shuffling the visible input must give
# normalized RMSE above 0.90 and more than three times the intended error.
# The primary prediction receives only amount; a separate target-tampering
# check verifies the mask. Require finite predictions, exactly 128 digest
# observations per update, frozen evaluation state, and checkpoint prediction
# stability within 0.0001 source units. Counts include repeated training epochs.
#
# ### Remaining work
#
# The ten-seed calibration retained the predeclared gates and fixed training
# budget. The task tests one smooth relationship with moderate skew. It
# does not establish calibrated intervals, arbitrary tail accuracy, drift
# adaptation, or extrapolation beyond the fitted target distribution.
#
# ### Complete experiment code
#

# %%
"""P067: learn a skewed numeric relationship through Quantile representations."""

from collections.abc import Iterator
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
from reporting import report

import relflow as rf

PROOF_ID = "P067"
BUDGET = 400
BATCH_SIZE = 128
TRAIN_ROWS = 4096
TEST_ROWS = 2048


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    amount = 50.0 * np.expm1(3.0 * rng.uniform(0.0, 1.0, rows))
    cost = 200.0 + 3.0 * amount + rng.normal(0.0, 10.0, rows)
    for value, target in zip(amount, cost, strict=True):
        yield {"amount": float(value), "cost": float(target)}


def build() -> rf.Model:
    model = rf.Model.xs(
        batch_size=BATCH_SIZE,
        amount=rf.Quantile,
        cost=rf.Quantile(objective="mse", mask=True),
    )
    model.optimizer = rf.adamw(learning_rate=0.002)
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    predictions = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    return np.asarray([row["/cost"]["content"] for row in predictions], dtype=np.float64)


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, rows=TRAIN_ROWS, seed=seed + 1),
        validate=partial(records, rows=1024, seed=seed + 2),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=BUDGET if steps is None else min(steps, BUDGET),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=data)
    model.eval()
    addresses = (rf.Address("amount"), rf.Address("cost"))
    normalization = {str(address): rf.Quantile.normalization(model, address) for address in addresses}
    rows = list(records(rows=TEST_ROWS, seed=seed + 3))
    targets = np.asarray([row["cost"] for row in rows])
    visible = [{"amount": row["amount"]} for row in rows]
    training_mean = float(np.mean([row["cost"] for row in records(rows=TRAIN_ROWS, seed=seed + 1)]))
    baseline_rmse = float(np.sqrt(np.mean((training_mean - targets) ** 2)))
    output = predict(model, visible)
    shuffled = np.random.default_rng(seed + 4).permutation(TEST_ROWS)
    control = predict(model, [visible[index] for index in shuffled])
    rmse = float(np.sqrt(np.mean((output - targets) ** 2)))
    control_rmse = float(np.sqrt(np.mean((control - targets) ** 2)))
    original = predict(model, rows[:128])
    tampered = predict(model, [{**row, "cost": row["cost"] + 1e6} for row in rows[:128]])
    target_delta = float(np.max(np.abs(original - tampered)))
    evaluation_frozen = normalization == {
        str(address): rf.Quantile.normalization(model, address) for address in addresses
    }
    with TemporaryDirectory(prefix="relflow-quantile-proof-") as directory:
        path = Path(directory) / "model.pt"
        model.save(path)
        restored = rf.Model.load(path).to(model.device)
        restored.eval()
        checkpoint_delta = float(np.max(np.abs(original - predict(restored, rows[:128]))))
        checkpoint_state = normalization == {
            str(address): rf.Quantile.normalization(restored, address) for address in addresses
        }
    metrics = {
        "steps": trainer.global_step,
        "train_rows": TRAIN_ROWS,
        "test_rows": TEST_ROWS,
        "training_mean": training_mean,
        "baseline_rmse": baseline_rmse,
        "noise_rmse": 10.0,
        "rmse": rmse,
        "normalized_rmse": rmse / baseline_rmse,
        "shuffled_rmse": control_rmse,
        "shuffled_normalized_rmse": control_rmse / baseline_rmse,
        "normalization": normalization,
        "target_tampering_max_delta": target_delta,
        "checkpoint_max_delta": checkpoint_delta,
    }
    checks = {
        "Held-out source-unit normalized RMSE below 0.25": rmse / baseline_rmse < 0.25,
        "Shuffled input normalized RMSE above 0.90": control_rmse / baseline_rmse > 0.90,
        "Shuffling more than triples the RMSE": control_rmse > 3.0 * rmse,
        "Predictions in original units are finite": bool(np.isfinite(output).all() and np.isfinite(control).all()),
        "Both digests count exactly the consumed training rows": all(
            state["count"] == trainer.global_step * BATCH_SIZE for state in normalization.values()
        ),
        "Evaluation preserves both training distributions": evaluation_frozen,
        "Hidden target changes do not affect predictions": target_delta < 1e-8,
        "Checkpoint preserves both training distributions": checkpoint_state,
        "Checkpoint predictions remain within 0.0001 source units": checkpoint_delta < 1e-4,
    }
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=6701)

# %% [markdown]
# </details>
