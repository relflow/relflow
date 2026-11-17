# %% [markdown]
# ---
# title: Does input jitter buy robustness at a reasonable clean-data cost?
# categories:
# - Representation
# description: Compare zero, modest, and excessive input jitter on a small fixed regression dataset with independent
#   nuisance coordinates.
# evidence-readout:
# - label: No jitter, noisy error
#   metric:
#   - arms
#   - none
#   - conditions
#   - strong_noise
#   - nrmse
#   format: number
# - label: Small jitter, noisy error
#   metric:
#   - arms
#   - small
#   - conditions
#   - strong_noise
#   - nrmse
#   format: number
# - label: Small jitter, clean error
#   metric:
#   - arms
#   - small
#   - conditions
#   - clean
#   - nrmse
#   format: number
# evidence-group: Representation
# evidence-diagram:
#   caption: The target is hidden. The diagram shows the informative representation; controls change only the declared
#     condition.
#   alt: Does input jitter buy robustness at a reasonable clean-data cost? Visible inputs lead to one hidden prediction
#     target.
# evidence-review:
#   question: Does input jitter buy robustness at a reasonable clean-data cost?
#   modeling: Fit identical regression models on the same small training set with no, modest, or excessive input jitter.
#     Test clean and noisy versions of the same held-out records while keeping targets unchanged. Compare noisy-input
#     error first, then inspect the clean-data cost.
#   comparison:
#     metric: Normalized RMSE
#     kind: Controlled comparison
#     conditions:
#     - label: No jitter · noisy test inputs
#       path:
#       - arms
#       - none
#       - conditions
#       - strong_noise
#       - nrmse
#       format: number
#     - label: Modest jitter · same noisy inputs
#       path:
#       - arms
#       - small
#       - conditions
#       - strong_noise
#       - nrmse
#       format: number
#     - label: Excessive jitter · same noisy inputs
#       path:
#       - arms
#       - excessive
#       - conditions
#       - strong_noise
#       - nrmse
#       format: number
#     definition: Root mean squared error divided by the reference scale defined in this experiment. Zero is exact;
#       one equals that reference scale.
#     direction: Lower error is better
#   note: Robustness gains are measured, not required by the pass criteria. Checks met alone does not establish that
#     jitter improved generalization.
# evidence-plot:
#   target: target
#   focus: When does training jitter help or harm generalization?
#   comparison: Compare clean and perturbed test conditions at each declared jitter level.
#   limit: No test condition selects the training configuration; the clean-data cost stays visible.
#   renderer: plot
# ---
#
# Compare zero, modest, and excessive input jitter on a small fixed regression dataset with independent
# nuisance coordinates.
#
# ## Example
#
# ```yaml
# signal: 0.5
# nuisance: [0.2, -0.7, 1.1, 0.4, -0.3, 0.0, 0.8, -1.0]
# target: 0.9915
# ```
#
# The answer follows sin(2 × signal) + 0.3 × signal, plus small independent noise. At signal 0.5 the
# noiseless answer is approximately 0.9915. The eight nuisance coordinates have no role in that rule.
#
# An extra noisy reading changes what the model observes, while the underlying answer stays the same.
# Jitter may encourage smoother predictions, but enough jitter also removes real information.
#
# ```typst
# #tree(node("record", kind: "root", children: (node("signal", type: "Number", detail: "Training jitter"), node("nuisance", type: "Vector(8)"), node("target", kind: "target", type: "Number"),)))
# ```
#
# ## Comparison
#
# Keep samples, initialization, optimizer, and update count matched across no-jitter, modest-jitter, and
# excessive-jitter arms. Evaluate clean observations and two measurement-noise levels using identical
# held-out latent records. Targets remain pristine.
#
# ## Result
#
# The numerical checks below are provisional, predeclared hypotheses, not measured successes. No full
# learning result is claimed until a matching run is recorded. Smoke runs establish execution only.
#
# ## Method
#
# ### Population, controls, and scoring
#
# Each arm receives 512 fixed training and 512 validation records, 500 updates, batch 64, and 2,048
# independent test records. Validation is not used for checkpoint selection. Number jitter is
# normalized; Vector jitter uses its continuous representation. Measurement noise has fixed source-unit
# scales, so it is a declared stress condition rather than a claim of exact distribution matching.
# Report clean and noisy errors plus the train-test gap. Checks require finite execution and basic
# learning; the sign of the paired robustness benefit is a research result, not an assumed win.
#

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.10",
#     "lightning",
#     "numpy",
#     "polars",
#     "pyarrow",
#     "pyyaml>=6",
#     "relflow>=0.5.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "reliability"
# types = ["Number"]
# conditions = ["noise"]
# ///

"""P090: measure clean/noisy generalization tradeoffs from input jitter."""

import sys
from collections.abc import Iterator
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
from policies import CONTROL
from reporting import experiment, report

BUDGET = 500
TRAIN_ROWS = 512
TEST_ROWS = 2048
LEVELS = {"none": 0.0, "small": 0.08, "excessive": 1.5}


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    signals = rng.uniform(-1, 1, rows)
    nuisance = rng.normal(size=(rows, 8))
    target = np.sin(2 * signals) + 0.3 * signals + rng.normal(0, 0.03, rows)
    for signal, noise, answer in zip(signals, nuisance, target, strict=True):
        yield {"signal": float(signal), "nuisance": noise.tolist(), "target": float(answer)}


def perturb(rows: list[dict], *, seed: int, scale: float) -> list[dict]:
    rng = np.random.default_rng(seed)
    signal_noise = rng.normal(0, scale, len(rows))
    nuisance_noise = rng.normal(0, scale, (len(rows), 8))
    return [
        {
            "signal": row["signal"] + float(signal_noise[index]),
            "nuisance": (np.asarray(row["nuisance"]) + nuisance_noise[index]).tolist(),
            "target": row["target"],
        }
        for index, row in enumerate(rows)
    ]


def prediction(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{key: value for key, value in row.items() if key != "target"} for row in rows]
    output = pl.DataFrame(model.predict(pa.Table.from_pylist(visible), batch_size=256))["predictions"].to_list()
    return np.asarray([row["target"]["content"] for row in output])


def fit(model: rf.Model, train: pa.Table, validate: pa.Table, seed: int, steps: int | None, accelerator: str) -> int:
    """Fit the declared fixed-data protocol without checkpoint selection."""
    data = rf.ArrowDataModule(train=train, validate=validate, num_workers=0, persistent_workers=False, pin_memory=False)
    model.fit(
        data,
        batch_size=64,
        learning_rate=0.003,
        weight_decay=0.01,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=BUDGET if steps is None else min(steps, BUDGET),
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    model.eval()
    return int(model.trainer.global_step)


@experiment("P090", seed=9001)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train = list(records(rows=TRAIN_ROWS, seed=seed + 1))
    validate = list(records(rows=512, seed=seed + 2))
    test = list(records(rows=TEST_ROWS, seed=seed + 3))
    actual = np.asarray([row["target"] for row in test])
    baseline = float(np.sqrt(np.mean((actual - np.mean([row["target"] for row in train])) ** 2)))
    metrics = {"budget": BUDGET, "train_rows": TRAIN_ROWS, "arms": {}}
    checks = {}
    for name, level in LEVELS.items():
        lit.seed_everything(seed, workers=True)
        jitter = rf.Jitter(add=level)

        class Model(rf.Model):
            signal = rf.Number(jitter=jitter)
            nuisance = rf.Vector(n_dim=8, jitter=jitter)
            target = rf.Number(mask=True, objective="mse")

        model = Model(preset=CONTROL.name)
        updates = fit(model, pa.Table.from_pylist(train), pa.Table.from_pylist(validate), seed, steps, accelerator)
        fitted = prediction(model, train)
        arm = {
            "steps": updates,
            "train_rmse": float(np.sqrt(np.mean((fitted - np.asarray([row["target"] for row in train])) ** 2))),
            "conditions": {},
        }
        for condition, scale in (("clean", 0.0), ("mild_noise", 0.08), ("strong_noise", 0.25)):
            selected = perturb(test, seed=seed + 4, scale=scale)
            predicted = prediction(model, selected)
            arm["conditions"][condition] = {
                "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
                "nrmse": float(np.sqrt(np.mean((predicted - actual) ** 2))) / baseline,
                "finite": bool(np.isfinite(predicted).all()),
            }
            checks[f"{name}, {condition}: finite predictions"] = arm["conditions"][condition]["finite"]
        arm["clean_test_minus_train_rmse"] = arm["conditions"]["clean"]["rmse"] - arm["train_rmse"]
        metrics["arms"][name] = arm
        checks[f"{name}: full budget retained"] = updates == BUDGET
    for name in ("none", "small"):
        checks[f"{name}: clean nRMSE below 0.50"] = metrics["arms"][name]["conditions"]["clean"]["nrmse"] < 0.50
    metrics["small_minus_none_strong_noise_nrmse"] = (
        metrics["arms"]["small"]["conditions"]["strong_noise"]["nrmse"]
        - metrics["arms"]["none"]["conditions"]["strong_noise"]["nrmse"]
    )
    metrics["small_minus_none_clean_nrmse"] = (
        metrics["arms"]["small"]["conditions"]["clean"]["nrmse"]
        - metrics["arms"]["none"]["conditions"]["clean"]["nrmse"]
    )
    return metrics, checks


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    import numpy as np

    from evidence.diagnostics.plots import axes_pair, dots

    m = bundle.record["metrics"]["arms"]
    arms = ["none", "small", "excessive"]
    conditions = ["clean", "mild_noise", "strong_noise"]
    fig, axes = axes_pair(width=10, height=4.8)
    matrix = np.array([[m[a]["conditions"][c]["nrmse"] for c in conditions] for a in arms])
    image = axes[0].imshow(matrix, cmap="YlOrRd", vmin=0, vmax=max(1, float(matrix.max())), aspect="auto")
    axes[0].set(
        xticks=range(3),
        xticklabels=["Clean", "Mild noise", "Strong noise"],
        yticks=range(3),
        yticklabels=["No jitter", "Small jitter (0.08)", "Excessive jitter (1.5)"],
        xlabel="Held-out input condition",
        title="Every jitter × evaluation condition",
    )
    for i, j in np.ndindex(matrix.shape):
        axes[0].text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="#243b35")
    fig.colorbar(image, ax=axes[0], fraction=0.035, label="Test nRMSE (lower is better)")
    dots(
        axes[1],
        ["No jitter", "Small jitter", "Excessive jitter"],
        {
            "Training RMSE": [m[a]["train_rmse"] for a in arms],
            "Clean held-out RMSE": [m[a]["conditions"]["clean"]["rmse"] for a in arms],
        },
        xlabel="RMSE in original target units",
    )
    return (
        fig,
        "The matrix includes all nine measured conditions, with 1 representing constant-guess error. The right panel separates training and clean held-out error on the same source-unit scale. Jitter stops at evaluation; adding too much can degrade the learned relationship.",
        {},
    )


if __name__ == "__main__":
    report(run)
