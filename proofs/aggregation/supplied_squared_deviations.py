# %% [markdown]
# ---
# title: Can it average prepared squared differences?
# categories:
# - Distribution statistics
# proof-id: P009
# description: Supply each value’s squared distance from the average, then predict the average of those
#   distances.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: Average prediction
#   metric:
#   - measured
#   - nrmse
#   format: error
# ---
#
# ## Example
#
# ```yaml
# items:
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
#   - {squared_deviation: 1.0}
# variance: 1.0
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-proof-supplied-squared-deviations
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, items, squared_deviation, variance. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("items", kind: "branch", repeated: true, width: 120pt, children: (
#     node("squared_deviation", width: 150pt, type: "Number"),
#   )),
#   node("variance", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Compare the prediction with the correct variance on new records.
#
# ## Result
#
# {{< proof P009 status >}}
#
# This tests the final averaging step. The inputs already contain the difficult parts of calculating variance.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P009 evidence >}}
#
# ### Run this experiment
#
# {{< proof P009 script >}}
#
# ### How it works
#
# The generator computes `(value - mean(value)) ** 2` before encoding. The
# item branch uses `attention=None` and `rf.Mean()`; the root uses learned
# attention to predict the average of eight supplied squared deviations.
#
# Mean averages encoded tokens, so the decoder still has to learn their numerical
# interpretation. Passing this diagnostic supports reduction and decoding of
# sufficient statistics. It does not prove learned centering or squaring; the
# [raw-value variance proof](raw-value-variance.html) checks those requirements.
#
# ### Remaining work
#
# Repeat this control across seeds and broaden collection lengths and edge
# cases. Keep population-versus-sample variance semantics explicit, and retain the
# raw-value comparison when claiming that dispersion was learned from inputs.
#
# The family’s promotion target is at least three core seeds and ten lightweight
# calibration seeds.
#
# ### Complete experiment code
#

# %%
"""Supplied squared deviations isolate collection averaging.

Mean learns population variance once squared deviations are supplied.

Run: uv run python proofs/run.py P009"""

from __future__ import annotations

from collections.abc import Iterator

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P009"
ITEMS = 8


def dispersion_records(*, rows: int, seed: int) -> Iterator[dict]:
    """Draw rows whose location and spread are statistically independent."""
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        basis = rng.normal(size=ITEMS)
        basis -= basis.mean()
        basis /= np.sqrt(np.mean(np.square(basis)))
        location = float(rng.uniform(-0.8, 0.8))
        scale = float(rng.uniform(0.12, 1.25))
        values = location + scale * basis
        items = [{"squared_deviation": float(np.square(value - location))} for value in values]
        yield {"items": items, "variance": scale * scale}


def prediction(model: rf.Model, observations: list[dict]) -> np.ndarray:
    inputs = [{"items": row["items"]} for row in observations]
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["/variance"]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def score(*, train: list[dict], test: list[dict], predicted: np.ndarray) -> dict[str, float]:
    """Compare held-out RMSE with the constant training-target mean."""
    actual = np.asarray([row["variance"] for row in test], dtype=np.float64)
    baseline = rmse(actual, float(np.asarray([row["variance"] for row in train], dtype=np.float64).mean()))
    measured = rmse(actual, predicted)
    return {"rmse": measured, "baseline_rmse": baseline, "nrmse": measured / baseline}


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    # Split seeds are independent; rerunning a generator reproduces the same records.
    train = list(dispersion_records(rows=768, seed=seed + 1))
    test = list(dispersion_records(rows=384, seed=seed + 3))
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=0.001),
        items=rf.Branch(length=ITEMS, n_layers=2, attention=None, reduction=rf.Mean(), squared_deviation=rf.Number),
        variance=rf.Number(mask=True, objective="mse"),
    )
    datamodule = rf.SyntheticDataModule(
        model=model,
        train=lambda: dispersion_records(rows=768, seed=seed + 1),
        validate=lambda: dispersion_records(rows=192, seed=seed + 2),
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
    measured = score(train=train, test=test, predicted=prediction(model, test))
    metrics = {"measured": measured, "steps": trainer.global_step}
    checks = {"Variance nRMSE below 0.20": bool(measured["nrmse"] < 0.2)}
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=3500)

# %% [markdown]
# </details>
