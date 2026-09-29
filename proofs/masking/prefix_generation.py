# %% [markdown]
# ---
# title: Prefix-only prediction and sequence rollout
# categories: [Dynamic masking]
# proof-id: P054
# description: Hide a sequence suffix, reconstruct only its next value, and feed predictions back for a short numerical rollout.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P054 status >}}
#
# ## Insights
#
# Dynamic selectors can impose a prefix-only encoder boundary for next-value
# prediction. The same model can then attempt a short autoregressive rollout.
# One prefix is exposed per observation; this is not a triangular attention mask
# that trains all next-token positions simultaneously. It tests numerical
# sequences of a fixed maximum length, not language generation, arbitrary
# forecasting, or causal discovery. A failed learning gate does not by itself
# imply that the visibility boundary leaked.
#
# ## Setup

# %%
"""P054: leakage-free prefix conditioning and free-running arithmetic-sequence rollout."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
from reporting import report

import relflow as rf

PROOF_ID = "P054"
LENGTH = 5
BUDGET = 800

# %% [markdown]
# ## Data and model
#
# A whole sequence draws `offset ~ Uniform(-1, 1)` and
# `increment ~ Uniform(-0.4, 0.4)`, then `value[t] = offset + t*increment`.
# Neither latent parameter is exposed. Choose a prefix length uniformly from
# 2, 3, and 4. The branch skips every suffix item; a leaf query requests only
# the first hidden value. Whole sequences are independently generated for
# training (4,096), validation (512), and testing (1,024).
#
# ```yaml
# events:
#   - {value: 0.2, future: false, next: false}
#   - {value: 0.5, future: false, next: false}
#   - {value: 0.8, future: true, next: true}
#   - {value: 1.1, future: true, next: false}
#   - {value: 1.4, future: true, next: false}
# ```
#
# Only 0.2 and 0.5 are encoder inputs; 0.8 is the single reconstruction label.
# Later values are neither encoder inputs nor reconstruction objectives.
# Standard field normalization observes pristine training values, including
# skipped positions, then stays frozen during evaluation. This is a
# sequence-held-out experiment, not chronological train/test evaluation.
# At generation time all suffix placeholders are zero and newly predicted
# values become visible one by one.
#
# ```yaml
# events:
#   - {value: 0.2, future: false, next: false}
#   - {value: 0.5, future: false, next: false}
#   - {value: 1002.0, future: true, next: true}
#   - {value: 1003.0, future: true, next: false}
#   - {value: 1004.0, future: true, next: false}
# ```
#
# This poisoned suffix must give the same next prediction. The original 0.8
# remains the evaluator's expected value.
#
# ```yaml
# events:
#   - {value: 0.2, future: false, next: false}
#   - {value: 0.5, future: false, next: false}
#   - {value: 0.79, future: false, next: false}
#   - {value: 0.0, future: true, next: true}
#   - {value: 0.0, future: true, next: false}
# ```
#
# A rollout might predict 0.79 instead of 0.8. The following step must use that
# generated value, so errors can accumulate; the rollout gate measures this.
#
# ```{typst}
# //| label: fig-proof-prefix-generation
# //| fig-cap: "A branch skip hides the suffix; a leaf query reconstructs just its first coordinate."
# //| fig-alt: "Record contains repeated events. The future selector skips an entire event; Number value is reconstructed only when next is true."
# #tree(node("record", kind: "root", children: (
#   node("events", kind: "branch", repeated: true, body: [Skip when `future`; no reduction], children: (
#     node("value", type: "Number", body: [Reconstruct when `next`]),
#   )),
# )))
# ```


# %%
def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        offset = float(rng.uniform(-1.0, 1.0))
        increment = float(rng.uniform(-0.4, 0.4))
        cut = int(rng.integers(2, LENGTH))
        yield {
            "events": [
                {"value": offset + index * increment, "future": index >= cut, "next": index == cut}
                for index in range(LENGTH)
            ]
        }


def build() -> rf.Model:
    return rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        batch_size=64,
        events=rf.Branch(
            length=LENGTH,
            reduction=None,
            mask=rf.Mask(query="future", skip=True, dropout=False),
            value=rf.Number(objective="mse", mask=rf.Mask(query="next", reconstruct=True)),
        ),
    )


def prediction(model: rf.Model, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    output = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    content = np.asarray(
        [[item["content"] for item in row["/events/value"]] for row in output], dtype=np.float64
    ).reshape(len(rows), LENGTH)
    inferred = np.asarray([[item["inferred"] for item in row["/events/value"]] for row in output], dtype=bool).reshape(
        len(rows), LENGTH
    )
    return content, inferred


def rollout(model: rf.Model, rows: list[dict]) -> np.ndarray:
    """Expose generated values only; never teacher-force the held-out suffix."""
    values = np.zeros((len(rows), LENGTH), dtype=np.float64)
    values[:, :2] = [[event["value"] for event in row["events"][:2]] for row in rows]
    for cut in range(2, LENGTH):
        inputs = [
            {
                "events": [
                    {"value": float(value), "future": index >= cut, "next": index == cut}
                    for index, value in enumerate(sequence)
                ]
            }
            for sequence in values
        ]
        content, _ = prediction(model, inputs)
        values[:, cut] = content[:, cut]
    return values


# %% [markdown]
# ## Training and controls
#
# Fit for 800 AdamW updates at learning rate 0.002. Score one-step predictions
# separately for all three prefix lengths, requiring nRMSE below 0.35. A rollout
# from the first two true values must achieve suffix nRMSE below 0.45 and beat
# persistence (repeating the second value). Baselines use training means at each
# position, not held-out statistics. Permuting entire visible prefixes between
# records retains the original labels and must give nRMSE above 0.85.
#
# Poison all suffix values, including the selected target: drift must remain
# below 1e-5. Training objectives and inference flags must select exactly one
# coordinate, while encoder presence selects exactly the prefix. The supplied
# correct suffix is used only as an evaluation label, never in the rollout.


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    model.optimizer = rf.adamw(learning_rate=0.002)
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, rows=4096, seed=seed + 1),
        validate=partial(records, rows=512, seed=seed + 2),
        seed=seed,
    )
    trainer = lit.Trainer(
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
    trainer.fit(model, datamodule=data)
    model.eval()
    train = list(records(rows=4096, seed=seed + 1))
    test = list(records(rows=1024, seed=seed + 3))
    actual = np.asarray([[event["value"] for event in row["events"]] for row in test])
    mean = np.asarray([[event["value"] for event in row["events"]] for row in train]).mean(axis=0)
    order = np.random.default_rng(seed + 4).permutation(len(test))
    metrics = {"steps": trainer.global_step, "prefixes": {}}
    checks = {}
    # A separate model audits target selection without fitting held-out moments.
    probe = build()
    for cut in range(2, LENGTH):
        selected = [
            {
                "events": [
                    {**event, "future": index >= cut, "next": index == cut} for index, event in enumerate(row["events"])
                ]
            }
            for row in test
        ]
        content, inferred = prediction(model, selected)
        poisoned = [
            {
                "events": [
                    {**event, "value": 1000.0 + index} if index >= cut else event
                    for index, event in enumerate(row["events"])
                ]
            }
            for row in selected
        ]
        shuffled = [
            {
                "events": [
                    {**event, "value": float(actual[other, index])} if index < cut else event
                    for index, event in enumerate(row["events"])
                ]
            }
            for row, other in zip(selected, order, strict=True)
        ]
        replaced, _ = prediction(model, poisoned)
        broken, _ = prediction(model, shuffled)
        baseline = float(np.sqrt(np.mean((actual[:, cut] - mean[cut]) ** 2)))
        measured = {
            "nrmse": float(np.sqrt(np.mean((content[:, cut] - actual[:, cut]) ** 2))) / baseline,
            "baseline_rmse": baseline,
            "shuffled_nrmse": float(np.sqrt(np.mean((broken[:, cut] - actual[:, cut]) ** 2))) / baseline,
            "future_value_drift": float(np.max(np.abs(content[:, cut] - replaced[:, cut]))),
        }
        metrics["prefixes"][cut] = measured
        checks[f"prefix {cut}: next-value nRMSE below 0.35"] = measured["nrmse"] < 0.35
        checks[f"prefix {cut}: shuffled prefix nRMSE above 0.85"] = measured["shuffled_nrmse"] > 0.85
        checks[f"prefix {cut}: future cannot affect prediction"] = measured["future_value_drift"] < 1e-5
        field = probe.encode(pa.Table.from_pylist(selected), strata="train")["/events/value"]
        expected = np.broadcast_to(np.arange(LENGTH) == cut, inferred.shape)
        visible = np.broadcast_to(np.arange(LENGTH) < cut, inferred.shape)
        checks[f"prefix {cut}: exactly the next coordinate is requested"] = bool(np.array_equal(inferred, expected))
        checks[f"prefix {cut}: exactly the next coordinate is trained"] = bool(
            np.array_equal(field.trainable.cpu().numpy().reshape(expected.shape), expected)
        )
        checks[f"prefix {cut}: only prefix values are encoder inputs"] = bool(
            np.array_equal(field.present.cpu().numpy().reshape(visible.shape), visible)
        )
    generated = rollout(model, test)
    baseline = float(np.sqrt(np.mean((actual[:, 2:] - mean[2:]) ** 2)))
    rmse = float(np.sqrt(np.mean((generated[:, 2:] - actual[:, 2:]) ** 2)))
    persistence = float(np.sqrt(np.mean((actual[:, 1:2] - actual[:, 2:]) ** 2)))
    metrics["rollout"] = {
        "rmse": rmse,
        "baseline_rmse": baseline,
        "nrmse": rmse / baseline,
        "persistence_rmse": persistence,
    }
    checks["Free-running suffix nRMSE below 0.45"] = metrics["rollout"]["nrmse"] < 0.45
    checks["Rollout beats persistence"] = rmse < persistence
    return metrics, checks


# %% [markdown]
# ## Evidence and remaining work
#
# {{< proof P054 evidence >}}
#
# Gates are provisional and declared before full runs. Repeat across seeds;
# distinguish failed forecasting from failed visibility controls. Nonlinear and
# stochastic dynamics, categorical sampling, longer horizons, and simultaneous
# autoregressive training remain outside this proof.
#
# ## Reproduce
#
# {{< proof P054 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=5401)
