# %% [markdown]
# ---
# title: Can a series forecast use the order of its steps?
# categories:
# - Forecasting
# description: Forecast the next steps of noisy two-channel seasonal series and compare with repeating the last step and
#   with a model whose history steps are shuffled.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Ordered history MAE
#   metric:
#   - ordered_mae
#   format: number
# - label: Repeat last step MAE
#   metric:
#   - last_step_mae
#   format: number
# - label: Shuffled history MAE
#   metric:
#   - shuffled_mae
#   format: number
# evidence-group: Temporal
# ---
#
# ## Example
#
# ```yaml
# history:
#   - [1.92, 0.97]
#   - [2.41, 1.18]
#   - [2.10, 1.07]
# future:
#   - [1.35, 0.66]
#   - [0.61, 0.31]
# ```
#
# Real records hold 48 history steps and 8 future steps. The future is shown here
# for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-temporal-series-forecast-requires-order
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with series, history, future. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("series", kind: "root", children: (
#   node("history", type: "Series", detail: "48 steps",),
#   node("future", kind: "target", type: "Series", detail: "horizon 8",),
# )))
# ```
#
# ## Comparison
#
# Compare a model reading ordered history with repeating the last observed step
# and with the same model trained and scored on histories whose steps are shuffled.
#
# ## Result
#
# {{< evidence P101 status >}}
#
# Ordered history beats repeating the last step. Shuffling keeps every value but
# erases the phase, so the shuffled model loses most of that advantage.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P101 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P101 script >}}
#
# ### How it works
#
# Each record is a sine wave with its own level, amplitude, phase, and a period
# of 8 or 12 steps, plus Gaussian noise. The second channel is a noisy half-scale
# copy of the first. The model reads 48 steps and forecasts the next 8 steps of
# both channels. Two history steps per training record are missing.
#
# The future depends on where the wave is in its cycle, which only step order
# reveals. The shuffled control permutes the steps of each history independently, in
# training and testing, keeping the values and level while removing the phase.
# Repeating the last step is the naive baseline; it ignores the cycle.
#
# Training, validation, and test records come from independent generator
# streams. Both models share architecture, seed, and 30 deterministic epochs.
# Mean absolute error is measured in source units over every test step and
# channel.
#
# ### Remaining work
#
# Repeat over three core seeds. Add irregular sampling, longer horizons, and
# real benchmark series. Compare with classical seasonal baselines.
#
# ### Complete experiment code
#

# %%
"""P101: a Series forecast depends on step order, not only on the values seen.

Seasonal phase is visible only through the order of history steps. Shuffling
each history keeps its values and level but removes that phase.
"""

from collections.abc import Callable, Iterator

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
import torch
from reporting import experiment, report

HISTORY, HORIZON, CHANNELS = 48, 8, 2


def waves(*, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """History and future arrays of shape ``(count, steps, CHANNELS)``."""
    rng = np.random.default_rng(seed)
    t = np.arange(HISTORY + HORIZON)[None, :]
    level = rng.uniform(-3.0, 3.0, (count, 1))
    amplitude = rng.uniform(0.5, 2.0, (count, 1))
    phase = rng.uniform(0.0, 2 * np.pi, (count, 1))
    period = rng.choice([8, 12], (count, 1))
    first = level + amplitude * np.sin(2 * np.pi * t / period + phase) + rng.normal(0.0, 0.1, t.shape[1:])
    second = 0.5 * first + rng.normal(0.0, 0.05, first.shape)
    steps = np.stack([first, second], axis=-1)
    return steps[:, :HISTORY], steps[:, HISTORY:]


def shuffle(history: np.ndarray, *, seed: int) -> np.ndarray:
    """Permute the steps of each record independently, keeping channels of a step together."""
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random(history.shape[:2]), axis=1)
    return np.take_along_axis(history, order[..., None], axis=1)


def records(history: np.ndarray, future: np.ndarray, *, missing: int, seed: int) -> Callable[[], Iterator[dict]]:
    """Records with ``missing`` unobserved history steps each."""
    rng = np.random.default_rng(seed)
    history = history.copy()
    for row in history:
        row[rng.choice(HISTORY, size=missing, replace=False)] = np.nan

    def generate() -> Iterator[dict]:
        for past, ahead in zip(history.tolist(), future.tolist(), strict=True):
            yield {"history": past, "future": ahead}

    return generate


def fit(
    train: Callable[[], Iterator[dict]],
    validate: Callable[[], Iterator[dict]],
    *,
    seed: int,
    steps: int | None,
    accelerator: str,
) -> rf.Model:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        history=rf.Series(channels=CHANNELS, length=HISTORY, overflow="tail", n_blocks=4),
        future=rf.Series(channels=CHANNELS, horizon=HORIZON, mask=True),
    )
    data = rf.SyntheticDataModule(train=train, validate=validate, seed=seed)
    model.fit(
        data,
        batch_size=64,
        learning_rate=0.003,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        fused=False,
        decay_bias=True,
        decay_1d=True,
        no_decay_name_fragments=(),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
        accelerator=accelerator,
        max_epochs=30,
        max_steps=steps if steps is not None else -1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    return model


def mae(model: rf.Model, history: np.ndarray, future: np.ndarray) -> float:
    """Mean absolute forecast error in source units over every step and channel."""
    predicted = model.predict(pa.table({"history": history.tolist()}), batch_size=256)
    content = predicted.column("predictions").combine_chunks().field("future").field("content").to_pylist()
    return float(np.abs(np.asarray(content) - future).mean())


@experiment("P101", seed=83)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train_history, train_future = waves(count=1200, seed=seed)
    validate_history, validate_future = waves(count=200, seed=seed + 1)
    test_history, test_future = waves(count=400, seed=seed + 2)

    ordered = fit(
        records(train_history, train_future, missing=2, seed=seed),
        records(validate_history, validate_future, missing=0, seed=seed),
        seed=seed,
        steps=steps,
        accelerator=accelerator,
    )
    ordered_mae = mae(ordered, test_history, test_future)

    shuffled = fit(
        records(shuffle(train_history, seed=seed), train_future, missing=2, seed=seed),
        records(shuffle(validate_history, seed=seed + 1), validate_future, missing=0, seed=seed),
        seed=seed,
        steps=steps,
        accelerator=accelerator,
    )
    shuffled_mae = mae(shuffled, shuffle(test_history, seed=seed + 2), test_future)

    last_step_mae = float(np.abs(test_history[:, -1:, :] - test_future).mean())
    return {
        "ordered_mae": ordered_mae,
        "last_step_mae": last_step_mae,
        "shuffled_mae": shuffled_mae,
    }, {
        "Ordered history MAE is at most half of repeating the last step": ordered_mae <= 0.5 * last_step_mae,
        "Shuffled history MAE is at least twice the ordered MAE": shuffled_mae >= 2.0 * ordered_mae,
    }


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
