# %% [markdown]
# ---
# title: Combining Weekday and Hour
# categories:
# - Calendar reasoning
# proof-id: P042
# description: Weekday and hour together identify a business window that neither coordinate
#   resolves alone.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# A business window depends on both the day and the time: weekday evenings
# and weekend mornings are closed, even though each shares one coordinate
# with an open observation.
#
# {{< proof P042 status >}}
#
# ## Insights
#
# **Weekday and hour together identify a business window that neither coordinate resolves alone.** A weekday
# evening and a weekend morning each share one coordinate with an open observation, so either
# single-coordinate representation loses a necessary distinction.
#
# The combined representation distinguishes the labels more accurately than either weekday or hour alone.
# The weekday fit does not reach all the skill available from its marginal signal; these are observed fits,
# not three equally optimized solutions.
#
# This supports learned coordinate composition in the constructed calendar. It does not cover holidays,
# timezones, or daylight-saving rules. Derive fixed business logic exactly in preprocessing when learning an
# approximation is unnecessary.
#
# ## Setup

# %%
"""P042: combine weekday and hour to identify an open business window.

Half the rows are open; the others are weekday/off-hour or weekend/work-hour
near misses. Either coordinate alone has an optimal accuracy of 0.75.
"""

from collections.abc import Callable, Iterator, Sequence
from datetime import date, datetime, time, timedelta
from functools import partial

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P042"

# %% [markdown]
# ## Examples
#
# The rule is open on weekdays from 09:00 through 16:00. Each displayed target
# is supervision hidden from the encoded inputs.
#
# ### Weekday during business hours
#
# ```yaml
# observed_at: 2025-01-06T09:00:00
# target: open
# ```
#
# Monday at 09:00 satisfies both conditions.
#
# ### Weekday outside business hours
#
# ```yaml
# observed_at: 2025-01-06T20:00:00
# target: closed
# ```
#
# The weekday is unchanged, but the hour changes the answer. A weekday-only
# representation cannot distinguish this near miss from the first record.
#
# ### Weekend during business hours
#
# ```yaml
# observed_at: 2025-01-11T09:00:00
# target: closed
# ```
#
# Saturday shares the first record's hour. An hour-only representation loses
# this distinction, while the two-coordinate representation contains both
# pieces needed to identify the open window.
#
# ## Synthetic data and controls


# %%
def records(*, start: date, weeks: int, rows: int, seed: int) -> Iterator[dict]:
    """Yield balanced open windows and two single-coordinate near misses."""
    if start.weekday() != 0:
        raise ValueError(f"business-window start must be a Monday, got {start.isoformat()}")
    if weeks < 1:
        raise ValueError(f"business-window weeks must be positive, got {weeks}")
    if rows % 4:
        raise ValueError(f"business-window rows must be divisible by four, got {rows}")
    business_hours = tuple(range(9, 17))
    off_hours = (*range(0, 9), *range(17, 24))
    observations = []
    for index in range(rows // 2):
        day = start + timedelta(weeks=index % weeks, days=index % 5)
        hour = business_hours[index % len(business_hours)]
        observations.append({"observed_at": datetime.combine(day, time(hour=hour)), "target": "open"})
    for index in range(rows // 4):
        day = start + timedelta(weeks=index % weeks, days=index % 5)
        hour = off_hours[index % len(off_hours)]
        observations.append({"observed_at": datetime.combine(day, time(hour=hour)), "target": "closed"})
    for index in range(rows // 4):
        day = start + timedelta(weeks=index % weeks, days=5 + index % 2)
        hour = business_hours[index % len(business_hours)]
        observations.append({"observed_at": datetime.combine(day, time(hour=hour)), "target": "closed"})
    for index in np.random.default_rng(seed).permutation(len(observations)):
        yield observations[index]


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-temporal-business-window-requires-composition
# //| fig-cap: "The combined route exposes weekday and hour together. Each control keeps only one coordinate; all hide the target."
# //| fig-alt: "Calendar contains DateParts observed at with weekday and hour together, compared with separate weekday-only and hour-only controls, and a Category open-or-closed target always hidden from input."
# #tree(node("calendar", kind: "root", children: (
#   node("observed_at", type: "DateParts", width: 150pt, body: [
#     - *Combined:* weekday + hour
#     - *Controls:* weekday or hour alone
#   ]),
#   node("target", kind: "target", type: "Category", detail: "open or closed", body: [
#     - *Input:* always hidden
#   ]),
# )))
# ```
#
# ## How it works
#
# Half the observations are weekdays during 09:00–16:00. A quarter are
# weekdays outside those hours, and a quarter are weekends during those hours.
# Neither coordinate alone identifies every label; a combined representation
# can distinguish the open window from both kinds of near miss.
#
# Three otherwise matched models expose `day_of_week`, `hour_of_day`, or both.
# They train for 20 deterministic epochs. Training, validation, and test
# use 1,024, 512, and 1,024 observations from separate periods beginning in
# 2017, 2021, and 2025.
#
# ## Training and evaluation


# %%
def fit(
    *,
    dateparts: Sequence[str],
    train: Callable[[], Iterator[dict]],
    validate: Callable[[], Iterator[dict]],
    epochs: int,
    seed: int,
    steps: int | None,
    accelerator: str,
) -> rf.Model:
    """Train on the selected calendar coordinates with the remaining schema fixed."""
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        name="calendar",
        d_model=64,
        n_layers=3,
        n_heads=4,
        batch_size=128,
        observed_at=rf.DateParts(dateparts=list(dateparts)),
        target=rf.Category(mask=True, p_unavailable=0.0),
    )
    model.optimizer = lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3)
    data = rf.SyntheticDataModule(model=model, train=train, validate=validate, seed=seed)
    trainer = lit.Trainer(
        accelerator=accelerator,
        max_epochs=epochs,
        max_steps=steps if steps is not None else -1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model=model, datamodule=data)
    return model


def accuracy(model: rf.Model, records: Callable[[], Iterator[dict]], accelerator: str) -> float:
    """Evaluate held-out labels without updating the Category vocabulary."""
    data = rf.SyntheticDataModule(model=model, test=records)
    trainer = lit.Trainer(
        accelerator=accelerator,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    metrics = trainer.test(model=model, datamodule=data, verbose=False)[0]
    return float(metrics["calendar.target/test.accuracy.content"])


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train = partial(records, start=date(2017, 1, 2), weeks=52, rows=1024, seed=seed + 1)
    validate = partial(records, start=date(2021, 1, 4), weeks=26, rows=512, seed=seed + 2)
    test = partial(records, start=date(2025, 1, 6), weeks=52, rows=1024, seed=seed + 3)
    day_only = fit(
        dateparts=("day_of_week",),
        train=train,
        validate=validate,
        epochs=20,
        seed=seed,
        steps=steps,
        accelerator=accelerator,
    )
    day_accuracy = accuracy(day_only, test, accelerator)
    hour_only = fit(
        dateparts=("hour_of_day",),
        train=train,
        validate=validate,
        epochs=20,
        seed=seed,
        steps=steps,
        accelerator=accelerator,
    )
    hour_accuracy = accuracy(hour_only, test, accelerator)
    composed = fit(
        dateparts=("day_of_week", "hour_of_day"),
        train=train,
        validate=validate,
        epochs=20,
        seed=seed,
        steps=steps,
        accelerator=accelerator,
    )
    composed_accuracy = accuracy(composed, test, accelerator)
    gap = composed_accuracy - max(day_accuracy, hour_accuracy)
    return {
        "day_only_accuracy": day_accuracy,
        "hour_only_accuracy": hour_accuracy,
        "composed_accuracy": composed_accuracy,
        "accuracy_gap": gap,
    }, {
        "Weekday-only accuracy is at most 0.82": day_accuracy <= 0.82,
        "Hour-only accuracy is at most 0.82": hour_accuracy <= 0.82,
        "Combined coordinates reach 0.95 accuracy": composed_accuracy >= 0.95,
        "Composition improves over either coordinate by at least 0.15": gap >= 0.15,
    }


# %% [markdown]
# ## Evidence
#
# {{< proof P042 evidence >}}
#
# ## Remaining work
#
# Repeat all three fits over three core seeds and at least ten calibration
# seeds. Extend to further coordinate combinations and characterize timezone
# and daylight-saving preprocessing independently of this naive-calendar test.
#
# ## Reproduce
#
# {{< proof P042 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=59)
