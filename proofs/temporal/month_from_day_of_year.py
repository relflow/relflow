# %% [markdown]
# ---
# title: Learning Month from Annual Phase
# categories:
# - Calendar reasoning
# proof-id: P044
# description: A model can learn month boundaries from day-of-year on unseen dates in
#   non-leap years.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# Does a cyclic annual coordinate contain enough information to learn the month?
# This proof exposes only `day_of_year`, then tests month predictions on dates
# and years withheld from training.
#
# {{< proof P044 status >}}
#
# ## Insights
#
# **In non-leap years, the day number is enough to learn which month a date belongs to.** DateParts supplies
# only day-of-year, and the model learns its month intervals from odd dates before testing on even dates in
# unseen years.
#
# Accuracy is high on intact observations and drops sharply when timestamps are permuted while labels stay
# fixed. That supports use of the date–month relationship rather than success from label frequencies alone.
# Evidence covers one model seed and the deliberately restricted calendar.
#
# For exact month extraction, expose the month coordinate or derive it in preprocessing. This proof tests
# learned periodicity, not general calendar reasoning; leap-year ambiguity requires additional information.
#
# ## Setup

# %%
"""P044: learn month from day-of-year on unseen dates and years.

Only non-leap years are used. Training dates are odd days of the month;
held-out dates are even days in new years. Timestamp permutation retains the
labels and date marginals while removing their relationship.
"""

import calendar
from collections.abc import Callable, Iterator, Sequence
from datetime import date, timedelta
from functools import partial

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P044"

# %% [markdown]
# ## Examples
#
# The model sees only the cyclic `day_of_year` coordinate. The original year
# and explicit month are not additional inputs; `target` is a masked Category.
#
# ### An unseen February date
#
# ```yaml
# observed_at: 2025-02-04
# target: February
# ```
#
# February 4 is ordinal day 35 in this non-leap year. It is an even day of the
# month, so it is outside the odd-day training selection.
#
# ### Another annual phase
#
# ```yaml
# observed_at: 2025-11-18
# target: November
# ```
#
# November 18 is ordinal day 322. The model must learn a different interval
# of the annual cycle rather than recall a particular training timestamp.
#
# ### Timestamp permutation
#
# ```yaml
# observed_at: 2025-11-18
# target: February
# ```
#
# A timestamp permutation can replace the first record's date while leaving
# its label in place. `February` is intentionally inconsistent with this date;
# the corruption control checks that breaking the association harms aggregate
# accuracy. It does not define a second calendar rule.
#
# ## Synthetic data and controls


# %%
def records(years: Sequence[int], *, parity: int) -> Iterator[dict]:
    """Yield non-leap dates selected by day-of-month parity."""
    for year in years:
        if calendar.isleap(year):
            raise ValueError(f"month periodicity requires non-leap years, got {year}")
        current = date(year, 1, 1)
        while current.year == year:
            if current.day % 2 == parity:
                yield {"observed_at": current, "target": calendar.month_name[current.month]}
            current += timedelta(days=1)


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-temporal-month-from-day-of-year
# //| fig-cap: "The input exposes only day-of-year, omitting the month coordinate. The month label is hidden from input."
# //| fig-alt: "Calendar contains DateParts observed at exposing only day-of-year and omitting the month coordinate, with a Category month target always hidden from input."
# #tree(node("calendar", kind: "root", children: (
#   node("observed_at", type: "DateParts", width: 150pt, body: [
#     - *Visible:* day of year
#     - *Month coordinate:* omitted
#   ]),
#   node("target", kind: "target", type: "Category", detail: "month", body: [
#     - *Input:* always hidden
#   ]),
# )))
# ```
#
# ## How it works
#
# In a non-leap calendar, day-of-year determines month. The model must learn
# the intervals in that cyclic representation. Training uses odd-numbered
# days of each month in 2017–2019; validation uses odd days in 2021. Test dates
# are even-numbered days in 2023 and 2025. All these years are non-leap.
#
# After 70 deterministic epochs, the test evaluates both the original
# held-out data and a version with timestamps permuted while labels stay put.
# Permutation preserves the marginals but destroys the useful relationship.
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
    return float(metrics[".target/test.accuracy.content"])


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train = partial(records, (2017, 2018, 2019), parity=1)
    validate = partial(records, (2021,), parity=1)
    test = list(records((2023, 2025), parity=0))
    model = fit(
        dateparts=("day_of_year",),
        train=train,
        validate=validate,
        epochs=70,
        seed=seed,
        steps=steps,
        accelerator=accelerator,
    )
    periodic_accuracy = accuracy(model, lambda: iter(test), accelerator)
    order = np.random.default_rng(seed + 1).permutation(len(test))
    permuted = [{**row, "observed_at": test[index]["observed_at"]} for row, index in zip(test, order, strict=True)]
    permuted_accuracy = accuracy(model, lambda: iter(permuted), accelerator)
    gap = periodic_accuracy - permuted_accuracy
    return {
        "periodic_accuracy": periodic_accuracy,
        "permuted_accuracy": permuted_accuracy,
        "accuracy_gap": gap,
    }, {
        "Unseen-date month accuracy reaches 0.90": periodic_accuracy >= 0.90,
        "Timestamp permutation accuracy is at most 0.20": permuted_accuracy <= 0.20,
        "Original dates exceed permutation by at least 0.65 accuracy": gap >= 0.65,
    }


# %% [markdown]
# ## Evidence
#
# {{< proof P044 evidence >}}
#
# ## Remaining work
#
# Repeat the gates over three core seeds and at least ten calibration seeds.
# Extend the family to week and multi-coordinate targets. Characterize timezone,
# daylight-saving, and leap-day preprocessing policies separately rather than
# assuming that DateParts adds those semantics.
#
# ## Reproduce
#
# {{< proof P044 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=41)
