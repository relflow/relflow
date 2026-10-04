# %% [markdown]
# ---
# title: Can day 60 reveal the month in a leap year?
# categories:
# - Calendar reasoning
# proof-id: P043
# description: Distinguish February 29 from March 1 when both can be day 60 of a year.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: Day number only
#   metric:
#   - day_only_accuracy
#   format: percent
# - label: Week-of-month added
#   metric:
#   - with_week_of_month_accuracy
#   format: percent
# ---
#
# ## Example
#
# ```yaml
# observed_at: 2024-02-29
# target: February
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-proof-temporal-leap-boundary-requires-context
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with calendar, observed_at, target. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("calendar", kind: "root", children: (
#   node("observed_at", type: "DateParts", width: 150pt,),
#   node("target", kind: "target", type: "Category", detail: "month",),
# )))
# ```
#
# ## Comparison
#
# Compare day-of-year alone with a model that also receives week-of-month.
#
# ## Result
#
# {{< proof P043 status >}}
#
# Day number alone cannot separate identical inputs with different answers. The added coordinate resolves these cases; general leap-year reasoning is not tested.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P043 evidence >}}
#
# ### Run this experiment
#
# {{< proof P043 script >}}
#
# ### How it works
#
# The ambiguous variant exposes only `day_of_year`. Every input coordinate
# is identical, with half the labels February and half March, so optimal
# accuracy is 0.50. The positive variant also exposes `week_of_month`.
#
# The train, validation, and test year ranges are 1901–1950, 1951–2000, and
# 2001–2050, with 512, 128, and 256 rows respectively. The ambiguous model
# trains for ten deterministic epochs; the identified model trains for 20.
# The experiment isolates available information but does not match the two
# training budgets.
#
# ### Remaining work
#
# Repeat the gates over three core seeds and at least ten calibration seeds.
# Investigate why adding the finer `day_of_month` coordinate optimizes less
# reliably than `week_of_month`. Test application policies that map leap dates
# onto a canonical non-leap calendar separately.
#
# ### Complete experiment code
#

# %%
"""P043: distinguish February 29 from March 1 at ordinal day sixty.

Day-of-year is identical for both labels. Adding week-of-month reveals their
difference. The original budgets are retained: ten epochs for day-only and
twenty for the richer input, so this is not an equal-budget comparison.
"""

import calendar
from collections.abc import Callable, Iterator, Sequence
from datetime import date
from functools import partial

import lightning.pytorch as lit
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P043"


def records(*, years: Sequence[int], rows: int) -> Iterator[dict]:
    """Yield balanced leap-day and ordinary March-boundary observations."""
    leap = [date(year, 2, 29) for year in years if calendar.isleap(year)]
    ordinary = [date(year, 3, 1) for year in years if not calendar.isleap(year)]
    if not leap or not ordinary:
        raise ValueError("leap-boundary data requires both leap and non-leap years")
    if rows % 2:
        raise ValueError(f"leap-boundary rows must be even, got {rows}")
    for index in range(rows // 2):
        yield {"observed_at": leap[index % len(leap)], "target": "February"}
        yield {"observed_at": ordinary[index % len(ordinary)], "target": "March"}


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
    train = partial(records, years=tuple(range(1901, 1951)), rows=512)
    validate = partial(records, years=tuple(range(1951, 2001)), rows=128)
    test = partial(records, years=tuple(range(2001, 2051)), rows=256)
    day_only = fit(
        dateparts=("day_of_year",),
        train=train,
        validate=validate,
        epochs=10,
        seed=seed,
        steps=steps,
        accelerator=accelerator,
    )
    ambiguous_accuracy = accuracy(day_only, test, accelerator)
    identified = fit(
        dateparts=("day_of_year", "week_of_month"),
        train=train,
        validate=validate,
        epochs=20,
        seed=seed,
        steps=steps,
        accelerator=accelerator,
    )
    identified_accuracy = accuracy(identified, test, accelerator)
    gap = identified_accuracy - ambiguous_accuracy
    return {
        "day_only_accuracy": ambiguous_accuracy,
        "with_week_of_month_accuracy": identified_accuracy,
        "accuracy_gap": gap,
    }, {
        "Day-of-year accuracy remains between 0.49 and 0.51": 0.49 <= ambiguous_accuracy <= 0.51,
        "Visible week-of-month accuracy reaches 0.95": identified_accuracy >= 0.95,
        "Visible week-of-month improves accuracy by at least 0.40": gap >= 0.40,
    }


if __name__ == "__main__":
    report(PROOF_ID, run, seed=53)

# %% [markdown]
# </details>
