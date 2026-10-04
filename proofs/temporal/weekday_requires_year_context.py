# %% [markdown]
# ---
# title: Can day-of-year alone identify the weekday?
# categories:
# - Calendar reasoning
# proof-id: P045
# description: Predict weekday from the day number while deliberately balancing different years across all
#   seven weekday answers.
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
# - label: Weekday supplied
#   metric:
#   - with_weekday_accuracy
#   format: percent
# ---
#
# ## Example
#
# ```yaml
# observed_at: 2001-01-01
# target: Monday
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-proof-temporal-weekday-requires-year-context
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with calendar, observed_at, target. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("calendar", kind: "root", children: (
#   node("observed_at", type: "DateParts", width: 150pt,),
#   node("target", kind: "target", type: "Category", detail: "weekday",),
# )))
# ```
#
# ## Comparison
#
# Compare day number alone with a model that also receives the weekday coordinate.
#
# ## Result
#
# {{< proof P045 status >}}
#
# The day-only model stays near one-in-seven chance. A source timestamp only helps through the coordinates exposed to the model.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P045 evidence >}}
#
# ### Run this experiment
#
# {{< proof P045 script >}}
#
# ### How it works
#
# The generator balances all seven weekday labels at every sampled ordinal
# day. With `dateparts=["day_of_year"]`, identical visible inputs therefore
# need seven different answers. The best accuracy is one seventh.
#
# The positive control uses
# `dateparts=["day_of_year", "day_of_week"]`, which supplies the missing
# distinction directly. This control tests a visible weekday coordinate;
# it does **not** test learning weekday arithmetic from a year coordinate.
#
# Training uses candidate dates from 1901–1935, validation from 1951–1985,
# and testing from 2001–2035, excluding leap years. Both models train for
# 14 deterministic epochs.
#
# ### Remaining work
#
# Repeat the positive and ambiguity gates over three core seeds and at least
# ten calibration seeds. Broaden the family's coordinate-composition targets
# while keeping balanced collisions that make information loss observable.
#
# ### Complete experiment code
#

# %%
"""P045: expose the weekday information erased by day-of-year alone.

Every sampled ordinal day has all seven weekday labels equally often. Adding
``day_of_week`` resolves these collisions; the model does not infer a hidden year.
"""

import calendar
from collections.abc import Callable, Iterator, Sequence
from datetime import date, timedelta
from functools import partial

import lightning.pytorch as lit
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P045"


def records(*, years: Sequence[int], ordinal_step: int = 5) -> Iterator[dict]:
    """Yield every weekday at each sampled non-leap ordinal day."""
    candidates: dict[tuple[int, int], date] = {}
    ordinals = range(1, 366, ordinal_step)
    for year in years:
        if calendar.isleap(year):
            continue
        first = date(year, 1, 1)
        for ordinal in ordinals:
            current = first + timedelta(days=ordinal - 1)
            candidates.setdefault((ordinal, current.weekday()), current)
    for ordinal in ordinals:
        for weekday in range(7):
            key = (ordinal, weekday)
            if key not in candidates:
                raise ValueError(f"years do not cover weekday {weekday} at ordinal day {ordinal}")
            yield {"observed_at": candidates[key], "target": calendar.day_name[weekday]}


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
    train = partial(records, years=tuple(range(1901, 1936)))
    validate = partial(records, years=tuple(range(1951, 1986)))
    test = partial(records, years=tuple(range(2001, 2036)))
    day_only = fit(
        dateparts=("day_of_year",),
        train=train,
        validate=validate,
        epochs=14,
        seed=seed,
        steps=steps,
        accelerator=accelerator,
    )
    ambiguous_accuracy = accuracy(day_only, test, accelerator)
    identified = fit(
        dateparts=("day_of_year", "day_of_week"),
        train=train,
        validate=validate,
        epochs=14,
        seed=seed,
        steps=steps,
        accelerator=accelerator,
    )
    identified_accuracy = accuracy(identified, test, accelerator)
    gap = identified_accuracy - ambiguous_accuracy
    return {
        "day_only_accuracy": ambiguous_accuracy,
        "with_weekday_accuracy": identified_accuracy,
        "accuracy_gap": gap,
    }, {
        "Day-of-year accuracy is at most 0.20": ambiguous_accuracy <= 0.20,
        "Visible weekday accuracy reaches 0.95": identified_accuracy >= 0.95,
        "Visible weekday improves accuracy by at least 0.70": gap >= 0.70,
    }


if __name__ == "__main__":
    report(PROOF_ID, run, seed=47)

# %% [markdown]
# </details>
