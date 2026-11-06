# %% [markdown]
# ---
# title: Can day-of-year alone identify the weekday?
# categories:
# - Calendar reasoning
# description: Predict weekday from the day number while deliberately balancing different years across all seven weekday
#   answers.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Day number only
#   metric:
#   - day_only_accuracy
#   format: percent
# - label: Weekday supplied
#   metric:
#   - with_weekday_accuracy
#   format: percent
# evidence-group: Temporal
# ---
#
# ## Example
#
# January 1, 2001 was Monday, as the illustration shows; January 1, 2002 was Tuesday. Both dates have
# ordinal day one. A model receiving only day-of-year cannot tell those cases apart even though their hidden
# weekday labels differ. The generator deliberately assembles all seven weekdays for each sampled ordinal
# day, so the ambiguity is repeated systematically rather than depending on an accidental imbalance in a few
# dates.
#
# ```yaml
# observed_at: 2001-01-01
# target: Monday
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-temporal-weekday-requires-year-context
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
# The two models receive day-of-year alone or day-of-year together with day-of-week. They train for the same
# number of epochs and are evaluated on a separate range of years. Adding day-of-week supplies the
# information needed to identify the answer directly; it does not ask the model to recover the missing year
# or learn a calendar algorithm. This positive control shows that the prediction machinery can read a
# sufficient representation.
#
# Compare day number alone with a model that also receives the weekday coordinate.
#
# ## Result
#
# With seven equally represented labels for each indistinguishable day number, about one correct answer in
# seven is the meaningful day-only reference. The richer-input accuracy should be read as evidence that the
# exposed weekday coordinate can be used, rather than evidence of year reasoning. A timestamp in the
# original record does not make all of its information available after encoding. The choice of coordinates
# determines which distinctions the downstream model can learn.
#
# {{< evidence P045 status >}}
#
# The day-only model stays near one-in-seven chance. A source timestamp only helps through the coordinates exposed to the model.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P045 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P045 script >}}
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
import relflow as rf
import torch
from reporting import experiment, report


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
        observed_at=rf.DateParts(dateparts=list(dateparts)),
        target=rf.Category(mask=True, p_unavailable=0.0),
    )
    data = rf.SyntheticDataModule(train=train, validate=validate, seed=seed)
    model.fit(
        data,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.003),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
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
    return model


def accuracy(model: rf.Model, records: Callable[[], Iterator[dict]], accelerator: str) -> float:
    """Evaluate held-out labels without updating the Category vocabulary."""
    data = rf.SyntheticDataModule(test=records)
    metrics = model.test(
        data,
        accelerator=accelerator,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        verbose=False,
        batch_size=128,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
    )[0]
    return float(metrics[".target/test.accuracy.content"])


@experiment("P045", seed=47)
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
    report(run)

# %% [markdown]
# </details>
