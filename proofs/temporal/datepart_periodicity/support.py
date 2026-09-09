"""Synthetic data and user guide for DateParts periodicity proofs.

Claim
-----
``DateParts`` exposes selected calendar coordinates as continuous cyclic
features. A model can learn a recurring relationship among those coordinates
on timestamps and years it never saw, provided the visible coordinates
actually determine the target.

What to do
----------
Use ``rf.DateParts(dateparts=[...])`` for recurring calendar behavior and list
every coordinate needed to identify the outcome. Cover phase boundaries in
the training data. For example, ``day_of_year`` determines ``month_of_year``
within non-leap calendars, while ``day_of_year`` plus ``week_of_month`` cleanly
separates the February/March boundary at ordinal day 60.

Omitting the target coordinate here is a benchmark of learned periodic
interaction, not a calendar-conversion recipe. If the required output is
literally a date part or fixed business-calendar rule, expose that DatePart or
derive it exactly in a :class:`relflow.Preprocessor` or the application.

What not to do
--------------
Do not assume that the original timestamp remains available after DateParts
selects its configured coordinates. ``day_of_year`` alone contains no year,
leap-year flag, or weekday. No model can recover distinctions erased at that
boundary. Do not interpret fitting one year's weekday pattern as calendar
reasoning; the relationship changes with the year.

Protocol and gates
------------------
The positive proof trains on odd days of the month from non-leap years and
tests on even dates from entirely unseen non-leap years. Month accuracy must be
at least 0.90. Permuting timestamps while retaining month labels must reduce
accuracy below 0.20 and by at least 0.65.

The weekday boundary balances all seven weekday labels at every represented
day-of-year coordinate. A day-of-year-only model is therefore bounded at 1/7
accuracy, while adding ``day_of_week`` must reach at least 0.95. The leap
boundary similarly pairs leap-year February 29 with non-leap March 1: both are
day 60. Day-of-year alone is bounded at 0.50, while adding ``week_of_month``
must reach at least 0.95.

The composition proof balances open business windows against two kinds of
near miss: a weekday outside business hours and a weekend during business
hours. Either coordinate alone has an optimal accuracy of 0.75; their combined
DateParts representation must reach at least 0.95 and improve by 0.15.

All models use the public ``Model`` and ``ArrowDataModule`` APIs, deterministic
CPU training, independently generated splits, and masked Category targets.

Status and current evidence
---------------------------
Provisional. The cases establish one positive periodic inference, two
constructive identifiability boundaries, and one two-coordinate composition
for one deterministic model seed. After 70 epochs, held-out month accuracy is
0.967 versus 0.110 after timestamp permutation. Weekday is 0.143 from
day-of-year and 1.000
after exposing day-of-week. The leap boundary is 0.500 from day-of-year and
1.000 after adding week-of-month. The business-window case reaches 1.000 from
both coordinates versus 0.500 from weekday and 0.750 from hour alone.

Further work
------------
Repeat the learned gates over at least three model seeds; expand the positive
case to week and multi-coordinate targets; investigate why the fine-grained
``day_of_year + day_of_month`` leap signal optimizes less reliably than the
larger ``week_of_month`` separation; characterize daylight-saving and timezone
preprocessing separately because DateParts does not add timezone semantics;
and test leap-day policies when applications map ordinal days onto a canonical
non-leap calendar.

Promotion criteria
------------------
Retain every positive, corruption, and identifiability gate over three core
seeds and a ten-seed calibration panel without weakening the thresholds.

Run all cases with::

    uv run pytest -n 0 proofs/temporal/datepart_periodicity -q
"""

from __future__ import annotations

import calendar
from collections.abc import Iterable, Sequence
from datetime import date, datetime, time, timedelta

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch

import relflow as rf


def dates(years: Iterable[int], *, parity: int) -> pa.Table:
    """Create non-leap calendar rows selected by day-of-month parity."""

    rows: list[dict[str, object]] = []
    for year in years:
        if calendar.isleap(year):
            raise ValueError(f"month periodicity requires non-leap years, got {year}")
        current = date(year, 1, 1)
        while current.year == year:
            if current.day % 2 == parity:
                rows.append(
                    {
                        "observed_at": current,
                        "target": calendar.month_name[current.month],
                    }
                )
            current += timedelta(days=1)
    return pa.Table.from_pylist(rows)


def permute_observed_at(table: pa.Table, *, seed: int) -> pa.Table:
    """Break timestamp/target association while retaining both marginals."""

    rng = np.random.default_rng(seed)
    observed = table["observed_at"].combine_chunks()
    shuffled = pa.array(np.asarray(observed.to_pylist(), dtype=object)[rng.permutation(len(table))])
    return table.set_column(table.schema.get_field_index("observed_at"), "observed_at", shuffled)


def balanced_weekdays(*, years: Sequence[int], ordinal_step: int = 5) -> pa.Table:
    """Balance every weekday label at each sampled non-leap ordinal day."""

    candidates: dict[tuple[int, int], date] = {}
    ordinals = range(1, 366, ordinal_step)
    for year in years:
        if calendar.isleap(year):
            continue
        first = date(year, 1, 1)
        for ordinal in ordinals:
            current = first + timedelta(days=ordinal - 1)
            candidates.setdefault((ordinal, current.weekday()), current)

    rows: list[dict[str, object]] = []
    for ordinal in ordinals:
        for weekday in range(7):
            key = (ordinal, weekday)
            if key not in candidates:
                raise ValueError(f"years do not cover weekday {weekday} at ordinal day {ordinal}")
            rows.append(
                {
                    "observed_at": candidates[key],
                    "target": calendar.day_name[weekday],
                }
            )
    return pa.Table.from_pylist(rows)


def leap_boundary(*, years: Sequence[int], rows: int) -> pa.Table:
    """Balance February 29 and March 1 observations sharing ordinal day 60."""

    leap = [date(year, 2, 29) for year in years if calendar.isleap(year)]
    ordinary = [date(year, 3, 1) for year in years if not calendar.isleap(year)]
    if not leap or not ordinary:
        raise ValueError("leap-boundary data requires both leap and non-leap years")
    if rows % 2:
        raise ValueError(f"leap-boundary rows must be even, got {rows}")

    observations: list[dict[str, object]] = []
    for index in range(rows // 2):
        observations.extend(
            (
                {"observed_at": leap[index % len(leap)], "target": "February"},
                {"observed_at": ordinary[index % len(ordinary)], "target": "March"},
            )
        )
    return pa.Table.from_pylist(observations)


def business_windows(*, start: date, weeks: int, rows: int, seed: int) -> pa.Table:
    """Create balanced open windows and two single-coordinate near misses."""

    if start.weekday() != 0:
        raise ValueError(f"business-window start must be a Monday, got {start.isoformat()}")
    if weeks < 1:
        raise ValueError(f"business-window weeks must be positive, got {weeks}")
    if rows % 4:
        raise ValueError(f"business-window rows must be divisible by four, got {rows}")

    business_hours = tuple(range(9, 17))
    off_hours = (*range(0, 9), *range(17, 24))
    observations: list[dict[str, object]] = []
    for index in range(rows // 2):
        weekday = index % 5
        hour = business_hours[index % len(business_hours)]
        day = start + timedelta(weeks=index % weeks, days=weekday)
        observations.append(
            {
                "observed_at": datetime.combine(day, time(hour=hour)),
                "target": "open",
            }
        )
    for index in range(rows // 4):
        weekday = index % 5
        hour = off_hours[index % len(off_hours)]
        day = start + timedelta(weeks=index % weeks, days=weekday)
        observations.append(
            {
                "observed_at": datetime.combine(day, time(hour=hour)),
                "target": "closed",
            }
        )
    for index in range(rows // 4):
        weekday = 5 + index % 2
        hour = business_hours[index % len(business_hours)]
        day = start + timedelta(weeks=index % weeks, days=weekday)
        observations.append(
            {
                "observed_at": datetime.combine(day, time(hour=hour)),
                "target": "closed",
            }
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(observations))
    return pa.Table.from_pylist([observations[index] for index in order])


def model(*, dateparts: Sequence[str], classes: int, seed: int) -> rf.Model:
    """Build the public DateParts-to-Category model used by every case."""

    lit.seed_everything(seed, workers=True)
    return rf.Model(
        name="calendar",
        d_model=64,
        n_layers=3,
        n_heads=4,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        observed_at=rf.DateParts(dateparts=list(dateparts)),
        target=rf.Category(mask=True, size=classes, p_unavailable=0.0),
    )


def fit(
    configured: rf.Model,
    *,
    train: pa.Table,
    validate: pa.Table,
    epochs: int,
    seed: int,
) -> None:
    """Fit one deterministic CPU model on Arrow-backed synthetic rows."""

    data = rf.ArrowDataModule(
        model=configured,
        train=train,
        validate=validate,
        seed=seed,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    trainer = lit.Trainer(
        accelerator="cpu",
        max_epochs=epochs,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model=configured, datamodule=data)


def accuracy(configured: rf.Model, table: pa.Table) -> float:
    """Measure held-out Category accuracy through ArrowDataModule."""

    data = rf.ArrowDataModule(
        model=configured,
        test=table,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    trainer = lit.Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    metrics = trainer.test(model=configured, datamodule=data, verbose=False)[0]
    return float(metrics["calendar.target/test.accuracy.content"])
