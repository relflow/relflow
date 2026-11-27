# %% [markdown]
# ---
# title: Can day 60 reveal the month in a leap year?
# description: Distinguish February 29 from March 1 when both can be day 60 of a year.
# evidence-readout:
# - label: Day number only
#   metric:
#   - day_only_accuracy
#   format: percent
# - label: Week-of-month added
#   metric:
#   - with_week_of_month_accuracy
#   format: percent
# evidence-group: Temporal
# evidence-review:
#   question: Can day 60 reveal the month in a leap year?
#   modeling: Choose dates where day 60 is either February 29 or March 1. With only day-of-year visible, the inputs
#     are identical but the answers differ. Compare that model with one also given week-of-month, which resolves the
#     ambiguity.
#   comparison:
#     kind: Recorded conditions
#     metric: Accuracy
#     conditions:
#     - label: Day number only
#       path:
#       - day_only_accuracy
#       format: percent
#     - label: Week-of-month added
#       path:
#       - with_week_of_month_accuracy
#       format: percent
#     definition: Fraction of predictions matching the expected answer, over the population named in each condition.
#     direction: Higher is better
#     domain:
#     - 0
#     - 1
#   note: The two runs use different training budgets. This is an information-availability comparison, not a matched
#     compute ablation.
# evidence-plot:
#   mode: temporal
#   target: target
#   focus: Which added calendar part resolves the date ambiguity?
#   comparison: Compare the day-only model with the declared extra calendar context.
#   limit: The plot follows the generator's actual calendar rule rather than a generic date illustration.
#   conditions:
#   - label: Day only
#     metric: day_only_accuracy
#   - label: Additional calendar context
#     metric: with_week_of_month_accuracy
# evidence-cost:
#   baseline: 11
#   candidate: 11
# ---
#
# ## Example
#
# February 29 in leap year 2024 is the sixtieth day of that year. March 1 in ordinary year 2025 is also day
# sixty, yet its hidden month label is March. The illustration shows the first case. If the model receives
# only day-of-year, these two dates become the same input with different correct answers. Week-of-month
# distinguishes a date at the end of February from one at the start of March.
#
# ```yaml
# observed_at: 2024-02-29
# target: February
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ## Comparison
#
# The generator balances leap-day February labels and ordinary-year March labels, then separates training,
# validation, and test years. The restricted model sees only the shared ordinal day. The richer model also
# sees week-of-month, which directly removes this ambiguity without requiring it to infer whether a hidden
# year is leap. The original budgets differ: ten epochs for the restricted model and twenty for the richer
# one, so this is not an equal-budget speed comparison.
#
# Compare day-of-year alone with a model that also receives week-of-month.
#
# ## Result
#
# Accuracy near 50% for day-only inputs is the expected consequence of balanced, indistinguishable cases.
# Higher accuracy with the additional coordinate shows that the selected representation provides
# distinguishing information. It does not show that the network derived the Gregorian leap-year rule from
# year numbers: those numbers are not exposed as inputs here. The result concerns this specific
# February/March collision, rather than general date arithmetic or every calendar boundary.
#
# Day number alone cannot separate identical inputs with different answers. The added coordinate resolves these cases; general leap-year reasoning is not tested.
#
# ## Method
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

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "lightning",
#     "pyyaml>=6",
#     "relflow>=0.6.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "representation"
# types = ["Period"]
# conditions = ["limited-information", "representations"]
# ///

"""P043: distinguish February 29 from March 1 at ordinal day sixty.

Day-of-year is identical for both labels. Adding week-of-month reveals their
difference. The original budgets are retained: ten epochs for day-only and
twenty for the richer input, so this is not an equal-budget comparison.
"""

import calendar
import sys
from collections.abc import Callable, Iterator, Sequence
from datetime import date
from functools import partial
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import relflow as rf
import torch
from reporting import experiment, report


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
    parts: Sequence[str],
    train: Callable[[], Iterator[dict]],
    validate: Callable[[], Iterator[dict]],
    epochs: int,
    seed: int,
    steps: int | None,
    accelerator: str,
) -> rf.Model:
    """Train on the selected calendar coordinates with the remaining schema fixed."""
    lit.seed_everything(seed, workers=True)

    class PredictionModel(rf.Model):
        observed_at = rf.Period(parts=list(parts))
        target = rf.Category(predict=True, p_unavailable=0.0)

    model = PredictionModel(d_model=64, n_layers=3, n_heads=4)
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


@experiment("P043", seed=53)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train = partial(records, years=tuple(range(1901, 1951)), rows=512)
    validate = partial(records, years=tuple(range(1951, 2001)), rows=128)
    test = partial(records, years=tuple(range(2001, 2051)), rows=256)
    day_only = fit(
        parts=("day_of_year",),
        train=train,
        validate=validate,
        epochs=10,
        seed=seed,
        steps=steps,
        accelerator=accelerator,
    )
    ambiguous_accuracy = accuracy(day_only, test, accelerator)
    identified = fit(
        parts=("day_of_year", "week_of_month"),
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
    report(run)
