# %% [markdown]
# ---
# title: Can it learn the month from the day number?
# categories:
# - Calendar reasoning
# evidence-id: P044
# description: Learn month boundaries from day-of-year in non-leap years.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Original dates
#   metric:
#   - periodic_accuracy
#   format: percent
# - label: Shuffled dates
#   metric:
#   - permuted_accuracy
#   format: percent
# evidence-group: Temporal
# ---
#
# ## Example
#
# ```yaml
# observed_at: 2025-02-04
# target: February
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-temporal-month-from-day-of-year
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
# Train on odd days of each month, test even days in unseen years, and shuffle timestamps while keeping month answers fixed.
#
# ## Result
#
# {{< evidence P044 status >}}
#
# The model learns the restricted calendar relationship. Exact month extraction is simpler in preprocessing, and leap-year ambiguity needs additional information.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P044 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P044 script >}}
#
# ### How it works
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
# ### Remaining work
#
# Repeat the gates over three core seeds and at least ten calibration seeds.
# Extend the family to week and multi-coordinate targets. Characterize timezone,
# daylight-saving, and leap-day preprocessing policies separately rather than
# assuming that DateParts adds those semantics.
#
# ### Complete experiment code
#

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
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P044"


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


if __name__ == "__main__":
    report(PROOF_ID, run, seed=41)

# %% [markdown]
# </details>
