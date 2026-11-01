# %% [markdown]
# ---
# title: Can day-of-year alone identify the weekday?
# categories:
# - Calendar reasoning
# evidence-id: P045
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
# evidence-result:
#   accelerator: cpu
#   checks:
#     Day-of-year accuracy is at most 0.20: true
#     Visible weekday accuracy reaches 0.95: true
#     Visible weekday improves accuracy by at least 0.70: true
#   duration_seconds: 5.355
#   metrics:
#     accuracy_gap: 0.8571428507566452
#     day_only_accuracy: 0.1428571492433548
#     with_weekday_accuracy: 1.0
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 3a24dc4bbe82e1da897b9c5c029eea8ebc138d6958acd0e582f7a1dd84b9ed70
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 789f19298ce5
#     packages:
#       lightning: 2.6.4
#       numpy: 2.4.6
#       polars: 2.0.0
#       pyarrow: 24.0.0
#       pydantic: 2.13.4
#       relflow: 0.4.1
#       tensordict: 0.12.4
#       torch: 2.12.0+cpu
#       torchmetrics: 1.9.0
#     platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.41
#     python: 3.12.16
#     relflow_path: /tmp/runtime/lib/python3.12/site-packages/relflow
#     relflow_revision: 41d6f5fb74c0ba9f0be104c896913a55adb125f5
#     relflow_source_sha256: 00fdf397aeefb3a4e705d431850e0a6294eef142989513fddcb903bb1aefa873
#     script_sha256: 19e4bc9c0092edfa756262c71ffae059cc7d88fd00faab4403d35b61ed4f6dce
#     source_sha256: 262dfe543d8ac4ddc2a1bbe30b4b5832687a6e6ad7f45a5f0d474d29eaa83815
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 47
#   started_at: '2026-11-01T01:46:17.165189+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41261482839-1
#   candidate:
#     revision: 41d6f5fb74c0ba9f0be104c896913a55adb125f5
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   baseline:
#     revision: 1add6d8919e686dc63b7fa6c4cdbdc437be2b570
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#   manifest_sha256: f4b3bfe706e43e38785ab5bed94c3f51eb44fe7314d0a1b2f0dd3c6f3f7bef92
#   passed: true
#   github_run:
#     event: pull_request
#     head_branch: dev/evidence-in-docs
#     head_sha: 8bb7764d8186c743eb554cb4a91dedb205ad4cb9
#     html_url: https://github.com/relflow/relflow/actions/runs/41261482839
#     pull_requests:
#     - base_sha: 1add6d8919e686dc63b7fa6c4cdbdc437be2b570
#       head_sha: 8bb7764d8186c743eb554cb4a91dedb205ad4cb9
#       number: 64
#     run_attempt: 1
#     run_id: 41261482839
#   release: https://github.com/relflow/docs/releases/tag/ci-41261482839-attempt-1
#   reports_sha256: bc67a1cb3725cbf9c5a8deb65844297d70f6d2122cd3b7de5c1d15f56603864f
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
# Compare day number alone with a model that also receives the weekday coordinate.
#
# ## Result
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
from reporting import report

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
