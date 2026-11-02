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
#   duration_seconds: 5.566
#   metrics:
#     accuracy_gap: 0.8571428507566452
#     day_only_accuracy: 0.1428571492433548
#     with_weekday_accuracy: 1.0
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 3a24dc4bbe82e1da897b9c5c029eea8ebc138d6958acd0e582f7a1dd84b9ed70
#     cuda: null
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: 374a3aa52829
#     packages:
#       lightning: 2.6.4
#       numpy: 2.4.6
#       polars: 2.0.0
#       pyarrow: 24.0.0
#       pydantic: 2.13.4
#       relflow: 0.4.2
#       tensordict: 0.12.4
#       torch: 2.12.0+cpu
#       torchmetrics: 1.9.0
#     platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.41
#     python: 3.12.16
#     relflow_path: /tmp/runtime/lib/python3.12/site-packages/relflow
#     relflow_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     relflow_source_sha256: 77953506049ed76c92adc5f32a9bcc32432a788068a4e366b2f78bcf709f29a2
#     script_sha256: dfbab2589236506bac6259ef6306fe6be5f4b640944b9dfd3be828bfe7516e40
#     source_sha256: b0740916630472313c34fbc7370e84bed40d7a3ecfa73c91a12eaf8bde631973
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 47
#   started_at: '2026-11-02T17:52:52.863353+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41494093900-1
#   candidate:
#     payload_sha256: 2169c968a7733085889a6eed6e895bcb2590059bcc46243d49a61ffacc42f67f
#     revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   baseline:
#     payload_sha256: 2a6af4bd3869faac23f6b7327f2c59e8f5b367580b2f2fa609710958775c08ed
#     revision: 7dec5e860dc389c37c3642d8904867249e831773
#     wheel_sha256: d3da50b65e0c0c7b98059f095b82bf6bf5d0ea0c5ccd0f302015118075f1dd85
#   evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#   executor_revision: 92b3c887eabc598ee2fc6c9dba4e05b9166cb8ba
#   manifest_sha256: 4f0926f3fc366fd8d14bf728b2d06f4a475d61852679176704816c110ae51cfd
#   passed: true
#   github_run:
#     event: push
#     head_branch: main
#     head_sha: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     html_url: https://github.com/relflow/relflow/actions/runs/41494093900
#     path: .github/workflows/ci.yml
#     run_attempt: 1
#     run_id: 41494093900
#     workflow_id: 403231172
#   release: https://github.com/relflow/relflow/releases/tag/evidence-ci-41494093900-attempt-1
#   reports_sha256: b1bfee5622ffbd389b1777443dc05e6d8f08dccff9e2540a601f0c2608948594
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
