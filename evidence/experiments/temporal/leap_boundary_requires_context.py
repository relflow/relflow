# %% [markdown]
# ---
# title: Can day 60 reveal the month in a leap year?
# categories:
# - Calendar reasoning
# evidence-id: P043
# description: Distinguish February 29 from March 1 when both can be day 60 of a year.
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
# - label: Week-of-month added
#   metric:
#   - with_week_of_month_accuracy
#   format: percent
# evidence-group: Temporal
# evidence-result:
#   accelerator: cpu
#   checks:
#     Day-of-year accuracy remains between 0.49 and 0.51: true
#     Visible week-of-month accuracy reaches 0.95: true
#     Visible week-of-month improves accuracy by at least 0.40: true
#   duration_seconds: 4.531
#   metrics:
#     accuracy_gap: 0.5
#     day_only_accuracy: 0.5
#     with_week_of_month_accuracy: 1.0
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: d4776427888fc84cdf369c240c4f49aa622dd081dfa20913ec924665828f9787
#     cuda: null
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: d0dec292d5ec
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
#     script_sha256: e7e86caf7538e062d222bba83c2f4962bd43e93a0bb6dd9074c4a5d046ed6e5e
#     source_sha256: 158fbdfe18c38d0c85ec0d5a6a9d5760ceabb8d9074f781c4ad854f8405d8a8f
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 53
#   started_at: '2026-11-02T17:52:50.542651+00:00'
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
# observed_at: 2024-02-29
# target: February
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-temporal-leap-boundary-requires-context
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
# {{< evidence P043 status >}}
#
# Day number alone cannot separate identical inputs with different answers. The added coordinate resolves these cases; general leap-year reasoning is not tested.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P043 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P043 script >}}
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
import relflow as rf
import torch
from reporting import report

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
