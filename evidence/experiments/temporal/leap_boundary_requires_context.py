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
#   duration_seconds: 4.479
#   metrics:
#     accuracy_gap: 0.5
#     day_only_accuracy: 0.5
#     with_week_of_month_accuracy: 1.0
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: d4776427888fc84cdf369c240c4f49aa622dd081dfa20913ec924665828f9787
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 14c9f0ee7fb6
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
#     script_sha256: c9bb43f4747884abd535928c5f6387da148db5315868da9cb16c6de583fd1d19
#     source_sha256: ec0db522c77506642ddf8602ece111214e7eb83f25f9c2c811f41627d12b5337
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 53
#   started_at: '2026-11-01T01:46:03.316117+00:00'
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
