# %% [markdown]
# ---
# title: Does it need both weekday and hour?
# categories:
# - Calendar reasoning
# evidence-id: P042
# description: Predict whether a time lies in a weekday business-hours window.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Weekday only
#   metric:
#   - day_only_accuracy
#   format: percent
# - label: Hour only
#   metric:
#   - hour_only_accuracy
#   format: percent
# - label: Both
#   metric:
#   - composed_accuracy
#   format: percent
# evidence-group: Temporal
# evidence-result:
#   accelerator: cpu
#   checks:
#     Combined coordinates reach 0.95 accuracy: true
#     Composition improves over either coordinate by at least 0.15: true
#     Hour-only accuracy is at most 0.82: true
#     Weekday-only accuracy is at most 0.82: true
#   duration_seconds: 18.662
#   metrics:
#     accuracy_gap: 0.25
#     composed_accuracy: 1.0
#     day_only_accuracy: 0.75
#     hour_only_accuracy: 0.71875
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 3f74f2e5ca20108ec239838aefa854ffb18ab75781fa2c9a4d470efa3fb4d6ac
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 1c591129f67e
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
#     script_sha256: 3dc532c03fee071a2a7acc3c8a0c8efb9a1dad8e4c5e58f2c6ef3b467dc83e89
#     source_sha256: 88564336a32a0335a6996508fba195fb505d73bc0aa89d20b9c79915980e29da
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 59
#   started_at: '2026-11-01T01:45:52.762825+00:00'
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
# observed_at: 2025-01-06T09:00:00
# target: open
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-temporal-business-window-requires-composition
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with calendar, observed_at, target. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("calendar", kind: "root", children: (
#   node("observed_at", type: "DateParts", width: 150pt,),
#   node("target", kind: "target", type: "Category", detail: "open or closed",),
# )))
# ```
#
# ## Comparison
#
# Train with only weekday, only hour, and both together. Each single coordinate leaves ambiguous cases.
#
# ## Result
#
# {{< evidence P042 status >}}
#
# Combining the coordinates resolves this constructed task. Holidays, time zones, and daylight-saving rules are not covered.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P042 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P042 script >}}
#
# ### How it works
#
# Half the observations are weekdays during 09:00–16:00. A quarter are
# weekdays outside those hours, and a quarter are weekends during those hours.
# Neither coordinate alone identifies every label; a combined representation
# can distinguish the open window from both kinds of near miss.
#
# Three otherwise matched models expose `day_of_week`, `hour_of_day`, or both.
# They train for 20 deterministic epochs. Training, validation, and test
# use 1,024, 512, and 1,024 observations from separate periods beginning in
# 2017, 2021, and 2025.
#
# ### Remaining work
#
# Repeat all three fits over three core seeds and at least ten calibration
# seeds. Extend to further coordinate combinations and characterize timezone
# and daylight-saving preprocessing independently of this naive-calendar test.
#
# ### Complete experiment code
#

# %%
"""P042: combine weekday and hour to identify an open business window.

Half the rows are open; the others are weekday/off-hour or weekend/work-hour
near misses. Either coordinate alone has an optimal accuracy of 0.75.
"""

from collections.abc import Callable, Iterator, Sequence
from datetime import date, datetime, time, timedelta
from functools import partial

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P042"


def records(*, start: date, weeks: int, rows: int, seed: int) -> Iterator[dict]:
    """Yield balanced open windows and two single-coordinate near misses."""
    if start.weekday() != 0:
        raise ValueError(f"business-window start must be a Monday, got {start.isoformat()}")
    if weeks < 1:
        raise ValueError(f"business-window weeks must be positive, got {weeks}")
    if rows % 4:
        raise ValueError(f"business-window rows must be divisible by four, got {rows}")
    business_hours = tuple(range(9, 17))
    off_hours = (*range(0, 9), *range(17, 24))
    observations = []
    for index in range(rows // 2):
        day = start + timedelta(weeks=index % weeks, days=index % 5)
        hour = business_hours[index % len(business_hours)]
        observations.append({"observed_at": datetime.combine(day, time(hour=hour)), "target": "open"})
    for index in range(rows // 4):
        day = start + timedelta(weeks=index % weeks, days=index % 5)
        hour = off_hours[index % len(off_hours)]
        observations.append({"observed_at": datetime.combine(day, time(hour=hour)), "target": "closed"})
    for index in range(rows // 4):
        day = start + timedelta(weeks=index % weeks, days=5 + index % 2)
        hour = business_hours[index % len(business_hours)]
        observations.append({"observed_at": datetime.combine(day, time(hour=hour)), "target": "closed"})
    for index in np.random.default_rng(seed).permutation(len(observations)):
        yield observations[index]


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
    train = partial(records, start=date(2017, 1, 2), weeks=52, rows=1024, seed=seed + 1)
    validate = partial(records, start=date(2021, 1, 4), weeks=26, rows=512, seed=seed + 2)
    test = partial(records, start=date(2025, 1, 6), weeks=52, rows=1024, seed=seed + 3)
    day_only = fit(
        dateparts=("day_of_week",),
        train=train,
        validate=validate,
        epochs=20,
        seed=seed,
        steps=steps,
        accelerator=accelerator,
    )
    day_accuracy = accuracy(day_only, test, accelerator)
    hour_only = fit(
        dateparts=("hour_of_day",),
        train=train,
        validate=validate,
        epochs=20,
        seed=seed,
        steps=steps,
        accelerator=accelerator,
    )
    hour_accuracy = accuracy(hour_only, test, accelerator)
    composed = fit(
        dateparts=("day_of_week", "hour_of_day"),
        train=train,
        validate=validate,
        epochs=20,
        seed=seed,
        steps=steps,
        accelerator=accelerator,
    )
    composed_accuracy = accuracy(composed, test, accelerator)
    gap = composed_accuracy - max(day_accuracy, hour_accuracy)
    return {
        "day_only_accuracy": day_accuracy,
        "hour_only_accuracy": hour_accuracy,
        "composed_accuracy": composed_accuracy,
        "accuracy_gap": gap,
    }, {
        "Weekday-only accuracy is at most 0.82": day_accuracy <= 0.82,
        "Hour-only accuracy is at most 0.82": hour_accuracy <= 0.82,
        "Combined coordinates reach 0.95 accuracy": composed_accuracy >= 0.95,
        "Composition improves over either coordinate by at least 0.15": gap >= 0.15,
    }


if __name__ == "__main__":
    report(PROOF_ID, run, seed=59)

# %% [markdown]
# </details>
