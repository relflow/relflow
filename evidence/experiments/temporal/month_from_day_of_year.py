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
# evidence-result:
#   accelerator: cpu
#   checks:
#     Original dates exceed permutation by at least 0.65 accuracy: true
#     Timestamp permutation accuracy is at most 0.20: true
#     Unseen-date month accuracy reaches 0.90: true
#   duration_seconds: 12.806
#   metrics:
#     accuracy_gap: 0.8519552871584892
#     periodic_accuracy: 0.9441340565681458
#     permuted_accuracy: 0.09217876940965652
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: b44b9c5bbe327eae6295bb6caf5f91ca58af34aa65784efcff81fec72d478193
#     cuda: null
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: 6aed2c95e420
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
#     script_sha256: e2d9d98f059701f9705a65698f0acc981ecdcb31b49a6eba6d913e3f2d192845
#     source_sha256: 0b8cd43ea9ba30ce61cc011635113a0519d87aaba7cfd413cb8431f352f93f96
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 41
#   started_at: '2026-11-02T17:52:52.003658+00:00'
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
