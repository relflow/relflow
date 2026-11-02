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
#   duration_seconds: 12.261
#   metrics:
#     accuracy_gap: 0.8519552871584892
#     periodic_accuracy: 0.9441340565681458
#     permuted_accuracy: 0.09217876940965652
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: b44b9c5bbe327eae6295bb6caf5f91ca58af34aa65784efcff81fec72d478193
#     cuda: null
#     evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     gpu: null
#     host: 06a3d12d624b
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
#     relflow_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     relflow_source_sha256: 77953506049ed76c92adc5f32a9bcc32432a788068a4e366b2f78bcf709f29a2
#     script_sha256: 89d63c29a93ff3bdd63de6f5d56a33a221aeec75a04e751b2a270f4f97f7f5fb
#     source_sha256: bbf700129452e705cbdb40c66576364805b11934ee233e28ab138a156946c394
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   seed: 41
#   started_at: '2026-11-02T04:18:40.408126+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41386396488-1
#   candidate:
#     revision: 7dec5e860dc389c37c3642d8904867249e831773
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   baseline:
#     revision: 7dec5e860dc389c37c3642d8904867249e831773
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#   executor_revision: 349b2202e6390a5fe0cbdf09a0c5972e495351ed
#   manifest_sha256: 1f096d9b4dcdf1b2235963715e8b41c325d2a989f2cf15da7c154ddfb9bfb08d
#   passed: true
#   github_run:
#     event: workflow_dispatch
#     head_branch: main
#     head_sha: 7dec5e860dc389c37c3642d8904867249e831773
#     html_url: https://github.com/relflow/relflow/actions/runs/41386396488
#     path: .github/workflows/ci.yml
#     run_attempt: 1
#     run_id: 41386396488
#     workflow_id: 403231172
#   release: https://github.com/relflow/relflow/releases/tag/evidence-ci-41386396488-attempt-1
#   reports_sha256: fca05e4cdb79410b5090075bab03e4fc6d4f2fbc70e798f208976f4f1d8a7ff9
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
