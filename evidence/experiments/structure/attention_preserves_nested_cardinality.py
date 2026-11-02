# %% [markdown]
# ---
# title: Does the grouping of values matter?
# categories:
# - Hierarchical statistics
# evidence-id: P040
# description: Predict the largest session total when the same six amounts are divided into different sessions.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Model error
#   metric:
#   - test_rmse
#   format: number
# - label: Error after discarding grouping
#   metric:
#   - flat_oracle_rmse
#   format: number
# evidence-group: Structure
# evidence-result:
#   accelerator: cpu
#   checks:
#     Attention improves over the flattened oracle by at least 25%: true
#     Attention recovers at least 75% of the pair delta: true
#   duration_seconds: 3.121
#   metrics:
#     flat_oracle_rmse: 1.0647465517891985
#     predicted_pair_delta: 1.9371915736199545
#     recovered_pair_delta_fraction: 0.9635957218854316
#     steps: 80
#     test_rmse: 0.17535235301382399
#     true_pair_delta: 2.010377930933031
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 6815df82b77f5c64829e1da572223f4bb27c4d10f8bd5321dba0ce8e8e5512ec
#     cuda: null
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: c1d047d1667d
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
#     script_sha256: 2d5b5809e9f68bc457781395eb50590b25a52ad804a438b0b165d6c641cdf430
#     source_sha256: 36f910f2bea92538536b2c6354aab2fec27cd80d24918beec0c8ec07747f50a2
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 2601
#   started_at: '2026-11-02T17:52:33.639734+00:00'
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
# sessions:
#   - transactions:
#       - amount: 0.5
#   - transactions:
#       - amount: 0.5
#       - amount: 0.5
#       - amount: 0.5
#       - amount: 0.5
#   - transactions:
#       - amount: 0.5
# global_total: 3.0
# largest_session_total: 2.0
# ```
#
# largest_session_total is hidden. global_total is an explanatory value, not a model input.
#
# ```{typst}
# //| label: fig-evidence-structure-attention-preserves-nested-cardinality
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with record, sessions, transactions, amount, largest_session_total. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("record", kind: "root", width: 120pt, children: (
#   node("sessions", kind: "branch", repeated: true, width: 120pt, children: (
#     node("transactions", kind: "branch", repeated: true, width: 120pt, children: (
#       node("amount", type: "Number"),
#     )),
#   )),
#   node("largest_session_total", width: 180pt, kind: "target", type: "Number",),
# )))
# ```
#
# ## Comparison
#
# Pair records with identical flat values but different grouping. Compare with the best prediction available after discarding that grouping.
#
# ## Result
#
# {{< evidence P040 status >}}
#
# The model retains useful information about the hierarchy. Only two arrangements of equal amounts are tested; general nested arithmetic remains a larger task.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P040 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P040 script >}}
#
# ### How it works
#
# The session, transaction, and root contexts all use Attention reduction.
# The reducer adds a projected sum and an explicit count contribution alongside
# normalized attention. This additional path lets equal tokens carry different
# mass when their repetition count changes.
#
# Each pair uses the same six amounts, with session sizes `(2, 2, 2)` versus
# `(1, 4, 1)`. The amount varies between pairs. The model trains on 128 pairs,
# validates on 32, and tests on 64 independently generated pairs, after 80
# deterministic steps.
#
# The comparison oracle sees only flattened values. It must predict the same
# answer for both members of a pair; their target average is its optimal
# squared-error prediction. Beating it requires retaining session structure.
#
# ### Remaining work
#
# Add nested global totals and largest-session averages. Vary amounts within
# sessions, test more partitions, and vary Attention output counts and nested
# capacities. Repeat the nested gates over three paired core seeds and at least
# ten calibration seeds without weakening either threshold.
#
# ### Complete experiment code
#

# %%
"""P040: retain local counts before selecting the largest session total.

Paired records contain the same six values partitioned as (2, 2, 2) and
(1, 4, 1). Flattening loses the answer. Attention must beat that erased-input
oracle and recover the difference between the paired targets.
"""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P040"


def records(*, pairs: int, seed: int) -> Iterator[dict]:
    """Yield adjacent regroupings of the same six equal transaction values."""
    rng = np.random.default_rng(seed)
    for _ in range(pairs):
        amount = float(rng.uniform(0.4, 1.6))
        for lengths in ((2, 2, 2), (1, 4, 1)):
            yield {
                "sessions": [{"transactions": [{"amount": amount} for _ in range(length)]} for length in lengths],
                "global_total": 6 * amount,
                "largest_session_total": max(lengths) * amount,
            }


def prediction(model: rf.Model, rows: list[dict], source: str, target: str) -> np.ndarray:
    """Predict one hidden scalar from its visible repeated context."""
    inputs = [{source: row[source]} for row in rows]
    output = model.predict(inputs).to_pylist()
    return np.asarray([row["predictions"][target]["content"] for row in output], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    data_seed = (seed - 2601) % (2**32)
    train = partial(records, pairs=128, seed=data_seed + 44)
    validate = partial(records, pairs=32, seed=data_seed + 55)
    test = list(records(pairs=64, seed=data_seed + 66))
    flattened = [[item for session in row["sessions"] for item in session["transactions"]] for row in test]
    if not all(flattened[index] == flattened[index + 1] for index in range(0, len(flattened), 2)):
        raise ValueError("regrouping pairs must contain identical flattened values")

    model = rf.Model(
        d_model=24,
        n_layers=1,
        n_heads=4,
        reduction=rf.Attention(),
        sessions=rf.Branch(
            length=3,
            reduction=rf.Attention(),
            transactions=rf.Branch(length=4, reduction=rf.Attention(), amount=rf.Number),
        ),
        largest_session_total=rf.Number(mask=True, objective="mse"),
    )
    data = rf.SyntheticDataModule(train=train, validate=validate, seed=seed)
    model.fit(
        data,
        batch_size=64,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.003),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
        accelerator=accelerator,
        max_epochs=-1,
        max_steps=steps if steps is not None else 80,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer = model.trainer

    targets = np.asarray([row["largest_session_total"] for row in test])
    predictions = prediction(model, test, "sessions", "largest_session_total")
    paired_targets = targets.reshape(-1, 2)
    flat_oracle = np.repeat(paired_targets.mean(axis=1), 2)
    oracle_rmse = rmse(targets, flat_oracle)
    test_rmse = rmse(targets, predictions)
    predicted_delta = float(np.abs(np.diff(predictions.reshape(-1, 2), axis=1)).mean())
    true_delta = float(np.abs(np.diff(paired_targets, axis=1)).mean())
    return {
        "steps": trainer.global_step,
        "flat_oracle_rmse": oracle_rmse,
        "test_rmse": test_rmse,
        "predicted_pair_delta": predicted_delta,
        "true_pair_delta": true_delta,
        "recovered_pair_delta_fraction": predicted_delta / true_delta,
    }, {
        "Attention improves over the flattened oracle by at least 25%": bool(
            np.isfinite(test_rmse) and test_rmse < oracle_rmse * 0.75
        ),
        "Attention recovers at least 75% of the pair delta": predicted_delta >= true_delta * 0.75,
    }


if __name__ == "__main__":
    report(PROOF_ID, run, seed=2601)

# %% [markdown]
# </details>
