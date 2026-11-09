# %% [markdown]
# ---
# title: "Does Duration learn the same fixed-unit signal as a prepared Number?"
# categories: ["Temporal"]
# description: "Compare elapsed-time features derived from two timestamps or one configured endpoint with an equivalent numeric feature."
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: "Two-query Duration error"
#   metric: ["arms", "duration", "nrmse"]
#   format: number
# - label: "Prepared Number error"
#   metric: ["arms", "number", "nrmse"]
#   format: number
# - label: "Configured-start error"
#   metric: ["arms", "static", "nrmse"]
#   format: number
# evidence-group: Temporal
# evidence-requires:
# - "relflow stage/0.5.0 wheel exporting Duration"
# ---
#
# Compare elapsed-time features derived from two timestamps or one configured endpoint with an
# equivalent numeric feature.
#
# ## Example
#
# ```yaml
# started_at: "2026-01-01T12:00:00Z"
# ended_at: "2026-01-03T00:00:00Z"
# target: 3.0
# ```
#
# The two instants are 1.5 days apart, so the noiseless target is 3.0. Negative intervals are also
# valid; the sign comes from end minus start. The hidden target contains a small independent noise term
# in generated records.
#
# This research experiment needs the stage/0.5.0 wheel that exports Duration. The current core package
# does not yet provide that API. Days are fixed 86,400-second intervals here; months and years remain
# outside this experiment.
#
# ```{typst}
# //| label: fig-evidence-p099
# //| fig-cap: "The target is hidden. The diagram shows the informative representation; controls change only the declared condition."
# //| fig-alt: "Does Duration learn the same fixed-unit signal as a prepared Number? Visible inputs lead to one hidden prediction target."
# #tree(node("record", kind: "root", children: (node("elapsed", type: "Duration", detail: "End minus start, days"), node("target", kind: "target", type: "Number"),)))
# ```
#
# ## Comparison
#
# Compare a two-query Duration, a configured-start Duration, and a prederived Number using the same
# latent elapsed times and numerical targets. Translating both endpoints by 371 days should preserve the
# two-query model's predictions.
#
# ## Result
#
# {{< evidence P099 status >}}
#
# The numerical checks below are provisional, predeclared hypotheses, not measured successes. No full
# learning result is claimed until a matching run is recorded. Smoke runs establish execution only.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P099 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P099 script >}}
#
# ### Population, controls, and scoring
#
# Each matched arm receives 4,096 training, 512 validation, and 2,048 test rows and 400 updates without
# checkpoint selection. Starts span years and elapsed times are fractional signed days. Source columns
# are explicit UTC Arrow timestamps. The Number baseline is computed from the exact source endpoints,
# rather than an independently rounded duration. This probes the convenience type's learning
# equivalence, not exact date arithmetic performed by neural weights. Query broadcasting and malformed
# timestamp rejection remain separate package contracts.
#
# ### Complete experiment code

# %%
"""P099: compare fixed-unit Duration features with explicitly derived Number inputs."""

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import experiment, report

BUDGET = 400
TRAIN_ROWS = 4096
TEST_ROWS = 2048
ANCHOR = datetime(2026, 1, 1, tzinfo=timezone.utc)


def records(*, rows: int, seed: int, static: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    starts = rng.integers(-2000, 2000, rows)
    days = rng.uniform(-10.0, 10.0, rows)
    noise = rng.normal(0, 0.03, rows)
    for start_day, elapsed, error in zip(starts, days, noise, strict=True):
        start = ANCHOR if static else ANCHOR + timedelta(days=int(start_day))
        end = start + timedelta(days=float(elapsed))
        actual = (end - start).total_seconds() / 86400.0
        yield {"started_at": start, "ended_at": end, "elapsed": actual, "target": 2 * actual + float(error)}


def table(rows: list[dict], *, arm: str, targets: bool = True) -> pa.Table:
    if arm == "number":
        columns = {"elapsed": pa.array([row["elapsed"] for row in rows], type=pa.float64())}
    else:
        timestamp = pa.timestamp("us", tz="UTC")
        columns = {name: pa.array([row[name] for row in rows], type=timestamp) for name in ("started_at", "ended_at")}
    if targets:
        columns["target"] = pa.array([row["target"] for row in rows], type=pa.float64())
    return pa.table(columns)


def build(arm: str) -> rf.Model:
    if not hasattr(rf, "Duration"):
        raise RuntimeError(
            "P099 requires the stage/0.5.0 relflow wheel exporting rf.Duration; the ordinary core wheel cannot execute this research experiment"
        )
    if arm == "number":
        feature = rf.Number
    elif arm == "static":
        feature = rf.Duration(query="ended_at", start=ANCHOR, granularity="days")
    else:
        feature = rf.Duration(query=("started_at", "ended_at"), granularity="days")
    return rf.Model.xs(elapsed=feature, target=rf.Number(mask=True, objective="mse"))


def prediction(model: rf.Model, rows: list[dict], arm: str) -> np.ndarray:
    output = pa.table(model.predict(table(rows, arm=arm, targets=False), batch_size=256))["predictions"].to_pylist()
    return np.asarray([row["target"]["content"] for row in output], dtype=np.float64)


def fit(model: rf.Model, train: pa.Table, validate: pa.Table, seed: int, steps: int | None, accelerator: str) -> int:
    """Fit the declared fixed-data protocol without checkpoint selection."""
    data = rf.ArrowDataModule(train=train, validate=validate, num_workers=0, persistent_workers=False, pin_memory=False)
    model.fit(
        data,
        batch_size=64,
        learning_rate=0.003,
        weight_decay=0.01,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=BUDGET if steps is None else min(steps, BUDGET),
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    model.eval()
    return int(model.trainer.global_step)


@experiment("P099", seed=9901)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    metrics = {"budget": BUDGET, "arms": {}}
    checks = {}
    predictions = {}
    for arm in ("duration", "static", "number"):
        lit.seed_everything(seed, workers=True)
        model = build(arm)
        train = list(records(rows=TRAIN_ROWS, seed=seed + 1, static=arm == "static"))
        validation = list(records(rows=512, seed=seed + 2, static=arm == "static"))
        test = list(records(rows=TEST_ROWS, seed=seed + 3, static=arm == "static"))
        updates = fit(model, table(train, arm=arm), table(validation, arm=arm), seed, steps, accelerator)
        actual = np.asarray([row["target"] for row in test])
        predicted = prediction(model, test, arm)
        predictions[arm] = predicted
        baseline = float(np.sqrt(np.mean((actual - np.mean([row["target"] for row in train])) ** 2)))
        measured = {
            "steps": updates,
            "nrmse": float(np.sqrt(np.mean((predicted - actual) ** 2))) / baseline,
            "finite": bool(np.isfinite(predicted).all()),
        }
        if arm == "duration":
            shifted = [
                {
                    **row,
                    "started_at": row["started_at"] + timedelta(days=371),
                    "ended_at": row["ended_at"] + timedelta(days=371),
                }
                for row in test
            ]
            measured["joint_endpoint_shift_max_drift"] = float(
                np.max(np.abs(predicted - prediction(model, shifted, arm)))
            )
        metrics["arms"][arm] = measured
        checks[f"{arm}: full budget retained"] = updates == BUDGET
        checks[f"{arm}: finite predictions"] = measured["finite"]
        checks[f"{arm}: held-out nRMSE below 0.25"] = measured["nrmse"] < 0.25
    metrics["duration_number_prediction_rmse"] = float(
        np.sqrt(np.mean((predictions["duration"] - predictions["number"]) ** 2))
    )
    checks["Duration stays within 0.05 nRMSE of derived Number"] = (
        metrics["arms"]["duration"]["nrmse"] <= metrics["arms"]["number"]["nrmse"] + 0.05
    )
    checks["Moving both endpoints preserves elapsed-time predictions"] = (
        metrics["arms"]["duration"]["joint_endpoint_shift_max_drift"] < 1e-5
    )
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
