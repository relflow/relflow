# %% [markdown]
# ---
# title: Does Duration learn the same fixed-unit signal as a prepared Number?
# categories:
# - Temporal
# description: Compare elapsed-time features derived from two timestamps or one configured endpoint with an equivalent
#   numeric feature.
# evidence-readout:
# - label: Two-query Duration error
#   metric:
#   - arms
#   - duration
#   - nrmse
#   format: number
# - label: Prepared Number error
#   metric:
#   - arms
#   - number
#   - nrmse
#   format: number
# - label: Configured-start error
#   metric:
#   - arms
#   - static
#   - nrmse
#   format: number
# evidence-group: Temporal
# evidence-diagram:
#   caption: The target is hidden. The diagram shows the informative representation; controls change only the declared
#     condition.
#   alt: Does Duration learn the same fixed-unit signal as a prepared Number? Visible inputs lead to one hidden prediction
#     target.
# evidence-review:
#   question: Does Duration learn the same fixed-unit signal as a prepared Number?
#   modeling: Generate explicit UTC timestamp endpoints and a numeric target determined by their signed fractional-day
#     difference. Compare a two-query Duration, a Duration with a configured start, and the exact same elapsed value
#     prepared as Number.
#   comparison:
#     metric: Normalized RMSE
#     kind: Controlled comparison
#     conditions:
#     - label: Prepared numeric elapsed time
#       path:
#       - arms
#       - number
#       - nrmse
#       format: number
#     - label: Duration from two endpoints
#       path:
#       - arms
#       - duration
#       - nrmse
#       format: number
#     - label: Duration with configured start
#       path:
#       - arms
#       - static
#       - nrmse
#       format: number
#     definition: Root mean squared error divided by the reference scale defined in this experiment. Zero is exact;
#       one equals that reference scale.
#     direction: Lower error is better
#   note: Duration computes elapsed time before learning. This tests equivalent learnable signals, not neural date
#     arithmetic; translating both endpoints is checked separately.
# evidence-plot:
#   target: target
#   focus: Does elapsed-time Duration behave like the equivalent prepared Number?
#   comparison: Compare held-out predictions, error distributions and normalized error for two-query Duration, a fixed
#     start, prepared Number and shuffled training durations.
#   limit: One seed and a fixed synthetic task test numeric equivalence. Joint endpoint translation is an arithmetic
#     invariant, not learned calendar reasoning.
#   renderer: plot
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
# Days are fixed 86,400-second intervals here; months and years remain outside
# this experiment.
#
# ```typst
# #tree(node("record", kind: "root", children: (node("elapsed", type: "Duration", detail: "End minus start, days"), node("target", kind: "target", type: "Number"),)))
# ```
#
# ## Comparison
#
# Compare a two-query Duration, a configured-start Duration, and a prederived Number using the same
# latent elapsed times and numerical targets. Translating both endpoints by 371 days should preserve the
# two-query model's predictions.
# A Number control shuffles only the training durations, preserving their distribution while breaking
# their relationship with the target. Its validation and test observations stay intact.
#
# ## Result
#
# The numerical checks below are provisional, predeclared hypotheses, not measured successes. No full
# learning result is claimed until a matching run is recorded. Smoke runs establish execution only.
#
# ## Method
#
# ### Population, controls, and scoring
#
# Each matched arm receives 4,096 training, 512 validation, and 2,048 test rows and 400 updates without
# checkpoint selection. Starts span years and elapsed times are fractional signed days. Source columns
# are explicit UTC Arrow timestamps. The Number baseline is computed from the exact source endpoints,
# rather than an independently rounded duration. This probes the convenience type's learning
# equivalence, not exact date arithmetic performed by neural weights. Query broadcasting and malformed
# timestamp rejection remain separate package contracts.
# The additional shuffled Number control uses the same samples and 400-update budget. Its training
# permutation uses seed + 4 without changing any of the three original arms.
#

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.10",
#     "lightning",
#     "numpy",
#     "pyarrow",
#     "pyyaml>=6",
#     "relflow>=0.5.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "representation"
# types = ["Number", "Duration"]
# conditions = ["changed-associations", "transformations", "representations"]
# ///

"""P099: compare fixed-unit Duration features with explicitly derived Number inputs."""

import sys
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from policies import CONTROL
from reporting import experiment, report

BUDGET = 400
TRAIN_ROWS = 4096
TEST_ROWS = 2048
ANCHOR = datetime(2026, 1, 1, tzinfo=timezone.utc)


def records(*, rows: int, seed: int, static: bool = False, shuffled: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    starts = rng.integers(-2000, 2000, rows)
    days = rng.uniform(-10.0, 10.0, rows)
    noise = rng.normal(0, 0.03, rows)
    prepared = []
    for start_day, elapsed, error in zip(starts, days, noise, strict=True):
        start = ANCHOR if static else ANCHOR + timedelta(days=int(start_day))
        end = start + timedelta(days=float(elapsed))
        actual = (end - start).total_seconds() / 86400.0
        prepared.append({"started_at": start, "ended_at": end, "elapsed": actual, "target": 2 * actual + float(error)})
    if shuffled:
        order = np.random.default_rng(seed + 3).permutation(rows)
        prepared = [{**row, "elapsed": prepared[index]["elapsed"]} for row, index in zip(prepared, order, strict=True)]
    yield from prepared


def table(rows: list[dict], *, arm: str, targets: bool = True) -> pa.Table:
    if arm in ("number", "shuffled"):
        columns = {"elapsed": pa.array([row["elapsed"] for row in rows], type=pa.float64())}
    else:
        timestamp = pa.timestamp("us", tz="UTC")
        columns = {name: pa.array([row[name] for row in rows], type=timestamp) for name in ("started_at", "ended_at")}
    if targets:
        columns["target"] = pa.array([row["target"] for row in rows], type=pa.float64())
    return pa.table(columns)


def build(arm: str) -> rf.Model:
    if arm in ("number", "shuffled"):
        feature = rf.Number
    elif arm == "static":
        feature = rf.Duration(query="ended_at", start=ANCHOR, granularity="days")
    else:
        feature = rf.Duration(query=("started_at", "ended_at"), granularity="days")

    class Model(rf.Model):
        elapsed = feature
        target = rf.Number(mask=True, objective="mse")

    model = Model(preset=CONTROL.name)
    return model


def prediction(model: rf.Model, rows: list[dict], arm: str) -> np.ndarray:
    return (
        model.predict(table(rows, arm=arm, targets=False), batch_size=256)["predictions"]
        .struct.field("target")
        .struct.field("content")
        .to_numpy()
    )


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
    for arm in ("duration", "static", "number", "shuffled"):
        lit.seed_everything(seed, workers=True)
        model = build(arm)
        train = list(records(rows=TRAIN_ROWS, seed=seed + 1, static=arm == "static", shuffled=arm == "shuffled"))
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
        if arm != "shuffled":
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
    checks["Configured-start Duration stays within 0.05 nRMSE of derived Number"] = (
        metrics["arms"]["static"]["nrmse"] <= metrics["arms"]["number"]["nrmse"] + 0.05
    )
    checks["Shuffled training durations retain nRMSE at least 0.90"] = metrics["arms"]["shuffled"]["nrmse"] >= 0.90
    checks["Duration beats shuffled training by at least 0.50 nRMSE"] = (
        metrics["arms"]["shuffled"]["nrmse"] - metrics["arms"]["duration"]["nrmse"] >= 0.50
    )
    return metrics, checks


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    from evidence.diagnostics.plots import COLORS, dots, finite

    "Compare elapsed-time readouts with the prepared and shuffled numeric controls."
    pairs = bundle.pairs(spec["target"])
    arms = ("duration", "static", "number", "shuffled")
    labels = ("Two-query Duration", "Configured-start Duration", "Prepared Number", "Shuffled training Number")
    measured = bundle.record["metrics"]
    selected = []
    for arm in arms:
        matches = [pair for pair in pairs if pair["context"].get("arm") == arm and "shifted" not in pair["call"]]
        if len(matches) != 1:
            raise ValueError(
                f"{bundle.manifest['id']}: expected one held-out prediction array for {arm}, got {len(matches)}"
            )
        selected.append(matches[0])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    primary = selected[0]
    actual, predicted = (finite(primary["actual"]), finite(primary["predicted"]))
    density = axes[0].hexbin(actual, predicted, gridsize=30, mincnt=1, bins="log", cmap="viridis")
    fig.colorbar(density, ax=axes[0], fraction=0.04, label="Held-out rows (log count)")
    bounds = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axes[0].plot(bounds, bounds, color="gray", ls="--", lw=1)
    axes[0].set(title="Duration predicts the held-out target", xlabel="Actual target", ylabel="Predicted target")
    axes[0].set_aspect("equal", adjustable="box")
    drift = measured["arms"]["duration"]["joint_endpoint_shift_max_drift"]
    axes[0].text(
        0.03,
        0.97,
        f"n = {len(actual):,}\n371-day endpoint shift\nmax prediction drift = {drift:.3g}",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
    )
    examples = []
    for index, (pair, label) in enumerate(zip(selected, labels, strict=True)):
        truth, prediction = (finite(pair["actual"]), finite(pair["predicted"]))
        errors = np.sort(np.abs(truth - prediction))
        axes[1].step(
            errors, np.arange(1, len(errors) + 1) / len(errors), where="post", color=COLORS[index], label=label
        )
        examples.append(
            {
                "label": label,
                "field": spec["target"],
                "population": "test",
                "n": len(truth),
                "records": [
                    {
                        "coordinate": [i],
                        "input": pair["rows"][i],
                        "actual": float(truth[i]),
                        "predicted": float(prediction[i]),
                    }
                    for i in range(min(16, len(truth)))
                ],
            }
        )
    axes[1].set(
        title="How much error remains?",
        xlabel="Absolute target error",
        ylabel="Fraction of held-out rows",
        ylim=(0, 1.02),
    )
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].grid(alpha=0.18)
    axes[1].legend(fontsize=8, loc="lower right")
    dots(
        axes[2],
        labels,
        {"Held-out nRMSE": [measured["arms"][arm]["nrmse"] for arm in arms]},
        xlabel="RMSE / training-mean predictor RMSE",
        reference=(1, "Training-mean reference"),
    )
    axes[2].axvline(0.25, color=COLORS[0], ls="--", lw=1, label="Informative-arm criterion")
    axes[2].set(title="Does the input carry information?", xlim=(0, None))
    axes[2].legend(fontsize=8)
    note = f"All four arms use 2,048 independent held-out rows and 400 updates. The shuffled control permutes only training durations; test inputs remain intact. The dashed diagonal is exact prediction; the error curves retain every held-out row. Duration versus prepared Number prediction RMSE is {measured['duration_number_prediction_rmse']:.4g}. The endpoint-shift annotation uses the same trained model. One seed; no uncertainty interval."
    return (fig, note, {"kind": "numeric", "conditions": examples})


if __name__ == "__main__":
    report(run)
