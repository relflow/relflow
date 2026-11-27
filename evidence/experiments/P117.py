# %% [markdown]
# ---
# title: Can coordinates predict unseen positions across the dateline?
# description: Predict held-out geographic positions from numeric clues and compare Coordinates with Vector and a
#   shuffled control.
# evidence-readout:
# - label: Constant training medoid
#   metric:
#   - baseline
#   - mae_m
#   format: number
# - label: Vector
#   metric:
#   - arms
#   - vector
#   - overall
#   - mae_m
#   format: number
# - label: Coordinates
#   metric:
#   - arms
#   - coordinates
#   - overall
#   - mae_m
#   format: number
# evidence-group: Representation
# evidence-review:
#   question: Can coordinates predict unseen positions across the dateline?
#   modeling: Use independent numeric clues to predict new positions in a cluster crossing the date line. Compare
#     Coordinates, Vector, and shuffled clues with the same observations and budgets.
#   comparison:
#     metric: Great-circle MAE (m)
#     kind: Controlled comparison
#     conditions:
#     - label: Constant training medoid
#       path:
#       - baseline
#       - mae_m
#       format: number
#     - label: Vector
#       path:
#       - arms
#       - vector
#       - overall
#       - mae_m
#       format: number
#     - label: Coordinates
#       path:
#       - arms
#       - coordinates
#       - overall
#       - mae_m
#       format: number
#     definition: Mean spherical great-circle distance between predicted and observed positions, in meters.
#     direction: Lower error is better
#   note: Positions are synthetic surface points; this does not measure roads, routing or real-world accuracy.
# evidence-plot:
#   renderer: plot
#   target: target
#   focus: Can coordinates predict unseen positions across the dateline?
#   comparison: Predict held-out geographic positions from numeric clues and compare Coordinates with Vector and a
#     shuffled control.
#   limit: Positions are synthetic surface points; this does not measure roads, routing or real-world accuracy.
# ---
#
# The numeric clues determine latitude and longitude on either side of the date line. Training and evaluation use independently sampled clues, and the position is hidden at inference.
#
# ## Example
#
# ```yaml
# north: 0.5
# east: -0.5
# position: [20.5, 178.5]
# ```
#
# ## Comparison
#
# Use independent numeric clues to predict new positions in a cluster crossing the date line. Compare Coordinates, Vector, and shuffled clues with the same observations and budgets.
# Independent splits, matched seeds, equal update budgets, and a negative control are retained.
#
# ## Result
#
# Measurements and checks are recorded independently; one synthetic task does not establish general performance.
#
# ## Method
#
# Each arm uses the same declared budget, optimizer, model/data seeds, and independent held-out observations.
# Per-arm metrics retain subgroup results and aligned predictions.
# Content and predicted value states are recorded separately; all test targets are valued.
#
# Generate 4,096 training, 512 validation, and 1,024 independent held-out pairs of
# clues uniformly in [-1, 1]. Latitude is 20 + north; longitude is 179.5 + 2 × east,
# wrapped to [-180, 180]. Run 500 updates per arm. The medoid is selected from 128
# training points using all training positions. For the unconstrained Vector output,
# clip latitude and wrap longitude before computing spherical error; retain raw
# invalid-coordinate counts and outputs. Score the two sides of the date line separately.
#
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["lightning", "matplotlib>=3.10", "numpy", "polars", "pyarrow", "pyyaml>=6", "relflow>=0.5.0", "torch"]
#
# [tool.relflow.evidence]
# capability = "representation"
# types = ["Coordinates", "Vector", "Number"]
# conditions = ["unseen-inputs", "changed-associations", "representations"]
# ///

"""P117: predict new geographic positions across the date line."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
from policies import CONTROL
from reporting import experiment, report
from visibility import TARGET

BUDGET = 500
TRAIN_ROWS = 4096
VALIDATION_ROWS = 512
TEST_ROWS = 1024
RADIUS_M = 6_371_009.0
ARMS = ("coordinates", "vector", "shuffled")


class CoordinateModel(rf.Model):
    north = rf.Number
    east = rf.Number
    position = rf.Coordinates(query=("latitude", "longitude"), **TARGET)


class VectorModel(rf.Model):
    north = rf.Number
    east = rf.Number
    position = rf.Vector(n_dim=2, **TARGET)


def distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Independent spherical haversine distance in meters."""
    first, second = np.deg2rad(first), np.deg2rad(second)
    dlat = first[..., 0] - second[..., 0]
    dlon = first[..., 1] - second[..., 1]
    a = np.sin(dlat / 2) ** 2 + np.cos(first[..., 0]) * np.cos(second[..., 0]) * np.sin(dlon / 2) ** 2
    return 2 * RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def records(*, rows: int, seed: int):
    rng = np.random.default_rng(seed)
    for index in range(rows):
        north, east = rng.uniform(-1.0, 1.0, 2)
        latitude = 20.0 + north
        longitude = (179.5 + 2.0 * east + 180.0) % 360.0 - 180.0
        yield {
            "case": f"{seed}:{index}",
            "north": float(north),
            "east": float(east),
            "position": [float(latitude), float(longitude)],
        }


def frame(rows: list[dict], *, targets: bool = True, shuffle: bool = False, seed: int = 0) -> pl.DataFrame:
    indices = np.arange(len(rows))
    if shuffle:
        indices = np.random.default_rng(seed).permutation(indices)
    data = {"north": [rows[int(i)]["north"] for i in indices], "east": [rows[int(i)]["east"] for i in indices]}
    if targets:
        points = [row["position"] for row in rows]
        data.update(position=points, latitude=[point[0] for point in points], longitude=[point[1] for point in points])
    return pl.DataFrame(data)


def fit(
    model: rf.Model, train: pl.DataFrame, validate: pl.DataFrame, seed: int, steps: int | None, accelerator: str
) -> int:
    data = rf.PolarsDataModule(
        train=train, validate=validate, seed=seed, num_workers=0, persistent_workers=False, pin_memory=False
    )
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


def score(actual: np.ndarray, predicted: np.ndarray, selected: np.ndarray) -> dict:
    errors = distance(predicted[selected], actual[selected])
    return {"rows": int(len(errors)), "mae_m": float(np.mean(errors)), "rmse_m": float(np.sqrt(np.mean(errors**2)))}


@experiment("P117", seed=11701)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train = list(records(rows=TRAIN_ROWS, seed=seed + 1))
    validate = list(records(rows=VALIDATION_ROWS, seed=seed + 2))
    test = list(records(rows=TEST_ROWS, seed=seed + 3))
    training = np.asarray([row["position"] for row in train])
    actual = np.asarray([row["position"] for row in test])
    # A deterministic subset gives a bounded training-only medoid reference.
    candidates = training[:128]
    medoid = candidates[np.argmin(distance(candidates[:, None, :], training[None, :, :]).mean(axis=1))]
    baseline = score(actual, np.broadcast_to(medoid, actual.shape), np.ones(len(test), dtype=bool))
    groups = {"overall": np.ones(len(test), dtype=bool), "west": actual[:, 1] < 0, "east": actual[:, 1] >= 0}
    metrics = {
        "budget": BUDGET,
        "rows": {"train": len(train), "validation": len(validate), "test": len(test)},
        "baseline": {**baseline, "position": medoid},
        "test": {"case": [r["case"] for r in test], "position": actual},
        "arms": {},
    }
    checks = {}
    for arm in ARMS:
        lit.seed_everything(seed, workers=True)
        model = (VectorModel if arm == "vector" else CoordinateModel)(preset=CONTROL.name)
        updates = fit(
            model, frame(train, shuffle=arm == "shuffled", seed=seed + 4), frame(validate), seed, steps, accelerator
        )
        output = model.predict(frame(test, targets=False), batch_size=256)["predictions"].to_list()
        predicted = np.asarray([row["position"]["content"] for row in output], dtype=np.float64)
        inferred = all(row["position"]["inferred"] for row in output)
        states = [max(row["position"]["state"], key=row["position"]["state"].get) for row in output]
        if not np.isfinite(predicted).all():
            raise ValueError(f"{arm} produced nonfinite positions")
        invalid = (np.abs(predicted[:, 0]) > 90) | (np.abs(predicted[:, 1]) > 180)
        # Apply a declared geographic interpretation to unconstrained Vector outputs.
        geographic = predicted.copy()
        geographic[:, 0] = np.clip(geographic[:, 0], -90, 90)
        geographic[:, 1] = (geographic[:, 1] + 180) % 360 - 180
        measured = {name: score(actual, geographic, selected) for name, selected in groups.items()}
        measured.update(
            steps=updates,
            inferred=inferred,
            state_accuracy=float(np.mean(np.asarray(states) == "valued")),
            states=states,
            invalid_raw=int(invalid.sum()),
            parameters=sum(p.numel() for p in model.parameters()),
            predictions=predicted,
        )
        metrics["arms"][arm] = measured
        checks[f"{arm}: full budget retained"] = updates == BUDGET
        checks[f"{arm}: finite inferred output"] = inferred and bool(np.isfinite(predicted).all())
        checks[f"{arm}: valued state accuracy at least 0.95"] = measured["state_accuracy"] >= 0.95
    checks["Coordinates improve on medoid"] = (
        metrics["arms"]["coordinates"]["overall"]["mae_m"] < 0.8 * baseline["mae_m"]
    )
    checks["Shuffled clues retain medoid-scale error"] = (
        metrics["arms"]["shuffled"]["overall"]["mae_m"] > 0.8 * baseline["mae_m"]
    )
    return metrics, checks


def plot(bundle, spec):
    """Compare spherical errors overall and on each side of the date line."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    from evidence.diagnostics.plots import dots

    metrics = bundle.record["metrics"]
    arms = metrics["arms"]
    labels = ("Coordinates", "Vector", "Shuffled clues")
    names = ("coordinates", "vector", "shuffled")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    dots(
        axes[0],
        ("Training medoid", *labels),
        {"Held-out MAE": [metrics["baseline"]["mae_m"]] + [arms[name]["overall"]["mae_m"] for name in names]},
        xlabel="Great-circle MAE (km; lower is better)",
    )
    dots(
        axes[1],
        ("West of date line", "East of date line"),
        {
            label: [arms[name][side]["mae_m"] for side in ("west", "east")]
            for name, label in zip(names, labels, strict=True)
        },
        xlabel="Great-circle MAE (km; lower is better)",
    )
    for axis in axes:
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}"))
    axes[0].set_title("Overall")
    axes[1].set_title("Held-out side")
    paths = [["baseline", "mae_m"]] + [
        ["arms", name, group, "mae_m"] for name in names for group in ("overall", "west", "east")
    ]
    return (
        fig,
        "Independent synthetic positions straddle the date line. Axes show kilometers; records retain meters. Vector outputs are clipped and wrapped before spherical scoring; raw invalid counts remain in the record.",
        {"metric_paths": paths},
    )


if __name__ == "__main__":
    report(run)
