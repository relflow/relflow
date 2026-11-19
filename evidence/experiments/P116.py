# %% [markdown]
# ---
# title: Can coordinates locate fresh positions on a sphere?
# description: Compare Coordinates and Vector on independently sampled positions and an independently shuffled control.
# evidence-readout:
# - label: Shuffled locations
#   metric:
#   - arms
#   - shuffled
#   - overall
#   - nrmse
#   format: number
# - label: Vector
#   metric:
#   - arms
#   - vector
#   - overall
#   - nrmse
#   format: number
# - label: Coordinates
#   metric:
#   - arms
#   - coordinates
#   - overall
#   - nrmse
#   format: number
# evidence-group: Representation
# evidence-review:
#   question: Can coordinates locate fresh positions on a sphere?
#   modeling: Predict spherical distance to a fixed reference from a fresh location. Compare the same numeric pair
#     represented by Coordinates or Vector, plus a shuffled-position control.
#   comparison:
#     metric: Normalized RMSE
#     kind: Controlled comparison
#     conditions:
#     - label: Shuffled locations
#       path:
#       - arms
#       - shuffled
#       - overall
#       - nrmse
#       format: number
#     - label: Vector
#       path:
#       - arms
#       - vector
#       - overall
#       - nrmse
#       format: number
#     - label: Coordinates
#       path:
#       - arms
#       - coordinates
#       - overall
#       - nrmse
#       format: number
#     definition: Root mean squared error divided by the training-mean predictor error on the same split.
#     direction: Lower error is better
#   note: This synthetic task tests geographic representation, not road or walking routes.
# evidence-plot:
#   mode: summary
#   target: target
#   focus: Can coordinates locate fresh positions on a sphere?
#   comparison: Compare Coordinates and Vector on independently sampled positions and an independently shuffled control.
#   limit: This synthetic task tests geographic representation, not road or walking routes.
# ---
#
# Distances are in thousands of kilometers on a spherical Earth. Fresh held-out coordinates include dateline and high-latitude subgroups. The illustrative example uses a fixed reference near 25 degrees north; the answer is hidden at inference.
#
# ## Example
#
# ```yaml
# point: [0.0, 180.0]
# distance: 2.78
# ```
#
# ## Comparison
#
# Predict spherical distance to a fixed reference from a fresh location. Compare the same numeric pair represented by Coordinates or Vector, plus a shuffled-position control.
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
# Generate 4,096 training, 512 validation, and 1,536 held-out positions uniformly over
# latitude -75 to 75 and longitude -180 to 180. Use a spherical haversine oracle in
# thousands of kilometers, a fixed reference at (25, 179.75), and 500 updates per arm.
# Score overall, date-line and high-latitude groups. The Vector arm has a different
# embedder and parameter count; these measurements do not guarantee a ranking.
#
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["lightning", "numpy", "polars", "pyarrow", "pyyaml>=6", "relflow>=0.5.0", "torch"]
#
# [tool.relflow.evidence]
# capability = "representation"
# types = ["Coordinates", "Vector", "Number"]
# conditions = ["unseen-inputs", "changed-associations", "representations"]
# ///

"""P116: learn spherical distance from held-out coordinate inputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
from policies import CONTROL
from reporting import experiment, report

BUDGET = 500
TRAIN_ROWS = 4096
VALIDATION_ROWS = 512
TEST_ROWS = 1536
RADIUS_KM = 6371.009
REFERENCE = (25.0, 179.75)
ARMS = ("coordinates", "vector", "shuffled")


class CoordinateModel(rf.Model):
    point = rf.Coordinates
    distance = rf.Number(mask=True, objective="mse")


class VectorModel(rf.Model):
    point = rf.Vector(n_dim=2)
    distance = rf.Number(mask=True, objective="mse")


def distance(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Independent haversine reference in thousands of kilometers."""
    lat = np.deg2rad(latitude)
    lon = np.deg2rad(longitude)
    ref_lat, ref_lon = np.deg2rad(REFERENCE)
    a = np.sin((lat - ref_lat) / 2) ** 2 + np.cos(lat) * np.cos(ref_lat) * np.sin((lon - ref_lon) / 2) ** 2
    return (2 * RADIUS_KM / 1000) * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def records(*, rows: int, seed: int):
    rng = np.random.default_rng(seed)
    latitudes = rng.uniform(-75.0, 75.0, rows)
    longitudes = rng.uniform(-180.0, 180.0, rows)
    targets = distance(latitudes, longitudes)
    for index, (latitude, longitude, target) in enumerate(zip(latitudes, longitudes, targets, strict=True)):
        yield {"case": f"{seed}:{index}", "point": [float(latitude), float(longitude)], "target": float(target)}


def frame(rows: list[dict], *, targets: bool = True, shuffle: bool = False, seed: int = 0) -> pl.DataFrame:
    indices = np.arange(len(rows))
    if shuffle:
        indices = np.random.default_rng(seed).permutation(indices)
    data = {"point": [rows[int(index)]["point"] for index in indices]}
    if targets:
        data["distance"] = [row["target"] for row in rows]
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


def score(actual: np.ndarray, predicted: np.ndarray, mean: float, selected: np.ndarray) -> dict:
    expected, value = actual[selected], predicted[selected]
    rmse = float(np.sqrt(np.mean((value - expected) ** 2)))
    baseline = float(np.sqrt(np.mean((expected - mean) ** 2)))
    return {
        "rows": int(len(expected)),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(value - expected))),
        "nrmse": rmse / baseline,
    }


@experiment("P116", seed=11601)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train = list(records(rows=TRAIN_ROWS, seed=seed + 1))
    validate = list(records(rows=VALIDATION_ROWS, seed=seed + 2))
    test = list(records(rows=TEST_ROWS, seed=seed + 3))
    expected = np.asarray([row["target"] for row in test])
    mean = float(np.mean([row["target"] for row in train]))
    points = np.asarray([row["point"] for row in test])
    groups = {
        "overall": np.ones(len(test), dtype=bool),
        "date_line": np.abs(points[:, 1]) >= 165,
        "high_latitude": np.abs(points[:, 0]) >= 55,
    }
    metrics = {
        "budget": BUDGET,
        "rows": {"train": len(train), "validation": len(validate), "test": len(test)},
        "reference": REFERENCE,
        "train_mean": mean,
        "test": {"case": [r["case"] for r in test], "point": points, "target": expected},
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
        predicted = np.asarray([row["distance"]["content"] for row in output], dtype=np.float64)
        inferred = all(row["distance"]["inferred"] for row in output)
        states = [max(row["distance"]["state"], key=row["distance"]["state"].get) for row in output]
        if not np.isfinite(predicted).all():
            raise ValueError(f"{arm} produced nonfinite predictions")
        measured = {name: score(expected, predicted, mean, selected) for name, selected in groups.items()}
        measured.update(
            steps=updates,
            inferred=inferred,
            state_accuracy=float(np.mean(np.asarray(states) == "valued")),
            states=states,
            parameters=sum(p.numel() for p in model.parameters()),
            predictions=predicted,
        )
        metrics["arms"][arm] = measured
        checks[f"{arm}: full budget retained"] = updates == BUDGET
        checks[f"{arm}: finite inferred output"] = inferred and bool(np.isfinite(predicted).all())
        checks[f"{arm}: valued state accuracy at least 0.95"] = measured["state_accuracy"] >= 0.95
    checks["Coordinates held-out nRMSE below 0.8"] = metrics["arms"]["coordinates"]["overall"]["nrmse"] < 0.8
    checks["Shuffled locations retain nRMSE above 0.8"] = metrics["arms"]["shuffled"]["overall"]["nrmse"] > 0.8
    checks["Coordinates improve on shuffled locations"] = (
        metrics["arms"]["coordinates"]["overall"]["nrmse"] + 0.1 < metrics["arms"]["shuffled"]["overall"]["nrmse"]
    )
    return metrics, checks


if __name__ == "__main__":
    report(run)
