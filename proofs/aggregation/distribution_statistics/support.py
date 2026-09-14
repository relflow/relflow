"""Shared protocol and user guide for distribution-statistic proofs.

Claim
-----
A repeated numerical field should support statistics that depend on more than
its arithmetic mean. In particular, a learned branch should be able to infer
variance from raw values, and aligned sibling ``x`` and ``y`` fields should
support covariance. The user should not have to supply squares, deviations,
or products merely to expose those relationships to the decoder.

What to do
----------
Keep values whose interaction matters on the same repeated item. For paired
statistics, model ``x`` and ``y`` as sibling fields of one item branch so that
their coordinate is explicit. The raw covariance proof uses
``reduction=None`` on that branch, preserving its aligned field-coordinate
evidence for the scalar answer. Use a learned reduction when the sufficient
statistic is not already known. Compare against a supplied-statistic control
when diagnosing convergence: it tells you whether the problem is the local
nonlinearity or the collection reduction.

Number embeddings expose a monotone standardized scalar lane alongside their
Fourier representation. Branches mix visible sibling fields at the same item
coordinate before the branch routes its encoded tokens onward. Preserving those
coordinates gives the natural paired schema a learned path for cross-moments
without requiring a user-supplied product.

What not to do
--------------
Do not replace raw values with their mean when the target needs dispersion.
Do not put aligned ``x`` and ``y`` values in independently reduced branches;
that discards the assertion that ``x[i]`` belongs with ``y[i]``. A supplied
``squared_deviation`` field is a diagnostic rung, not the preferred public
schema: computing it requires precisely the domain knowledge the learned
model is meant to acquire.

That omission is specific to this learned-interaction benchmark. If variance,
covariance, or another statistic is a known contractual output, calculate it
exactly in an :class:`relflow.Preprocessor` or the application rather than
substituting a learned approximation.

Why the controls are identifiable
----------------------------------
Variance rows are constructed with an independently random location and
spread. Every row's deviations have exactly zero mean and unit mean square
before scaling, so the target is the squared scale. Location alone carries no
spread information. The held-out matched-mean pair changes only spread.

Covariance rows use centered orthogonal basis vectors. ``x`` and ``y`` have
independently random locations and scales, while a random coefficient controls
their covariance. Shuffling ``y`` within a row preserves both univariate
marginals exactly and destroys only pair alignment. Thus a model that remains
accurate against the original target after shuffling did not use covariance.

Protocol and gates
------------------
All cases use deterministic CPU training, independently generated train,
validation, and test rows, eight items per row, and public ``Model`` and
``ArrowDataModule`` APIs. Accuracy is test RMSE divided by the RMSE of the
training-target mean.

The supplied squared-deviation control must reach variance nRMSE below 0.20.
The desired raw-value path must reach nRMSE below 0.30 and predict a materially
larger variance for a matched-mean high-spread row. The aligned covariance
path must reach nRMSE below 0.35, while the within-row ``y`` shuffle must
worsen nRMSE by at least 0.35.

Status
------
Provisional established suite. Supplied sufficient-statistic controls, raw
variance, and raw aligned covariance each clear their hard gates at one
deterministic seed. These results establish learned routes through the current
architecture; they do not make variance or covariance exact operators.

Current evidence
----------------
* At seed 3500, supplied squared deviations clear the 0.20 variance nRMSE gate.
* At seed 3504, raw values clear the 0.30 variance nRMSE gate and predict the
  required separation between matched-mean low- and high-spread rows.
* At seed 3513, supplied centered cross-products clear the 0.20 covariance
  nRMSE gate.
* At seed 3508, raw aligned ``x``/``y`` pairs clear the 0.35 covariance nRMSE
  gate, while shuffling only ``y`` worsens nRMSE by at least 0.35. This causal
  gap is the evidence that the model uses item-local pairing rather than only
  the two univariate marginals.
Remaining work
--------------
* Repeat supported paths across at least three core seeds and ten lightweight
  calibration seeds before promoting numerical thresholds.
* Add variable-cardinality variance with unbiased-versus-population semantics
  stated explicitly.
* Add RMS, standard deviation, higher moments, quantiles, and robust spread.
* Add Pearson correlation with varying offsets and scales; covariance alone
  does not prove that the decoder learned normalization by both variances.
* Cover missing values, duplicate observations, extreme outliers, and
  zero-variance rows.
* Add exact joint-pair permutation checks and ablate coordinate mixing.
* Compare token-preserving ``reduction=None`` with compressed Attention at
  matched parameter and output-token budgets.

Promotion criteria
------------------
Every claimed capability must pass its accuracy and metamorphic gates over at
least three core seeds. The supplied-statistic control must remain easy, raw
variance must discriminate matched means, and covariance must consistently
degrade when only item-local pairing is corrupted.

Run all cases with::

    uv run pytest -n 0 proofs/aggregation/distribution_statistics -q
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch

import relflow as rf

ITEMS = 8
MomentSource = Literal["raw", "squared_deviation", "cross_deviation"]
Target = Literal["variance", "covariance", "correlation"]


def dispersion_records(*, rows: int, seed: int, source: MomentSource) -> pa.Table:
    """Draw rows whose location and spread are statistically independent."""

    rng = np.random.default_rng(seed)
    observations: list[dict[str, object]] = []
    for _ in range(rows):
        basis = rng.normal(size=ITEMS)
        basis -= basis.mean()
        basis /= np.sqrt(np.mean(np.square(basis)))
        location = float(rng.uniform(-0.8, 0.8))
        scale = float(rng.uniform(0.12, 1.25))
        values = location + scale * basis
        if source == "raw":
            items = [{"value": float(value)} for value in values]
        else:
            items = [{"squared_deviation": float(np.square(value - location))} for value in values]
        observations.append({"items": items, "variance": scale * scale})
    return pa.Table.from_pylist(observations)


def covariance_records(*, rows: int, seed: int) -> pa.Table:
    """Draw paired fields with controlled means, scales, and covariance."""

    rng = np.random.default_rng(seed)
    observations: list[dict[str, object]] = []
    for _ in range(rows):
        x_basis = rng.normal(size=ITEMS)
        x_basis -= x_basis.mean()
        x_basis /= np.sqrt(np.mean(np.square(x_basis)))

        orthogonal = rng.normal(size=ITEMS)
        orthogonal -= orthogonal.mean()
        orthogonal -= np.mean(orthogonal * x_basis) * x_basis
        orthogonal /= np.sqrt(np.mean(np.square(orthogonal)))

        correlation = float(rng.uniform(-0.88, 0.88))
        y_basis = correlation * x_basis + np.sqrt(1.0 - correlation * correlation) * orthogonal
        x_scale = float(rng.uniform(0.35, 1.45))
        y_scale = float(rng.uniform(0.35, 1.45))
        x_location = float(rng.uniform(-0.8, 0.8))
        y_location = float(rng.uniform(-0.8, 0.8))
        x_values = x_location + x_scale * x_basis
        y_values = y_location + y_scale * y_basis
        observations.append(
            {
                "items": [{"x": float(x), "y": float(y)} for x, y in zip(x_values, y_values, strict=True)],
                "covariance": correlation * x_scale * y_scale,
                "correlation": correlation,
            }
        )
    return pa.Table.from_pylist(observations)


def covariance_sufficient_records(*, rows: int, seed: int) -> pa.Table:
    """Expose per-item centered products while retaining the same targets."""

    rows_with_pairs = covariance_records(rows=rows, seed=seed).to_pylist()
    observations: list[dict[str, object]] = []
    for row in rows_with_pairs:
        x = np.asarray([item["x"] for item in row["items"]], dtype=np.float64)
        y = np.asarray([item["y"] for item in row["items"]], dtype=np.float64)
        cross_deviation = (x - x.mean()) * (y - y.mean())
        observations.append(
            {
                "items": [{"cross_deviation": float(value)} for value in cross_deviation],
                "covariance": row["covariance"],
            }
        )
    return pa.Table.from_pylist(observations)


def shuffle_y(table: pa.Table, *, seed: int) -> pa.Table:
    """Break within-row pairing while preserving every marginal value."""

    rng = np.random.default_rng(seed)
    rows = table.to_pylist()
    changed = 0
    for row in rows:
        items = row["items"]
        offset = int(rng.integers(1, len(items)))
        shifted = np.roll([item["y"] for item in items], offset)
        changed += sum(item["y"] != y for item, y in zip(items, shifted, strict=True))
        row["items"] = [{"x": item["x"], "y": float(y)} for item, y in zip(items, shifted, strict=True)]
    if changed == 0:
        raise AssertionError("pair shuffle did not change any item-local associations")
    return pa.Table.from_pylist(rows, schema=table.schema)


def model(*, source: MomentSource | Literal["paired"], target: Target) -> rf.Model:
    """Build one public-schema distribution-statistic model."""

    if source == "raw":
        fields: dict[str, object] = {"value": rf.Number}
        reduction: rf.ReductionConfig = rf.Attention(n_layers=2)
    elif source in ("squared_deviation", "cross_deviation"):
        fields = {source: rf.Number}
        reduction = rf.Mean()
    else:
        fields = {"x": rf.Number, "y": rf.Number}
        reduction = None
    branch_options: dict[str, object] = {
        "length": ITEMS,
        "n_layers": 2,
        "reduction": reduction,
        **fields,
    }
    if source in ("squared_deviation", "cross_deviation"):
        branch_options["attention"] = None
    return rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=1e-3),
        items=rf.Branch(**branch_options),
        **{target: rf.Number(mask=True, objective="mse")},
    )


def data(configured: rf.Model, train: pa.Table, validate: pa.Table, *, seed: int) -> rf.ArrowDataModule:
    """Build a deterministic in-memory Arrow data module."""

    return rf.ArrowDataModule(
        model=configured,
        train=train,
        validate=validate,
        seed=seed,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )


def trainer(*, steps: int) -> lit.Trainer:
    """Build a quiet deterministic CPU trainer."""

    return lit.Trainer(
        accelerator="cpu",
        max_epochs=-1,
        max_steps=steps,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )


def prediction(configured: rf.Model, table: pa.Table, target: str) -> np.ndarray:
    """Run inference and unwrap one numerical prediction field."""

    output = configured.predict(table.select(["items"]))
    field = output["predictions"].combine_chunks().field(f"record/{target}")
    return np.asarray(field.field("content").to_numpy(zero_copy_only=False), dtype=np.float64)


def column(table: pa.Table, name: str) -> np.ndarray:
    """Return one Arrow column as a dense float array."""

    return np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    """Calculate root mean squared error."""

    return float(np.sqrt(np.mean(np.square(actual - predicted))))


@dataclass(frozen=True)
class Score:
    """Accuracy relative to the training-target mean."""

    rmse: float
    baseline_rmse: float
    nrmse: float


def score(*, train: pa.Table, test: pa.Table, predicted: np.ndarray, target: str) -> Score:
    """Normalize test RMSE by the constant training-target mean."""

    actual = column(test, target)
    baseline = rmse(actual, float(column(train, target).mean()))
    measured = rmse(actual, predicted)
    return Score(rmse=measured, baseline_rmse=baseline, nrmse=measured / baseline)


def diagnostics(label: str, measured: Score) -> str:
    """Render one compact assertion diagnostic."""

    return f"{label}: rmse={measured.rmse:.4f}, baseline={measured.baseline_rmse:.4f}, nrmse={measured.nrmse:.4f}"
