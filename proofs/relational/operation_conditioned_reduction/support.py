"""Synthetic data, model construction, and metrics for operation selection.

This benchmark asks whether a model can learn approximate operator selection.
When an exact sum, mean, minimum, or maximum is a contractual calculation,
compute it upstream in a :class:`relflow.Preprocessor` or the application
rather than substituting a learned prediction.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch

import relflow as rf

OPERATIONS = ("sum", "mean", "min", "max")


def reduce(values: np.ndarray, operation: str) -> float:
    """Execute one synthetic reduction outside RelFlow."""

    match operation:
        case "sum":
            return float(values.sum())
        case "mean":
            return float(values.mean())
        case "min":
            return float(values.min())
        case "max":
            return float(values.max())
        case _:
            raise ValueError(f"unknown reduction operation: {operation!r}")


def values(rng: np.random.Generator, length: int) -> np.ndarray:
    """Draw bounded asymmetric values whose reductions are usually distinct."""

    if length < 4:
        raise ValueError("reduction bags require at least four values")
    middle = rng.uniform(-0.2, 0.6, size=length - 2)
    low = rng.uniform(-1.3, -0.7, size=1)
    high = rng.uniform(0.9, 1.7, size=1)
    return np.concatenate((middle, low, high))


def operation_table(*, bags: int, length: int, seed: int) -> pa.Table:
    """Expand each independently drawn bag into its four reduction requests."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for bag in range(bags):
        bag_values = values(rng, length)
        order = rng.permutation(length)
        items = [{"value": float(bag_values[index])} for index in order]
        for operation in OPERATIONS:
            rows.append(
                {
                    "bag": bag,
                    "operation": operation,
                    "items": items,
                    "answer": reduce(bag_values, operation),
                }
            )
    return pa.Table.from_pylist(rows)


def permute_items(table: pa.Table, *, seed: int) -> pa.Table:
    """Reorder each bag once while retaining all values and targets."""

    rng = np.random.default_rng(seed)
    reordered: dict[int, list[dict[str, object]]] = {}
    rows = table.to_pylist()
    for row in rows:
        bag = int(row["bag"])
        if bag not in reordered:
            items = row["items"]
            reordered[bag] = [items[index] for index in rng.permutation(len(items))]
        row["items"] = reordered[bag]
    return pa.Table.from_pylist(rows, schema=table.schema)


def hide_operation(table: pa.Table) -> pa.Table:
    """Retain the visible request field while making every operation null."""

    index = table.schema.get_field_index("operation")
    return table.set_column(index, "operation", pa.nulls(len(table), type=pa.string()))


def model(*, length: int) -> rf.Model:
    return rf.Model(
        name="request",
        d_model=64,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        batch_size=128,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=1e-3),
        items=rf.Branch(
            length=length,
            n_layers=2,
            reduction=rf.Attention(n_layers=2),
            value=rf.Number,
        ),
        answer=rf.Number(mask=True, objective="mse"),
        operation=rf.Category(size=len(OPERATIONS), p_unavailable=0.0),
    )


def fit(configured: rf.Model, train: pa.Table, validate: pa.Table, *, steps: int) -> None:
    datamodule = rf.ArrowDataModule(
        model=configured,
        train=train,
        validate=validate,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    trainer = lit.Trainer(
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
    trainer.fit(model=configured, datamodule=datamodule)


def predict(configured: rf.Model, table: pa.Table) -> np.ndarray:
    requests = table.drop(["answer"])
    output = configured.predict(requests)
    predictions = output["predictions"].combine_chunks()
    answer = predictions.field("request/answer")
    return np.asarray(answer.field("content").to_numpy(zero_copy_only=False))


@dataclass(frozen=True)
class Score:
    rmse: float
    baseline_rmse: float
    nrmse: float


def scores(
    *,
    train: pa.Table,
    test: pa.Table,
    predicted: np.ndarray,
    keys: Iterable[str],
) -> dict[tuple[str, ...], Score]:
    """Compute train-mean normalized RMSE for every requested metric cell."""

    train_rows = train.select([*keys, "answer"]).to_pylist()
    test_rows = test.select([*keys, "answer"]).to_pylist()
    means: dict[tuple[str, ...], float] = {}
    for key in {tuple(str(row[name]) for name in keys) for row in train_rows}:
        targets = [float(row["answer"]) for row in train_rows if tuple(str(row[name]) for name in keys) == key]
        means[key] = float(np.mean(targets))

    result: dict[tuple[str, ...], Score] = {}
    cells = {tuple(str(row[name]) for name in keys) for row in test_rows}
    for cell in cells:
        indices = np.asarray(
            [index for index, row in enumerate(test_rows) if tuple(str(row[name]) for name in keys) == cell]
        )
        target = np.asarray([float(test_rows[index]["answer"]) for index in indices])
        rmse = float(np.sqrt(np.mean(np.square(predicted[indices] - target))))
        baseline_rmse = float(np.sqrt(np.mean(np.square(means[cell] - target))))
        result[cell] = Score(rmse=rmse, baseline_rmse=baseline_rmse, nrmse=rmse / baseline_rmse)
    return result


def diagnostics(result: dict[tuple[str, ...], Score]) -> str:
    return "\n".join(
        f"{','.join(cell)}: rmse={score.rmse:.4f}, baseline={score.baseline_rmse:.4f}, nrmse={score.nrmse:.3f}"
        for cell, score in sorted(result.items())
    )
