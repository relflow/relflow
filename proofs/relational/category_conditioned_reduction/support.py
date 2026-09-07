"""Synthetic data, models, and metrics for category-conditioned reduction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch

import relflow as rf

OPERATIONS = ("sum", "mean", "min", "max")
GROUPS = ("A", "B", "C")


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


def grouped_table(*, bags: int, items_per_group: int, seed: int) -> pa.Table:
    """Expand interleaved grouped bags into every group-operation request."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    length = len(GROUPS) * items_per_group
    for bag in range(bags):
        bag_values = np.concatenate([values(rng, items_per_group) + rng.uniform(-1.5, 1.5) for _ in GROUPS])
        labels = np.repeat(np.asarray(GROUPS), items_per_group)
        order = rng.permutation(length)
        items = [{"group": str(labels[index]), "value": float(bag_values[index])} for index in order]
        for group in GROUPS:
            selected = bag_values[labels == group]
            for operation in OPERATIONS:
                rows.append(
                    {
                        "bag": bag,
                        "selected_group": group,
                        "operation": operation,
                        "items": items,
                        "answer": reduce(selected, operation),
                    }
                )
    return pa.Table.from_pylist(rows)


def filtered_mean_table(*, bags: int, items_per_group: int, seed: int) -> pa.Table:
    """Generate the smallest group-filtering rung with one fixed reduction."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for bag in range(bags):
        labels = np.repeat(np.asarray(GROUPS), items_per_group)
        bag_values = np.concatenate(
            [rng.uniform(-1.5, 1.5) + rng.uniform(-0.15, 0.15, size=items_per_group) for _ in GROUPS]
        )
        order = rng.permutation(len(labels))
        items = [{"group": str(labels[index]), "value": float(bag_values[index])} for index in order]
        for group in GROUPS:
            rows.append(
                {
                    "bag": bag,
                    "selected_group": group,
                    "operation": "mean",
                    "items": items,
                    "answer": reduce(bag_values[labels == group], "mean"),
                }
            )
    return pa.Table.from_pylist(rows)


def permute_item_groups(table: pa.Table, *, seed: int) -> pa.Table:
    """Break Category/value association while retaining each marginal."""

    rng = np.random.default_rng(seed)
    corrupted: dict[int, list[dict[str, object]]] = {}
    rows = table.to_pylist()
    for row in rows:
        bag = int(row["bag"])
        if bag not in corrupted:
            items = row["items"]
            labels = [item["group"] for item in items]
            shuffled = rng.permutation(labels).tolist()
            corrupted[bag] = [
                {"group": group, "value": item["value"]} for item, group in zip(items, shuffled, strict=True)
            ]
        row["items"] = corrupted[bag]
    return pa.Table.from_pylist(rows, schema=table.schema)


def model(
    *,
    length: int,
    item_reduction: rf.ReductionConfig | None,
    root_reduction: rf.ReductionConfig | None,
) -> rf.Model:
    return rf.Model(
        name="request",
        d_model=64,
        n_layers=2,
        n_heads=4,
        reduction=root_reduction,
        batch_size=128,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=1e-3),
        items=rf.Branch(
            length=length,
            n_layers=2,
            reduction=item_reduction,
            value=rf.Number,
            group=rf.Category(size=len(GROUPS), p_unavailable=0.0),
        ),
        answer=rf.Number(mask=True, objective="mse"),
        operation=rf.Category(size=len(OPERATIONS), p_unavailable=0.0),
        selected_group=rf.Category(size=len(GROUPS), p_unavailable=0.0),
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
