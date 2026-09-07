"""Synthetic data and metrics for the item-alignment proof."""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa

import relflow as rf

LENGTH = 5


def records(*, rows: int, seed: int, permute_targets: bool = False) -> pa.Table:
    rng = np.random.default_rng(seed)
    target_rng = np.random.default_rng(seed + 10_000)
    orders: list[dict[str, object]] = []
    for row in range(rows):
        length = row % (LENGTH + 1)
        quantity = rng.uniform(0.25, 2.0, size=length)
        unit_price = rng.uniform(-1.5, 1.5, size=length)
        subtotal = quantity * unit_price
        if permute_targets and length > 1:
            subtotal = subtotal[target_rng.permutation(length)]
        orders.append(
            {
                "items": [
                    {
                        "quantity": float(item_quantity),
                        "unit_price": float(item_price),
                        "subtotal": float(item_subtotal),
                    }
                    for item_quantity, item_price, item_subtotal in zip(
                        quantity,
                        unit_price,
                        subtotal,
                        strict=True,
                    )
                ]
            }
        )
    return pa.Table.from_pylist(orders)


def requests(table: pa.Table) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "items": [
                    {
                        "quantity": item["quantity"],
                        "unit_price": item["unit_price"],
                    }
                    for item in row
                ]
            }
            for row in table["items"].to_pylist()
        ]
    )


def actual(table: pa.Table) -> np.ndarray:
    return np.asarray(
        [item["subtotal"] for row in table["items"].to_pylist() for item in row],
        dtype=np.float64,
    )


def predicted(model: rf.Model, table: pa.Table) -> tuple[np.ndarray, list[list[dict[str, object]]]]:
    output = model.predict(requests(table))
    rows = output["predictions"].combine_chunks().field("order/items/subtotal").to_pylist()
    lengths = [len(row) for row in table["items"].to_pylist()]
    values = [
        float(coordinate["content"]) for row, length in zip(rows, lengths, strict=True) for coordinate in row[:length]
    ]
    return np.asarray(values, dtype=np.float64), rows


def rmse(target: np.ndarray, estimate: np.ndarray | float) -> float:
    return math.sqrt(float(np.mean(np.square(target - estimate))))
