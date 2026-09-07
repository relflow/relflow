"""Synthetic data, routes, and metrics for associative recall."""

from __future__ import annotations

import math
from typing import Literal

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch

import relflow as rf

TRAIN_ROWS = 2048
VALIDATE_ROWS = 512
TEST_ROWS = 1024
PAIR_COUNT = 2
MAX_STEPS = 800
Route = Literal["compressed", "preserved"]


def records(
    *,
    rows: int,
    pairs: int,
    seed: int,
    namespace: str,
    broken_identity: bool = False,
    aligned: bool = False,
) -> pa.Table:
    """Generate unseen, observation-local key/value associations as Arrow."""

    rng = np.random.default_rng(seed)
    observations: list[dict[str, object]] = []
    for row in range(rows):
        keys = [f"{namespace}-{row:06d}-{pair:02d}" for pair in range(pairs)]
        values = rng.uniform(-1.0, 1.0, size=pairs)
        query_keys = list(keys)
        if broken_identity:
            query_keys = query_keys[1:] + query_keys[:1]

        source_order = rng.permutation(pairs)
        query_order = source_order if aligned else rng.permutation(pairs)
        source = [
            {
                "entity_id": keys[index],
                "value": float(values[index]),
            }
            for index in source_order
        ]
        target = [
            {
                "entity_id": query_keys[index],
                "value": float(values[index]),
            }
            for index in query_order
        ]

        memory = [{**item, "role": "source", "is_query": False} for item in source]
        memory.extend({**item, "role": "query", "is_query": True} for item in target)
        if not aligned:
            rng.shuffle(memory)
        observations.append({"memory": memory})

    return pa.Table.from_pylist(observations)


def model(*, pairs: int, route: Route) -> rf.Model:
    reduction = rf.Attention() if route == "compressed" else None
    return rf.Model(
        name="association",
        d_model=64,
        n_layers=2,
        n_heads=4,
        reduction=reduction,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        memory=rf.Branch(
            length=2 * pairs,
            n_layers=2,
            reduction=reduction,
            entity_id=rf.Hash(n_hashes=4, n_bands=8),
            role=rf.Category(size=2, p_unavailable=0.0),
            is_query=rf.Boolean,
            value=rf.Number(
                mask=rf.Mask(
                    query="is_query",
                    dropout=False,
                    reconstruct=True,
                ),
                objective="mse",
            ),
        ),
    )


def train(
    *,
    pairs: int = PAIR_COUNT,
    seed: int = 17,
    aligned: bool = False,
    route: Route,
) -> rf.Model:
    lit.seed_everything(seed, workers=True)
    configured = model(pairs=pairs, route=route)
    data = rf.ArrowDataModule(
        model=configured,
        train=records(
            rows=TRAIN_ROWS,
            pairs=pairs,
            seed=seed + 1,
            namespace="train",
            aligned=aligned,
        ),
        validate=records(
            rows=VALIDATE_ROWS,
            pairs=pairs,
            seed=seed + 2,
            namespace="validate",
            aligned=aligned,
        ),
        seed=seed,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    trainer = lit.Trainer(
        accelerator="cpu",
        max_steps=MAX_STEPS,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        check_val_every_n_epoch=5,
    )
    trainer.fit(model=configured, datamodule=data)
    return configured


def query_values(table: pa.Table) -> np.ndarray:
    values: list[float] = []
    for row in table["memory"].to_pylist():
        for item in row:
            if item["is_query"]:
                values.append(float(item["value"]))
    return np.asarray(values, dtype=np.float64)


def predicted_query_values(configured: rf.Model, table: pa.Table) -> np.ndarray:
    address = "association/memory/value"
    rows = configured.predict(table)["predictions"].combine_chunks().field(address).to_pylist()
    values: list[float] = []
    source_rows = table["memory"].to_pylist()
    for source, prediction in zip(source_rows, rows, strict=True):
        for item, coordinate in zip(source, prediction, strict=True):
            if item["is_query"]:
                values.append(float(coordinate["content"]))
    return np.asarray(values, dtype=np.float64)


def normalized_rmse(configured: rf.Model, table: pa.Table) -> float:
    actual = query_values(table)
    predicted = predicted_query_values(configured, table)
    rmse = math.sqrt(float(np.mean(np.square(predicted - actual))))
    baseline = math.sqrt(float(np.mean(np.square(actual))))
    return rmse / baseline
