"""Synthetic data, routes, and metrics for sibling entity transfer."""

from __future__ import annotations

import math
from typing import Literal

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch

import relflow as rf

Identity = Literal["hash", "category"]
Route = Literal["compressed", "preserved"]

TRAIN_ROWS = 2048
VALIDATE_ROWS = 512
TEST_ROWS = 1024
PAIR_COUNT = 2
MAX_STEPS = 600


def records(
    *,
    rows: int,
    pairs: int,
    seed: int,
    namespace: str,
    identity: Identity = "hash",
    broken_identity: bool = False,
) -> pa.Table:
    """Generate observation-local key/value associations in sibling branches."""

    rng = np.random.default_rng(seed)
    observations: list[dict[str, object]] = []
    for row in range(rows):
        keys = (
            [f"{namespace}-{row:06d}-{pair:02d}" for pair in range(pairs)]
            if identity == "hash"
            else [f"entity-{chr(ord('A') + pair)}" for pair in range(pairs)]
        )
        values = rng.uniform(-1.0, 1.0, size=pairs)
        query_keys = list(keys)
        if broken_identity:
            query_keys = query_keys[1:] + query_keys[:1]

        source_order = rng.permutation(pairs)
        query_order = rng.permutation(pairs)
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
        observations.append({"source": source, "target": target})

    return pa.Table.from_pylist(observations)


def model(*, pairs: int, identity: Identity = "hash", route: Route) -> rf.Model:
    def identity_field():
        if identity == "hash":
            return rf.Hash(n_hashes=4, n_bands=8)
        return rf.Category(size=pairs, p_unavailable=0.0)

    reduction = rf.Attention() if route == "compressed" else None
    return rf.Model(
        name="association",
        d_model=64,
        n_layers=2,
        n_heads=4,
        reduction=reduction,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        source=rf.Branch(
            length=pairs,
            n_layers=2,
            reduction=reduction,
            entity_id=identity_field(),
            value=rf.Number,
        ),
        target=rf.Branch(
            length=pairs,
            n_layers=2,
            reduction=reduction,
            entity_id=identity_field(),
            value=rf.Number(mask=True, objective="mse"),
        ),
    )


def train(
    *,
    pairs: int = PAIR_COUNT,
    seed: int = 17,
    identity: Identity = "hash",
    route: Route,
) -> rf.Model:
    lit.seed_everything(seed, workers=True)
    configured = model(pairs=pairs, identity=identity, route=route)
    data = rf.ArrowDataModule(
        model=configured,
        train=records(
            rows=TRAIN_ROWS,
            pairs=pairs,
            seed=seed + 1,
            namespace="train",
            identity=identity,
        ),
        validate=records(
            rows=VALIDATE_ROWS,
            pairs=pairs,
            seed=seed + 2,
            namespace="validate",
            identity=identity,
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
    return np.asarray(
        [float(item["value"]) for row in table["target"].to_pylist() for item in row],
        dtype=np.float64,
    )


def predicted_query_values(configured: rf.Model, table: pa.Table) -> np.ndarray:
    rows = configured.predict(table)["predictions"].combine_chunks().field("association/target/value").to_pylist()
    return np.asarray(
        [float(coordinate["content"]) for row in rows for coordinate in row],
        dtype=np.float64,
    )


def normalized_rmse(configured: rf.Model, table: pa.Table) -> float:
    actual = query_values(table)
    predicted = predicted_query_values(configured, table)
    rmse = math.sqrt(float(np.mean(np.square(predicted - actual))))
    baseline = math.sqrt(float(np.mean(np.square(actual))))
    return rmse / baseline
