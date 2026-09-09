"""Shared data, training, and scoring for unseen Hash-equality proofs."""

from __future__ import annotations

from typing import Literal

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch

import relflow as rf

Identity = Literal["hash", "category"]


def records(*, rows: int, seed: int, namespace: str, shuffle_targets: bool = False) -> pa.Table:
    """Create balanced equality pairs in one identity namespace."""

    rng = np.random.default_rng(seed)
    target = np.tile(np.array([False, True]), (rows + 1) // 2)[:rows]
    rng.shuffle(target)
    left_index = rng.integers(0, rows * 2, size=rows)
    other_index = rng.integers(0, rows * 2 - 1, size=rows)
    other_index += other_index >= left_index
    right_index = np.where(target, left_index, other_index)
    labels = target.copy()
    if shuffle_targets:
        rng.shuffle(labels)
    return pa.table(
        {
            "left_id": [f"{namespace}-{value}" for value in left_index],
            "right_id": [f"{namespace}-{value}" for value in right_index],
            "equal": labels,
        }
    )


def score(model: rf.Model, table: pa.Table) -> float:
    """Return Boolean content AUC on one test table."""

    data = rf.ArrowDataModule(
        model=model,
        test=table,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    trainer = lit.Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    metrics = trainer.test(model=model, datamodule=data, verbose=False)[0]
    return float(metrics["identity.equal/test.auc.content"])


def train(
    *,
    identity: Identity,
    train_rows: pa.Table,
    validate_rows: pa.Table,
    seed: int,
) -> rf.Model:
    """Fit one identity model."""

    lit.seed_everything(seed, workers=True)
    field = rf.Hash(n_hashes=4) if identity == "hash" else rf.Category(size=8192, p_unavailable=0.0)
    model = rf.Model(
        name="identity",
        d_model=48,
        n_layers=2,
        n_heads=4,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        left_id=field,
        right_id=rf.Hash(n_hashes=4) if identity == "hash" else rf.Category(size=8192, p_unavailable=0.0),
        equal=rf.Boolean(mask=True),
    )
    data = rf.ArrowDataModule(
        model=model,
        train=train_rows,
        validate=validate_rows,
        seed=seed,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    trainer = lit.Trainer(
        accelerator="cpu",
        max_epochs=20,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model=model, datamodule=data)
    return model
