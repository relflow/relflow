"""Shared synthetic process and trainer for signal-detection calibration."""

from __future__ import annotations

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch

import relflow as rf


def records(*, rows: int, seed: int, signal: bool) -> pa.Table:
    """Generate a balanced target with either a matching or random Boolean."""

    rng = np.random.default_rng(seed)
    target = np.tile(np.array([False, True]), (rows + 1) // 2)[:rows]
    rng.shuffle(target)
    independent = rng.integers(0, 2, size=rows).astype(bool)
    tags = np.array(["red", "blue", "round", "square", "hot", "cold"])
    tag_rows: list[list[str]] = []
    for _ in range(rows):
        selected = rng.choice(tags, size=int(rng.integers(0, 4)), replace=False)
        tag_rows.append(selected.tolist())

    return pa.table(
        {
            "x": rng.normal(size=rows),
            "segment": [f"segment-{value}" for value in rng.integers(0, 4, size=rows)],
            "tags": tag_rows,
            "leak": target if signal else independent,
            "target": target,
        }
    )


def fit_and_test(*, signal: bool, seed: int = 7) -> float:
    """Fit one calibration model and return held-out AUC."""

    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        name="calibration",
        d_model=24,
        n_layers=1,
        n_heads=4,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=5e-3),
        x=rf.Number,
        segment=rf.Category(size=4, p_unavailable=0.0),
        tags=rf.Set(size=6, p_unavailable=0.0),
        leak=rf.Boolean,
        target=rf.Boolean(mask=True),
    )
    data = rf.ArrowDataModule(
        model=model,
        train=records(rows=1024, seed=seed + 1, signal=signal),
        validate=records(rows=512, seed=seed + 2, signal=signal),
        test=records(rows=2048, seed=seed + 3, signal=signal),
        seed=seed,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    trainer = lit.Trainer(
        accelerator="cpu",
        max_epochs=12,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model=model, datamodule=data)
    metrics = trainer.test(model=model, datamodule=data, verbose=False)[0]
    return float(metrics["calibration.target/test.auc.content"])
