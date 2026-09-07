"""Synthetic data, model routes, and metrics for argmax retrieval."""

from __future__ import annotations

from typing import Literal

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch

import relflow as rf

LENGTH = 3
Route = Literal["summarized", "preserved"]


def records(*, rows: int, seed: int, break_pairs: bool = False) -> pa.Table:
    rng = np.random.default_rng(seed)
    scores = rng.uniform(-2.0, 2.0, size=(rows, LENGTH))
    payloads = rng.normal(0.0, 1.0, size=(rows, LENGTH))
    winners = scores.argmax(axis=1)
    answers = payloads[np.arange(rows), winners]
    visible_payloads = np.roll(payloads, shift=1, axis=1) if break_pairs else payloads
    items = [
        [
            {"score": float(score), "payload": float(payload)}
            for score, payload in zip(row_scores, row_payloads, strict=True)
        ]
        for row_scores, row_payloads in zip(scores, visible_payloads, strict=True)
    ]
    return pa.Table.from_pylist(
        [{"items": row_items, "answer": float(answer)} for row_items, answer in zip(items, answers, strict=True)]
    )


def requests(table: pa.Table) -> pa.Table:
    return table.drop(["answer"])


def predictions(model: rf.Model, table: pa.Table) -> np.ndarray:
    result = model.predict(requests(table))
    payload = pc.struct_field(result["predictions"], "retrieval/answer")
    content = pc.struct_field(payload, "content")
    return content.to_numpy(zero_copy_only=False)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def fit(route: Route, train: pa.Table, validate: pa.Table, *, seed: int) -> rf.Model:
    """Fit one explicitly reduced route on a shared synthetic split."""

    root_reduction = rf.Attention(n_outputs=LENGTH) if route == "summarized" else None
    item_reduction = rf.Attention() if route == "summarized" else None
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        name="retrieval",
        d_model=48,
        n_layers=1,
        n_heads=4,
        reduction=root_reduction,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        items=rf.Branch(
            length=LENGTH,
            overflow="error",
            n_layers=2,
            n_heads=4,
            reduction=item_reduction,
            score=rf.Number,
            payload=rf.Number,
        ),
        answer=rf.Number(mask=True),
    )
    data = rf.ArrowDataModule(
        model=model,
        train=train,
        validate=validate,
        seed=seed,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    trainer = lit.Trainer(
        accelerator="cpu",
        max_epochs=25,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model=model, datamodule=data)
    return model
