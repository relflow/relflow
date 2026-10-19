"""Record a seeded 0.3.2 training movie; rendering never runs training.

Run with the runtime environment: .venv/bin/python explainer-video/compute_learning.py
Fixed, label-free probes are evaluated between updates, in eval mode. Attention
is reconstructed from the reducer's actual projected queries and keys. Stored
weights are an unselected top-left 8x8 slice of that reducer's query projection.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import math
import platform
import subprocess
from pathlib import Path

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch

import relflow as rf

HERE = Path(__file__).resolve().parent


def order(rng):
    items = [
        {
            "sku": f"S{int(rng.integers(20)):02d}",
            "quantity": int(rng.integers(1, 6)),
            "price": float(np.round(rng.lognormal(3, 0.6), 2)),
        }
        for _ in range(int(rng.integers(1, 9)))
    ]
    return {"line_items": items, "returned": any(it["sku"] in ("S03", "S11") and it["quantity"] >= 3 for it in items)}


def record():
    assert importlib.metadata.version("relflow") == "0.3.2"
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    train = [order(rng) for _ in range(4096)]
    test = [order(rng) for _ in range(512)]
    base = [
        {"sku": s, "quantity": q, "price": p}
        for s, q, p in zip(
            ("S02", "S07", "S11", "S15", "S03", "S19"), (2, 4, 4, 1, 1, 2), (15.0, 23.0, 18.0, 29.0, 9.0, 42.0)
        )
    ]
    kept = [dict(it, quantity=1) if i == 2 else dict(it) for i, it in enumerate(base)]
    moved = [base[2], *base[:2], *base[3:]]
    probes = [{"line_items": x} for x in (base, kept, moved)]
    evaluation = pa.Table.from_pylist(probes + [{"line_items": r["line_items"]} for r in test])
    model = rf.Model(
        d_model=64,
        n_layers=1,
        n_heads=4,
        embed=True,
        line_items=rf.Branch(length=8, sku=rf.Category, quantity=rf.Number, price=rf.Number),
        returned=rf.Boolean(mask=True),
    )
    attn = model.nodes["/line_items"].encoder.pool.blocks[0].attention
    weight = attn.q_proj.weight
    captured, snapshots, losses, update = {}, [], [], {}

    def capture(mod, args, kwargs):
        captured.update(
            q=args[0].detach(),
            k=args[1].detach(),
            pad=kwargs.get("key_padding_mask"),
            qp=kwargs.get("query_positions"),
            kp=kwargs.get("key_positions"),
        )

    def snapshot(step):
        was_training = model.training
        model.eval()
        handle = attn.register_forward_pre_hook(capture, with_kwargs=True)
        # One probe at a time avoids tying the displayed layout to another row's length.
        attention = []
        for probe in probes:
            model.predict(pa.Table.from_pylist([probe]))
            with torch.inference_mode():
                q = attn.rotate(attn.splitheads(attn.q_proj(captured["q"]), attn.nhead), captured["qp"])
                k = attn.rotate(attn.splitheads(attn.k_proj(captured["k"]), attn.n_kv_heads), captured["kp"])
                scores = q @ k.transpose(-1, -2) / math.sqrt(attn.head_dim)
                if captured["pad"] is not None:
                    scores = scores.masked_fill(captured["pad"][:, None, None, :], -torch.inf)
                attention.append(scores.softmax(-1)[0, :, 0].tolist())
        handle.remove()
        out = model.predict(evaluation).to_pylist()
        assert all(len(r["representations"]) == 1 and r["representations"][0]["path"] == [] for r in out)
        prob = [r["predictions"]["returned"]["content"]["probability"] for r in out]
        emb = [r["representations"][0]["embedding"] for r in out[3:]]
        y = np.array([r["returned"] for r in test], dtype=float)
        p = np.clip(prob[3:], 1e-7, 1 - 1e-7)
        snapshots.append(
            {
                "step": step,
                "attention": attention,
                "prob": prob[:3],
                "embedding": emb,
                "weights": weight[:8, :8].detach().tolist(),
                "test_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                "test_accuracy": float(((p > 0.5) == y).mean()),
            }
        )
        model.train(was_training)
        print(f"step {step}: loss {snapshots[-1]['test_loss']:.4f}, probes {prob[:3]}", flush=True)

    class Track(lit.Callback):
        def on_train_start(self, trainer, module):
            snapshot(0)

        def on_before_optimizer_step(self, trainer, module, optimizer):
            if trainer.global_step == 15:
                update["step"] = 16
                update["before"] = weight[:8, :8].detach().clone().tolist()
                update["gradient"] = weight.grad[:8, :8].detach().clone().tolist()
                groups = {
                    "SKU embeddings": model.nodes["/line_items/sku"].embedder,
                    "item encoder": model.nodes["/line_items"].encoder.encoder,
                    "summary query": attn,
                    "target decoder": model.nodes["/returned"].decoder,
                }
                update["gradient_norms"] = {
                    name: math.sqrt(sum(float(p.grad.square().sum()) for p in mod.parameters() if p.grad is not None))
                    for name, mod in groups.items()
                }

        def on_train_batch_end(self, trainer, module, outputs, batch, idx):
            losses.append(float(outputs["loss"]))
            step = trainer.global_step
            if step == 16:
                update["after"] = weight[:8, :8].detach().tolist()
            if step in (1, 4, 8, 16, 32, 64, 96, 128, 192, 256):
                snapshot(step)

    data = rf.ArrowDataModule(
        model=model, train=pa.Table.from_pylist(train), num_workers=0, persistent_workers=False, pin_memory=False
    )
    model.fit(
        data,
        batch_size=128,
        learning_rate=2e-3,
        weight_decay=0.01,
        max_epochs=8,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        callbacks=[Track()],
        early_stopping=False,
        rollback=False,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    # One common, final-checkpoint PCA basis: movement cannot come from rotating the camera.
    final = np.asarray(snapshots[-1]["embedding"])
    center = final.mean(0)
    _, _, vt = np.linalg.svd(final - center, full_matrices=False)
    for s in snapshots:
        s["xy"] = ((np.asarray(s.pop("embedding")) - center) @ vt[:2].T).tolist()
    payload = {
        "provenance": {
            "version": "0.3.2",
            "runtime_tree": subprocess.check_output(
                ["git", "rev-parse", "HEAD:src/relflow"], cwd=HERE.parent, text=True
            ).strip(),
            "runtime_reference": "2bb35adb87b6fefe615864b170c196e7509f81fe",
            "seed": 0,
            "torch": torch.__version__,
            "lightning": lit.__version__,
            "pyarrow": pa.__version__,
            "numpy": np.__version__,
            "machine": platform.machine(),
        },
        "train_rows": len(train),
        "test_rows": len(test),
        "probes": probes,
        "labels": [r["returned"] for r in test],
        "snapshots": snapshots,
        "loss": losses,
        "update": update,
        "layout": "field-major: six sku, six quantity, six price tokens; all four heads of the first summary query",
        "projection": "same final PCA basis and center at every checkpoint; all 512 held-out orders",
        "optimizer": {"name": "AdamW", "learning_rate": 0.002, "weight_decay": 0.01, "batch_size": 128},
    }
    (HERE / "data/learning_current.json").write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    record()
