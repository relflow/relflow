"""Train a repeat-purchase model on fictional records, outside the docs build."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from importlib.metadata import version
from pathlib import Path

import lightning.pytorch as lit
import pyarrow as pa
import pyarrow.parquet as pq
import relflow as rf
import torch

SEED = 42


class Order(rf.Branch):
    amount = rf.Number
    category = rf.Category
    days_ago = rf.Number


class Customer(rf.Model):
    days_since_signup = rf.Number
    orders = Order(length=32, overflow="tail")
    returns_in_30d = rf.Boolean(predict=True)


def schema(*, labeled: bool) -> pa.Schema:
    order = pa.struct([("amount", pa.float64()), ("category", pa.string()), ("days_ago", pa.float64())])
    fields = [
        pa.field("customer_id", pa.string()),
        pa.field("days_since_signup", pa.float64()),
        pa.field("orders", pa.list_(pa.field("element", order))),
    ]
    if labeled:
        fields.append(pa.field("returns_in_30d", pa.bool_()))
    return pa.schema(fields)


def observations(count: int, *, prefix: str, seed: int) -> list[dict]:
    """Sample fictional future outcomes using only recency known at the cutoff."""
    rng = random.Random(seed)
    records = []
    for ordinal in range(count):
        recent = rng.choice((3, 7, 14, 45, 60, 90))
        days = sorted(rng.sample(range(recent + 1, 181), rng.randrange(1, 8)), reverse=True) + [recent]
        orders = [
            {"amount": round(rng.uniform(10, 150), 2), "category": rng.choice(("books", "home", "sports")), "days_ago": float(day)}
            for day in days
        ]
        records.append({
            "customer_id": f"{prefix}-{ordinal:04d}",
            "days_since_signup": float(rng.randrange(365, 1096)),
            "orders": orders,
            "returns_in_30d": rng.random() < (0.85 if recent <= 14 else 0.15),
        })
    return records


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    lit.seed_everything(SEED, workers=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    requests = [
        {"customer_id": "recent", "days_since_signup": 730.0, "orders": [
            {"amount": 80.0, "category": "home", "days_ago": 75.0},
            {"amount": 35.0, "category": "books", "days_ago": 3.0},
        ]},
        {"customer_id": "less-recent", "days_since_signup": 730.0, "orders": [
            {"amount": 80.0, "category": "home", "days_ago": 150.0},
            {"amount": 35.0, "category": "books", "days_ago": 90.0},
        ]},
        {"customer_id": "occasional", "days_since_signup": 400.0, "orders": [
            {"amount": 45.0, "category": "sports", "days_ago": 14.0},
        ]},
    ]
    tables = {
        "train": pa.Table.from_pylist(observations(512, prefix="train", seed=SEED), schema=schema(labeled=True)),
        "validation": pa.Table.from_pylist(observations(128, prefix="validation", seed=SEED + 1), schema=schema(labeled=True)),
        "requests": pa.Table.from_pylist(requests, schema=schema(labeled=False)),
    }
    for name, table in tables.items():
        pq.write_table(table, output / f"{name}.parquet")
    data = rf.ArrowDataModule(train=str(output / "train.parquet"), validate=str(output / "validation.parquet"), num_workers=0, pin_memory=False)
    settings = dict(batch_size=128, learning_rate=1e-3, max_epochs=6, accelerator="cpu", devices=1,
                    precision="32-true", deterministic=True, logger=False, enable_progress_bar=False,
                    enable_model_summary=False, num_sanity_val_steps=0)
    if args.smoke:
        settings.update(max_epochs=1, limit_train_batches=1, limit_val_batches=1)
    model = Customer(preset="small")
    model.fit(data, checkpoint_dir=output / "checkpoints", **settings)
    model.save(output / "customer.rf")
    loaded = rf.Model.load(output / "customer.rf")
    predicted = loaded.predict(tables["requests"], retain=("customer_id",), batch_size=128)
    predicted.write_parquet(output / "predictions.parquet")
    readable = [
        {"customer_id": row["inputs"]["customer_id"], "return_probability": row["predictions"]["returns_in_30d"]["content"]["probability"]}
        for row in predicted.iter_rows(named=True)
    ]
    receipt = {
        "purpose": "Fictional workflow demonstration; not evidence of application quality or competitive performance.",
        "seed": SEED, "smoke": args.smoke, "architecture_preset": "small", "fit_settings": settings,
        "threads": 1, "python_version": platform.python_version(),
        "versions": {name: version(name) for name in ("relflow", "torch", "lightning", "pyarrow", "polars")},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data": {name: {"rows": table.num_rows, "sha256": hashlib.sha256((output / f"{name}.parquet").read_bytes()).hexdigest()} for name, table in tables.items()},
        "fictional_rule": "Return labels are sampled with probability 0.85 when the latest order is at most 14 days old; otherwise 0.15.",
        "predictions": readable,
    }
    (output / "recorded-run.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(readable, indent=2, allow_nan=False))
    print(f"Saved data, model, predictions, and recorded-run.json in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("repeat-purchase-run"))
    parser.add_argument("--smoke", action="store_true", help="Run one training and validation batch to check the workflow.")
    run(parser.parse_args())
