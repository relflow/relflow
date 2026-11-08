"""Compare EMA policies on P078 training and validation observations only."""

import argparse
import hashlib
import json
import time
from functools import partial
from pathlib import Path

import lightning.pytorch as lit
import numpy as np
import regime_recovery as proof
import torch

import relflow as rf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ema", choices=("none", "0.9", "0.99", "0.999"), required=True)
    parser.add_argument("--learning-rate", type=float, choices=(0.003, 0.006), default=0.003)
    parser.add_argument("--steps", type=int, choices=(2, 1800), default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    lit.seed_everything(args.seed, workers=True)
    started = time.perf_counter()
    options = {"partition_seed": args.seed * 100, "no_regime": False}
    training = list(proof.records(rows=4096, seed=args.seed * 100 + 1, **options))
    validation = list(proof.records(rows=1024, seed=args.seed * 100 + 2, **options))
    model = proof.build(4)
    data = rf.SyntheticDataModule(
        train=partial(proof.records, rows=4096, seed=args.seed * 100 + 1, **options),
        validate=partial(proof.records, rows=1024, seed=args.seed * 100 + 2, **options),
        seed=args.seed,
    )
    callbacks = [] if args.ema == "none" else [rf.EMA(decay=float(args.ema))]
    model.fit(
        data,
        batch_size=128,
        learning_rate=args.learning_rate,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator="cpu",
        devices=1,
        max_steps=args.steps,
        max_epochs=-1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )
    model.eval()
    assignments = rf.Group.assignments(model, rf.Address("entity"))
    assigned = np.array([assignments[f"entity-{i:04d}"]["group"] for i in range(proof.ENTITY_COUNT)])
    target = np.asarray([row["y"] for row in validation])
    prediction = proof.predict(model, validation)
    mean = np.mean([row["y"] for row in training])
    rmse = float(np.sqrt(np.mean((target - prediction) ** 2)))
    baseline = float(np.sqrt(np.mean((target - mean) ** 2)))
    result = {
        "purpose": "validation-only EMA exploration; not a P078 test result",
        "seed": args.seed,
        "ema": args.ema,
        "learning_rate": args.learning_rate,
        "steps": model.trainer.global_step,
        "train_rows": len(training),
        "validation_rows": len(validation),
        "adjusted_rand": proof.adjusted_rand(proof.partition(options["partition_seed"]), assigned),
        "validation_rmse": rmse,
        "validation_normalized_rmse": rmse / baseline,
        "assignments": assigned.tolist(),
        "duration_seconds": time.perf_counter() - started,
        "proof_sha256": hashlib.sha256(Path(proof.__file__).read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
