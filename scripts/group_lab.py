"""Exploratory Group optimization on P078 training and validation data.

This is a search, not a replacement for P078's preregistered test. Run each
configuration once per seed; keep its JSON and logs even when it fails.
"""

import argparse
import hashlib
import json
import platform
import time
from functools import partial
from pathlib import Path

import lightning.pytorch as lit
import numpy as np
import torch
from regime_recovery import ENTITY_COUNT, adjusted_rand, partition, predict, records

import relflow as rf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    lit.seed_everything(args.seed, workers=True)
    start = time.time()
    partition_seed = args.seed * 100
    options = {"partition_seed": partition_seed, "no_regime": False}
    training = list(records(rows=4096, seed=args.seed * 100 + 1, **options))
    validation = list(records(rows=1024, seed=args.seed * 100 + 2, **options))
    model = rf.Model.xs(
        entity=rf.Group(n_groups=args.groups, p_unavailable=0.0),
        x=rf.Number,
        y=rf.Number(mask=True, objective="mse"),
    )
    data = rf.SyntheticDataModule(
        train=partial(records, rows=4096, seed=args.seed * 100 + 1, **options),
        validate=partial(records, rows=1024, seed=args.seed * 100 + 2, **options),
        seed=args.seed,
    )
    model.fit(
        data,
        batch_size=128,
        learning_rate=args.learning_rate,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        accelerator=args.accelerator,
        devices=1,
        max_steps=args.steps,
        max_epochs=-1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=[rf.EMA(decay=0.99)],
    )
    model.eval()
    assignments = rf.Group.assignments(model, rf.Address("entity"))
    assigned = np.array([assignments[f"entity-{i:04d}"]["group"] for i in range(ENTITY_COUNT)])
    target = np.asarray([row["y"] for row in validation])
    prediction = predict(model, validation)
    mean = np.mean([row["y"] for row in training])
    rmse = float(np.sqrt(np.mean((target - prediction) ** 2)))
    baseline = float(np.sqrt(np.mean((target - mean) ** 2)))
    result = {
        "purpose": "validation-only exploratory search; not a P078 test result",
        "seed": args.seed,
        "groups": args.groups,
        "learning_rate": args.learning_rate,
        "steps": model.trainer.global_step,
        "train_rows": len(training),
        "validation_rows": len(validation),
        "adjusted_rand": adjusted_rand(partition(partition_seed), assigned),
        "validation_rmse": rmse,
        "validation_normalized_rmse": rmse / baseline,
        "occupied_groups": int(len(np.unique(assigned))),
        "assignments": assigned.tolist(),
        "duration_seconds": time.time() - start,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if args.accelerator == "gpu" else "cpu",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
