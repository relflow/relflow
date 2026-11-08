"""Compare P078's repeated generator with pre-materialized fixed Arrow rows."""

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import regime_recovery as proof
import torch


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("synthetic", "arrow"), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    materializations = []
    if args.mode == "arrow":

        def materialize(*, train, validate, seed):
            started = time.perf_counter()
            training = pa.Table.from_pylist(list(train()))
            validation = pa.Table.from_pylist(list(validate()))
            if (training.num_rows, validation.num_rows) != (4096, 1024):
                raise ValueError("P078 source size changed; re-evaluate fixed Arrow materialization")
            materializations.append(time.perf_counter() - started)
            return proof.rf.ArrowDataModule(train=training, validate=validation, seed=seed)

        proof.rf.SyntheticDataModule = materialize
    steps = 2 if args.smoke else 600
    proof.BUDGET = steps
    torch.set_num_threads(1)
    started = datetime.now(UTC).isoformat()
    clock = time.perf_counter()
    metrics, checks = proof.run(args.seed, None, "cpu")
    if args.mode == "arrow" and len(materializations) != 2:
        raise RuntimeError(f"P078 must materialize exactly two arms, observed {len(materializations)}")
    record = {
        "purpose": "exploratory data-ingress timing; not canonical paired Evidence CI",
        "started_at": started,
        "duration_seconds": time.perf_counter() - clock,
        "seed": args.seed,
        "mode": args.mode,
        "budget": steps,
        "learning_rate": 0.003,
        "ema_decay": 0.99,
        "threads": 1,
        "accelerator": "cpu",
        "platform": platform.platform(),
        "torch": torch.__version__,
        "wheel_sha256": os.environ["GROUP_WHEEL_SHA256"],
        "proof_sha256": digest(Path(proof.__file__)),
        "runner_sha256": digest(Path(__file__)),
        "materialization_seconds": materializations,
        "metrics": metrics,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
