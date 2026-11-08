"""Probe Group on crossed response rules and imbalanced identity groups."""

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import regime_recovery as proof
import torch


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--case", choices=("crossed", "imbalanced"), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    counts = np.array([16, 16, 16, 16]) if args.case == "crossed" else np.array([24, 8, 8, 24])
    if args.case == "crossed":
        proof.INTERCEPTS = np.asarray((-0.3, -0.1, 0.1, 0.3))
        proof.SLOPES = np.asarray((-3.0, -1.0, 1.0, 3.0))
    else:

        def partition(seed):
            groups = np.repeat(np.arange(proof.REGIME_COUNT), counts)
            return np.random.default_rng(seed).permutation(groups)

        proof.partition = partition
    weights = counts / counts.sum()
    if not np.isclose(weights @ proof.INTERCEPTS, 0) or not np.isclose(weights @ proof.SLOPES, 0):
        raise ValueError("The no-regime control must have zero conditional mean for every x")
    steps = 2 if args.smoke else 4800
    proof.BUDGET = steps
    torch.set_num_threads(1)
    started = datetime.now(UTC).isoformat()
    clock = time.perf_counter()
    metrics, checks = proof.run(args.seed, None, "cpu")
    stable = metrics["group_counts"]["4"]["stable"]
    measured = np.bincount(stable["expected_partition"], minlength=4)
    checks["Declared identity counts match the generator"] = bool(np.array_equal(measured, counts))
    checks["No-regime population has zero conditional mean"] = bool(
        np.isclose(weights @ proof.INTERCEPTS, 0) and np.isclose(weights @ proof.SLOPES, 0)
    )
    result = {
        "purpose": "exploratory crossed or imbalanced regimes; not canonical paired Evidence CI",
        "started_at": started,
        "duration_seconds": time.perf_counter() - clock,
        "seed": args.seed,
        "case": args.case,
        "budget": steps,
        "learning_rate": 0.003,
        "ema_decay": 0.99,
        "identity_counts": counts.tolist(),
        "intercepts": proof.INTERCEPTS.tolist(),
        "slopes": proof.SLOPES.tolist(),
        "noise_std": 0.1,
        "threads": 1,
        "accelerator": "cpu",
        "platform": platform.platform(),
        "torch": torch.__version__,
        "wheel_sha256": os.environ["GROUP_WHEEL_SHA256"],
        "proof_sha256": digest(Path(proof.__file__)),
        "runner_sha256": digest(Path(__file__)),
        "metrics": metrics,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
