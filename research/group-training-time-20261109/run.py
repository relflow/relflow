"""Compare P078's original budget with a longer, otherwise unchanged run."""

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import regime_recovery as proof
import torch


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, choices=(2, 600, 4800), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    started = datetime.now(UTC).isoformat()
    clock = time.perf_counter()
    # The published proof caps every invocation at 600; this exploration uses
    # its unchanged generator, fit options, evaluation, and checks for longer.
    proof.BUDGET = 4800
    metrics, checks = proof.run(args.seed, args.steps, "cpu")
    result = {
        "purpose": "exploratory training-budget comparison; not canonical Evidence CI",
        "started_at": started,
        "duration_seconds": time.perf_counter() - clock,
        "seed": args.seed,
        "budget": args.steps,
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
