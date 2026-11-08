"""Time the unchanged P078 proof with one or two Torch CPU threads."""

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
    parser.add_argument("--threads", type=int, choices=(1, 2), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    steps = 2 if args.smoke else 600
    proof.BUDGET = steps
    torch.set_num_threads(args.threads)
    started = datetime.now(UTC).isoformat()
    clock = time.perf_counter()
    metrics, checks = proof.run(args.seed, None, "cpu")
    record = {
        "purpose": "exploratory thread timing; not canonical paired Evidence CI",
        "started_at": started,
        "duration_seconds": time.perf_counter() - clock,
        "seed": args.seed,
        "budget": steps,
        "threads": args.threads,
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
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
