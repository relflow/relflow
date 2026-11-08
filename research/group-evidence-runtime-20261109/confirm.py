"""Compare a selected setting with P078 on fresh seeds and full controls."""

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
    parser.add_argument("--setting", choices=("baseline", "faster"), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    steps, rate = (4800, 0.003) if args.setting == "baseline" else (1800, 0.006)
    if args.smoke:
        steps = 2
    original_fit = proof.rf.Model.fit
    calls = []

    def fit(model, *positional, **keywords):
        if keywords.get("learning_rate") != 0.003:
            raise ValueError("P078 learning rate changed; re-evaluate the experiment adapter")
        keywords["learning_rate"] = rate
        calls.append(rate)
        return original_fit(model, *positional, **keywords)

    # Override only the two declared experimental settings; proof.run owns its
    # generator, data, controls, EMA, evaluation, and original named checks.
    proof.rf.Model.fit = fit
    proof.BUDGET = steps
    torch.set_num_threads(1)
    started = datetime.now(UTC).isoformat()
    clock = time.perf_counter()
    metrics, checks = proof.run(args.seed, None, "cpu")
    if len(calls) != 2:
        raise RuntimeError(f"P078 must train exactly two arms, observed {len(calls)}")
    record = {
        "purpose": "independent-seed comparison; not canonical paired Evidence CI",
        "started_at": started,
        "duration_seconds": time.perf_counter() - clock,
        "seed": args.seed,
        "setting": args.setting,
        "budget": steps,
        "learning_rate": rate,
        "ema_decay": 0.99,
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
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
