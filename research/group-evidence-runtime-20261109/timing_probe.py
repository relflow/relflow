"""Measure P078's existing trajectory callback without changing its checks."""

import hashlib
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import regime_recovery as proof
import torch

OUTPUT = Path(os.environ["GROUP_PROFILE_OUTPUT"])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    original = proof.AssignmentTrajectory
    samples = []

    class TimedTrajectory(original):
        def on_validation_epoch_end(self, trainer, pl_module):
            start = time.perf_counter()
            try:
                return super().on_validation_epoch_end(trainer, pl_module)
            finally:
                samples.append(time.perf_counter() - start)

    proof.AssignmentTrajectory = TimedTrajectory
    torch.set_num_threads(1)
    start = time.perf_counter()
    started = datetime.now(UTC).isoformat()
    metrics, checks = proof.run(7801, None, "cpu")
    record = {
        "purpose": "exploratory timing; not canonical paired Evidence CI",
        "started_at": started,
        "duration_seconds": time.perf_counter() - start,
        "seed": 7801,
        "threads": 1,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "wheel_sha256": os.environ["GROUP_WHEEL_SHA256"],
        "proof_sha256": digest(Path(proof.__file__)),
        "runner_sha256": digest(Path(__file__)),
        "trajectory_callback_seconds": samples,
        "metrics": metrics,
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
