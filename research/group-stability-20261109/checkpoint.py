"""Compare P078's fixed run with validation-selected early stopping."""

import argparse
import hashlib
import json
import math
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import regime_recovery as proof
import torch

from relflow.architecture.checkpoint import RollbackCheckpoint


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy", choices=("fixed", "validation"), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    steps = 32 if args.smoke else 4800
    proof.BUDGET = steps
    original_fit = proof.rf.Model.fit
    selections = []

    def fit(model, *positional, **keywords):
        if keywords.get("early_stopping") is not False or keywords.get("rollback") is not False:
            raise ValueError("P078 training policy changed; review the experiment adapter")
        if args.policy == "validation":
            checkpoint_dir = args.output.parent / "checkpoints" / args.output.stem / str(len(selections))
            if checkpoint_dir.exists():
                raise FileExistsError(checkpoint_dir)
            keywords.update(
                early_stopping=True,
                rollback=True,
                patience=20,
                min_delta=0.0,
                checkpoint_dir=checkpoint_dir,
                enable_checkpointing=True,
            )
        result = original_fit(model, *positional, **keywords)
        trainer = model.trainer
        selected = None
        if args.policy == "validation":
            checkpoint = next(callback for callback in trainer.callbacks if isinstance(callback, RollbackCheckpoint))
            path = Path(checkpoint.best_model_path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            score = float(checkpoint.best_model_score)
            if not math.isfinite(score) or not 0 < saved["global_step"] <= trainer.global_step <= steps:
                raise ValueError("Validation checkpoint must have a finite score within the declared budget")
            restored = model.state_dict()
            if restored.keys() != saved["state_dict"].keys() or any(
                isinstance(value, torch.Tensor) and not torch.equal(value.cpu(), restored[name].cpu())
                for name, value in saved["state_dict"].items()
            ):
                raise ValueError("Model tensors do not match the validation-selected checkpoint")
            selected = {
                "step": saved["global_step"],
                "validation_loss": score,
                "file": str(path.relative_to(args.output.parent)),
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
        selections.append({"stopped_at": trainer.global_step, "selected": selected})
        return result

    proof.rf.Model.fit = fit
    torch.set_num_threads(1)
    started = datetime.now(UTC).isoformat()
    clock = time.perf_counter()
    metrics, checks = proof.run(args.seed, None, "cpu")
    if len(selections) != 2:
        raise RuntimeError(f"P078 must train two arms, observed {len(selections)}")
    result = {
        "purpose": "fresh-seed validation-policy exploration; not canonical Evidence CI",
        "started_at": started,
        "duration_seconds": time.perf_counter() - clock,
        "seed": args.seed,
        "policy": args.policy,
        "max_steps_per_arm": steps,
        "learning_rate": 0.003,
        "ema_decay": 0.99,
        "validation_patience_epochs": 20 if args.policy == "validation" else None,
        "validation_min_delta": 0.0 if args.policy == "validation" else None,
        "threads": 1,
        "accelerator": "cpu",
        "platform": platform.platform(),
        "torch": torch.__version__,
        "wheel_sha256": os.environ["GROUP_WHEEL_SHA256"],
        "proof_sha256": digest(Path(proof.__file__)),
        "runner_sha256": digest(Path(__file__)),
        "selections": {"stable": selections[0], "no_regime": selections[1]},
        "metrics": metrics,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
