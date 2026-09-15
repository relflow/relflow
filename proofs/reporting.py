"""Command-line options and persistent evidence for standalone experiments."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from source import fingerprint

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "proofs/results.yaml"
Experiment = Callable[[int, int | None, str], tuple[dict[str, Any], dict[str, bool]]]


class EvidenceDumper(yaml.SafeDumper):
    """Keep prose and tracebacks readable in the evidence file."""


def represent_string(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|" if "\n" in value else None)


EvidenceDumper.add_representer(str, represent_string)


def plain(value: Any) -> Any:
    """Convert numerical scalars and arrays to portable YAML values."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot record {type(value).__name__}; return numerical metrics or ordinary YAML values.")


def provenance(script: Path) -> dict[str, Any]:
    """Identify the actual experiment and model source, including uncommitted edits."""
    import torch

    import relflow

    package = Path(relflow.__file__).resolve().parent
    files = [
        (path.relative_to(ROOT).as_posix(), path)
        for path in (script, Path(__file__), Path(__file__).with_name("source.py"))
    ]
    files.extend(
        ("src/relflow/" + path.relative_to(package).as_posix(), path) for path in sorted(package.rglob("*.py"))
    )
    digest = hashlib.sha256()
    for name, path in files:
        digest.update(name.encode())
        digest.update(path.read_bytes())
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return {
        "revision": revision.stdout.strip() or None,
        "source_sha256": digest.hexdigest(),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "code_sha256": fingerprint(script.read_text()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host": platform.node(),
        "relflow_path": str(package),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "relflow",
                "torch",
                "lightning",
                "numpy",
                "pyarrow",
                "polars",
                "torchmetrics",
                "tensordict",
                "pydantic",
            )
        },
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def save(identifier: str, result: dict[str, Any], output: Path) -> None:
    """Append one run atomically; lock the read/update for concurrent lab processes."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(output.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        document = yaml.safe_load((output if output.exists() else RESULTS).read_text())
        if document.get("schema") != 1 or identifier not in document.get("proofs", {}):
            raise ValueError(f"{output}: no proof {identifier} in evidence schema 1")
        document["proofs"][identifier]["runs"].append(plain(result))
        with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as pending:
            yaml.dump(document, pending, Dumper=EvidenceDumper, sort_keys=False, allow_unicode=True, width=110)
            temporary = Path(pending.name)
        try:
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)


def report(identifier: str, experiment: Experiment, *, seed: int) -> None:
    """Run an experiment and record every outcome, including unmet gates and errors."""
    parser = argparse.ArgumentParser(description=experiment.__module__ + " · " + identifier)
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--steps", type=int, help="Cap each training stage for a smoke run; not capability evidence.")
    parser.add_argument("--accelerator", choices=("cpu", "gpu", "mps", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=RESULTS)
    options = parser.parse_args()
    if options.steps is not None and options.steps < 1:
        parser.error("--steps must be positive")
    if options.threads < 1:
        parser.error("--threads must be positive")

    import torch

    torch.set_num_threads(options.threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    script = Path(experiment.__code__.co_filename).resolve()
    started = time.perf_counter()
    result: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": "full" if options.steps is None else "smoke",
        "seed": options.seed,
        "steps_override": options.steps,
        "accelerator": options.accelerator,
        "threads": options.threads,
        "provenance": {},
    }
    error = None
    try:
        result["provenance"] = provenance(script)
        metrics, checks = experiment(options.seed, options.steps, options.accelerator)
        result["metrics"] = plain(metrics)
        result["checks"] = {name: bool(value) for name, value in checks.items()}
        if not checks:
            raise ValueError(f"{identifier}: the experiment must return its behavioral checks")
        result["outcome"] = "met" if all(result["checks"].values()) else "not_met"
    except Exception as exception:
        error = exception
        result["outcome"] = "error"
        result["error"] = traceback.format_exc()
    result["duration_seconds"] = round(time.perf_counter() - started, 3)
    save(identifier, result, options.output)
    print(json.dumps({"proof": identifier, **result}, indent=2))
    print(f"Recorded {identifier} in {options.output}")
    if error is not None:
        raise SystemExit(1)


__all__ = ["report"]
