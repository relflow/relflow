"""Run standalone experiments by their stable IDs, serially by default."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from source import registry


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    entries = registry(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="*", help="Evidence IDs, such as P014; omit to run all experiments.")
    parser.add_argument("--list", action="store_true", help="List stable IDs and their scripts without training.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--accelerator", choices=("cpu", "gpu", "mps", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, help="Directory for independent JSON records.")
    options = parser.parse_args()
    selected = options.experiments or list(entries)
    unknown = sorted(set(selected) - entries.keys())
    if unknown:
        parser.error(f"Unknown evidence IDs: {', '.join(unknown)}. Use --list to see available experiments.")
    if options.list:
        for identifier in selected:
            print(f"{identifier}  {entries[identifier]['script']}")
        return

    arguments = []
    for name in ("seed", "steps", "accelerator", "threads"):
        value = getattr(options, name)
        if value is not None:
            arguments.extend([f"--{name}", str(value)])
    errors = []
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(root / "experiments"), environment.get("PYTHONPATH")))
    )
    for identifier in selected:
        print(f"\nRunning {identifier}: {entries[identifier]['script']}", flush=True)
        output = (
            ["--output", str(options.output_dir / f"{identifier}-{time.time_ns()}.json")] if options.output_dir else []
        )
        result = subprocess.run(
            [sys.executable, str(root / entries[identifier]["script"]), *arguments, *output],
            check=False,
            env=environment,
        )
        if result.returncode:
            errors.append(identifier)
    if errors:
        print(f"Execution errors: {', '.join(errors)}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
