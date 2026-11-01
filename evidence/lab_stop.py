"""Stop only a run's verified coordinator and labelled containers."""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import subprocess
from pathlib import Path


def stop(output: Path) -> None:
    output = output.resolve()
    marker = output.parent / (output.name + ".pid")
    try:
        pid = int(marker.read_text())
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        cwd = (Path("/proc") / str(pid) / "cwd").resolve()
        script = (cwd / command[1]).resolve()
        recorded_output = (cwd / command[command.index("--output") + 1]).resolve()
        if pid > 1 and script == Path(__file__).with_name("lab_run.py").resolve() and recorded_output == output:
            os.kill(pid, signal.SIGTERM)
    except (OSError, ValueError, IndexError):
        pass
    label = hashlib.sha256(output.name.encode()).hexdigest()[:20]
    containers = subprocess.check_output(
        ["docker", "ps", "--quiet", "--filter", "label=relflow.evidence=" + label], text=True
    ).split()
    if containers:
        # The coordinator may be removing the same containers after SIGTERM.
        subprocess.run(
            ["docker", "rm", "--force", *containers],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        remaining = subprocess.check_output(
            ["docker", "ps", "--quiet", "--filter", "label=relflow.evidence=" + label], text=True
        ).split()
        if remaining:
            raise RuntimeError(f"Evidence containers are still running: {remaining}")
    marker.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    stop(parser.parse_args().output)
