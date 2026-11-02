"""Stop only a run's verified coordinator and labelled containers."""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import subprocess
import time
from pathlib import Path


def coordinator(pid: int, output: Path, *, proc: Path = Path("/proc")) -> str | None:
    """Identify this coordinator incarnation before signalling a possibly recycled PID."""
    if pid <= 1:
        return None
    try:
        process = proc / str(pid)
        fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in {"Z", "X", "x"} or not fields[19].isdigit():
            return None
        command = (process / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        cwd = (process / "cwd").resolve(strict=True)
        script = (cwd / command[1]).resolve()
        recorded_output = (cwd / command[command.index("--output") + 1]).resolve()
        if script == Path(__file__).with_name("lab_run.py").resolve() and recorded_output == output:
            return fields[19]
    except (OSError, ValueError, IndexError):
        pass
    return None


def stop(output: Path, *, timeout: float = 20) -> None:
    """Drain only this run, tolerating concurrent removals within a bounded deadline."""
    output = output.resolve()
    marker = output.parent / (output.name + ".pid")
    try:
        pid = int(marker.read_text())
    except (OSError, ValueError):
        pid = 0
    started = coordinator(pid, output)
    beginning = time.monotonic()
    deadline = beginning + timeout
    if started is not None:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    label = hashlib.sha256(output.name.encode()).hexdigest()[:20]
    quiet = 0
    killed = False
    containers = []
    docker_error = None
    while time.monotonic() < deadline:
        alive = started is not None and coordinator(pid, output) == started
        if alive and not killed and time.monotonic() - beginning >= 2:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            killed = True
        try:
            containers = subprocess.check_output(
                ["docker", "ps", "--all", "--quiet", "--filter", "label=relflow.evidence=" + label],
                text=True,
                timeout=min(2, max(0.01, deadline - time.monotonic())),
            ).split()
            docker_error = None
            if containers:
                # Workers and the coordinator may be removing these same containers.
                subprocess.run(
                    ["docker", "rm", "--force", *containers],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=min(2, max(0.01, deadline - time.monotonic())),
                )
            quiet = quiet + 1 if not alive and not containers else 0
            if quiet >= 2:
                marker.unlink(missing_ok=True)
                return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            docker_error = type(error).__name__
            quiet = 0
        time.sleep(min(0.2, max(0, deadline - time.monotonic())))
    alive = started is not None and coordinator(pid, output) == started
    raise RuntimeError(
        f"Evidence cleanup exceeded {timeout:g}s: coordinator={pid if alive else None}, "
        f"containers={containers}, docker_error={docker_error}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    stop(parser.parse_args().output)
