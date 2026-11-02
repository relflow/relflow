"""Cancellation cleanup may stop only the coordinator and containers it owns."""

import hashlib
import signal
from pathlib import Path

import lab_stop
import pytest
from ssh_run import cleanup


@pytest.mark.parametrize("owned", [False, True])
def test_stop_verifies_pid_ownership_and_limits_container_removal(tmp_path, monkeypatch, owned):
    output = tmp_path / "owned-run"
    marker = tmp_path / "owned-run.pid"
    marker.write_text("54321\n")
    script = Path(lab_stop.__file__).with_name("lab_run.py") if owned else tmp_path / "unrelated.py"
    command = "\0".join(["python3", str(script), "--output", str(output)]).encode()
    original = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes", lambda path: command if str(path) == "/proc/54321/cmdline" else original(path)
    )
    signals, commands = [], []
    responses = iter(["owned-container\n", ""])
    monkeypatch.setattr(lab_stop.os, "kill", lambda pid, value: signals.append((pid, value)))
    monkeypatch.setattr(
        lab_stop.subprocess, "check_output", lambda command, **kwargs: commands.append(command) or next(responses)
    )
    monkeypatch.setattr(lab_stop.subprocess, "run", lambda command, **kwargs: commands.append(command))
    lab_stop.stop(output)
    assert signals == ([(54321, signal.SIGTERM)] if owned else [])
    label = hashlib.sha256(output.name.encode()).hexdigest()[:20]
    assert commands == [
        ["docker", "ps", "--quiet", "--filter", "label=relflow.evidence=" + label],
        ["docker", "rm", "--force", "owned-container"],
        ["docker", "ps", "--quiet", "--filter", "label=relflow.evidence=" + label],
    ]
    assert not marker.exists()


@pytest.mark.parametrize("run_id", ["../other-run", "run;command", "-option"])
def test_cleanup_rejects_unscoped_run_names(run_id):
    with pytest.raises(ValueError, match="Invalid evidence run ID"):
        cleanup("user@lab", "ssh", run_id)
