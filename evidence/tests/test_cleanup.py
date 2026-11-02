"""Cancellation cleanup may stop only the coordinator and containers it owns."""

import hashlib
import signal
import subprocess
from pathlib import Path

import lab_stop
import pytest
from ssh_run import cleanup


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(lab_stop.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(lab_stop.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    return now


@pytest.mark.parametrize("owned", [False, True])
def test_coordinator_checks_script_output_and_process_start_identity(tmp_path, owned):
    output = tmp_path / "owned-run"
    proc = tmp_path / "proc"
    process = proc / "54321"
    process.mkdir(parents=True)
    script = Path(lab_stop.__file__).with_name("lab_run.py") if owned else tmp_path / "unrelated.py"
    command = "\0".join(["python3", str(script), "--output", str(output)]).encode()
    (process / "cmdline").write_bytes(command)
    (process / "cwd").symlink_to(tmp_path, target_is_directory=True)
    (process / "stat").write_text("54321 (python with spaces) S " + "0 " * 18 + "123456 0\n")
    assert lab_stop.coordinator(54321, output, proc=proc) == ("123456" if owned else None)
    assert lab_stop.coordinator(54321, tmp_path / "other-run", proc=proc) is None
    assert lab_stop.coordinator(1, output, proc=proc) is None


@pytest.mark.parametrize("owned", [False, True])
def test_stop_verifies_pid_ownership_and_limits_container_removal(tmp_path, monkeypatch, clock, owned):
    output = tmp_path / "owned-run"
    marker = tmp_path / "owned-run.pid"
    marker.write_text("54321\n")
    signals, commands = [], []
    monkeypatch.setattr(lab_stop, "coordinator", lambda *args: "start" if owned and not signals else None)
    responses = iter(["owned-container\n", "", ""])
    monkeypatch.setattr(lab_stop.os, "kill", lambda pid, value: signals.append((pid, value)))
    monkeypatch.setattr(
        lab_stop.subprocess, "check_output", lambda command, **kwargs: commands.append(command) or next(responses)
    )
    monkeypatch.setattr(lab_stop.subprocess, "run", lambda command, **kwargs: commands.append(command))
    lab_stop.stop(output)
    assert signals == ([(54321, signal.SIGTERM)] if owned else [])
    label = hashlib.sha256(output.name.encode()).hexdigest()[:20]
    assert commands == [
        ["docker", "ps", "--all", "--quiet", "--filter", "label=relflow.evidence=" + label],
        ["docker", "rm", "--force", "owned-container"],
        ["docker", "ps", "--all", "--quiet", "--filter", "label=relflow.evidence=" + label],
        ["docker", "ps", "--all", "--quiet", "--filter", "label=relflow.evidence=" + label],
    ]
    assert not marker.exists()


def test_stop_drains_concurrent_removal_and_a_late_created_worker(tmp_path, monkeypatch, clock):
    output = tmp_path / "cancelled-run"
    label = hashlib.sha256(output.name.encode()).hexdigest()[:20]
    responses = iter(["first-worker\n", "first-worker\n", "", "late-worker\n", "", ""])
    events = []
    monkeypatch.setattr(lab_stop, "coordinator", lambda *args: None)

    def listing(command, **kwargs):
        assert command == ["docker", "ps", "--all", "--quiet", "--filter", "label=relflow.evidence=" + label]
        assert 0 < kwargs["timeout"] <= 2
        value = next(responses)
        events.append(("ps", value.split()))
        return value

    def remove(command, **kwargs):
        assert command[:3] == ["docker", "rm", "--force"]
        assert 0 < kwargs["timeout"] <= 2 and kwargs["check"] is False
        events.append(("rm", command[3:]))
        # Another remover may already be removing this container; retry the scoped listing.
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(lab_stop.subprocess, "check_output", listing)
    monkeypatch.setattr(lab_stop.subprocess, "run", remove)
    lab_stop.stop(output)
    assert [value for operation, value in events if operation == "rm"] == [
        ["first-worker"],
        ["first-worker"],
        ["late-worker"],
    ]
    assert events[-2:] == [("ps", []), ("ps", [])]


def test_stubborn_coordinator_is_killed_only_after_grace_and_identity_recheck(tmp_path, monkeypatch, clock):
    output = tmp_path / "owned-run"
    (tmp_path / "owned-run.pid").write_text("54321\n")
    signals = []
    monkeypatch.setattr(
        lab_stop, "coordinator", lambda *args: None if signals and signals[-1][1] == signal.SIGKILL else "original"
    )
    monkeypatch.setattr(lab_stop.os, "kill", lambda pid, value: signals.append((pid, value)))
    monkeypatch.setattr(lab_stop.subprocess, "check_output", lambda *args, **kwargs: "")
    lab_stop.stop(output)
    assert signals == [(54321, signal.SIGTERM), (54321, signal.SIGKILL)]
    assert 2 <= clock[0] < 20


def test_recycled_coordinator_pid_is_never_killed(tmp_path, monkeypatch, clock):
    output = tmp_path / "owned-run"
    (tmp_path / "owned-run.pid").write_text("54321\n")
    signals = []
    monkeypatch.setattr(lab_stop, "coordinator", lambda *args: "recycled" if signals else "original")
    monkeypatch.setattr(lab_stop.os, "kill", lambda pid, value: signals.append((pid, value)))
    monkeypatch.setattr(lab_stop.subprocess, "check_output", lambda *args, **kwargs: "")
    lab_stop.stop(output)
    assert signals == [(54321, signal.SIGTERM)]


def test_remaining_workers_fail_within_deadline_and_preserve_marker(tmp_path, monkeypatch, clock):
    output = tmp_path / "owned-run"
    marker = tmp_path / "owned-run.pid"
    marker.write_text("54321\n")
    monkeypatch.setattr(lab_stop, "coordinator", lambda *args: None)
    monkeypatch.setattr(lab_stop.subprocess, "check_output", lambda *args, **kwargs: "stuck-worker\n")
    monkeypatch.setattr(lab_stop.subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="containers=\\['stuck-worker'\\]"):
        lab_stop.stop(output, timeout=0.7)
    assert clock[0] == pytest.approx(0.7)
    assert marker.exists()


def test_unresponsive_docker_is_bounded_and_does_not_claim_cleanup(tmp_path, monkeypatch, clock):
    output = tmp_path / "owned-run"
    marker = tmp_path / "owned-run.pid"
    marker.write_text("54321\n")
    monkeypatch.setattr(lab_stop, "coordinator", lambda *args: None)

    def listing(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(lab_stop.subprocess, "check_output", listing)
    with pytest.raises(RuntimeError, match="docker_error=TimeoutExpired"):
        lab_stop.stop(output, timeout=0.7)
    assert clock[0] == pytest.approx(0.7)
    assert marker.exists()


@pytest.mark.parametrize("run_id", ["../other-run", "run;command", "-option"])
def test_cleanup_rejects_unscoped_run_names(run_id):
    with pytest.raises(ValueError, match="Invalid evidence run ID"):
        cleanup("user@lab", "ssh", run_id)
