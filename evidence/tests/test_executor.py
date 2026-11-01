"""Experiment revisions are data on the host, and execute only inside containers."""

import argparse
import io
import tarfile
from types import SimpleNamespace

import ssh_run
from evidence_ci import plan


def test_planning_uses_trusted_parser_without_importing_suite_helpers(tmp_path):
    family = tmp_path / "experiments/family"
    family.mkdir(parents=True)
    (tmp_path / "experiments/source.py").write_text("raise RuntimeError('suite helper imported on host')\n")
    (family / "signal.py").write_text(
        "# %% [markdown]\n# ---\n# evidence-id: P901\n# ---\n"
        "# %%\nraise RuntimeError('experiment executed on host')\nreport(seed=123)\n"
    )
    manifest = plan(
        run_id="separate-suite", candidate={}, baseline={}, evidence_sha="suite", executor_sha="executor", root=tmp_path
    )
    assert len(manifest["tasks"]) == 2
    assert {task["id"] for task in manifest["tasks"]} == {"P901"}
    assert {task["seed"] for task in manifest["tasks"]} == {123}
    assert {task["executor_sha"] for task in manifest["tasks"]} == {"executor"}


def test_dispatch_keeps_executor_and_suite_in_separate_namespaces(tmp_path, monkeypatch):
    executor, suite = tmp_path / "executor", tmp_path / "suite"
    for root, body in ((executor, "trusted executor"), (suite, "untrusted suite coordinator")):
        root.mkdir()
        (root / "lab_run.py").write_text(body)
    candidate, baseline, requirements = [
        tmp_path / name for name in ("candidate.whl", "baseline.whl", "requirements.txt")
    ]
    for path in (candidate, baseline, requirements):
        path.write_text("fixture")
    monkeypatch.setattr(ssh_run, "ROOT", executor)

    def git(command, *, cwd, **kwargs):
        if command[1] == "rev-parse":
            return ("e" if cwd == executor else "a") * 40 + "\n"
        if command[1] == "status":
            return ""
        assert command[1] == "ls-files"
        return b"lab_run.py\0"

    monkeypatch.setattr(ssh_run.subprocess, "check_output", git)
    calls, payload = [], []
    reports = io.BytesIO()
    with tarfile.open(fileobj=reports, mode="w:gz") as archive:
        item = tarfile.TarInfo("summary.json")
        item.size = 2
        archive.addfile(item, io.BytesIO(b"{}"))

    def remote(command, **kwargs):
        calls.append(command[-1])
        if "input" in kwargs:
            payload.append(kwargs["input"])
        return SimpleNamespace(returncode=0, stdout=reports.getvalue())

    monkeypatch.setattr(ssh_run.subprocess, "run", remote)
    ssh_run.run(
        argparse.Namespace(
            suite_root=suite,
            target="lab",
            transport="ssh",
            output=tmp_path / "owned-run",
            candidate_wheel=candidate,
            baseline_wheel=baseline,
            requirements=requirements,
            candidate_sha="candidate",
            baseline_sha="base",
            workers=8,
            steps=None,
            ids=None,
        )
    )
    with tarfile.open(fileobj=io.BytesIO(payload[0]), mode="r:gz") as archive:
        assert archive.extractfile("executor/lab_run.py").read() == b"trusted executor"
        assert archive.extractfile("suite/lab_run.py").read() == b"untrusted suite coordinator"
    dispatch = calls[1]
    assert "executor/lab_run.py" in dispatch and "suite/lab_run.py" not in dispatch
    assert "--executor-sha " + "e" * 40 in dispatch
    assert "--evidence-sha " + "a" * 40 in dispatch
    assert "--suite-root relflow-evidence/runs/owned-run/suite" in dispatch
    assert "executor/lab_stop.py" in calls[-1]
