"""Dispatch isolated evidence to lab over SSH/Tailscale and retrieve the reports."""

from __future__ import annotations

import argparse
import io
import re
import shlex
import signal
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

from evidence_ci import ROOT


def connection(target: str, transport: str) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@:-]*", target):
        raise ValueError("Invalid SSH target")
    if transport == "tailscale":
        return ["tailscale", "ssh", target]
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", target]


def extract_reports(data: bytes, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        if sum(member.size for member in members) > 1024 * 1024 * 1024:
            raise ValueError("Report archive exceeds the 1 GiB transfer bound")
        seen = set()
        for member in members:
            name = PurePosixPath(member.name)
            if member.isdir():
                continue
            if not member.isfile() or name.is_absolute() or ".." in name.parts or name.as_posix() in seen:
                raise ValueError("Unsafe report archive member")
            seen.add(name.as_posix())
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as destination:
                destination.write(source.read())


def cleanup(target: str, transport: str, run_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) or run_id in {".", ".."}:
        raise ValueError("Invalid evidence run ID")
    remote = "relflow-evidence/runs/" + run_id
    script = remote + "/executor/lab_stop.py"
    command = (
        "if test -f "
        + shlex.quote(script)
        + "; then "
        + shlex.join(["python3", script, remote + "/results/" + run_id])
        + "; fi"
    )
    subprocess.run([*connection(target, transport), command], check=True, timeout=30)


def run(options: argparse.Namespace) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", options.output.name) or options.output.name in {".", ".."}:
        raise ValueError("Use a simple unique run directory name")
    revisions = {}
    for role, directory in (("executor", ROOT), ("suite", options.suite_root.resolve())):
        revisions[role] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=directory, text=True
        )
        if dirty:
            raise ValueError(f"Commit {role} changes before dispatch so its recorded revision is exact")
    command = connection(options.target, options.transport)
    remote = "relflow-evidence/runs/" + options.output.name
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for role, directory in (("executor", ROOT), ("suite", options.suite_root.resolve())):
            files = subprocess.check_output(["git", "ls-files", "-z"], cwd=directory).decode().split("\0")
            for name in filter(None, files):
                path = directory / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError("Evidence dispatch requires regular tracked source files")
                archive.add(path, arcname=role + "/" + name, recursive=False)
        for role, wheel in (("candidate", options.candidate_wheel), ("baseline", options.baseline_wheel)):
            if wheel.is_symlink() or not wheel.is_file():
                raise ValueError("Supply regular wheel files for dispatch")
            archive.add(wheel, arcname=f"inputs/{role}/{wheel.name}", recursive=False)
        archive.add(options.requirements, arcname="inputs/requirements.txt", recursive=False)
    destination = shlex.quote(remote)
    subprocess.run(
        [*command, f"test ! -e {destination} && mkdir -p {destination} && tar -xz -C {destination}"],
        input=payload.getvalue(),
        check=True,
    )
    remote_output = remote + "/results/" + options.output.name
    arguments = [
        "python3",
        remote + "/executor/lab_run.py",
        "--candidate-wheel",
        remote + "/inputs/candidate/" + options.candidate_wheel.name,
        "--baseline-wheel",
        remote + "/inputs/baseline/" + options.baseline_wheel.name,
        "--candidate-sha",
        options.candidate_sha,
        "--baseline-sha",
        options.baseline_sha,
        "--evidence-sha",
        revisions["suite"],
        "--executor-sha",
        revisions["executor"],
        "--suite-root",
        remote + "/suite",
        "--requirements",
        remote + "/inputs/requirements.txt",
        "--output",
        remote_output,
        "--workers",
        str(options.workers),
    ]
    if options.steps is not None:
        arguments += ["--steps", str(options.steps)]
    if options.ids:
        arguments += ["--ids", *options.ids]
    try:
        result = subprocess.run([*command, shlex.join(arguments)], check=False)
        reports = subprocess.run(
            [*command, "tar -cz -C " + shlex.quote(remote_output) + " ."], stdout=subprocess.PIPE, check=True
        )
        extract_reports(reports.stdout, options.output)
    finally:
        cleanup(options.target, options.transport, options.output.name)
    print("Retrieved evidence reports to", options.output, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    for signum in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(signum, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--transport", choices=("ssh", "tailscale"), default="ssh")
    parser.add_argument("--candidate-wheel", required=True, type=Path)
    parser.add_argument("--baseline-wheel", required=True, type=Path)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--baseline-sha", required=True)
    parser.add_argument("--requirements", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--ids", nargs="+")
    options = parser.parse_args()
    if options.output.exists():
        parser.error("Use a new output directory for each run")
    run(options)


if __name__ == "__main__":
    main()
