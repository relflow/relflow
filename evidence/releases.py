"""Supported Evidence comparisons and immutable release asset names."""

import re
import subprocess
import tomllib
from collections.abc import Iterable
from pathlib import Path

VERSION = re.compile(r"[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?")
MINIMUM = (0, 5)
BASELINE = "b868c450249de1c68c843506dca512bce7a6b333"
LEGACY = {
    "publication": "publication.json",
    "reports": "evidence-reports.tar.gz",
    "inputs": "evidence-inputs.tar.gz",
    "provenance": "github-run.json",
    "plots": "proof-visualizations.zip",
    "measurements": "measurements.md",
    "checksums": "SHA256SUMS",
}


def supported(version: str) -> bool:
    if not VERSION.fullmatch(version):
        raise ValueError(f"Evidence requires a normalized package version, got {version!r}")
    parts = re.match(r"(\d+)\.(\d+)", version)
    return parts is not None and tuple(map(int, parts.groups())) >= MINIMUM


def event_base(project: Path, candidate: str, event: dict, name: str) -> str:
    """Read the actual comparison base from the authenticated workflow event."""
    if name == "pull_request":
        return event["pull_request"]["base"]["sha"]
    if name == "merge_group":
        return event["merge_group"]["base_sha"]
    if name == "push":
        if event.get("before", "").strip("0"):
            return event["before"]
    elif name == "workflow_dispatch":
        if event.get("inputs", {}).get("evidence_base"):
            return event["inputs"]["evidence_base"]
    else:
        raise ValueError("Evidence requires a supported CI event")
    return subprocess.check_output(["git", "rev-parse", candidate + "^1"], cwd=project, text=True).strip()


def comparison(project: Path, candidate: str, base: str, *, forced: bool = False) -> dict:
    """Keep the 0.5 protocol on supported SDKs without relabelling the event base."""
    if any(re.fullmatch(r"[0-9a-f]{40}", revision) is None for revision in (candidate, base)):
        raise ValueError("Evidence comparisons require full candidate and event-base revisions")
    candidate_version = package_version(project, candidate)
    if not supported(candidate_version):
        raise ValueError(f"Evidence candidate {candidate} requires relflow >=0.5.0, found {candidate_version}")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, candidate], cwd=project, capture_output=True, check=False
    )
    if ancestry.returncode:
        if not forced:
            raise ValueError(f"Evidence event base {base} must be an available ancestor of {candidate}")
        baseline, reason = BASELINE, "rewritten-push"
    elif supported(package_version(project, base)):
        baseline, reason = base, "event-base"
    else:
        baseline, reason = BASELINE, "pre-0.5"
    baseline_version = package_version(project, baseline)
    if not supported(baseline_version):
        raise ValueError(f"Evidence baseline {baseline} requires relflow >=0.5.0, found {baseline_version}")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", baseline, candidate], cwd=project, capture_output=True, check=False
    )
    if ancestry.returncode:
        raise ValueError(f"Evidence baseline {baseline} must be an available ancestor of {candidate}")
    return {
        "schema": 1,
        "event_base": base,
        "baseline": baseline,
        "reason": reason,
        "candidate_version": candidate_version,
        "baseline_version": baseline_version,
    }


def package_version(project: Path, revision: str) -> str:
    """Read package metadata as data; never import the selected SDK."""
    content = subprocess.check_output(["git", "show", revision + ":pyproject.toml"], cwd=project, text=True)
    return tomllib.loads(content)["project"]["version"]


def filenames(version: str) -> dict[str, str]:
    if not VERSION.fullmatch(version):
        raise ValueError(f"Evidence assets require a normalized package version, got {version!r}")
    prefix = f"relflow-{version}-evidence"
    return {
        "publication": f"{prefix}.json",
        "reports": f"{prefix}-reports.tar.gz",
        "inputs": f"{prefix}-inputs.tar.gz",
        "provenance": f"{prefix}-provenance.json",
        "plots": f"{prefix}-plots.zip",
        "measurements": f"{prefix}-measurements.md",
        "checksums": f"{prefix}-checksums.txt",
    }


def inventory(version: str, available: Iterable[str]) -> dict[str, str]:
    """Read old immutable releases; partial versioned inventories never fall back."""
    current = filenames(version)
    return current if set(current.values()).intersection(available) else LEGACY
