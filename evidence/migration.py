"""Verify the reviewed Cluster-to-Group baseline source inventory.

This verifier does not change Evidence planning or authorize an unpaired result.
The trusted executor must integrate and pin the migration policy separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BASELINE = "c0cae6df38010e232c0f12c966eaa6a06fae4b94"
MANIFEST = "migrations/cluster_to_group/manifest.json"
MANIFEST_SHA256 = "deb99890b6eb0395f45e287556d8985158c22b326a7ba881d8a186f1314c8db7"
IDENTIFIERS = frozenset({"P018", "P019", "P020", "P065", "P078", "P095"})


def regular(root: Path, relative: str) -> Path:
    """Reject aliases, traversal and files outside the supplied Evidence root."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Migration Evidence root must be a regular directory: {root}")
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"Migration input must be a relative Evidence path: {relative!r}")
    resolved_root = root.resolve(strict=True)
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Migration input cannot contain a symbolic link: {relative}")
    if not current.is_file() or not current.resolve().is_relative_to(resolved_root):
        raise ValueError(f"Migration input must be a regular contained file: {relative}")
    return current


def verify_sources(root: Path, baseline_revision: str) -> dict:
    """Verify the approved baseline and each byte-identical retired proof."""
    if baseline_revision != BASELINE:
        raise ValueError(f"Migration baseline must be {BASELINE}, got {baseline_revision}")
    manifest_path = regular(root, MANIFEST)
    payload = manifest_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Migration manifest differs from the reviewed executor policy")
    manifest = json.loads(payload)
    if manifest.get("schema") != 1 or manifest.get("baseline_revision") != BASELINE:
        raise ValueError("Migration manifest has an invalid schema or baseline")
    entries = manifest.get("proofs")
    if not isinstance(entries, list) or len(entries) != len(IDENTIFIERS):
        raise ValueError("Migration must include all six retired Cluster proofs")
    identifiers = [entry.get("id") for entry in entries]
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != IDENTIFIERS:
        raise ValueError("Migration has missing, duplicate or unexpected Cluster proof IDs")
    for entry in entries:
        path = regular(root, entry["archive"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Migration proof {entry['id']} differs from the reviewed baseline source")
    return manifest


def transition(root: Path, baseline_revision: str, catalog: dict) -> list[dict]:
    """Bind retired sources to distinct registered replacement IDs, without comparing them."""
    manifest = verify_sources(root, baseline_revision)
    old = {entry["id"] for entry in manifest["proofs"]}
    if old & catalog.keys():
        raise ValueError("Retired Cluster proof IDs cannot be reused by the candidate catalog")
    replacements = [entry["proposed_replacement"] for entry in manifest["proofs"] if entry["proposed_replacement"]]
    if len(replacements) != len(set(replacements)) or set(replacements) & old:
        raise ValueError("Migration replacements must have unique new proof IDs")
    for identifier in replacements:
        entry = catalog.get(identifier)
        if not isinstance(entry, dict) or entry.get("lane") != "core":
            raise ValueError(f"Migration replacement {identifier} must be registered as a core experiment")
    return manifest["proofs"]


def compare_checks(entries: list[dict], baseline: dict, candidate: dict) -> dict:
    """Classify verified checks separately; callers must verify execution provenance first."""
    old = {entry["id"] for entry in entries}
    new = {entry["proposed_replacement"] for entry in entries if entry["proposed_replacement"]}
    if old & new:
        raise ValueError("Migration cannot reuse a retired ID for a replacement")
    if set(baseline) != old or set(candidate) != new:
        raise ValueError("Migration check comparison requires every retired and replacement result exactly once")
    if len(old) != len(entries) or len(new) != sum(bool(entry["proposed_replacement"]) for entry in entries):
        raise ValueError("Migration check comparison contains duplicate proof IDs")
    unmet = {"baseline_unmet": [], "candidate_unmet": []}
    for role, records in (("baseline", baseline), ("candidate", candidate)):
        for identifier, result in sorted(records.items()):
            checks = result.get("checks") if isinstance(result, dict) else None
            if (
                not isinstance(checks, dict)
                or not checks
                or any(not isinstance(name, str) or not name.strip() for name in checks)
                or any(type(value) is not bool for value in checks.values())
            ):
                raise ValueError(f"Migration {role} proof {identifier} has invalid checks")
            expected_outcome = "met" if all(checks.values()) else "not_met"
            if result.get("outcome") != expected_outcome:
                raise ValueError(f"Migration {role} proof {identifier} has an inconsistent outcome")
            unmet[f"{role}_unmet"].extend(
                {"id": identifier, "check": name} for name, passed in checks.items() if not passed
            )
    return unmet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    manifest = verify_sources(args.root, args.baseline)
    print(f"Verified {len(manifest['proofs'])} Cluster sources for baseline {args.baseline}")


if __name__ == "__main__":
    main()
