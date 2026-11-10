"""Static plot bundle validation shared by CI publication and documentation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from zipfile import ZipFile

LIMIT = 128 * 1024**2


def validate(review: dict, asset: Callable[[str], bytes]) -> dict:
    if review.get("schema") != 1 or not isinstance(review.get("entries"), dict):
        raise ValueError("Evidence review must declare schema 1 and per-proof entries")
    if review.get("errors") or set(review["entries"]) != set(review["expected"]):
        raise ValueError("Evidence review must contain its complete selection without rendering errors")
    for identifier, entry in review["entries"].items():
        if not re.fullmatch(r"P\d{3}", identifier):
            raise ValueError("Evidence review must use canonical proof identifiers")
        needed = {entry["data"]}
        needed.update(f"{figure['name']}-{theme}.svg" for figure in entry["figures"] for theme in ["light", "dark"])
        if not needed <= entry["assets"].keys():
            raise ValueError(f"{identifier}: figures and example data must be checksum-bound assets")
        for name, expected in entry["assets"].items():
            if len(Path(name).parts) != 1 or Path(name).is_absolute() or not name.startswith(identifier):
                raise ValueError(f"{identifier}: review asset must be a contained proof file: {name}")
            if hashlib.sha256(asset(name)).hexdigest() != expected:
                raise ValueError(f"{identifier}: review asset changed: {name}; regenerate its figures")
        data = json.loads(asset(entry["data"]))
        if data["id"] != identifier or data["figures"] != entry["figures"]:
            raise ValueError(f"{identifier}: figure data belongs to a different proof or inventory")
        if data.get("record") is not None:
            name = f"{identifier}-record.json"
            if (
                name not in entry["assets"]
                or hashlib.sha256(asset(name)).hexdigest() != entry["record_sha256"]
                or json.loads(asset(name)) != data["record"]
            ):
                raise ValueError(f"{identifier}: figure data must retain the exact captured record")
    return review


def read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > LIMIT:
        raise ValueError("Evidence plot bundle must be a regular file below 128 MiB")
    with ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) > 1000 or len(set(names)) != len(names) or sum(entry.file_size for entry in entries) > LIMIT:
            raise ValueError("Evidence plot bundle exceeds its bounded unique inventory")
        if any(len(Path(name).parts) != 1 or Path(name).is_absolute() for name in names):
            raise ValueError("Evidence plot bundle must contain flat relative file names")
        review = validate(json.loads(archive.read("review.json")), archive.read)
        expected = {"review.json"} | {name for entry in review["entries"].values() for name in entry["assets"]}
        if set(names) != expected:
            raise ValueError("Evidence plot bundle differs from its reviewed inventory")
    return review


def install(path: Path, target: Path) -> dict:
    review = read(path)
    target.mkdir(parents=True, exist_ok=True)
    with ZipFile(path) as archive:
        for entry in archive.infolist():
            destination = target / entry.filename
            if destination.is_symlink():
                raise ValueError("Evidence plot installation cannot replace a symbolic link")
            destination.write_bytes(archive.read(entry))
    return review


def verify(path: Path, manifest: dict, records: list[dict]) -> dict:
    """Require every measured figure to use its exact paired candidate record."""
    review = read(path)
    ci = review.get("ci") or {}
    if (
        ci.get("run_id") != manifest["run_id"]
        or ci.get("candidate_revision") != manifest["candidate"]["revision"]
        or ci.get("baseline_revision") != manifest["baseline"]["revision"]
        or ci.get("evidence_revision") != manifest["evidence_sha"]
        or ci.get("executor_sha") != manifest["executor_sha"]
        or ci.get("capture_sha256") != manifest["observations"]["capture_sha256"]
    ):
        raise ValueError("Evidence plots belong to a different paired execution")
    candidate_only = {
        entry["proposed_replacement"] for entry in manifest.get("migration", []) if entry["proposed_replacement"]
    }
    if set(ci.get("candidate_only", [])) != candidate_only:
        raise ValueError("Evidence plot migration scope differs from the measured task plan")
    candidates = {r["task"]["id"]: r for r in records if r["task"]["role"] == "candidate"}
    with ZipFile(path) as archive:
        measured = set()
        for identifier, entry in review["entries"].items():
            data = json.loads(archive.read(entry["data"]))
            if data.get("record") is None:
                if identifier in candidates:
                    raise ValueError(f"{identifier}: measured candidate was replaced by an illustration")
                continue
            measured.add(identifier)
            record = candidates.get(identifier)
            if (
                record is None
                or data["record"] != record["result"]
                or entry["source_sha256"] != record["task"]["script_sha256"]
            ):
                raise ValueError(f"{identifier}: plots differ from the canonical candidate measurement or source")
            scope = (
                "candidate-only Evidence CI migration"
                if identifier in candidate_only
                else "paired Evidence CI candidate"
            )
            if data.get("scope") != scope:
                raise ValueError(f"{identifier}: plots must identify their measured candidate scope")
    if measured != candidates.keys():
        raise ValueError("Evidence plots omit or duplicate candidate measurements")
    return review
