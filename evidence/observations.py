"""Bounded, checksum-bound proof observations kept outside the learning report."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

LIMIT = 1024**3
MEMBERS = 262144


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def check_name(name: str) -> None:
    if name not in {"record.json", "frames.json", "proof.py", "manifest.json"} and not re.fullmatch(
        r"observations/[a-f0-9]{64}\.arrow", name
    ):
        raise ValueError(f"Proof observation has an unexpected member: {name}")


def package(source: Path, destination: Path) -> dict:
    manifest = json.loads((source / "manifest.json").read_text())
    assets = dict(manifest["artifacts"])
    assets["manifest.json"] = {"sha256": digest(source / "manifest.json")}
    if len(assets) > MEMBERS:
        raise ValueError("Proof observations exceed the bounded member inventory")
    total = 0
    with ZipFile(destination, "x", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, expected in sorted(assets.items()):
            check_name(name)
            path = source / name
            if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
                raise ValueError(f"Proof observation must be a regular contained file: {name}")
            total += path.stat().st_size
            if total > LIMIT:
                raise ValueError("Proof observations exceed the one GiB uncompressed bound")
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise ValueError(f"Proof observation changed before archiving: {name}")
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, payload)
    return {
        "schema": 1,
        "sha256": digest(destination),
        "bytes": destination.stat().st_size,
        "manifest_sha256": assets["manifest.json"]["sha256"],
        "capture_sha256": manifest["capture_sha256"],
    }


def read(path: Path, expected: dict | None = None) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > LIMIT:
        raise ValueError("Proof observation archive must be a regular file below one GiB")
    if expected is not None and (digest(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]):
        raise ValueError("Proof observation archive differs from its execution receipt")
    with ZipFile(path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(members) > MEMBERS or len(set(names)) != len(names) or sum(m.file_size for m in members) > LIMIT:
            raise ValueError("Proof observation archive exceeds its unique bounded inventory")
        for name in names:
            check_name(name)
        if archive.getinfo("manifest.json").file_size > 64 * 1024**2:
            raise ValueError("Proof observation manifest exceeds 64 MiB")
        payload = archive.read("manifest.json")
        if expected is not None and hashlib.sha256(payload).hexdigest() != expected["manifest_sha256"]:
            raise ValueError("Proof observation manifest differs from its execution receipt")
        manifest = json.loads(payload)
        if set(names) != set(manifest["artifacts"]) | {"manifest.json"}:
            raise ValueError("Proof observation archive differs from its manifest inventory")
        for name in ["record.json", "proof.py", "frames.json"]:
            if archive.getinfo(name).file_size > 64 * 1024**2:
                raise ValueError(f"Proof observation metadata exceeds 64 MiB: {name}")
            data = archive.read(name)
            if len(data) > 64 * 1024**2 or hashlib.sha256(data).hexdigest() != manifest["artifacts"][name]["sha256"]:
                raise ValueError(f"Proof observation metadata differs from its manifest: {name}")
        record = json.loads(archive.read("record.json"))
        if hashlib.sha256(archive.read("proof.py")).hexdigest() != manifest["source_sha256"]:
            raise ValueError("Proof observation source differs from its manifest")
    return {"manifest": manifest, "record": record, "manifest_sha256": hashlib.sha256(payload).hexdigest()}


def unpack(path: Path, destination: Path, expected: dict | None = None) -> dict:
    value = read(path, expected)
    destination.mkdir(parents=True, exist_ok=False)
    with ZipFile(path) as archive:
        for member in archive.infolist():
            target = destination / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = archive.read(member)
            if (
                member.filename != "manifest.json"
                and hashlib.sha256(payload).hexdigest() != value["manifest"]["artifacts"][member.filename]["sha256"]
            ):
                raise ValueError(f"Proof observation content changed: {member.filename}")
            target.write_bytes(payload)
    return value
