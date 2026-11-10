"""Durable receipts and atomic completion for a single CLI job."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import yaml

from relflow.data.sources import Source

__all__ = ["ExecutionError", "Job", "digest", "manifest"]


class ExecutionError(RuntimeError):
    """An already-reserved job failed and retained its terminal receipt."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__(report["error"]["message"])


def digest(path: str | Path) -> str:
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def manifest(sources: dict) -> dict[str, Any]:
    result = {}
    for split, selected in sources.items():
        if not isinstance(selected, Source):
            continue
        snapshot = selected.manifest or selected.discover()
        selected.manifest = snapshot
        result[str(split)] = {
            "schema": str(snapshot.schema),
            "files": [{"path": item.path, "size": item.size, "modified": item.modified} for item in snapshot.files],
            "identity": "file metadata; content not hashed",
        }
    return result


@dataclass
class Job:
    """An exclusively reserved directory with one atomic terminal receipt."""

    path: Path
    command: str
    configuration: dict[str, Any]
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def metadata(self) -> Path:
        return self.path / "_metadata" if self.command == "predict" else self.path

    def write(self, name: str, value: Any) -> None:
        path = self.metadata / name
        temporary = path.with_name(path.name + ".partial")
        text = (
            yaml.safe_dump(value, sort_keys=False)
            if name.endswith(".yaml")
            else json.dumps(value, indent=2, allow_nan=False)
        )
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(path)

    def __enter__(self) -> Job:
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata.mkdir(exist_ok=True)
        self.report = {
            "schema_version": 1,
            "command": self.command,
            "status": "running",
            "output": str(self.path),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "versions": {name: version(name) for name in ("relflow", "torch", "lightning", "pyarrow", "polars")},
        }
        self.write("resolved.yaml", self.configuration)
        self.write("report.json", self.report)
        return self

    def __exit__(self, kind, error, traceback) -> Literal[False]:
        if error is not None:
            self.report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
            self.report["finished_at"] = datetime.now(timezone.utc).isoformat()
            self.write("report.json", self.report)
            if isinstance(error, Exception):
                raise ExecutionError(self.report) from error
        return False

    def complete(self, **fields: Any) -> dict[str, Any]:
        self.report.update(fields, status="success", finished_at=datetime.now(timezone.utc).isoformat())
        self.write("report.json", self.report)
        marker = self.path / "_SUCCESS.partial"
        marker.write_text("", encoding="utf-8")
        marker.replace(self.path / "_SUCCESS")
        return self.report
