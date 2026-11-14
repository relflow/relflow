"""Durable receipts and atomic completion for a single CLI job."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import yaml

from relflow.data.sources import Source
from relflow.structs.enums import Strata

__all__ = ["Job", "digest", "snapshot"]


def digest(path: str | Path) -> str:
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def snapshot(sources: Mapping[Strata, Source]) -> dict[str, Any]:
    """Freeze the files used by each split and describe that exact selection."""
    result = {}
    for split, selected in sources.items():
        if selected.manifest is None:
            selected.manifest = selected.discover()
        manifest = selected.manifest
        result[str(split)] = {
            "schema": str(manifest.schema),
            "files": [{"path": item.path, "size": item.size, "modified": item.modified} for item in manifest.files],
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
        self.report.update(
            schema_version=1,
            command=self.command,
            status="running",
            output=str(self.path),
            started_at=datetime.now(timezone.utc).isoformat(),
            versions={name: version(name) for name in ("relflow", "torch", "lightning", "pyarrow", "polars")},
        )
        self.write("resolved.yaml", self.configuration)
        self.write("report.json", self.report)
        return self

    def __exit__(self, kind, error, traceback) -> Literal[False]:
        if error is not None:
            self.report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
            self.report["finished_at"] = datetime.now(timezone.utc).isoformat()
            self.write("report.json", self.report)
        return False

    def complete(self, **fields: Any) -> dict[str, Any]:
        self.report.update(fields, status="success", finished_at=datetime.now(timezone.utc).isoformat())
        self.write("report.json", self.report)
        marker = self.path / "_SUCCESS.partial"
        marker.write_text("", encoding="utf-8")
        marker.replace(self.path / "_SUCCESS")
        return self.report
