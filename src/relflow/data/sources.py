"""Lazy file selection and Arrow readers, independent of model computation."""

from __future__ import annotations

import fnmatch
import os
import posixpath
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path, PurePosixPath
from typing import Any, TypeAlias
from urllib.parse import unquote

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as fs

Location: TypeAlias = str | os.PathLike[str]
Filesystem: TypeAlias = str | Callable[[], fs.FileSystem] | None

FORMATS = {
    ".parquet": "parquet",
    ".csv": "csv",
    ".jsonl": "json",
    ".ndjson": "json",
    ".arrow": "ipc",
    ".ipc": "ipc",
    ".feather": "ipc",
    ".orc": "orc",
}


def matches(path: str, pattern: str) -> bool:
    """Match path segments, with ** spanning zero or more directories."""

    parts = path.split("/")
    positions = {0}
    for segment in pattern.split("/"):
        if not positions:
            return False
        if segment == "**":
            positions = set(range(min(positions), len(parts) + 1))
        else:
            positions = {
                index + 1 for index in positions if index < len(parts) and fnmatch.fnmatchcase(parts[index], segment)
            }
    return len(parts) in positions


def resolve(location: str, filesystem: fs.FileSystem | None) -> tuple[fs.FileSystem, str]:
    """Resolve a URI without interpreting path glob characters as URI syntax."""

    if filesystem is not None:
        if "://" in location:
            raise ValueError("use filesystem-relative source paths when filesystem is explicitly configured")
        return filesystem, filesystem.normalize_path(location)
    if "://" not in location:
        filesystem = fs.LocalFileSystem()
        return filesystem, filesystem.normalize_path(location)
    scheme, remainder = location.split("://", 1)
    authority, _, key = remainder.partition("/")
    filesystem, base = fs.FileSystem.from_uri(f"{scheme}://{authority}/")
    path = posixpath.join(base, unquote(key))
    return filesystem, filesystem.normalize_path(path)


@dataclass(frozen=True, slots=True)
class File:
    """Available file identity metadata for a frozen source selection."""

    path: str
    size: int | None
    modified: int | None


@dataclass(frozen=True, slots=True)
class Manifest:
    """Serializable file selection and schema shared by source consumers."""

    files: tuple[File, ...]
    format: ds.FileFormat
    schema: pa.Schema
    partition_base_dir: str


@dataclass(slots=True)
class Source:
    """A restartable file source whose live Arrow resources stay process-local."""

    locations: tuple[str, ...]
    explicit: bool
    format: str | ds.FileFormat | None
    match: re.Pattern[str] | None
    schema: pa.Schema | None
    filesystem: Filesystem
    partitioning: Any
    partition_base_dir: str | None
    context: str = "file"
    manifest: Manifest | None = field(default=None, init=False, repr=False)
    dataset: ds.Dataset | None = field(default=None, init=False, repr=False)
    process: int | None = field(default=None, init=False, repr=False)

    @property
    def label(self) -> str:
        return f"{self.context} source {self.locations!r}"

    def __getstate__(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self) if item.init or item.name == "manifest"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            setattr(self, name, value)
        self.dataset = None
        self.process = None

    def connect(self) -> tuple[fs.FileSystem, tuple[str, ...]]:
        """Open the configured filesystem and resolve all paths in its namespace."""

        filesystem = None
        if isinstance(self.filesystem, str):
            filesystem, base = fs.FileSystem.from_uri(self.filesystem)
            if base and base != "/":
                filesystem = fs.SubTreeFileSystem(base, filesystem)
        elif self.filesystem is not None:
            filesystem = self.filesystem()
            if not isinstance(filesystem, fs.FileSystem):
                raise TypeError(f"{self.label}: filesystem factory must return a pyarrow FileSystem")

        owner = None
        paths = []
        for location in self.locations:
            current, path = resolve(location, filesystem)
            if owner is not None and not current.equals(owner):
                raise ValueError(f"{self.label}: all files must use the same filesystem")
            owner = current
            paths.append(path)
        assert owner is not None
        return owner, tuple(paths)

    def select(self) -> tuple[fs.FileSystem, tuple[File, ...], str]:
        """Deterministically select existing files within one filesystem."""

        filesystem, paths = self.connect()
        selected: dict[str, File] = {}
        roots = []
        for path in paths:
            magic = re.search(r"[*?\[]", path) if not self.explicit else None
            if magic:
                root = posixpath.dirname(path[: magic.start()]) or "."
                candidates = filesystem.get_file_info(fs.FileSelector(root, recursive=True))
            else:
                info = filesystem.get_file_info(path)
                if info.type == fs.FileType.Directory and not self.explicit:
                    root = path
                    candidates = filesystem.get_file_info(fs.FileSelector(path, recursive=True))
                elif info.type == fs.FileType.File:
                    root = posixpath.dirname(path)
                    candidates = [info]
                else:
                    raise FileNotFoundError(
                        f"{self.label}: expected an existing {'file' if self.explicit else 'file or directory'} at {path!r}"
                    )
            roots.append(root)
            for info in candidates:
                if info.type != fs.FileType.File:
                    continue
                relative = posixpath.relpath(info.path, root)
                if any(part.startswith((".", "_")) for part in PurePosixPath(relative).parts):
                    continue
                normalized = posixpath.normpath(info.path)
                if magic and not matches(normalized, posixpath.normpath(path)):
                    continue
                if self.match is not None and not self.match.fullmatch(PurePosixPath(info.path).name):
                    continue
                selected[normalized] = File(normalized, info.size, info.mtime_ns)
        if not selected:
            raise FileNotFoundError(f"{self.label}: no files selected; check the path, glob, and match pattern")
        base = self.partition_base_dir if self.partition_base_dir is not None else posixpath.commonpath(roots)
        return filesystem, tuple(selected[path] for path in sorted(selected)), base

    def discover(self) -> Manifest:
        """Freeze selection and validate each file before sharing its scan schema."""

        try:
            filesystem, files, base = self.select()
            format = self.format
            if format is None:
                formats = {FORMATS.get(PurePosixPath(item.path).suffix.lower()) for item in files}
                if None in formats or len(formats) != 1:
                    raise ValueError(
                        f"{self.label}: mixed or unknown file formats; narrow the selection or pass format explicitly"
                    )
                format = formats.pop()
            dataset = ds.dataset(
                [item.path for item in files],
                filesystem=filesystem,
                format=format,
                schema=self.schema,
                partitioning=self.partitioning,
                partition_base_dir=base,
                exclude_invalid_files=False,
            )
            physical = None
            for fragment in dataset.get_fragments():
                actual = fragment.physical_schema
                if self.schema is None:
                    if physical is not None and not physical.equals(actual, check_metadata=True):
                        raise TypeError(
                            f"{self.label}: schema changed in {fragment.path!r}; expected {physical}, got {actual}; supply a compatible schema explicitly"
                        )
                    physical = actual
                else:
                    partition_names = ds.get_partition_keys(fragment.partition_expression)
                    missing = set(self.schema.names) - set(actual.names) - set(partition_names)
                    if missing:
                        raise TypeError(f"{self.label}: {fragment.path!r} is missing declared fields {sorted(missing)}")
            return Manifest(files, dataset.format, dataset.schema, base)
        except (pa.ArrowException, OSError) as error:
            raise type(error)(f"{self.label}: {error}") from error

    def __call__(self) -> Iterator[pa.RecordBatch]:
        if self.manifest is None:
            self.manifest = self.discover()
        manifest = self.manifest
        # A source may have been used before a fork; never reuse its native state.
        if self.process != os.getpid():
            self.dataset = None
        try:
            if self.dataset is None:
                filesystem, _ = self.connect()
                self.dataset = ds.dataset(
                    [item.path for item in manifest.files],
                    filesystem=filesystem,
                    format=manifest.format,
                    schema=manifest.schema,
                    partitioning=self.partitioning,
                    partition_base_dir=manifest.partition_base_dir,
                    exclude_invalid_files=False,
                )
                self.process = os.getpid()
            current = self.dataset.filesystem.get_file_info([item.path for item in manifest.files])
            for expected, actual in zip(manifest.files, current, strict=True):
                if actual.type != fs.FileType.File or (actual.size, actual.mtime_ns) != (
                    expected.size,
                    expected.modified,
                ):
                    raise RuntimeError(
                        f"{self.label}: selected file {expected.path!r} changed; create a new source for the new snapshot"
                    )
            emitted = False
            with self.dataset.scanner().to_reader() as reader:
                for batch in reader:
                    emitted = True
                    yield batch
            if not emitted:
                yield pa.RecordBatch.from_arrays(
                    [pa.array([], type=item.type) for item in manifest.schema], schema=manifest.schema
                )
        except (pa.ArrowException, OSError) as error:
            raise type(error)(f"{self.label}: {error}") from error


def source(
    location: Location | Sequence[Location],
    *,
    format: str | ds.FileFormat | None = None,
    match: str | re.Pattern[str] | None = None,
    schema: pa.Schema | None = None,
    filesystem: Filesystem = None,
    partitioning: Any = None,
    partition_base_dir: Location | None = None,
) -> Source:
    """Describe files, a directory, or a glob without opening or listing data.

    Infer Parquet, CSV, JSON Lines, IPC/Feather v2, and ORC from file suffixes,
    or supply an Arrow FileFormat with parsing options. ``match`` full-matches
    each basename. With an explicit filesystem, use filesystem-relative paths.
    Factories must be importable when using spawned data workers.
    """

    explicit = not isinstance(location, (str, os.PathLike))
    if explicit and (not isinstance(location, Sequence) or not location):
        raise TypeError("source location must be a path or a nonempty sequence of file paths")
    locations = tuple(location) if explicit else (location,)
    normalized = []
    for item in locations:
        if not isinstance(item, (str, os.PathLike)) or not os.fspath(item):
            raise TypeError(f"source location must contain nonempty paths; got {item!r}")
        path = os.fspath(item)
        if not isinstance(path, str):
            raise TypeError("source paths must be strings, not bytes")
        if "://" not in path and filesystem is None:
            path = os.path.abspath(Path(path).expanduser())
        normalized.append(path)
    if format is not None and not isinstance(format, (str, ds.FileFormat)):
        raise TypeError("source format must be an Arrow format name or FileFormat")
    if match is not None and not isinstance(match, (str, re.Pattern)):
        raise TypeError("source match must be a string or compiled regular expression")
    pattern = re.compile(match) if match is not None else None
    if pattern is not None and not isinstance(pattern.pattern, str):
        raise TypeError("source match must be a text pattern, not bytes")
    if schema is not None and not isinstance(schema, pa.Schema):
        raise TypeError("source schema must be a pyarrow Schema")
    if filesystem is not None and not isinstance(filesystem, str) and not callable(filesystem):
        raise TypeError("source filesystem must be a URI or a factory returning a pyarrow FileSystem")
    base = None if partition_base_dir is None else os.fspath(partition_base_dir)
    if base is not None and filesystem is None and all("://" not in item for item in normalized):
        base = os.path.abspath(Path(base).expanduser())
    return Source(tuple(normalized), explicit, format, pattern, schema, filesystem, partitioning, base)


__all__ = ["source"]
