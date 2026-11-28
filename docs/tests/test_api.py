from __future__ import annotations

import sys
from pathlib import Path

import pytest

from docs.api import extract


@pytest.fixture
def package(tmp_path: Path) -> Path:
    source = tmp_path / "src/relflow"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text(
        '''"""A source-only package."""
from .types import Derived as Number, Root as Model
from .types import Alias as RequestBase
from .types import operation as preprocess
from missing_runtime import External
raise RuntimeError("the documentation must never execute this package")
'''
    )
    (source / "types.py").write_text(
        '''from typing import Annotated, ClassVar, TypeAlias, overload

class Leaf:
    """Shared field documentation."""
    active: bool = True
    def normalization(self, value: float) -> float:
        """Normalize an observation."""
        return value

Alias: TypeAlias = Leaf

class Numeric(Alias):
    """Numbers retain their source units."""
    offset: int = 4
    jitter: object = Field(default_factory=ExplosiveFactory)

class Derived(Numeric):
    offset: int = 8
    active: ClassVar[None] = None
    kind: ClassVar[str] = "derived"
    @property
    def logarithmic(self) -> bool:
        """Whether log scaling is active."""
        return False

class Root:
    @overload
    def __init__(self, value: str) -> None: ...
    @overload
    def __init__(self, value: int) -> None: ...
    def __init__(self, value: str | int) -> None: ...

def operation(source: str, /, *paths: str, limit: int = 8, **options: object) -> list[str]:
    """Read the requested paths.

    Args:
        source: The source to read.
        limit: Maximum number of rows.

    Returns:
        The selected paths.

    Raises:
        ValueError: When a path is invalid.
    """
    raise RuntimeError("never execute reference examples")
'''
    )
    return tmp_path


def test_reference_resolves_reexports_and_typealias_bases_without_importing_sdk(package: Path):
    number = extract(package, ["rf.Number"])[0]
    fields = {field["name"]: field for field in number["fields"]}
    assert number["text"] == "Numbers retain their source units."
    assert number["source"]["file"] == "src/relflow/types.py"
    assert number["documentationSource"]["line"] < number["source"]["line"]
    assert "active" not in fields
    assert fields["kind"]["default"] == "'derived'"
    assert fields["offset"]["default"] == "8"
    assert fields["offset"]["inherited"] is False
    assert fields["jitter"]["default"] == "ExplosiveFactory()"
    assert fields["jitter"]["factory"] is True
    assert fields["logarithmic"]["property"] is True
    assert number["methods"][0]["name"] == "rf.Number.normalization"
    assert "relflow" not in sys.modules
    assert "missing_runtime" not in sys.modules


def test_signature_keeps_parameter_kinds_defaults_and_source_documentation(package: Path):
    entry = extract(package, ["rf.preprocess"])[0]
    signature = entry["signatures"][0]
    assert "source: str, /, *paths: str, limit: int = 8, **options: object" in signature["code"]
    parameters = {parameter["name"]: parameter for parameter in signature["parameters"]}
    assert parameters["source"]["kind"] == "positional-only"
    assert parameters["source"]["default"] is None
    assert parameters["limit"]["default"] == "8"
    assert parameters["limit"]["description"] == "Maximum number of rows."
    assert signature["returns"] == [{"type": "list[str]", "description": "The selected paths."}]
    assert signature["raises"] == [{"type": "ValueError", "description": "When a path is invalid."}]


def test_overloads_and_external_reexports_do_not_fabricate_runtime_signatures(package: Path):
    model, external = extract(package, ["rf.Model", "rf.External"])
    assert len(model["overloads"]) == 2
    assert model["signatures"][0]["code"] == "rf.Model(value: str | int)"
    assert external["external"] == "missing_runtime.External"
    assert external["signatures"] == []
    assert external["source"]["file"] == "src/relflow/__init__.py"
    assert external["definition"] == "from missing_runtime import External"


def test_source_changes_refresh_reference_without_an_authored_snapshot(package: Path):
    source = package / "src/relflow/types.py"
    before = extract(package, ["rf.Number"])[0]
    source.write_text(
        source.read_text().replace("source units", "original units").replace("offset: int = 8", "offset: int = 16")
    )
    after = extract(package, ["rf.Number"])[0]
    assert before["text"] != after["text"]
    assert after["text"] == "Numbers retain their original units."
    assert next(field for field in after["fields"] if field["name"] == "offset")["default"] == "16"


def test_missing_local_symbol_fails_instead_of_publishing_stale_content(package: Path):
    with pytest.raises(KeyError):
        extract(package, ["rf.Removed"])
