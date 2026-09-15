"""Literate proof metadata is static, and prose does not change experiment identity."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def source() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "proofs/source.py"
    spec = importlib.util.spec_from_file_location("proof_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fingerprint_ignores_documentation_and_formatting(source: ModuleType) -> None:
    original = '"""An experiment."""\nvalue = sum([1, 2])\n'
    documented = (
        "# %% [markdown]\n# ---\n# title: New explanation\n# ---\n"
        "# A rewritten insight, with different Markdown.\n\n# %%\n"
        '\n"""An experiment."""\n\nvalue = sum(\n    [1, 2],\n)  # More explanation.\n'
    )
    expected = hashlib.sha256(ast.dump(ast.parse(original), include_attributes=False).encode()).hexdigest()
    assert source.fingerprint(original) == source.fingerprint(documented) == expected


@pytest.mark.parametrize(
    "changed",
    [
        '"""An experiment."""\nvalue = sum([1, 3])\n',
        '"""An experiment."""\nvalue = max([1, 2])\n',
        '"""A different runtime docstring."""\nvalue = sum([1, 2])\n',
    ],
)
def test_fingerprint_detects_python_changes(source: ModuleType, changed: str) -> None:
    assert source.fingerprint('"""An experiment."""\nvalue = sum([1, 2])\n') != source.fingerprint(changed)


def test_metadata_preserves_nested_yaml(source: ModuleType) -> None:
    script = (
        "# %% [markdown]\n# ---\n# proof-id: P014\n# title: Weighted sums\n"
        "# categories: [Aggregation]\n# execute:\n#   enabled: false\n#   echo: true\n"
        "# description: |\n#   First line.\n#   Second line.\n# ---\n# Explanation.\n"
        "# %%\nraise RuntimeError('must not execute')\n"
    )
    assert source.metadata(script) == {
        "proof-id": "P014",
        "title": "Weighted sums",
        "categories": ["Aggregation"],
        "execute": {"enabled": False, "echo": True},
        "description": "First line.\nSecond line.",
    }


@pytest.mark.parametrize(
    ("script", "message"),
    [
        ("print('no header')", "must begin with a '# %% \\[markdown\\]' cell"),
        ("# %% [markdown]\n# title: Missing delimiter\n", "must begin with a commented '# ---'"),
        ("# %% [markdown]\n# ---\n# title: Incomplete\n", "must end with a commented '# ---'"),
        ("# %% [markdown]\n# ---\ntitle = 'Not a comment'\n", "line 3 must be a Python comment"),
        ("# %% [markdown]\n# ---\n# title: [\n# ---\n", "YAML header is invalid"),
        ("# %% [markdown]\n# ---\n# - item\n# ---\n", "must be a mapping with string keys"),
        ("# %% [markdown]\n# ---\n# 1: item\n# ---\n", "must be a mapping with string keys"),
    ],
)
def test_metadata_rejects_malformed_header(source: ModuleType, script: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        source.metadata(script)


def test_reading_source_never_executes_it(source: ModuleType, tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    script = (
        "# %% [markdown]\n# ---\n# proof-id: P001\n# ---\n# %%\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "raise RuntimeError('must not execute')\n"
    )
    assert source.metadata(script) == {"proof-id": "P001"}
    assert len(source.fingerprint(script)) == 64
    assert not marker.exists()
    unsafe_yaml = "# %% [markdown]\n# ---\n# value: !!python/object/apply:builtins.eval ['1 + 1']\n# ---\n"
    with pytest.raises(ValueError, match="YAML header is invalid"):
        source.metadata(unsafe_yaml)
