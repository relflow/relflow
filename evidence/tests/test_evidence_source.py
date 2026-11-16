"""Literate evidence metadata is static, and prose does not change experiment identity."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def source() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "evidence/experiments/source.py"
    spec = importlib.util.spec_from_file_location("evidence_source", path)
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


def test_inline_dependencies_are_static_and_do_not_change_python_identity(source):
    original = (
        '# /// script\n# requires-python = ">=3.12"\n'
        '# dependencies = ["relflow[torchjd]", "numpy>=2"]\n# ///\n'
        "raise AssertionError('dependency discovery must not execute this script')\n"
    )
    changed = original.replace("numpy>=2", "numpy>=3")
    assert source.environment(original) == {
        "requires-python": ">=3.12",
        "dependencies": ["relflow[torchjd]", "numpy>=2"],
    }
    assert source.fingerprint(original) == source.fingerprint(changed)
    assert hashlib.sha256(original.encode()).hexdigest() != hashlib.sha256(changed.encode()).hexdigest()
    assert source.environment("# Historical suite without inline dependencies\n") is None


def test_tags_belong_to_each_script_and_do_not_require_imports(source):
    script = (
        "# /// script\n# dependencies = []\n# [tool.relflow.evidence]\n"
        '# capability = "structure"\n# types = ["Hash"]\n# conditions = ["unseen-identities"]\n# ///\n'
        "raise AssertionError('tag discovery must not execute a model')\n"
    )
    assert source.tags(script) == {"capability": "structure", "types": ["Hash"], "conditions": ["unseen-identities"]}
    assert source.tags("# /// script\n# dependencies = []\n# ///\n") is None
    for changed in (
        script.replace('types = ["Hash"]', 'types = ["Hash", "Hash"]'),
        script.replace('conditions = ["unseen-identities"]', "conditions = []"),
        script.replace('capability = "structure"', 'capability = ["structure"]'),
        script.replace('types = ["Hash"]', 'extra = ["Hash"]'),
    ):
        with pytest.raises(ValueError, match="Evidence"):
            source.tags(changed)


@pytest.mark.parametrize(
    "block",
    [
        "# /// script\n# dependencies = []\n# ///\n# /// script\n# dependencies = []\n# ///\n",
        "# /// script\n# dependencies = []\n",
        "# /// script\ndependencies = []\n# ///\n",
        "# /// script\n# dependencies = [\n# ///\n",
        '# /// script\n# requires-python = ">=3.12"\n# ///\n',
        '# /// script\n# dependencies = "relflow"\n# ///\n',
        "# /// script\n# dependencies = [3]\n# ///\n",
        '# /// script\n# dependencies = [" "]\n# ///\n',
        "# /// script\n# dependencies = []\n# requires-python = 3.12\n# ///\n",
        '# /// script\n# dependencies = []\n# tool = "uv"\n# ///\n',
    ],
)
def test_invalid_inline_environments_fail_closed(source, block):
    with pytest.raises(ValueError, match="PEP 723"):
        source.environment(block)


def test_metadata_preserves_nested_yaml(source: ModuleType) -> None:
    script = (
        "# %% [markdown]\n# ---\n# title: Weighted sums\n"
        "# categories: [Aggregation]\n# execute:\n#   enabled: false\n#   echo: true\n"
        "# description: |\n#   First line.\n#   Second line.\n# ---\n# Explanation.\n"
        "# %%\nfrom reporting import experiment, report\n"
        "@experiment('P014', seed=14)\n"
        "def run(seed, steps, accelerator):\n    raise RuntimeError('must not execute')\n"
    )
    assert source.metadata(script) == {
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
        "# %% [markdown]\n# ---\n# title: Signal\n# ---\n# %%\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "raise RuntimeError('must not execute')\n"
        "from reporting import experiment, report\n"
        "@experiment('P001', seed=1)\n"
        "def run(seed, steps, accelerator):\n    return {}, {'signal': True}\n"
    )
    assert source.metadata(script) == {"title": "Signal"}
    assert len(source.fingerprint(script)) == 64
    assert not marker.exists()
    unsafe_yaml = "# %% [markdown]\n# ---\n# value: !!python/object/apply:builtins.eval ['1 + 1']\n# ---\n"
    with pytest.raises(ValueError, match="YAML header is invalid"):
        source.metadata(unsafe_yaml)


def test_frontmatter_updates_preserve_code_and_nested_results(source):
    body = (
        "# %%\nfrom reporting import experiment, report\n"
        "@experiment('P001', seed=1)\n"
        "def run(seed, steps, accelerator):\n    raise RuntimeError('not executed')\n"
    )
    script = "# %% [markdown]\n# ---\n# title: Signal\n# ---\n" + body
    header = source.metadata(script)
    header["description"] = "First line\n---\nLast line"
    header["evidence-result"] = {"seed": 42, "checks": {"signal": True}, "metrics": {"error": 0.125}}
    updated = source.frontmatter(script, header)
    assert source.metadata(updated) == header
    assert source.frontmatter(updated, header) == updated
    assert source.fingerprint(updated) == source.fingerprint(script)
    assert updated.endswith(body)


def test_discovery_rejects_duplicate_ids_instead_of_losing_an_experiment(source, tmp_path):
    import pytest

    family = tmp_path / "experiments/family"
    family.mkdir(parents=True)
    text = (
        "# %% [markdown]\n# ---\n# title: Signal\n# ---\n# %%\n"
        "from reporting import experiment, report\n"
        "@experiment('P001', seed=1)\n"
        "def run(seed, steps, accelerator):\n    raise RuntimeError('not executed')\n"
    )
    (family / "first.py").write_text(text)
    (family / "second.py").write_text(text)
    with pytest.raises(ValueError, match="duplicate evidence ID"):
        source.registry(tmp_path)
