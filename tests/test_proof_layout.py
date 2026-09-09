"""Structural contracts for the separately collected modeling-proof suite."""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[1] / "proofs"
NUMBERED_PREFIX = re.compile(r"^(?:test_)?p\d{2}(?:_|\.|$)", re.IGNORECASE)


def proof_directories() -> list[Path]:
    return sorted({path.parent for path in ROOT.glob("*/*/test_*.py")})


def module_docstring(path: Path) -> str:
    return ast.get_docstring(ast.parse(path.read_text())) or ""


def top_level_test_functions(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    ]


def test_proofs_use_semantic_directories_without_sidecar_work_documents() -> None:
    work_documents = sorted(ROOT.rglob("WORK.md"))
    numbered_parts = sorted(
        path
        for path in ROOT.rglob("*")
        if not any(part == "__pycache__" for part in path.parts) and NUMBERED_PREFIX.match(path.name)
    )

    assert not work_documents, f"move proof status into Python docstrings: {work_documents}"
    assert not numbered_parts, f"replace numbered proof names with semantic names: {numbered_parts}"


def test_each_proof_variant_has_one_documented_test_module() -> None:
    directories = proof_directories()
    assert directories, "no proof directories were discovered"

    basenames = Counter(path.name for directory in directories for path in directory.glob("test_*.py"))
    duplicates = sorted(name for name, count in basenames.items() if count > 1)
    assert not duplicates, f"proof test module names must be globally unique for pytest collection: {duplicates}"

    for directory in directories:
        for path in sorted(directory.glob("test_*.py")):
            functions = top_level_test_functions(path)
            source = path.read_text()
            guide = module_docstring(path)
            assert len(functions) == 1, f"{path} must contain exactly one top-level test, found {functions}"
            assert guide.strip(), f"{path} must explain its proof variant in a module docstring"
            assert "pytestmark = pytest.mark.proof" in source, f"{path} must carry the proof marker"

            examples = re.findall(r"```yaml\n(.*?)\n```", guide, flags=re.DOTALL)
            assert examples, f"{path} must include a fenced YAML input/output example"
            assert any("input:" in example and "expected_output:" in example for example in examples), (
                f"{path} YAML example must contain input and expected_output trees"
            )


def test_each_proof_directory_has_complete_python_user_guide() -> None:
    for directory in proof_directories():
        guide = "\n".join(module_docstring(path) for path in sorted(directory.glob("*.py"))).lower()
        sections = {
            "claim": ("claim",),
            "user guidance": ("user guidance", "what to do", "\nuse\n", "use "),
            "avoidance guidance": ("what not to do", "avoid", "do not "),
            "reason": ("why", "because", "distinction matters"),
            "protocol/gate": ("protocol", "gate"),
            "status": ("status",),
            "evidence": ("evidence",),
            "promotion": ("promotion",),
            "run command": ("uv run pytest", "make proofs"),
        }
        missing = [label for label, variants in sections.items() if not any(item in guide for item in variants)]
        if "remaining work" not in guide and "further work" not in guide:
            missing.append("remaining/further work")
        assert not missing, f"{directory} docstrings are missing guide sections: {missing}"
