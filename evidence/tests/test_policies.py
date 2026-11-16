"""The shared Evidence exception owns immutable configuration, never scientific code."""

import ast
from pathlib import Path


def test_internal_policy_module_contains_configuration_only():
    source = Path(__file__).resolve().parents[1] / "experiments/policies.py"
    syntax = ast.parse(source.read_text())
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(syntax))
    constructors = {ast.unparse(node.func) for node in ast.walk(syntax) if isinstance(node, ast.Call)}
    assert constructors == {
        "rf.presets.register",
        "rf.presets.Preset",
        "rf.presets.BranchDefaults",
        "rf.presets.LeafDefaults",
        "rf.Attention",
    }
    assert [ast.unparse(node) for node in syntax.body if isinstance(node, (ast.Import, ast.ImportFrom))] == [
        "import relflow as rf"
    ]
