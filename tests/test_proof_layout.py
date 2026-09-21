"""Contracts for standalone experiments registered by stable proof IDs."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROOFS = ROOT / "proofs"


def test_registered_proofs_have_unique_standalone_scripts() -> None:
    document = yaml.safe_load((PROOFS / "results.yaml").read_text())
    assert document["schema"] == 1
    entries = document["proofs"]
    assert entries, "register experiments in proofs/results.yaml"

    paths = Counter(entry["script"] for entry in entries.values())
    assert all(count == 1 for count in paths.values()), "each proof ID must own one experiment script"
    expected = {ROOT / path for path in paths}
    actual = set(PROOFS.rglob("*.py")) - {
        PROOFS / name for name in ("reporting.py", "run.py", "source.py", "render.py")
    }
    assert actual == expected, f"proof scripts and results registry disagree: {actual ^ expected}"
    assert not list(PROOFS.rglob("test_*.py")), "learning experiments belong in standalone scripts"
    pages = Counter(entry["page"] for entry in entries.values())
    assert all(count == 1 for count in pages.values()), "each proof ID must own one published page"
    assert not list((ROOT / "docs/proofs").glob("*/*.qmd")), "author proof pages in their experiment scripts"

    for identifier, entry in entries.items():
        assert entry["historical"]["status"] in {"Passing", "Limited", "Partial"}, identifier
        path = ROOT / entry["script"]
        assert path.parent.parent == PROOFS and path.is_file(), path
        page = ROOT / entry["page"]
        assert page.suffix == ".py" and page.parent.parent == ROOT / "docs/proofs", page
        assert path.parent.name == page.parent.name, f"{path}: keep the script in its documented category"
        module = ast.parse(path.read_text(), filename=str(path))
        declared = [
            ast.literal_eval(node.value)
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "PROOF_ID" for target in node.targets)
        ]
        assert declared == [identifier], f"{path}: PROOF_ID must match {identifier}"
        assert ast.get_docstring(module), f"{path}: explain the experiment's claim"


def test_experiments_keep_generation_and_modeling_in_their_own_script() -> None:
    entries = yaml.safe_load((PROOFS / "results.yaml").read_text())["proofs"]
    sibling_modules = {Path(entry["script"]).stem for entry in entries.values()}
    for entry in entries.values():
        path = ROOT / entry["script"]
        module = ast.parse(path.read_text(), filename=str(path))
        imports = []
        relflow_names = set()
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
                relflow_names.update(alias.asname or alias.name for alias in node.names if alias.name == "relflow")
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, f"{path}: relative experiment imports hide part of the proof"
                imports.append(node.module or "")

        forbidden = [
            name
            for name in imports
            if name.split(".")[0] in {"pytest", *sibling_modules}
            or (name.startswith("proofs.") and name != "proofs.reporting")
            or name == "proofs"
        ]
        assert not forbidden, f"{path}: keep the experiment self-contained: {forbidden}"
        constructors = set()
        for node in ast.walk(module):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            constructor = node.func
            if (
                constructor.attr in {"xs", "sm", "md", "lg", "xl"}
                and isinstance(constructor.value, ast.Attribute)
                and constructor.value.attr == "Model"
            ):
                constructor = constructor.value
            if isinstance(constructor.value, ast.Name) and constructor.value.id in relflow_names:
                constructors.add(constructor.attr)
        assert {"Model", "SyntheticDataModule"} <= constructors, f"{path}: include the model and synthetic splits"
        assert any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(module)), (
            f"{path}: include the synthetic record generator"
        )
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
            for node in module.body
        ), f"{path}: expose an experiment rather than a collected test"


def test_experiments_expose_the_shared_reporting_entrypoint() -> None:
    entries = yaml.safe_load((PROOFS / "results.yaml").read_text())["proofs"]
    for entry in entries.values():
        path = ROOT / entry["script"]
        module = ast.parse(path.read_text(), filename=str(path))
        functions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
        assert "run" in functions, f"{path}: missing callable experiment"
        assert [argument.arg for argument in functions["run"].args.args] == ["seed", "steps", "accelerator"], path
        guards = [
            node
            for node in module.body
            if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"
        ]
        assert len(guards) == 1, f"{path}: protect CLI execution with a main guard"
        reports = [
            node
            for node in ast.walk(guards[0])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "report"
        ]
        assert len(reports) == 1, f"{path}: record the experiment through shared reporting"
        assert [ast.unparse(argument) for argument in reports[0].args] == ["PROOF_ID", "run"], path
        assert any(keyword.arg == "seed" for keyword in reports[0].keywords), f"{path}: declare a reproducible seed"
