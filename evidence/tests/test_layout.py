"""Contracts for standalone experiments registered by stable proof IDs."""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import yaml
from experiments.source import registry

ROOT = Path(__file__).resolve().parents[1]
PROOFS = ROOT / "experiments"


def test_registered_proofs_have_unique_standalone_scripts() -> None:
    entries = registry(ROOT)
    assert entries, "declare each experiment in its frontmatter"

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

    for identifier, entry in entries.items():
        path = ROOT / entry["script"]
        assert path.parent.parent == PROOFS and path.is_file(), path
        page = Path(entry["page"])
        assert page.suffix == ".py" and page.parent.parent == Path("docs/evidence"), page
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
    entries = registry(ROOT)
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
                if constructor.attr.endswith("DataModule"):
                    assert not node.args and all(keyword.arg != "model" for keyword in node.keywords), (
                        f"{path}:{node.lineno}: let the model or Trainer loop bind the data module automatically"
                    )
        assert {"Model", "SyntheticDataModule"} <= constructors, f"{path}: include the model and synthetic splits"
        assert any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(module)), (
            f"{path}: include the synthetic record generator"
        )
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
            for node in module.body
        ), f"{path}: expose an experiment rather than a collected test"


def test_experiments_keep_explicit_training_protocols_on_model_fit() -> None:
    entries = registry(ROOT)
    for entry in entries.values():
        path = ROOT / entry["script"]
        module = ast.parse(path.read_text(), filename=str(path))
        calls = [node for node in ast.walk(module) if isinstance(node, ast.Call)]
        assert not any(ast.unparse(call.func) == "lit.Trainer" for call in calls), path
        fits = [call for call in calls if ast.unparse(call.func) == "model.fit"]
        assert len(fits) == 1, path
        options = {keyword.arg: keyword.value for keyword in fits[0].keywords}
        assert "batch_size" in options and "max_steps" in options and "max_epochs" in options, path
        assert ast.literal_eval(options["early_stopping"]) is False, path
        assert ast.literal_eval(options["rollback"]) is False, path
        assert ast.literal_eval(options["precision"]) == "32-true", path
        if "devices" in options:
            assert ast.literal_eval(options["strategy"]) == "auto", path
        else:
            expected = ast.parse(
                '"ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto"',
                mode="eval",
            ).body
            assert ast.dump(options["strategy"], include_attributes=False) == ast.dump(
                expected, include_attributes=False
            ), path
        assert ("optimizer" in options) != ("learning_rate" in options), path
        for call in calls:
            if ast.unparse(call.func) in {"rf.Model", "rf.Model.xs"}:
                assert all(keyword.arg not in {"optimizer", "batch_size"} for keyword in call.keywords), path
            if ast.unparse(call.func) == "rf.Cluster":
                assert all(keyword.arg != "n_clusters" for keyword in call.keywords), path
            if ast.unparse(call.func) == "rf.Address":
                assert all(
                    "/" not in argument.value
                    for argument in call.args
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
                ), path
        assert not any(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Attribute) and target.attr == "optimizer" for target in node.targets)
            for node in ast.walk(module)
        ), path


def test_experiments_expose_the_shared_reporting_entrypoint() -> None:
    entries = registry(ROOT)
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


def markdown(source: Path) -> str:
    """Read authored proof prose without importing the experiment."""
    lines: list[str] = []
    active = False
    for line in source.read_text().splitlines():
        if line.startswith("# %%"):
            active = line.strip() == "# %% [markdown]"
            if active:
                lines.append("")
        elif active:
            assert not line.strip() or line.startswith("#"), f"{source}: Markdown cells must contain comments"
            lines.append(line.removeprefix("#").removeprefix(" "))
    return "\n".join(lines).strip()


def test_every_registered_proof_has_a_small_readable_overview() -> None:
    entries = registry(ROOT)
    documented: Counter[str] = Counter()
    for entry in entries.values():
        page = ROOT / entry["script"]
        text = markdown(page)
        header = re.match(r"\A---\s*\n(.*?)\n---(?:\n|\Z)", text, flags=re.DOTALL)
        assert header, f"{page}: missing proof metadata"
        metadata = yaml.safe_load(header[1])
        identifier = metadata["evidence-id"]
        assert identifier in entries, f"{page}: unregistered proof ID {identifier}"
        documented[identifier] += 1
        assert ROOT / entries[identifier]["script"] == page, f"{page}: registered script does not match {identifier}"
        assert metadata.get("title") and metadata.get("description"), page
        assert metadata.get("categories"), f"{page}: missing proof family"
        assert "evidence-source" not in metadata and "evidence-status" not in metadata, (
            f"{page}: status is derived from the measured result"
        )
        sections = set(re.findall(r"\{\{<\s*evidence\s+([^\s>]+)\s+([^\s>]+)\s*>\}\}", text))
        expected = {(identifier, part) for part in ("status", "evidence", "script")}
        assert sections == expected, f"{page}: render recorded evidence and reproduction commands for {identifier}"
        overview, details = text.split('<details class="evidence-details">', maxsplit=1)
        assert "<summary>Experiment details and code</summary>" in details, page
        assert f"{{{{< evidence {identifier} evidence >}}}}" in details, page
        assert f"{{{{< evidence {identifier} script >}}}}" in details, page
        assert metadata["toc"] is False, f"{page}: optional details must not fill the contents menu"
        assert re.findall(r"^## (.+)$", overview, flags=re.MULTILINE) == ["Example", "Comparison", "Result"], page
        result = overview.split("## Result\n", maxsplit=1)[1]
        insight = re.sub(r"\{\{<.*?>\}\}", "", result).strip()
        assert insight, f"{page}: interpret the result and name its limits in the authored script"
        assert 1 <= len(metadata["evidence-readout"]) <= 3, f"{page}: keep the result comparison small"
        assert "```{typst}" in text and "#tree(" in text, f"{page}: missing model tree"
        assert "//| fig-alt:" in text, f"{page}: missing tree description"
        assert len(re.findall(r"^```yaml\s*$", text, flags=re.MULTILINE)) == 1, f"{page}: show one example record"
        assert "#tree(" not in details, f"{page}: keep one diagram beside the example"

    assert set(documented) == set(entries), (
        f"proofs missing documentation: {sorted(entries.keys() - documented.keys())}"
    )
    assert all(count == 1 for count in documented.values()), "each proof ID must have exactly one docs page"
