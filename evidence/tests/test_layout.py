"""Contracts for standalone experiments registered by stable proof IDs."""

from __future__ import annotations

import ast
import re
import sys
from collections import Counter
from pathlib import Path

import yaml
from experiments.source import environment, registry, tags

ROOT = Path(__file__).resolve().parents[1]
PROOFS = ROOT / "experiments"


def test_shared_rendering_and_publication_never_list_experiment_ids():
    shared = [
        *(ROOT.parent / "docs/starlight/src/evidence").glob("*.ts"),
        ROOT / "diagnostics/plots.py",
        ROOT / "record.py",
    ]
    for path in shared:
        assert not re.search(r"\bP\d{3}\b", path.read_text()), f"{path}: move experiment-specific data into its script"
    for path in ("diagnostics/views.json", "timings.json"):
        assert not (ROOT / path).exists(), f"{path}: experiments own their metadata"
    assert not (ROOT.parent / "docs/starlight/src/evidence/reviews.ts").exists()


def test_every_experiment_declares_its_imported_dependencies_without_importing_models():
    for entry in registry(ROOT, measurements=False).values():
        path = ROOT / entry["script"]
        source = path.read_text()
        assert not re.search(r"\]\([^)]*\.qmd(?:[)#?])", source), f"{path}: link to a current Starlight route"
        declared = environment(source)
        assert declared is not None, f"{path}: declare an isolated PEP 723 environment"
        assert declared["requires-python"] == ">=3.12"
        classified = tags(source)
        assert classified is not None, f"{path}: own tags in [tool.relflow.evidence]"
        assert classified["capability"] in {"computation", "structure", "representation", "training", "reliability"}
        assert classified["conditions"], f"{path}: review the evaluation conditions"
        dependencies = {re.split(r"[\[ <>=!~;@]", value)[0].lower() for value in declared["dependencies"]}
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module.split(".")[0]
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom) and node.module
        }
        expected = imports - sys.stdlib_module_names - {"reporting", "policies", "evidence"}
        assert expected <= dependencies, f"{path}: missing dependencies {expected - dependencies}"
        assert {"pyyaml", "torch", "relflow"} <= dependencies, f"{path}: reporter dependencies are explicit"
        assert "Path(__file__).resolve().parent" in source
        if "torchjd" in expected:
            assert "relflow[torchjd]>=0.5.0" in declared["dependencies"]
        if "transformers" in expected:
            extras = {
                extra
                for requirement in declared["dependencies"]
                if (match := re.fullmatch(r"relflow\[([\w,-]+)\].*", requirement))
                for extra in match[1].split(",")
            }
            media = {name.lower() for name in classified["types"] if name in {"Text", "Image", "Audio", "Video"}}
            assert extras & {"text", "image", "audio", "video"} and media <= extras, (
                f"{path}: select the relevant encoder extras"
            )


def test_registered_proofs_have_unique_standalone_scripts() -> None:
    entries = registry(ROOT)
    assert entries, "declare each experiment with @experiment"

    paths = Counter(entry["script"] for entry in entries.values())
    assert all(count == 1 for count in paths.values()), "each proof ID must own one experiment script"
    expected = {ROOT / path for path in paths}
    actual = set(PROOFS.rglob("*.py"))
    assert not (ROOT / "research").exists(), "keep canonical experiments together"
    actual -= {PROOFS / name for name in ("policies.py", "reporting.py", "run.py", "source.py")}
    assert actual == expected, f"proof scripts and results registry disagree: {actual ^ expected}"
    assert not list(PROOFS.rglob("test_*.py")), "learning experiments belong in standalone scripts"
    pages = Counter(entry["page"] for entry in entries.values())
    assert all(count == 1 for count in pages.values()), "each proof ID must own one published page"

    for identifier, entry in entries.items():
        path = ROOT / entry["script"]
        assert path.parent == PROOFS and path.name == f"{identifier}.py" and path.is_file(), path
        page = Path(entry["page"])
        assert page == Path(f"docs/evidence/{identifier}.py"), page
        module = ast.parse(path.read_text(), filename=str(path))
        declared = [
            ast.literal_eval(decorator.args[0])
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            for decorator in node.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "experiment"
        ]
        assert declared == [identifier], f"{path}: one @experiment declaration must own {identifier}"
        assert not any(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "PROOF_ID" for target in node.targets)
            for node in module.body
        ), f"{path}: the decorator owns the ID without a duplicate constant"
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
        model_classes = {
            node.name
            for node in ast.walk(module)
            if isinstance(node, ast.ClassDef)
            and any(
                isinstance(base, ast.Attribute)
                and isinstance(base.value, ast.Name)
                and base.value.id in relflow_names
                and base.attr == "Model"
                for base in node.bases
            )
        }
        constructors = set()
        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            constructor = node.func
            if isinstance(constructor.value, ast.Name) and constructor.value.id in relflow_names:
                constructors.add(constructor.attr)
                if constructor.attr.endswith("DataModule"):
                    assert not node.args and all(keyword.arg != "model" for keyword in node.keywords), (
                        f"{path}:{node.lineno}: let the model or Trainer loop bind the data module automatically"
                    )
        assert model_classes and constructors & {
            "SyntheticDataModule",
            "ArrowDataModule",
            "PolarsDataModule",
        }, f"{path}: include the model and independently generated splits"
        assert any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(module)), (
            f"{path}: include the synthetic record generator"
        )
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
            for node in module.body
        ), f"{path}: expose an experiment rather than a collected test"


def test_experiments_declare_models_in_classes():
    for entry in registry(ROOT, measurements=False).values():
        path = ROOT / entry["script"]
        syntax = ast.parse(path.read_text())
        for node in ast.walk(syntax):
            if isinstance(node, ast.Call):
                assert ast.unparse(node.func) != "rf.Model", (
                    f"{path}:{node.lineno}: declare the model's children in a Model class; "
                    "inline Branch definitions remain valid inside that declaration"
                )


def test_experiments_keep_explicit_training_protocols_on_model_fit() -> None:
    entries = registry(ROOT)
    for entry in entries.values():
        path = ROOT / entry["script"]
        module = ast.parse(path.read_text(), filename=str(path))
        calls = [node for node in ast.walk(module) if isinstance(node, ast.Call)]
        model_classes = {
            node.name
            for node in ast.walk(module)
            if isinstance(node, ast.ClassDef) and any(ast.unparse(base) == "rf.Model" for base in node.bases)
        }
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
            if ast.unparse(call.func) in model_classes:
                assert all(keyword.arg not in {"optimizer", "batch_size"} for keyword in call.keywords), path
            if ast.unparse(call.func) == "rf.Group":
                options = {keyword.arg: keyword.value for keyword in call.keywords}
                assert "n_groups" in options, path
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
        assert [ast.unparse(argument) for argument in reports[0].args] == ["run"], path
        assert not reports[0].keywords, f"{path}: the decorator owns reporting metadata"
        declarations = [
            decorator
            for decorator in functions["run"].decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "experiment"
        ]
        assert len(declarations) == 1, f"{path}: register the existing run function once"
        seeds = [ast.literal_eval(keyword.value) for keyword in declarations[0].keywords if keyword.arg == "seed"]
        assert len(seeds) == 1 and type(seeds[0]) is int, f"{path}: declare one reproducible integer seed"


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
    for identifier, entry in entries.items():
        page = ROOT / entry["script"]
        text = markdown(page)
        header = re.match(r"\A---\s*\n(.*?)\n---(?:\n|\Z)", text, flags=re.DOTALL)
        assert header, f"{page}: missing proof metadata"
        metadata = yaml.safe_load(header[1])
        assert "evidence-id" not in metadata, f"{page}: the decorator owns the ID"
        assert entry["header"]["evidence-id"] == identifier
        assert identifier in entries, f"{page}: unregistered proof ID {identifier}"
        documented[identifier] += 1
        assert ROOT / entries[identifier]["script"] == page, f"{page}: registered script does not match {identifier}"
        assert metadata.get("title") and metadata.get("description"), page
        assert metadata.get("evidence-group"), f"{page}: missing proof family"
        assert not {"categories", "evidence-diagram"} & metadata.keys(), f"{page}: remove unused presentation metadata"
        assert "evidence-source" not in metadata and "evidence-status" not in metadata, (
            f"{page}: status is derived from the measured result"
        )
        assert not {"execute", "code-fold", "toc"} & metadata.keys(), f"{page}: remove retired presentation options"
        assert "{{<" not in text and "evidence-details" not in text, f"{page}: use plain Markdown sections"
        overview, details = text.split("## Method", maxsplit=1)
        assert details.strip(), f"{page}: retain the scientific method"
        assert re.findall(r"^## (.+)$", overview, flags=re.MULTILINE) == ["Example", "Comparison", "Result"], page
        insight = overview.split("## Result\n", maxsplit=1)[1].strip()
        assert insight, f"{page}: interpret the result and name its limits in the authored script"
        assert 1 <= len(metadata["evidence-readout"]) <= 3, f"{page}: keep the result comparison small"
        assert "```typst" not in text and "#tree(" not in text, f"{page}: remove unused diagram source"
        assert len(re.findall(r"^```yaml\s*$", text, flags=re.MULTILINE)) == 1, f"{page}: show one example record"

    assert set(documented) == set(entries), (
        f"proofs missing documentation: {sorted(entries.keys() - documented.keys())}"
    )
    assert all(count == 1 for count in documented.values()), "each proof ID must have exactly one docs page"
