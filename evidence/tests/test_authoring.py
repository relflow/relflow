"""Scaffolding writes safe, discoverable sources without executing experiments."""

import ast
import builtins
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile

import pytest
from experiments import run as authoring
from reporting import experiment
from source import Experiment, declaration, environment, metadata, registry


@pytest.fixture
def root(tmp_path):
    path = tmp_path / "evidence"
    path.mkdir()
    return path


def existing(root, identifier):
    path = root / "experiments" / f"{identifier}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# %% [markdown]\n# ---\n# title: Existing question\n# ---\n"
        f"@experiment({identifier!r}, seed=123)\n"
        "def run(seed, steps, accelerator):\n"
        "    raise AssertionError('Existing experiments must not execute while authoring')\n"
        "raise AssertionError('Existing experiments must not import while authoring')\n"
    )
    return path


def contents(root):
    return {
        path.relative_to(root).as_posix(): (
            ("symlink", str(path.readlink()))
            if path.is_symlink()
            else ("file", path.read_bytes())
            if path.is_file()
            else ("directory", None)
        )
        for path in sorted(root.rglob("*"))
    }


def test_create_uses_next_numeric_id_without_reusing_gaps_or_publication_data(root, monkeypatch):
    existing(root, "P001")
    existing(root, "P009")
    (root / "results.json").write_text("Invalid publication JSON must not be read by source authoring")
    original = contents(root)
    monkeypatch.chdir(root.parent)
    imported = builtins.__import__

    def reject_models(name, *args, **kwargs):
        if name.split(".")[0] in {"relflow", "torch", "lightning"}:
            pytest.fail(f"Scaffolding must not import model dependency {name}")
        return imported(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_models)
    path = authoring.create(root, title="A new question?", group="Structure")
    assert path == root / "experiments/P010.py"
    source = path.read_text()
    assert declaration(source) == Experiment("P010", 0)
    assert environment(source)["dependencies"] == ["relflow>=0.6.0", "pyyaml>=6", "torch"]
    header = metadata(source)
    assert header["title"] == "A new question?"
    assert header["evidence-group"] == "Structure"
    assert "evidence-id" not in header
    discovered = registry(root, measurements=False)
    assert set(discovered) == {"P001", "P009", "P010"}
    assert discovered["P010"]["script"] == "experiments/P010.py"
    assert all(entry["run"] is None for entry in discovered.values())
    assert {key: value for key, value in contents(root).items() if key in original} == original


def test_empty_catalog_starts_at_one_with_reviewable_group_placeholder(root):
    path = authoring.create(root, title="A new question?")
    assert path.name == "P001.py"
    assert declaration(path.read_text()) == Experiment("P001", 0)
    assert metadata(path.read_text())["evidence-group"] == "Unreviewed"


def test_next_identifier_uses_numeric_order(root):
    for identifier in ("P9", "P010", "P100"):
        existing(root, identifier)
    path = authoring.create(root, title="Next question?")
    assert path.name == "P101.py"
    assert declaration(path.read_text()) == Experiment("P101", 0)


def test_explicit_declaration_and_presentation_values_remain_literal(root):
    title = 'Can "quoted": values\nstay literal?'
    group = 'Science: "quoted"\ngroup'
    path = authoring.create(root, title=title, group=group, identifier="P123", seed=2**32 - 1)
    assert path.name == "P123.py"
    source = path.read_text()
    assert declaration(source) == Experiment("P123", 2**32 - 1)
    assert metadata(source)["title"] == title
    assert metadata(source)["evidence-group"] == group
    assert "evidence-id" not in metadata(source)


@pytest.mark.parametrize(
    "options",
    [
        {"identifier": "Pbad"},
        {"identifier": ""},
        {"identifier": "../outside"},
        {"identifier": "/tmp/P001"},
        {"identifier": "P001.py"},
        {"identifier": "family/P001"},
        {"seed": -1},
        {"seed": 2**32},
        {"seed": True},
        {"seed": 1.0},
        {"title": ""},
        {"title": " \n "},
        {"group": ""},
        {"group": " \n "},
    ],
)
def test_invalid_declaration_or_presentation_does_not_create_directories(root, options):
    before = contents(root)
    with pytest.raises(ValueError):
        authoring.create(root, **{"title": "New question?", **options})
    assert contents(root) == before


@pytest.mark.parametrize("kind", ["file", "directory", "dangling-symlink"])
def test_existing_target_is_never_overwritten(root, kind):
    path = root / "experiments/P007.py"
    path.parent.mkdir(parents=True)
    if kind == "file":
        path.write_text("Preserve these exact bytes")
    elif kind == "directory":
        path.mkdir()
    else:
        path.symlink_to(root / "missing.py")
    before = contents(root)
    with pytest.raises(FileExistsError):
        authoring.create(root, title="New question?", identifier="P007")
    assert contents(root) == before


@pytest.mark.parametrize("component", ["root", "experiments"])
def test_symlink_directories_cannot_redirect_scaffolding(root, tmp_path, component):
    outside = tmp_path / "outside"
    outside.mkdir()
    selected = root
    if component == "root":
        selected = tmp_path / "linked_evidence"
        selected.symlink_to(root, target_is_directory=True)
    else:
        (root / "experiments").symlink_to(outside, target_is_directory=True)
    before = contents(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        authoring.create(selected, title="New question?")
    assert contents(tmp_path) == before


def test_generated_experiment_declares_a_model_and_cannot_claim_a_passing_result(root):
    path = authoring.create(root, title="An unimplemented question?")
    syntax = ast.parse(path.read_text())
    model = next(node for node in syntax.body if isinstance(node, ast.ClassDef))
    assert model.name == "Model" and ast.unparse(model.bases[0]) == "rf.Model"
    run = next(node for node in syntax.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    namespace = {"experiment": experiment}
    exec(compile(ast.Module(body=[run], type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["run"].experiment == Experiment("P001", 0)
    with pytest.raises(NotImplementedError):
        namespace["run"](0, None, "cpu")


@pytest.mark.parametrize("explicit", [False, True])
def test_new_cli_writes_scaffold_without_spawning_experiments(root, monkeypatch, capsys, explicit):
    existing(root, "P007")
    monkeypatch.setattr(authoring, "ROOT", root)

    def reject_execution(*args, **kwargs):
        pytest.fail("--new must not launch a child experiment")

    monkeypatch.setattr(authoring.subprocess, "run", reject_execution)
    arguments = ["run.py", "--new", "--title", "A new question?"]
    if explicit:
        arguments.extend(["--group", "New science", "--id", "P123", "--seed", "456"])
    monkeypatch.setattr(sys, "argv", arguments)
    authoring.main()
    identifier = "P123" if explicit else "P008"
    source = (root / f"experiments/{identifier}.py").read_text()
    assert declaration(source) == (Experiment("P123", 456) if explicit else Experiment("P008", 0))
    assert metadata(source)["title"] == "A new question?"
    assert metadata(source)["evidence-group"] == ("New science" if explicit else "Unreviewed")
    assert f"Created {identifier}: evidence/experiments/{identifier}.py" in capsys.readouterr().out


@pytest.mark.parametrize(
    "arguments",
    [
        ["P001", "--new", "--title", "A question?"],
        ["--new", "--list"],
        ["--new"],
        ["--new", "--title", "A question?", "--steps", "1"],
        ["--new", "--title", "A question?", "--accelerator", "gpu"],
        ["--new", "--title", "A question?", "--threads", "2"],
        ["--new", "--title", "A question?", "--output-dir", "results"],
        ["--new", "--title", "A question?", "--wheel", "relflow-0.5.0-py3-none-any.whl"],
        ["--title", "A question?"],
        ["--group", "A group"],
        ["--id", "P001"],
        ["--new", "../outside"],
        ["--new", "family/new_trial"],
        ["--new", "--title", "A question?", "--seed", "-1"],
        ["--new", "--title", "A question?", "--id", "Pbad"],
    ],
)
def test_invalid_cli_combinations_fail_before_any_write_or_execution(root, monkeypatch, arguments):
    monkeypatch.setattr(authoring, "ROOT", root)
    monkeypatch.setattr(sys, "argv", ["run.py", *arguments])

    def reject_execution(*args, **kwargs):
        pytest.fail("Invalid authoring commands must not launch experiments")

    monkeypatch.setattr(authoring.subprocess, "run", reject_execution)
    before = contents(root)
    with pytest.raises(SystemExit) as error:
        authoring.main()
    assert error.value.code == 2
    assert contents(root) == before


def test_scripts_without_inline_dependencies_cannot_run(root, monkeypatch):
    existing(root, "P007")
    monkeypatch.setattr(authoring, "ROOT", root)
    monkeypatch.setattr(sys, "argv", ["run.py", "P007"])
    with pytest.raises(SystemExit) as error:
        authoring.main()
    assert error.value.code == 2


@pytest.mark.parametrize("supplied", [False, True])
def test_uv_executes_in_isolation_from_an_unrelated_directory(root, monkeypatch, tmp_path, supplied):
    if shutil.which("uv") is None:
        pytest.skip("uv is required to exercise isolated script execution")
    wheel = tmp_path / "relflow-0.5.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("relflow/__init__.py", 'VERSION = "supplied-wheel"\n')
        archive.writestr(
            "relflow-0.5.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: relflow\nVersion: 0.5.0\nProvides-Extra: torchjd\n",
        )
        archive.writestr(
            "relflow-0.5.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr("relflow-0.5.0.dist-info/RECORD", "")
    script = authoring.create(root, title="Isolation probe?")
    # This probe has no model dependency: exercise the same uv launch with a tiny
    # wheel and a sibling import, without training or reaching a package index.
    source = script.read_text()
    header = source.split("# %%\n")[0]
    script.write_text(
        header + '# %%\n# /// script\n# requires-python = ">=3.12"\n'
        '# dependencies = ["relflow[torchjd]"]\n'
        '# [tool.relflow.evidence]\n# capability = "representation"\n# types = []\n'
        '# conditions = ["noise"]\n# ///\n'
        "import json, os, sys\nfrom pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parent))\n"
        "import relflow, shared_fixture\n"
        "def experiment(*args, **kwargs):\n    return lambda f: f\n"
        '@experiment("P001", seed=0)\ndef run(seed, steps, accelerator):\n    pass\n'
        'print(json.dumps({"version": relflow.VERSION, "package": relflow.__file__, '
        '"shared": shared_fixture.VALUE, "prefix": sys.prefix, '
        '"wheel_sha256": os.environ.get("EVIDENCE_WHEEL_SHA256"), "arguments": sys.argv[1:]}))\n'
    )
    (root / "experiments").mkdir(exist_ok=True)
    (root / "experiments/shared_fixture.py").write_text('VALUE = "sibling-helper"\n')
    trap = tmp_path / "editable"
    (trap / "relflow").mkdir(parents=True)
    (trap / "relflow/__init__.py").write_text("raise AssertionError('editable checkout selected')\n")
    monkeypatch.setenv("PYTHONPATH", str(trap))
    monkeypatch.setenv("VIRTUAL_ENV", str(trap))
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.setenv("UV_NO_INDEX", "1")
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("UV_PYTHON_DOWNLOADS", "never")
    if not supplied:
        monkeypatch.setenv("UV_FIND_LINKS", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(authoring, "ROOT", root)
    arguments = ["run.py", "P001", "--steps", "1"]
    if supplied:
        arguments.extend(["--wheel", str(wheel)])
    monkeypatch.setattr(sys, "argv", arguments)
    results = []
    launch = subprocess.run

    def capture(command, **kwargs):
        result = launch(command, **kwargs, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        results.append(json.loads(result.stdout))
        return result

    monkeypatch.setattr(authoring.subprocess, "run", capture)
    authoring.main()
    assert len(results) == 1
    result = results[0]
    assert result["version"] == "supplied-wheel"
    assert result["shared"] == "sibling-helper"
    if supplied:
        assert result["wheel_sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert result["prefix"] != sys.prefix
    assert str(trap) not in result["package"]
    assert result["arguments"] == ["--steps", "1", "--accelerator", "cpu", "--threads", "1"]


def test_listing_uses_static_sources_without_reading_measurements_or_importing_experiments(root, monkeypatch, capsys):
    existing(root, "P007")
    (root / "results.json").write_text("Invalid publication JSON must not be read by listing")
    monkeypatch.setattr(authoring, "ROOT", root)
    monkeypatch.setattr(sys, "argv", ["run.py", "--list"])
    before = contents(root)
    authoring.main()
    assert capsys.readouterr().out == "P007  experiments/P007.py\n"
    assert contents(root) == before
