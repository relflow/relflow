"""Scaffolding writes safe, discoverable sources without executing experiments."""

import builtins
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile

import pytest
from experiments import run as authoring
from source import Experiment, declaration, environment, metadata, registry


@pytest.fixture
def root(tmp_path):
    path = tmp_path / "evidence"
    path.mkdir()
    return path


def existing(root, target, identifier, *, group="Existing science"):
    path = root / "experiments" / target
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# %% [markdown]\n# ---\n# title: Existing question\n"
        f"# evidence-group: {json.dumps(group)}\n# ---\n"
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
    existing(root, "family/first.py", "P001", group="Family science")
    existing(root, "other/last.py", "P009")
    (root / "results.json").write_text("Invalid publication JSON must not be read by source authoring")
    original = contents(root)
    monkeypatch.chdir(root.parent)
    imported = builtins.__import__

    def reject_models(name, *args, **kwargs):
        if name.split(".")[0] in {"relflow", "torch", "lightning"}:
            pytest.fail(f"Scaffolding must not import model dependency {name}")
        return imported(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_models)
    path = authoring.create(root, "family/new_trial")

    assert path == root / "experiments/family/new_trial.py"
    source = path.read_text()
    assert declaration(source) == Experiment("P010", 0)
    assert environment(source)["dependencies"] == ["relflow>=0.5.0", "pyyaml>=6", "torch"]
    header = metadata(source)
    assert header["title"] == "New trial"
    assert header["evidence-group"] == "Family science"
    assert "evidence-id" not in header
    discovered = registry(root, measurements=False)
    assert set(discovered) == {"P001", "P009", "P010"}
    assert discovered["P010"]["script"] == "experiments/family/new_trial.py"
    assert all(entry["run"] is None for entry in discovered.values())
    assert {key: value for key, value in contents(root).items() if key in original} == original


def test_empty_catalog_starts_at_one_and_accepts_optional_python_suffix(root):
    path = authoring.create(root, "new_family/new_trial.py")
    assert declaration(path.read_text()) == Experiment("P001", 0)
    assert metadata(path.read_text())["evidence-group"] == "New Family"


def test_next_identifier_uses_numeric_order(root):
    existing(root, "family/nine.py", "P9")
    existing(root, "family/ten.py", "P010")
    existing(root, "family/hundred.py", "P100")
    path = authoring.create(root, "family/next")
    assert declaration(path.read_text()) == Experiment("P101", 0)


def test_explicit_declaration_and_presentation_values_remain_literal(root):
    title = 'Can "quoted": values\nstay literal?'
    group = 'Science: "quoted"\ngroup'
    path = authoring.create(root, "family/explicit", title=title, group=group, identifier="P123", seed=2**32 - 1)
    source = path.read_text()
    assert declaration(source) == Experiment("P123", 2**32 - 1)
    assert metadata(source)["title"] == title
    assert metadata(source)["evidence-group"] == group
    assert "evidence-id" not in metadata(source)


def test_ambiguous_existing_groups_fall_back_to_family_name(root):
    existing(root, "new_family/first.py", "P001", group="First science")
    existing(root, "new_family/second.py", "P002", group="Second science")
    path = authoring.create(root, "new_family/third")
    assert metadata(path.read_text())["evidence-group"] == "New Family"


@pytest.mark.parametrize(
    "target",
    [
        "",
        "name",
        "/tmp/family/name.py",
        "../outside",
        "family/../outside",
        "family/../../outside",
        "family/./name",
        "family//name",
        "family/",
        "family/nested/name",
        "family/name.txt",
        "family/name.py.py",
        "family/.hidden",
        "Family/name",
        "family/Name",
        "family/has-hyphen",
        "family/has space",
        "family/name;command",
        "family\\name",
        "family/reporting",
        "family/source.py",
        "family/run",
        "family/render.py",
    ],
)
def test_invalid_targets_do_not_create_directories_or_files(root, target):
    before = contents(root)
    with pytest.raises(ValueError):
        authoring.create(root, target)
    assert contents(root) == before


@pytest.mark.parametrize(
    "options",
    [
        {"identifier": "Pbad"},
        {"identifier": ""},
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
        authoring.create(root, "new_family/new_trial", **options)
    assert contents(root) == before


def test_duplicate_identifier_fails_before_creating_a_new_family(root):
    existing(root, "existing/original.py", "P007")
    before = contents(root)
    with pytest.raises(ValueError, match="already registered"):
        authoring.create(root, "new_family/new_trial", identifier="P007")
    assert contents(root) == before


@pytest.mark.parametrize("kind", ["file", "directory", "dangling-symlink"])
def test_existing_target_is_never_overwritten(root, kind):
    path = root / "experiments/family/existing.py"
    path.parent.mkdir(parents=True)
    if kind == "file":
        path.write_text("Preserve these exact bytes")
    elif kind == "directory":
        path.mkdir()
    else:
        path.symlink_to(root / "missing.py")
    before = contents(root)
    with pytest.raises((FileExistsError, ValueError)):
        authoring.create(root, "family/existing")
    assert contents(root) == before


@pytest.mark.parametrize("component", ["root", "experiments", "family"])
def test_symlink_directories_cannot_redirect_scaffolding(root, tmp_path, component):
    outside = tmp_path / "outside"
    outside.mkdir()
    selected = root
    if component == "root":
        selected = tmp_path / "linked_evidence"
        selected.symlink_to(root, target_is_directory=True)
    elif component == "experiments":
        (root / "experiments").symlink_to(outside, target_is_directory=True)
    else:
        (root / "experiments").mkdir()
        (root / "experiments/family").symlink_to(outside, target_is_directory=True)
    before = contents(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        authoring.create(selected, "family/new_trial")
    assert contents(tmp_path) == before


def test_generated_experiment_is_importable_but_cannot_claim_a_passing_result(root):
    path = authoring.create(root, "family/unimplemented")
    spec = importlib.util.spec_from_file_location("unimplemented_scaffold", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.run.experiment == Experiment("P001", 0)
    with pytest.raises(NotImplementedError):
        module.run(0, None, "cpu")


@pytest.mark.parametrize("explicit", [False, True])
def test_new_cli_writes_scaffold_without_spawning_experiments(root, monkeypatch, capsys, explicit):
    existing(root, "family/original.py", "P007", group="Family science")
    monkeypatch.setattr(authoring, "ROOT", root)

    def reject_execution(*args, **kwargs):
        pytest.fail("--new must not launch a child experiment")

    monkeypatch.setattr(authoring.subprocess, "run", reject_execution)
    arguments = ["run.py", "--new", "family/new_trial"]
    if explicit:
        arguments.extend(["--title", "A new question?", "--group", "New science", "--id", "P123", "--seed", "456"])
    else:
        arguments.extend(["--accelerator", "cpu", "--threads", "1"])
    monkeypatch.setattr(sys, "argv", arguments)
    authoring.main()
    source = (root / "experiments/family/new_trial.py").read_text()
    assert declaration(source) == (Experiment("P123", 456) if explicit else Experiment("P008", 0))
    assert metadata(source)["title"] == ("A new question?" if explicit else "New trial")
    assert metadata(source)["evidence-group"] == ("New science" if explicit else "Family science")
    assert (
        f"Created {'P123' if explicit else 'P008'}: evidence/experiments/family/new_trial.py" in capsys.readouterr().out
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["P001", "--new", "family/new_trial"],
        ["--new", "family/new_trial", "--list"],
        ["--new", "family/new_trial", "--steps", "1"],
        ["--new", "family/new_trial", "--accelerator", "gpu"],
        ["--new", "family/new_trial", "--threads", "2"],
        ["--new", "family/new_trial", "--output-dir", "results"],
        ["--new", "family/new_trial", "--wheel", "relflow-0.5.0-py3-none-any.whl"],
        ["--title", "A question?"],
        ["--group", "A group"],
        ["--id", "P001"],
        ["--new", "../outside"],
        ["--new", "family/new_trial", "--seed", "-1"],
        ["--new", "family/new_trial", "--id", "Pbad"],
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


def test_dependency_free_historical_scripts_remain_listable_but_cannot_run_without_a_declaration(root, monkeypatch):
    existing(root, "family/original.py", "P007")
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
    script = authoring.create(root, "family/probe")
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
        'sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))\n'
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
    existing(root, "family/original.py", "P007")
    (root / "results.json").write_text("Invalid publication JSON must not be read by listing")
    monkeypatch.setattr(authoring, "ROOT", root)
    monkeypatch.setattr(sys, "argv", ["run.py", "--list"])
    before = contents(root)
    authoring.main()
    assert capsys.readouterr().out == "P007  experiments/family/original.py\n"
    assert contents(root) == before
