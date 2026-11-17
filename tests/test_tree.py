"""The repository admits owned source files and rejects misplaced or generated artifacts."""

import subprocess
import sys
import tomllib

import pytest
import yaml

from scripts.tree import ROOT, violations

POLICY = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["relflow"]["tree"]


@pytest.mark.parametrize(
    "name",
    [
        "README.md",
        "scripts/tree.py",
        "src/relflow/new_type.py",
        "src/relflow/data/packing.py",
        "tests/data/test_packing.py",
        "docs/tests/test_tree.py",
        "docs/starlight/src/content/docs/new-guide.mdx",
        "docs/starlight/src/components/NewPanel.astro",
        "docs/starlight/tests/fixtures/example.mjs",
        "evidence/experiments/P001.py",
        "evidence/experiments/P1234.py",
        "native/src/cache/module.rs",
        "native/python/relflow_native/module.pyi",
    ],
)
def test_owned_source_paths_are_allowed(tmp_path, name):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("source")
    assert violations(tmp_path, POLICY, [name]) == []


@pytest.mark.parametrize(
    "name",
    [
        "notes.md",
        "scratch.py",
        "report.json",
        "scripts/one_off_benchmark.py",
        "research/new/probe.py",
        "src/relflow/report.json",
        "src/relflow/result.pkl",
        "tests/recorded-run.json",
        "docs/temporary.py",
        "docs/.gitignore",
        "docs/content-review.md",
        "docs/assets/benchmarks/results.json",
        "docs/starlight/src/probe.py",
        "docs/site/index.html",
        "evidence/results.json",
        "evidence/archives/report.json",
        "evidence/research/text/proof.py",
        "evidence/diagnostics/views.json",
        "evidence/timings.json",
        "docs/starlight/src/evidence/reviews.ts",
        "evidence/runs/result.json",
        "evidence/experiments/group/nested/proof.py",
        "evidence/experiments/aggregation/proof.py",
        "evidence/experiments/median.py",
        "evidence/experiments/p001.py",
        "src/relflow/__pycache__/module.py",
        "docs/starlight/src/node_modules/module.ts",
        "native/target/module.rs",
    ],
)
def test_misplaced_files_and_tracked_outputs_are_rejected(tmp_path, name):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("unowned")
    assert violations(tmp_path, POLICY, [name]), name


def test_size_and_symlink_checks_cannot_be_bypassed_by_an_allowed_extension(tmp_path):
    name = "src/relflow/model.py"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * (POLICY["max-bytes"] + 1))
    assert "exceeds" in violations(tmp_path, POLICY, [name])[0]
    path.unlink()
    outside = tmp_path / "outside"
    outside.write_text("source")
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("This platform cannot create unprivileged symlinks")
    assert "symlink" in violations(tmp_path, POLICY, [name])[0]


def test_git_inventory_rejects_force_added_ignored_artifacts(tmp_path, monkeypatch):
    # Commit hooks export Git paths; this fixture must own its repository and index.
    for name in subprocess.check_output(["git", "rev-parse", "--local-env-vars"], text=True).splitlines():
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/tree.py").write_bytes((ROOT / "scripts/tree.py").read_bytes())
    (tmp_path / "pyproject.toml").write_bytes((ROOT / "pyproject.toml").read_bytes())
    (tmp_path / ".gitignore").write_text("evidence/runs/\n")
    artifact = tmp_path / "evidence/runs/result.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}")
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    command = [sys.executable, str(tmp_path / "scripts/tree.py")]
    assert subprocess.run(command, cwd=tmp_path, capture_output=True).returncode == 0
    subprocess.run(["git", "add", "--force", "evidence/runs/result.json"], cwd=tmp_path, check=True)
    rejected = subprocess.run([*command, "--tracked"], cwd=tmp_path, capture_output=True, text=True)
    assert rejected.returncode == 1
    assert "evidence/runs/result.json: generated output" in rejected.stderr


def test_structure_guard_precedes_builds_and_is_required_in_ci_and_hooks():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    assert "structure" in workflow["required"]["needs"]
    assert workflow["package"]["needs"] == workflow["docs"]["needs"] == "structure"
    assert any("scripts/tree.py --tracked" in step.get("run", "") for step in workflow["structure"]["steps"])
    hooks = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())["repos"][0]["hooks"]
    assert hooks[0]["id"] == "tree"
    assert hooks[0]["always_run"] and not hooks[0]["pass_filenames"]
