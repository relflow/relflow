from __future__ import annotations

import runpy
import subprocess
from pathlib import Path

import pytest

PUBLISH = runpy.run_path(str(Path(__file__).resolve().parents[1] / ".github/scripts/docs.py"))
discover = PUBLISH["discover"]
assemble = PUBLISH["assemble"]


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "repository"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    for key, value in (("user.name", "Docs Test"), ("user.email", "docs@example.invalid")):
        subprocess.run(["git", "config", key, value], cwd=root, check=True)

    commits = {}
    configurations = {
        "main": "docs/_quarto.yml",
        "dev/a-new-branch": "docs/_quarto.yml",
        "archive/mkdocs": "mkdocs.yml",
        "no-docs": "README.md",
    }
    for branch, filename in configurations.items():
        for config in (root / "docs/_quarto.yml", root / "mkdocs.yml"):
            config.unlink(missing_ok=True)
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"title: {branch}\n")
        subprocess.run(["git", "add", "--all"], cwd=root, check=True)
        subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", branch],
            cwd=root,
            check=True,
        )
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        commits[branch] = commit
        subprocess.run(["git", "update-ref", f"refs/remotes/origin/{branch}", commit], cwd=root, check=True)
    subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"], cwd=root, check=True
    )
    return root, commits


def test_discover_all_documented_branches_at_their_commits(repository: tuple[Path, dict[str, str]]) -> None:
    root, commits = repository
    branches = discover(root)["include"]
    by_name = {branch["name"]: branch for branch in branches}

    assert set(by_name) == {"main", "dev/a-new-branch", "archive/mkdocs"}
    assert branches[0]["name"] == "main"
    assert {name: branch["ref"] for name, branch in by_name.items()} == {
        name: commit for name, commit in commits.items() if name != "no-docs"
    }
    assert by_name["main"]["destination"] == ""
    assert by_name["dev/a-new-branch"]["destination"] == "branches/dev/a-new-branch"
    assert by_name["dev/a-new-branch"]["url"] == "https://relflow.github.io/relflow/branches/dev/a-new-branch/"
    assert len({branch["artifact"] for branch in branches}) == len(branches)
    assert all("/" not in branch["artifact"] for branch in branches)


def test_discover_requires_documented_main(repository: tuple[Path, dict[str, str]]) -> None:
    root, commits = repository
    subprocess.run(["git", "update-ref", "refs/remotes/origin/main", commits["no-docs"]], cwd=root, check=True)

    with pytest.raises(ValueError, match="main"):
        discover(root)


@pytest.fixture
def builds(tmp_path: Path) -> tuple[dict[str, list[dict[str, str]]], Path, Path]:
    artifacts = tmp_path / "artifacts"
    matrix = {
        "include": [
            {"name": "dev/topic", "artifact": "docs-topic", "destination": "branches/dev/topic"},
            {"name": "main", "artifact": "docs-main", "destination": ""},
            {"name": "archive/mkdocs", "artifact": "docs-old", "destination": "branches/archive/mkdocs"},
        ]
    }
    for branch in matrix["include"]:
        output = artifacts / branch["artifact"]
        (output / "assets").mkdir(parents=True)
        (output / "index.html").write_text(f'<a href="guide.html">{branch["name"]}</a>')
        (output / "guide.html").write_text(branch["name"])
        (output / "assets/style.css").write_text(f"/* {branch['name']} */")
    return matrix, artifacts, tmp_path / "site"


def test_assemble_preserves_each_branch_and_its_relative_assets(
    builds: tuple[dict[str, list[dict[str, str]]], Path, Path],
) -> None:
    matrix, artifacts, site = builds

    assemble(matrix, artifacts, site)

    for branch in matrix["include"]:
        output = site / branch["destination"]
        assert (output / "index.html").read_text() == f'<a href="guide.html">{branch["name"]}</a>'
        assert (output / "guide.html").read_text() == branch["name"]
        assert (output / "assets/style.css").read_text() == f"/* {branch['name']} */"


@pytest.mark.parametrize("missing", ["docs-main", "docs-topic"])
def test_assemble_refuses_incomplete_builds_before_creating_site(
    builds: tuple[dict[str, list[dict[str, str]]], Path, Path], missing: str
) -> None:
    matrix, artifacts, site = builds
    (artifacts / missing / "index.html").unlink()

    with pytest.raises(ValueError, match="Missing rendered index.html"):
        assemble(matrix, artifacts, site)

    assert not site.exists()


def test_assemble_refuses_to_overwrite_main_content_with_a_branch(
    builds: tuple[dict[str, list[dict[str, str]]], Path, Path],
) -> None:
    matrix, artifacts, site = builds
    collision = Path("branches/dev/topic/index.html")
    existing = artifacts / "docs-main" / collision
    existing.parent.mkdir(parents=True)
    existing.write_text("Content owned by main")

    with pytest.raises(ValueError, match="overwrite"):
        assemble(matrix, artifacts, site)

    assert (site / collision).read_text() == "Content owned by main"
    assert existing.read_text() == "Content owned by main"
