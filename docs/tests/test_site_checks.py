from __future__ import annotations

from pathlib import Path

import pytest

from docs.check import documents, rendered_checks, source_checks


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path.resolve()
    docs = root / "docs"
    docs.mkdir()
    (docs / "_quarto.yml").write_text(
        "engine: markdown\nexecute: {enabled: false, eval: false}\n"
        "website:\n  site-url: https://relflow.github.io/docs/\n"
        "  sidebar:\n    contents:\n      - href: index.qmd\n"
    )
    (root / "README.md").write_text("# RelFlow\n\n[Documentation](docs/README.md)\n")
    (docs / "README.md").write_text("# Documentation\n\n[Start](index.qmd)\n")
    (docs / "index.qmd").write_text("# Start\n\n[More](next.qmd#details)\n")
    (docs / "next.qmd").write_text("# Next\n\n## Details\n")
    return root


def test_static_checks_resolve_the_site_inside_its_own_repository(project: Path) -> None:
    assert source_checks(project) == []
    assert {path.relative_to(project).as_posix() for path in documents(project)} == {
        "README.md",
        "docs/README.md",
        "docs/index.qmd",
        "docs/next.qmd",
    }


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[Missing](absent.qmd)", "missing local link"),
        ("[Missing anchor](next.qmd#absent)", "missing link anchor"),
        ("[Escape](../../outside.md)", "escapes the documentation repository"),
        ("```{python}\nprint('executed')\n```", "plain python fences"),
        ("```python\nvalue = (\n```", "invalid python example"),
        ("```python\naddress = rf.Address('history/amount')\n```", "one child name"),
        ("```yaml\n- amount: 1\n```", "YAML examples must show one record"),
        ("{{< evidence P999 evidence >}}", "missing evidence fragment"),
    ],
)
def test_static_checks_reject_broken_authoring(project: Path, content: str, message: str) -> None:
    (project / "docs/index.qmd").write_text("# Start\n\n" + content + "\n")
    assert any(message in error for error in source_checks(project))


def test_static_checks_do_not_execute_python_examples(project: Path) -> None:
    (project / "docs/index.qmd").write_text(
        "# Start\n\n```python\nraise RuntimeError('documentation must stay static')\n```\n"
    )
    assert source_checks(project) == []


def test_static_checks_reject_a_document_execution_override(project: Path) -> None:
    (project / "docs/index.qmd").write_text("---\nexecute: {eval: true}\n---\n\n# Start\n")
    assert any("executable document" in error for error in source_checks(project))


def test_rendered_checks_resolve_github_pages_base_paths_and_real_anchors(project: Path) -> None:
    site = project / "docs/site"
    site.mkdir()
    (site / "index.html").write_text('<a href="/docs/next.html#details">Next</a>')
    (site / "next.html").write_text('<h1 id="details">Details</h1><a href="index.html">Back</a>')
    assert rendered_checks(project) == []


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('<a href="missing.html">Missing</a>', "missing rendered link"),
        ('<a href="#missing">Missing</a>', "missing rendered anchor"),
        ('<h1 id="duplicate"></h1><p id="duplicate"></p>', "duplicate anchors"),
        ('<div class="typst-render-error">Failure</div>', "rendering failed"),
        ("{{< evidence P001 status >}}", "shortcode remains unresolved"),
    ],
)
def test_rendered_checks_reject_broken_output(project: Path, content: str, message: str) -> None:
    site = project / "docs/site"
    site.mkdir()
    (site / "index.html").write_text(content)
    assert any(message in error for error in rendered_checks(project))


def test_build_configuration_has_no_machine_learning_dependencies() -> None:
    import tomllib

    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root / "docs/pyproject.toml").read_text())
    assert config["project"]["dependencies"] == ["pyyaml==6.0.3"]
    assert config["tool"]["uv"]["package"] is False
    for name in ("docs/check.py", "evidence/render.py"):
        text = (root / name).read_text()
        assert "import relflow" not in text and "import torch" not in text


def test_theme_resources_can_share_ids_while_content_anchors_stay_unique(project: Path) -> None:
    site = project / "docs/site"
    site.mkdir()
    for name in ("light.css", "dark.css"):
        (site / name).write_text("body {}")
    html = (
        '<html><head><link id="theme" rel="stylesheet" href="light.css">'
        '<link id="theme" rel="stylesheet" href="dark.css"></head>'
        '<body><h1 id="intro">Introduction</h1><a href="#theme">Theme resource</a>'
    )
    (site / "index.html").write_text(html + "</body></html>")
    assert rendered_checks(project) == []

    (site / "index.html").write_text(html + '<h2 id="intro">Duplicate heading</h2></body></html>')
    assert any("duplicate anchors ['intro']" in error for error in rendered_checks(project))
