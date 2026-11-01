from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

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
    "site_url",
    [
        "https://relflow.ai/",
        "https://relflow.ai/en/latest/",
        "https://relflow.readthedocs.io/en/v0.4.2/",
        "https://relflow--66.org.readthedocs.build/en/66/",
    ],
)
@pytest.mark.parametrize("page", ["next.html", "guides/index.html"])
def test_rendered_checks_validate_deployment_hosts_version_paths_and_canonical_urls(
    project: Path, site_url: str, page: str
) -> None:
    site = project / "docs/site"
    (site / "assets").mkdir(parents=True)
    (site / "assets/site.js").write_text("// Static asset\n")
    (site / "assets/logo.svg").write_text("<svg />")
    location = urlsplit(site_url)
    (site / "index.html").write_text(
        f'<link rel="canonical" href="{site_url}">'
        f'<script src="{site_url}assets/site.js"></script>'
        f'<img src="{location.path}assets/logo.svg">'
        f'<a href="//{location.netloc}{location.path}{page}#details">Next</a>'
        '<a href="https://external.example/missing.html">External page</a>'
    )
    destination = site / page
    destination.parent.mkdir(exist_ok=True)
    canonical = site_url + ("guides/" if page == "guides/index.html" else page)
    destination.write_text(
        f'<link rel="canonical" href="{canonical}"><h1 id="details">Details</h1><a href="{site_url}">Home</a>'
    )
    assert rendered_checks(project, site_url=site_url) == []


def test_rendered_checks_resolve_encoded_filenames_and_fragments(project: Path) -> None:
    site = project / "docs/site"
    site.mkdir()
    site_url = "https://relflow.ai/en/latest/"
    (site / "index.html").write_text(
        f'<link rel="canonical" href="{site_url}"><a href="{site_url}a%20guide.html#deep%20section">Guide</a>'
    )
    (site / "a guide.html").write_text(
        f'<link rel="canonical" href="{site_url}a%20guide.html"><h1 id="deep section">Details</h1>'
    )
    assert rendered_checks(project, site_url=site_url) == []


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("https://relflow.ai/en/latest/missing.html", "missing rendered link"),
        ("https://relflow.ai/en/latest/#absent", "missing rendered anchor"),
        ("//relflow.ai/en/latest/assets/missing.svg", "missing rendered link"),
        ("/assets/logo.svg", "escapes site base"),
        ("/en/latestish/assets/logo.svg", "escapes site base"),
        ("https://relflow.ai/en/v0.4.2/", "escapes site base"),
        ("https://relflow.ai/", "escapes site base"),
        ("/en/latest/%2e%2e/outside.html", "missing rendered link"),
    ],
)
def test_rendered_checks_reject_broken_internal_absolute_links_and_version_escapes(
    project: Path, target: str, message: str
) -> None:
    site = project / "docs/site"
    (site / "assets").mkdir(parents=True)
    (site / "assets/logo.svg").write_text("<svg />")
    (site / "index.html").write_text(
        f'<link rel="canonical" href="https://relflow.ai/en/latest/"><a href="{target}">Broken internal target</a>'
    )
    assert any(message in error for error in rendered_checks(project, site_url="https://relflow.ai/en/latest/"))


@pytest.mark.parametrize(
    "canonical",
    [
        "https://previous.example/en/latest/",
        "https://relflow.ai/",
        "https://relflow.ai/en/v0.4.2/",
        "https://relflow.ai/en/latest/next.html",
        "https://relflow.ai/en/latest/?old=true",
        "https://relflow.ai/en/latest/#fragment",
    ],
)
def test_rendered_checks_reject_canonical_urls_for_other_hosts_versions_or_pages(project: Path, canonical: str) -> None:
    site = project / "docs/site"
    site.mkdir()
    (site / "index.html").write_text(f'<link rel="canonical" href="{canonical}">')
    assert any(
        "incorrect canonical URL" in error
        for error in rendered_checks(project, site_url="https://relflow.ai/en/latest/")
    )


@pytest.mark.parametrize(
    ("canonical", "message"),
    [
        ("", "missing canonical URL"),
        (
            '<link rel="canonical" href="https://relflow.ai/en/latest/">'
            '<link rel="canonical" href="https://relflow.ai/en/latest/">',
            "expected one canonical URL",
        ),
    ],
)
def test_rendered_checks_require_one_canonical_url_for_an_explicit_deployment(
    project: Path, canonical: str, message: str
) -> None:
    site = project / "docs/site"
    site.mkdir()
    (site / "index.html").write_text(canonical)
    assert any(message in error for error in rendered_checks(project, site_url="https://relflow.ai/en/latest/"))


@pytest.mark.parametrize("site_url", ["", "/en/latest/", "ftp://relflow.ai/", "https://relflow.ai/?old=true"])
def test_rendered_checks_reject_invalid_deployment_urls(project: Path, site_url: str) -> None:
    site = project / "docs/site"
    site.mkdir()
    (site / "index.html").write_text("<h1>Start</h1>")
    assert any("absolute HTTP(S) URL" in error for error in rendered_checks(project, site_url=site_url))


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
