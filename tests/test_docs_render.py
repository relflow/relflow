from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

SPEC = importlib.util.spec_from_file_location(
    "render_docs", Path(__file__).resolve().parents[1] / ".github/scripts/render_docs.py"
)
assert SPEC is not None and SPEC.loader is not None
render_docs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(render_docs)


def test_quarto_preparation_preserves_static_examples_and_typst(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    config = {
        "project": {"type": "website", "output-dir": "old-site"},
        "website": {
            "navbar": {
                "left": [{"href": "index.qmd"}],
                "right": [{"text": "Versions", "menu": [{"text": "main"}]}, {"icon": "github"}],
            }
        },
        "filters": ["typst-render"],
        "engine": "jupyter",
        "jupyter": "python3",
        "execute": {"eval": True},
    }
    (docs / "_quarto.yml").write_text(yaml.safe_dump(config))
    body = """# Example

```python
model = rf.Model(amount=rf.Number)
```

```{python .marimo}
raise RuntimeError("Never execute this example")
```

~~~{python}
model.fit()
~~~

```{typst}
#tree(node("customer"))
```

Inline: `{python} model.schema`.
"""
    (docs / "index.qmd").write_text(
        "---\ntitle: Example\nengine: marimo\njupyter: python3\nexecute:\n  eval: true\n"
        "marimo-static-output: true\n---\n" + body
    )
    (docs / "other.qmd").write_text('# Other\n\n```{typst}\n#node("event")\n```\n')

    render_docs.prepare_quarto(tmp_path, "https://example.test/branches/dev/docs/")

    prepared = yaml.safe_load((docs / "_quarto.yml").read_text())
    assert prepared["project"]["output-dir"] == "site"
    assert prepared["website"]["site-url"] == "https://example.test/branches/dev/docs/"
    assert prepared["website"]["navbar"] == config["website"]["navbar"]
    assert prepared["filters"] == ["typst-render"]
    assert prepared["engine"] == "markdown"
    assert prepared["execute"] == {"eval": False}
    assert "jupyter" not in prepared
    page = (docs / "index.qmd").read_text()
    metadata = yaml.safe_load(page.split("---", 2)[1])
    assert metadata == {"title": "Example", "engine": "markdown", "execute": {"eval": False}}
    assert "```python\nmodel = rf.Model(amount=rf.Number)\n```" in page
    assert '```python\nraise RuntimeError("Never execute this example")\n```' in page
    assert "~~~python\nmodel.fit()\n~~~" in page
    assert '```{typst}\n#tree(node("customer"))\n```' in page
    assert "Inline: `model.schema`." in page
    assert "engine: markdown" in (docs / "other.qmd").read_text()


def test_mkdocs_preparation_disables_execution_and_runtime_inspection(tmp_path: Path) -> None:
    config = {
        "site_name": "Old docs",
        "site_url": "https://old.test/",
        "extra": {"version": {"provider": "mike"}, "generator": False},
        "plugins": [
            "search",
            {"mkdocs-jupyter": {"execute": True, "include_source": True}},
            {"mkdocstrings": {"handlers": {"python": {"options": {"show_source": True}}}}},
        ],
    }
    config_path = tmp_path / "mkdocs.yml"
    config_path.write_text(yaml.safe_dump(config))
    render_docs.prepare_mkdocs(tmp_path, "https://example.test/branches/old/docs/")
    prepared = yaml.safe_load(config_path.read_text())
    assert prepared["site_url"] == "https://example.test/branches/old/docs/"
    assert prepared["extra"] == config["extra"]
    assert prepared["plugins"][0] == "search"
    assert prepared["plugins"][1]["mkdocs-jupyter"] == {"execute": False, "include_source": True}
    handler = prepared["plugins"][2]["mkdocstrings"]["handlers"]["python"]
    assert handler["paths"] == ["src"]
    assert handler["options"] == {"show_source": True, "allow_inspection": False, "force_inspection": False}


@pytest.mark.parametrize("builder", ["quarto", "mkdocs"])
def test_render_uses_standalone_documentation_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, builder: str
) -> None:
    (tmp_path / "docs").mkdir()
    built = tmp_path / ("docs/site" if builder == "quarto" else "site")
    built.mkdir()
    (built / "index.html").write_text("<!doctype html>")
    config_path = tmp_path / ("docs/_quarto.yml" if builder == "quarto" else "mkdocs.yml")
    config_path.write_text("project: {}\n" if builder == "quarto" else "site_name: Docs\n")
    run = Mock()
    monkeypatch.setattr(render_docs.subprocess, "run", run)

    render_docs.render(tmp_path)

    run.assert_called_once()
    args, kwargs = run.call_args
    command = args[0]
    assert command[0] == "uvx"
    assert kwargs == {"cwd": tmp_path.resolve(), "check": True}
    assert "relflow" not in " ".join(command)
    if builder == "quarto":
        assert "quarto-cli==1.9.38" in command
        assert command[command.index("quarto") + 1 :] == ["render", "docs", "--no-execute"]
    else:
        assert command[command.index("mkdocs") + 1 :] == ["build", "--site-dir", str(tmp_path / "site")]
        assert any(item.startswith("mkdocs-jupyter") for item in command)
        assert any(item.startswith("mkdocstrings[python]") for item in command)
    assert (tmp_path / "docs/site/index.html").is_file()


def test_render_failure_is_not_silently_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/_quarto.yml").write_text("project: {}\n")
    run = Mock(side_effect=subprocess.CalledProcessError(1, ["quarto"]))
    monkeypatch.setattr(render_docs.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        render_docs.render(tmp_path)

    run.side_effect = None
    with pytest.raises(RuntimeError, match="did not produce docs/site/index.html"):
        render_docs.render(tmp_path)


def test_unknown_docs_layout_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No supported documentation config"):
        render_docs.render(tmp_path)


def test_successful_renderer_with_failed_typst_diagram_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/_quarto.yml").write_text("project: {}\n")

    def build(command: list[str], *, cwd: Path, check: bool) -> None:
        output = cwd / "docs/site"
        (output / "guides").mkdir(parents=True)
        (output / "index.html").write_text("<!doctype html>")
        (output / "guides/model.html").write_text('<div class="typst-render-error">Invalid tree</div>')

    monkeypatch.setattr(render_docs.subprocess, "run", build)
    with pytest.raises(RuntimeError, match="Typst diagram failed to render.*model.html"):
        render_docs.render(tmp_path)
