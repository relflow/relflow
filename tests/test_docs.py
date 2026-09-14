from __future__ import annotations

import ast
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

import relflow as rf

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def anchors(source: Path) -> set[str]:
    text = source.read_text()
    anchors = set(re.findall(r"\{#([^}]+)\}", text))

    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*$", text, flags=re.MULTILINE):
        heading = re.sub(r"\s*\{[^}]*\}\s*$", "", heading)
        heading = re.sub(r"`([^`]*)`", r"\1", heading)
        slug = re.sub(r"[^a-z0-9\s-]", "", heading.lower())
        slug = re.sub(r"\s+", "-", slug).strip("-")
        if slug:
            anchors.add(slug)

    return anchors


def test_docs_disable_python_execution() -> None:
    config = yaml.safe_load((DOCS / "_quarto.yml").read_text())
    assert config["execute"]["eval"] is False
    assert config["engine"] == "markdown"

    for source in DOCS.rglob("*.qmd"):
        text = source.read_text()
        header = re.match(r"\A---\s*\n(.*?)\n---(?:\n|\Z)", text, flags=re.DOTALL)
        metadata = (yaml.safe_load(header[1]) or {}) if header else {}

        assert metadata.get("execute", {}).get("eval") is not True, source
        assert "jupyter" not in metadata, source
        assert metadata.get("engine", config["engine"]) == "markdown", source
        if re.search(r"^(?:`{3,}|~{3,})\s*\{typst\}", text, flags=re.MULTILINE):
            # Quarto's engine inference for Typst cells ignores the project default.
            assert metadata.get("engine") == "markdown", source
        assert not re.search(r"^(?:`{3,}|~{3,})\s*\{[^}\n]*\bpython\b", text, flags=re.MULTILINE), source
        assert not re.search(r"`\{python\}", text), source


def test_static_python_examples_have_valid_syntax() -> None:
    for source in [ROOT / "README.md", *DOCS.rglob("*.qmd")]:
        text = source.read_text()
        for snippet in re.finditer(
            r"^(`{3,}|~{3,})python[ \t]*\n(.*?)^\1[ \t]*$", text, flags=re.MULTILINE | re.DOTALL
        ):
            padding = "\n" * text[: snippet.start(2)].count("\n")
            ast.parse(padding + snippet[2], filename=str(source))


def test_data_examples_are_single_yaml_records() -> None:
    for source in [ROOT / "README.md", *DOCS.rglob("*.qmd")]:
        text = source.read_text()
        assert not re.search(r"^(?:`{3,}|~{3,})json\b", text, flags=re.MULTILINE), source
        for snippet in re.finditer(r"^(`{3,}|~{3,})yaml[ \t]*\n(.*?)^\1[ \t]*$", text, flags=re.MULTILINE | re.DOTALL):
            assert isinstance(yaml.safe_load(snippet[2]), dict), f"{source}: expected one YAML record"


def test_quarto_navigation_and_relative_links_resolve() -> None:
    navigation = (DOCS / "_quarto.yml").read_text()
    hrefs = re.findall(r"^\s*-?\s*(?:href|logo|favicon|image):\s+([^\s]+)", navigation, flags=re.MULTILINE)
    missing_navigation = [href for href in hrefs if not urlsplit(href).scheme and not (DOCS / href).is_file()]
    assert missing_navigation == []

    missing_links: list[tuple[str, str]] = []
    missing_anchors: list[tuple[str, str]] = []
    for source in [ROOT / "README.md", *DOCS.rglob("*.qmd")]:
        text = source.read_text()
        links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
        links.extend(re.findall(r'(?:src|srcset)="([^\"]+)"', text))
        for target in links:
            link = urlsplit(target)
            if link.scheme or link.netloc:
                continue
            pathname = unquote(link.path)
            destination = (source.parent / pathname).resolve() if pathname else source.resolve()
            if not destination.is_file():
                missing_links.append((source.relative_to(ROOT).as_posix(), target))
            elif link.fragment and destination.suffix in {".qmd", ".md"} and link.fragment not in anchors(destination):
                missing_anchors.append((source.relative_to(ROOT).as_posix(), target))

    assert missing_links == []
    assert missing_anchors == []


def test_datatype_references_cover_public_options_and_states() -> None:
    common_fields = set(rf.RequestBase.model_fields)
    for name in ("Boolean", "Category", "Cluster", "DateParts", "Hash", "Number", "Set", "Text", "Vector"):
        request = getattr(rf, name)
        reference = (DOCS / f"data-types/{name.lower()}.qmd").read_text()
        for field_name, field in request.model_fields.items():
            if field_name in common_fields:
                continue
            public_name = field.serialization_alias or field.alias or field_name
            assert f"`{public_name}`" in reference, f"{name}.{public_name} is missing from its reference"

    data_types = (DOCS / "core-concepts/data-types.qmd").read_text()
    assert all(f"`{token.name}`" in data_types for token in rf.Tokens)
    branch = (DOCS / "data-types/branch.qmd").read_text()
    assert all(mode.value in branch for mode in rf.AttentionMode)
