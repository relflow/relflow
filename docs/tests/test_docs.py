from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

from evidence.experiments.source import registry

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def markdown(source: Path) -> str:
    """Read authored prose without treating experiment code as documentation."""
    text = source.read_text()
    if source.suffix != ".py":
        return text
    lines: list[str] = []
    active = False
    for line in text.splitlines():
        if line.startswith("# %%"):
            active = line.strip() == "# %% [markdown]"
            if active:
                lines.append("")
        elif active:
            assert not line.strip() or line.startswith("#"), f"{source}: Markdown cells must contain comments"
            lines.append(line.removeprefix("#").removeprefix(" "))
    return "\n".join(lines).strip()


def documents() -> dict[Path, str]:
    """Resolve canonical authored sources in their published page directories."""
    pages = {
        source: markdown(source)
        for source in [ROOT / "README.md", DOCS / "README.md", *DOCS.rglob("*.qmd")]
        if source.is_file() and not source.is_relative_to(DOCS / "site")
    }
    entries = registry(ROOT / "evidence")
    pages.update({ROOT / entry["page"]: markdown(ROOT / "evidence" / entry["script"]) for entry in entries.values()})
    return pages


def anchors(text: str) -> set[str]:
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
    assert config["execute"]["enabled"] is False
    assert config["execute"]["eval"] is False
    assert config["engine"] == "markdown"

    for source, text in documents().items():
        header = re.match(r"\A---\s*\n(.*?)\n---(?:\n|\Z)", text, flags=re.DOTALL)
        metadata = (yaml.safe_load(header[1]) or {}) if header else {}

        assert metadata.get("execute", {}).get("enabled") is not True, source
        assert metadata.get("execute", {}).get("eval") is not True, source
        assert "jupyter" not in metadata, source
        if source.suffix == ".py":
            assert metadata["execute"] == {"enabled": False, "eval": False}, source
            assert metadata["code-fold"] is True, source
        else:
            assert metadata.get("engine", config["engine"]) == "markdown", source
            if re.search(r"^(?:`{3,}|~{3,})\s*\{typst\}", text, flags=re.MULTILINE):
                # Quarto's engine inference for Typst cells ignores the project default.
                assert metadata.get("engine") == "markdown", source
        assert not re.search(r"^(?:`{3,}|~{3,})\s*\{[^}\n]*\bpython\b", text, flags=re.MULTILINE), source
        assert not re.search(r"`\{python\}", text), source


def test_diagram_roots_have_descriptive_labels() -> None:
    for source, text in documents().items():
        for label in re.findall(r'\bnode\("([^\"]*)",\s*kind:\s*"root"', text):
            assert label.strip() and not label.strip().startswith("/"), (
                f"{source}: root labels describe the observation"
            )


def test_static_python_examples_have_valid_syntax_and_address_components() -> None:
    for source, text in documents().items():
        for snippet in re.finditer(
            r"^(`{3,}|~{3,})python[ \t]*\n(.*?)^\1[ \t]*$", text, flags=re.MULTILINE | re.DOTALL
        ):
            padding = "\n" * text[: snippet.start(2)].count("\n")
            tree = ast.parse(padding + snippet[2], filename=str(source))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and ast.unparse(node.func).startswith("rf.")
                    and ast.unparse(node.func).endswith("DataModule")
                ):
                    assert not node.args and all(keyword.arg != "model" for keyword in node.keywords), (
                        f"{source}:{node.lineno}: let the model loop bind the data module automatically"
                    )
                if not isinstance(node, ast.Call) or ast.unparse(node.func) != "rf.Address":
                    continue
                for part in node.args:
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        assert "/" not in part.value, (
                            f"{source}:{node.lineno}: pass one child name per Address argument"
                        )


def test_data_examples_are_single_yaml_records() -> None:
    for source, text in documents().items():
        assert not re.search(r"^(?:`{3,}|~{3,})json\b", text, flags=re.MULTILINE), source
        for snippet in re.finditer(r"^(`{3,}|~{3,})yaml[ \t]*\n(.*?)^\1[ \t]*$", text, flags=re.MULTILINE | re.DOTALL):
            assert isinstance(yaml.safe_load(snippet[2]), dict), f"{source}: expected one YAML record"


def test_quarto_navigation_and_relative_links_resolve() -> None:
    pages = documents()
    published = {source.with_suffix(".html"): text for source, text in pages.items() if source.is_relative_to(DOCS)}
    entries = registry(ROOT / "evidence")
    downloads = {
        DOCS / f"evidence/_generated/{identifier}.py": ROOT / "evidence" / entry["script"]
        for identifier, entry in entries.items()
    }
    navigation = (DOCS / "_quarto.yml").read_text()
    hrefs = re.findall(r"^\s*-?\s*(?:href|logo|favicon|image):\s+([^\s]+)", navigation, flags=re.MULTILINE)
    missing_navigation = [
        href
        for href in hrefs
        if not urlsplit(href).scheme and not (DOCS / href).is_file() and DOCS / href not in published
    ]
    assert missing_navigation == []

    missing_links: list[tuple[str, str]] = []
    missing_anchors: list[tuple[str, str]] = []
    for source, text in pages.items():
        links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
        links.extend(re.findall(r'(?:src|srcset)="([^\"]+)"', text))
        for target in links:
            link = urlsplit(target)
            if link.scheme or link.netloc:
                continue
            pathname = unquote(link.path)
            directory = DOCS if pathname.startswith("/") else source.parent
            destination = (directory / pathname.lstrip("/")).resolve() if pathname else source.resolve()
            destination = downloads.get(destination, destination)
            if not destination.is_file() and destination not in published and destination not in pages:
                missing_links.append((source.relative_to(ROOT).as_posix(), target))
            elif link.fragment:
                target_text = published.get(destination, pages.get(destination))
                if target_text is None and destination.suffix in {".qmd", ".md"}:
                    target_text = markdown(destination)
                if target_text is not None and link.fragment not in anchors(target_text):
                    missing_anchors.append((source.relative_to(ROOT).as_posix(), target))

    assert missing_links == []
    assert missing_anchors == []


def test_every_registered_evidence_has_a_small_readable_overview() -> None:
    entries = registry(ROOT / "evidence")
    documented: Counter[str] = Counter()
    for entry in entries.values():
        page = ROOT / "evidence" / entry["script"]
        text = markdown(page)
        header = re.match(r"\A---\s*\n(.*?)\n---(?:\n|\Z)", text, flags=re.DOTALL)
        assert header, f"{page}: missing evidence metadata"
        metadata = yaml.safe_load(header[1])
        identifier = metadata["evidence-id"]
        assert identifier in entries, f"{page}: unregistered evidence ID {identifier}"
        documented[identifier] += 1
        assert ROOT / "evidence" / entries[identifier]["script"] == page, (
            f"{page}: registered script does not match {identifier}"
        )
        assert metadata.get("title") and metadata.get("description"), page
        assert metadata.get("categories"), f"{page}: missing evidence family"
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
        f"evidence missing documentation: {sorted(entries.keys() - documented.keys())}"
    )
    assert all(count == 1 for count in documented.values()), "each evidence ID must have exactly one docs page"
