from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

import relflow as rf

ROOT = Path(__file__).resolve().parents[1]
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
    pages = {source: markdown(source) for source in [ROOT / "README.md", *DOCS.rglob("*.qmd")]}
    entries = yaml.safe_load((ROOT / "proofs/results.yaml").read_text())["proofs"]
    pages.update({ROOT / entry["page"]: markdown(ROOT / entry["script"]) for entry in entries.values()})
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
    evidence = ROOT / "proofs/results.yaml"
    entries = yaml.safe_load(evidence.read_text())["proofs"]
    downloads = {DOCS / "proofs/_generated/results.yaml": evidence}
    downloads.update(
        {DOCS / f"proofs/_generated/{identifier}.py": ROOT / entry["script"] for identifier, entry in entries.items()}
    )
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


def test_datatype_references_cover_public_options_and_states() -> None:
    common_fields = set(rf.RequestBase.model_fields)
    for name in ("Boolean", "Category", "Cluster", "DateParts", "Hash", "Number", "Quantile", "Set", "Text", "Vector"):
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


def test_every_registered_proof_contains_illustrations_and_authored_insights() -> None:
    entries = yaml.safe_load((ROOT / "proofs/results.yaml").read_text())["proofs"]
    documented: Counter[str] = Counter()
    for entry in entries.values():
        page = ROOT / entry["script"]
        text = markdown(page)
        header = re.match(r"\A---\s*\n(.*?)\n---(?:\n|\Z)", text, flags=re.DOTALL)
        assert header, f"{page}: missing proof metadata"
        metadata = yaml.safe_load(header[1])
        identifier = metadata["proof-id"]
        assert identifier in entries, f"{page}: unregistered proof ID {identifier}"
        documented[identifier] += 1
        assert ROOT / entries[identifier]["script"] == page, f"{page}: registered script does not match {identifier}"
        assert metadata.get("title") and metadata.get("description"), page
        assert metadata.get("categories"), f"{page}: missing proof family"
        assert "proof-source" not in metadata and "proof-status" not in metadata, (
            f"{page}: source and status belong in proofs/results.yaml"
        )
        sections = set(re.findall(r"\{\{<\s*proof\s+([^\s>]+)\s+([^\s>]+)\s*>\}\}", text))
        expected = {(identifier, part) for part in ("status", "evidence", "script")}
        assert sections == expected, f"{page}: render recorded evidence and reproduction commands for {identifier}"
        assert entry["historical"]["insights"].strip(), f"{identifier}: missing recorded interpretation"
        assert entry["historical"]["evidence"].strip(), f"{identifier}: missing historical evidence"
        insights = re.search(r"^## Insights\n(.*?)(?=^## |\Z)", text, flags=re.MULTILINE | re.DOTALL)
        assert insights and insights[1].strip(), f"{page}: explain the core insight in the authored script"
        assert "{{<" not in insights[1], f"{page}: insights belong beside the experiment code"
        assert "```{typst}" in text and "#tree(" in text, f"{page}: missing model tree"
        assert "//| fig-alt:" in text, f"{page}: missing tree description"
        assert len(re.findall(r"^```yaml\s*$", text, flags=re.MULTILINE)) >= 3, f"{page}: show three example records"

    assert set(documented) == set(entries), (
        f"proofs missing documentation: {sorted(entries.keys() - documented.keys())}"
    )
    assert all(count == 1 for count in documented.values()), "each proof ID must have exactly one docs page"
