"""Validate static documentation without importing or executing its examples."""

from __future__ import annotations

import argparse
import ast
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlsplit

import yaml

from evidence.experiments.source import registry

ROOT = Path(__file__).resolve().parents[1]


def markdown(source: Path) -> str:
    """Read ordinary pages and Markdown cells from generated evidence listings."""
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
            if line.strip() and not line.startswith("#"):
                raise ValueError(f"Evidence Markdown cell must contain comments: {source}")
            lines.append(line.removeprefix("#").removeprefix(" "))
    return "\n".join(lines).strip()


def documents(root: Path = ROOT) -> dict[Path, str]:
    """Read authored pages and registered canonical evidence Markdown."""
    docs = root / "docs"
    pages = {
        source.resolve(): markdown(source)
        for source in [root / "README.md", docs / "README.md", *docs.rglob("*.qmd")]
        if source.is_file() and not source.is_relative_to(docs / "site")
    }
    if (root / "evidence/experiments").is_dir():
        entries = registry(root / "evidence")
        pages.update(
            {
                (root / entry["page"]).resolve(): markdown(root / "evidence" / entry["script"])
                for entry in entries.values()
            }
        )
    return pages


def anchors(text: str) -> set[str]:
    found = set(re.findall(r"\{#([^}\s]+)\}", text))
    found.update(re.findall(r'\bid=["\']([^"\']+)["\']', text))
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*$", text, re.MULTILINE):
        heading = re.sub(r"\s*\{[^}]*\}\s*$", "", heading)
        heading = re.sub(r"`([^`]*)`", r"\1", heading)
        slug = re.sub(r"[^a-z0-9\s-]", "", heading.lower())
        slug = re.sub(r"\s+", "-", slug).strip("-")
        if slug:
            found.add(slug)
    return found


def source_checks(root: Path = ROOT) -> list[str]:
    """Check authoring rules, syntax, navigation, and local source references."""
    errors: list[str] = []
    config = yaml.safe_load((root / "docs/_quarto.yml").read_text())
    if config.get("engine") != "markdown" or config.get("execute") != {"enabled": False, "eval": False}:
        errors.append("Project must use the Markdown engine with Python execution disabled")
    pages = documents(root)
    published = {
        source.with_suffix(".html"): text for source, text in pages.items() if source.is_relative_to(root / "docs")
    }
    for source, text in pages.items():
        label = source.relative_to(root).as_posix()
        header = re.match(r"\A---\s*\n(.*?)\n---(?:\n|\Z)", text, re.DOTALL)
        metadata = (yaml.safe_load(header[1]) or {}) if header else {}
        if (
            metadata.get("execute", {}).get("enabled") is True
            or metadata.get("execute", {}).get("eval") is True
            or "jupyter" in metadata
            or (source.suffix != ".py" and metadata.get("engine", "markdown") != "markdown")
        ):
            errors.append(f"{label}: executable document engine is not allowed")
        if re.search(r"^(?:`{3,}|~{3,})\s*\{[^}\n]*\bpython\b|`\{python\}", text, re.MULTILINE):
            errors.append(f"{label}: use plain python fences, not executable Python cells")
        for language in ("python", "yaml"):
            for match in re.finditer(
                rf"^(`{{3,}}|~{{3,}}){language}[ \t]*\n(.*?)^\1[ \t]*$", text, re.MULTILINE | re.DOTALL
            ):
                try:
                    if language == "python":
                        tree = ast.parse(match[2], filename=label)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Call) and ast.unparse(node.func) == "rf.Address":
                                if any(
                                    isinstance(part, ast.Constant) and isinstance(part.value, str) and "/" in part.value
                                    for part in node.args
                                ):
                                    errors.append(f"{label}: pass one child name per Address argument")
                    elif not isinstance(yaml.safe_load(match[2]), dict):
                        errors.append(f"{label}: YAML examples must show one record")
                except (SyntaxError, yaml.YAMLError) as error:
                    errors.append(f"{label}: invalid {language} example: {error}")
        for identifier, part in re.findall(r"\{\{<\s*evidence\s+(\S+)\s+(\S+)\s*>\}\}", text):
            if not (root / f"docs/evidence/_generated/{identifier}-{part}.md").is_file():
                errors.append(f"{label}: missing evidence fragment {identifier}-{part}")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            link = urlsplit(target)
            if link.scheme or link.netloc:
                continue
            pathname = unquote(link.path)
            directory = root / "docs" if pathname.startswith("/") else source.parent
            destination = (directory / pathname.lstrip("/")).resolve() if pathname else source
            if not destination.is_relative_to(root):
                errors.append(f"{label}: local link escapes the documentation repository: {target}")
            elif not destination.is_file() and destination not in published:
                errors.append(f"{label}: missing local link {target}")
            elif link.fragment:
                body = published.get(destination, pages.get(destination))
                if body is None and destination.suffix in {".md", ".qmd"}:
                    body = markdown(destination)
                if body is not None and link.fragment not in anchors(body):
                    errors.append(f"{label}: missing link anchor {target}")
    for href in re.findall(
        r"^\s*-?\s*(?:href|logo|favicon|image):\s+([^\s]+)", (root / "docs/_quarto.yml").read_text(), re.MULTILINE
    ):
        if not urlsplit(href).scheme and not (root / "docs" / href).is_file():
            errors.append(f"Navigation points to a missing file: {href}")
    return errors


class Page(HTMLParser):
    """Collect concrete rendered IDs and resource links without running JavaScript."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: Counter[str] = Counter()
        self.anchors: Counter[str] = Counter()
        self.links: list[str] = []
        self.canonical: list[str] = []
        self.in_head = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "head":
            self.in_head = True
        elif tag == "body":
            self.in_head = False
        attributes = dict(attrs)
        if tag == "link" and "canonical" in (attributes.get("rel") or "").lower().split():
            self.canonical.append(attributes.get("href") or "")
        resource = self.in_head and tag in {"link", "script", "style", "meta"}
        for name, value in attrs:
            if name == "id" and value:
                self.ids[value] += 1
                if not resource:
                    self.anchors[value] += 1
            if name in {"href", "src", "poster"} and value:
                self.links.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag == "head":
            self.in_head = False


def rendered_checks(root: Path = ROOT, *, site_url: str | None = None) -> list[str]:
    """Check rendered links, anchors, and diagrams; an explicit URL also requires canonical tags."""
    root = root.resolve()
    site = root / "docs/site"
    errors: list[str] = []
    if not (site / "index.html").is_file():
        return ["Rendered site is missing docs/site/index.html; run make render first"]
    config = yaml.safe_load((root / "docs/_quarto.yml").read_text())
    address = site_url if site_url is not None else config.get("website", {}).get("site-url", "")
    try:
        location = urlsplit(address)
    except ValueError as error:
        return [f"Invalid rendered site URL {address!r}: {error}"]
    if address and (
        location.scheme not in {"http", "https"} or not location.netloc or location.query or location.fragment
    ):
        return [f"Rendered site URL must be an absolute HTTP(S) URL without query or fragment: {address}"]
    if site_url is not None and not address:
        return ["Rendered site URL must be an absolute HTTP(S) URL without query or fragment: ''"]
    base = unquote(location.path).rstrip("/")
    pages: dict[Path, Page] = {}
    for source in site.rglob("*.html"):
        text = source.read_text()
        page = Page()
        page.feed(text)
        pages[source.resolve()] = page
        if 'class="typst-render-error"' in text or "{{< evidence " in text:
            errors.append(f"{source.relative_to(root)}: rendering failed or a evidence shortcode remains unresolved")
        duplicates = [name for name, count in page.anchors.items() if count > 1]
        if duplicates:
            errors.append(f"{source.relative_to(root)}: duplicate anchors {duplicates}")
    for source, page in pages.items():
        label = source.relative_to(root)
        if site_url is not None and not page.canonical:
            errors.append(f"{label}: missing canonical URL for rendered site {address}")
        if len(page.canonical) > 1:
            errors.append(f"{label}: expected one canonical URL, found {len(page.canonical)}")
        relative = quote(source.relative_to(site).as_posix())
        expected = urljoin(
            address.rstrip("/") + "/", relative.removesuffix("index.html") if source.name == "index.html" else relative
        )
        for canonical in page.canonical:
            if canonical != expected:
                errors.append(f"{label}: incorrect canonical URL {canonical!r}; expected {expected!r}")
        for target in page.links:
            link = urlsplit(target)
            if link.scheme and link.scheme not in {"http", "https"}:
                continue
            if link.netloc and link.netloc.lower() != location.netloc.lower():
                continue
            pathname = unquote(link.path or ("/" if link.netloc else ""))
            if pathname.startswith("/"):
                if base:
                    if pathname != base and not pathname.startswith(base + "/"):
                        errors.append(f"{label}: rendered link escapes site base {base}/: {target}")
                        continue
                    pathname = pathname[len(base) :]
                destination = (site / pathname.lstrip("/")).resolve()
            else:
                destination = (source.parent / pathname).resolve() if pathname else source
            if destination.is_dir():
                destination /= "index.html"
            if not destination.is_relative_to(site) or not destination.is_file():
                errors.append(f"{label}: missing rendered link {target}")
            elif link.fragment and destination in pages and unquote(link.fragment) not in pages[destination].ids:
                errors.append(f"{label}: missing rendered anchor {target}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendered", action="store_true")
    parser.add_argument(
        "--site-url", help="Deployment URL used to validate rendered version paths and canonical links."
    )
    arguments = parser.parse_args()
    if arguments.site_url is not None and not arguments.rendered:
        parser.error("--site-url requires --rendered")
    errors = source_checks()
    if arguments.rendered:
        errors.extend(rendered_checks(site_url=arguments.site_url))
    if errors:
        raise SystemExit("\n".join(errors))
    print(
        "Static documentation checks passed"
        + ("; rendered site links and diagrams passed" if arguments.rendered else "")
    )


if __name__ == "__main__":
    main()
