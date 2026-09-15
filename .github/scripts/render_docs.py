"""Render a disposable branch checkout without running its Python examples."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

import yaml


def prepare_quarto(root: Path, site_url: str | None) -> None:
    """Keep branch content and diagrams while removing notebook execution."""
    config_path = root / "docs/_quarto.yml"
    config = yaml.safe_load(config_path.read_text())
    config["engine"] = "markdown"
    config["execute"] = {"enabled": False, "eval": False}
    config.pop("jupyter", None)
    config["project"]["output-dir"] = "site"
    website = config.setdefault("website", {})
    if site_url:
        website["site-url"] = site_url
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    for source in (root / "docs").rglob("*.qmd"):
        text = source.read_text()
        header = re.match(r"\A---\s*\n(.*?)\n---(?:\n|\Z)", text, flags=re.DOTALL)
        metadata = (yaml.safe_load(header[1]) or {}) if header else {}
        body = text[header.end() :] if header else text
        metadata["engine"] = "markdown"
        metadata["execute"] = {"enabled": False, "eval": False}
        metadata.pop("jupyter", None)
        for key in tuple(metadata):
            if key.startswith("marimo-"):
                metadata.pop(key)
        body = re.sub(
            r"^([ \t]*(?:`{3,}|~{3,}))[ \t]*\{python\b[^}\n]*\}[ \t]*$",
            r"\1python",
            body,
            flags=re.MULTILINE,
        )
        body = re.sub(r"`\{python\}\s*([^`\n]+)`", r"`\1`", body)
        source.write_text(f"---\n{yaml.safe_dump(metadata, sort_keys=False)}---\n{body}")


def prepare_mkdocs(root: Path, site_url: str | None) -> None:
    """Render saved notebook outputs and extract API docs from source only."""
    config_path = root / "mkdocs.yml"
    config = yaml.safe_load(config_path.read_text())
    if site_url:
        config["site_url"] = site_url
    for plugin in config.get("plugins", []):
        if not isinstance(plugin, dict):
            continue
        if "mkdocs-jupyter" in plugin:
            plugin["mkdocs-jupyter"]["execute"] = False
        if "mkdocstrings" in plugin:
            handler = plugin["mkdocstrings"].setdefault("handlers", {}).setdefault("python", {})
            handler["paths"] = ["src"]
            handler.setdefault("options", {}).update(allow_inspection=False, force_inspection=False)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))


def render(root: Path, site_url: str | None = None) -> None:
    """Build supported current and historical docs into ``docs/site``."""
    root = root.resolve()
    output = root / "docs/site"
    built = output
    if (root / "docs/_quarto.yml").is_file():
        prepare_quarto(root, site_url)
        command = ["uvx", "--from", "quarto-cli==1.9.38", "quarto", "render", "docs", "--no-execute"]
    elif (root / "mkdocs.yml").is_file():
        prepare_mkdocs(root, site_url)
        # MkDocs rejects an output directory inside its docs source directory.
        built = root / "site"
        command = [
            "uvx",
            "--from",
            "mkdocs==1.6.1",
            "--with",
            "mkdocs-material>=9.6,<10",
            "--with",
            "mkdocs-jupyter>=0.26.3",
            "--with",
            "mkdocstrings[python]>=0.27",
            "mkdocs",
            "build",
            "--site-dir",
            str(built),
        ]
    else:
        raise ValueError(f"No supported documentation config in {root}: expected docs/_quarto.yml or mkdocs.yml")
    subprocess.run(command, cwd=root, check=True)
    if not (built / "index.html").is_file():
        raise RuntimeError(f"Documentation build in {root} did not produce docs/site/index.html")
    for page in built.rglob("*.html"):
        if 'class="typst-render-error"' in page.read_text():
            raise RuntimeError(f"Typst diagram failed to render in {page}")
    if built != output:
        if output.exists():
            shutil.rmtree(output)
        shutil.move(built, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Disposable checkout to prepare and render")
    args = parser.parse_args()
    render(args.root, os.environ.get("DOCS_SITE_URL"))


if __name__ == "__main__":
    main()
