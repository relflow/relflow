"""Render recorded proof evidence and source listings without running experiments."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import yaml
from source import fingerprint, metadata


def clean(root: Path) -> None:
    """Remove temporary proof inputs after Quarto has copied the rendered output."""
    docs = root / "docs/proofs"
    if not docs.exists():
        return
    entries = yaml.safe_load((root / "proofs/results.yaml").read_text())["proofs"]
    pages = [(root / entry["page"]).resolve() for entry in entries.values()]
    for page in pages:
        if page.parent.parent != docs.resolve() or page.suffix != ".py":
            raise ValueError(f"Cannot clean proof page outside docs/proofs/<family>/: {page}")
    for page in pages:
        page.unlink(missing_ok=True)
    for name in ("_generated", "__pycache__"):
        path = docs / name
        if path.exists():
            shutil.rmtree(path)
    (docs / "catalog.yaml").unlink(missing_ok=True)
    for path in docs.iterdir():
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    if not any(docs.iterdir()):
        docs.rmdir()


def latest(entry: dict[str, Any]) -> dict[str, Any] | None:
    return next((run for run in reversed(entry["runs"]) if run["mode"] == "full"), None)


def status(entry: dict[str, Any]) -> str:
    run = latest(entry)
    if run is None:
        return "Missing"
    if run["outcome"] == "error":
        return "Error"
    if run["outcome"] == "not_met":
        return "Failing"
    return entry["historical"]["status"]


def value(item: Any) -> str:
    if isinstance(item, float):
        return f"{item:.6g}"
    if isinstance(item, list):
        prefix = f"{len(item)} values; final 8: " if len(item) > 8 else ""
        return prefix + ", ".join(value(part) for part in item[-8:])
    if isinstance(item, dict):
        return "; ".join(f"{key}: {value(part)}" for key, part in item.items())
    return str(item).replace("|", "\\|").replace("\n", " ")


def evidence(entry: dict[str, Any]) -> str:
    sections = []
    runs = entry["runs"]
    run = latest(entry)
    if run is not None:
        sections.extend(
            [
                "### Latest full run",
                f"Seed **{run['seed']}**, `{run['accelerator']}`, recorded {run['started_at']}. "
                f"Outcome: **{run['outcome'].replace('_', ' ')}**.",
            ]
        )
        origin = run.get("provenance", {})
        if origin:
            sections.append(
                f"Source fingerprint: `{origin['source_sha256']}`. "
                f"Python {origin['python']}; Torch {origin['packages']['torch']}."
            )
        if run.get("error"):
            sections.append("```text\n" + run["error"] + "\n```")
        if run.get("metrics"):
            sections.append(
                "| Measurement | Value |\n| --- | --- |\n"
                + "\n".join(f"| {value(name)} | {value(measured)} |" for name, measured in run["metrics"].items())
            )
        if run.get("checks"):
            sections.append(
                "<details>\n<summary>Behavioral checks</summary>\n\n"
                "| Behavioral check | Outcome |\n| --- | --- |\n"
                + "\n".join(f"| {value(name)} | {'Met' if met else 'Not met'} |" for name, met in run["checks"].items())
                + "\n\n</details>"
            )
    if runs and runs[-1]["mode"] == "smoke":
        smoke = runs[-1]
        sections.append(
            f"The latest smoke run used seed {smoke['seed']} and a {smoke['steps_override']}-step cap "
            f"per training stage. Execution {'failed' if smoke['outcome'] == 'error' else 'completed'}; "
            "a shortened run does not establish learning capability or change the proof status."
        )
    sections.append("[Recorded results](../_generated/results.yaml).")
    return "\n\n".join(sections) + "\n"


def write(path: Path, content: str) -> None:
    """Preserve unchanged input timestamps so preview rendering can settle."""
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def render(root: Path) -> None:
    docs = root / "docs/proofs"
    source = root / "proofs/results.yaml"
    recorded = source.read_text()
    document = yaml.safe_load(recorded)
    if document.get("schema") != 1:
        raise ValueError(f"{source}: expected evidence schema 1")
    entries = document["proofs"]
    output = docs / "_generated"
    output.mkdir(parents=True, exist_ok=True)
    catalog = []
    pages = set()
    for identifier, entry in entries.items():
        script = root / entry["script"]
        code = script.read_text()
        header = metadata(code)
        if header["proof-id"] != identifier:
            raise ValueError(f"{script}: proof ID {header['proof-id']} does not match {identifier}")
        if header.get("execute", {}).get("enabled") is not False or header.get("execute", {}).get("eval") is not False:
            raise ValueError(f"{script}: proof documentation requires execute.enabled: false and execute.eval: false")
        page = root / entry["page"]
        if page.parent.parent != docs or page.suffix != ".py":
            raise ValueError(f"{identifier}: expected a generated script page under docs/proofs/<family>/")
        if page in pages:
            raise ValueError(f"{identifier}: another proof already uses page {page}")
        pages.add(page)
        page.parent.mkdir(exist_ok=True)
        write(page, code)
        run = latest(entry)
        state = status(entry)
        if run is None:
            title = state
            summary = "No full experiment has been recorded."
        else:
            title = f"Latest full run · {state}"
            met = sum(run.get("checks", {}).values())
            total = len(run.get("checks", {}))
            summary = (
                f"Seed {run['seed']} on `{run['accelerator']}`: {met} of {total} behavioral checks met. "
                "See the measurements below."
            )
            if run["outcome"] == "error":
                summary = "The latest full experiment encountered an execution error. See the recorded error below."
            elif run["outcome"] == "not_met":
                summary += " See the failed checks and their interpretation below."
            recorded_hash = run.get("provenance", {}).get("code_sha256")
            if recorded_hash and recorded_hash != fingerprint(code):
                summary += (
                    " The experiment code has changed since this run; these measurements describe its earlier version."
                )
        kind = "warning" if state in {"Failing", "Error", "Limited"} else "note"
        write(
            output / f"{identifier}-status.md",
            f'::: {{.callout-{kind} title="{identifier} · {title}"}}\n{summary}\n:::\n',
        )
        write(output / f"{identifier}-evidence.md", evidence(entry))
        write(
            output / f"{identifier}-script.md",
            f"Run by stable ID from the repository root:\n\n```bash\nuv run python proofs/run.py {identifier}\n```\n\n"
            f"Or run the self-contained script directly:\n\n```bash\nPYTHONPATH=proofs uv run python {entry['script']}\n```\n\n"
            f"Add `--accelerator gpu` for CUDA or `--seed 42` for another seeded experiment. "
            "`--steps 2` checks execution with a short training budget; it is recorded as a smoke run.\n\n"
            f"[Download the complete proof](../_generated/{identifier}.py).\n",
        )
        write(output / f"{identifier}.py", code)
        catalog.append(
            {
                "id": identifier,
                "title": header["title"],
                "categories": header["categories"],
                "path": page.relative_to(root / "docs").with_suffix(".html").as_posix(),
                "proof-status": state,
                "description": header["description"],
            }
        )
    for page in docs.glob("*/*.py"):
        if page.parent != output and page not in pages:
            page.unlink()
    write(docs / "catalog.yaml", yaml.safe_dump(catalog, sort_keys=False, allow_unicode=True))
    write(output / "results.yaml", recorded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="Remove generated proof inputs after rendering.")
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if options.clean:
        clean(root)
    else:
        render(root)
