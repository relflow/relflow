"""Render recorded evidence and source listings without running experiments."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Any

from evidence.experiments.source import fingerprint, frontmatter, registry


def clean(root: Path) -> None:
    """Remove temporary evidence inputs after Quarto has copied the rendered output."""
    docs = root / "docs/evidence"
    if not docs.exists():
        return
    entries = registry(root / "evidence")
    pages = [(root / entry["page"]).resolve() for entry in entries.values()]
    for page in pages:
        if page.parent.parent != docs.resolve() or page.suffix != ".py":
            raise ValueError(f"Cannot clean evidence page outside docs/evidence/<family>/: {page}")
    for page in pages:
        page.unlink(missing_ok=True)
        page.with_suffix(".quarto_ipynb").unlink(missing_ok=True)
        for temporary in page.parent.glob(f"{page.stem}.quarto_ipynb_*"):
            temporary.unlink()
    for name in ("_generated", "__pycache__"):
        path = docs / name
        if path.exists():
            shutil.rmtree(path)
    for path in docs.iterdir():
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    if not any(docs.iterdir()):
        docs.rmdir()


def status(entry: dict[str, Any]) -> str:
    run = entry["run"]
    if run is None:
        return "Missing"
    if run["outcome"] == "error":
        return "Error"
    if run["outcome"] == "not_met":
        return "Failing"
    return "Passing"


def value(item: Any) -> str:
    if isinstance(item, float):
        return f"{item:.6g}"
    if isinstance(item, list):
        prefix = f"{len(item)} values; final 8: " if len(item) > 8 else ""
        return prefix + ", ".join(value(part) for part in item[-8:])
    if isinstance(item, dict):
        return "; ".join(f"{key}: {value(part)}" for key, part in item.items())
    return str(item).replace("|", "\\|").replace("\n", " ")


def readout(header: dict[str, Any], run: dict[str, Any] | None) -> str:
    """Present a few authored comparisons directly from the recorded metrics."""
    rows = []
    formats = set()
    for selection in header.get("evidence-readout", []):
        label = selection["label"]
        path = selection["metric"]
        style = selection["format"]
        if not isinstance(path, list) or not path or style not in {"number", "percent", "error", "auc"}:
            raise ValueError(f"{header['evidence-id']}: readout {label!r} requires a metric path and supported format")
        measured = run.get("metrics", {}) if run else {}
        try:
            for part in path:
                measured = measured[part]
        except (KeyError, IndexError, TypeError):
            displayed = "Not recorded in this run"
        else:
            if isinstance(measured, bool) or not isinstance(measured, (int, float)):
                raise ValueError(
                    f"{header['evidence-id']}: readout {label!r} requires a numerical metric, got {measured!r}"
                )
            if not math.isfinite(measured):
                displayed = "Not available"
            elif style == "error":
                displayed = f"{measured * 100:.1f}% of baseline error"
            elif style == "percent":
                displayed = f"{measured * 100:.1f}%"
            elif style == "auc":
                displayed = f"{measured:.4f}"
            else:
                displayed = f"{measured:.4g}"
        rows.append(f"| {value(label)} | {displayed} |")
        formats.add(style)
    if not rows or run is None:
        return ""
    sections = ["| Comparison | Current measurement |\n| --- | --- |\n" + "\n".join(rows)]
    if "error" in formats:
        sections.append(
            "Baseline error compares prediction error with a constant guess. "
            "100% matches that guess; lower is better. Details specify the baseline and root-mean-square error calculation."
        )
    if "auc" in formats:
        sections.append(
            "These are AUC ranking scores: 0.5 is chance and 1 is perfect separation. They are not percentages of correct answers."
        )
    return "\n\n".join(sections) + "\n"


def evidence(entry: dict[str, Any]) -> str:
    sections = []
    run = entry["run"]
    if run is not None:
        sections.extend(
            [
                "### Current full run",
                f"Seed **{run['seed']}**, `{run['accelerator']}`, recorded {run['started_at']}. "
                f"Outcome: **{run['outcome'].replace('_', ' ')}**.",
            ]
        )
        origin = run.get("provenance", {})
        if origin:
            sections.append(
                f"Source fingerprint: `{origin['source_sha256']}`. "
                f"RelFlow {origin['packages']['relflow']}; Python {origin['python']}; "
                f"Torch {origin['packages']['torch']}. "
                f"Library revision: `{origin['relflow_revision']}`; "
                f"evidence revision: `{origin['evidence_revision']}`."
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
    report = entry["header"].get("evidence-report", {})
    report_url = report.get("release") or report.get("github_run", {}).get("html_url")
    if report_url:
        sections.append(f"[Complete run report]({report_url}).")
    return "\n\n".join(sections) + "\n"


def write(path: Path, content: str) -> None:
    """Preserve unchanged input timestamps so preview rendering can settle."""
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


GROUP_ORDER = [
    "Aggregation",
    "Structure",
    "State",
    "Identity",
    "Relational",
    "Representation",
    "Vocabulary",
    "Masking",
    "Mutations",
    "Temporal",
    "Composition",
    "Calibration",
    "Anomaly",
]


def render(root: Path) -> None:
    docs = root / "docs/evidence"
    entries = registry(root / "evidence")
    output = docs / "_generated"
    output.mkdir(parents=True, exist_ok=True)
    pages = set()
    for identifier, entry in entries.items():
        script = root / "evidence" / entry["script"]
        code = script.read_text()
        header = entry["header"]
        if header["evidence-id"] != identifier:
            raise ValueError(f"{script}: evidence ID {header['evidence-id']} does not match {identifier}")
        if header.get("execute", {}).get("enabled") is not False or header.get("execute", {}).get("eval") is not False:
            raise ValueError(
                f"{script}: evidence documentation requires execute.enabled: false and execute.eval: false"
            )
        page = root / entry["page"]
        if page.parent.parent != docs or page.suffix != ".py":
            raise ValueError(f"{identifier}: expected a generated script page under docs/evidence/<family>/")
        if page in pages:
            raise ValueError(f"{identifier}: another experiment already uses page {page}")
        pages.add(page)
        page.parent.mkdir(exist_ok=True)
        run = entry["run"]
        if run is not None and run["mode"] != "full":
            raise ValueError(f"{identifier}: only full runs may be published")
        state = status(entry)
        if run is None:
            title = state
            summary = "No full experiment has been recorded."
        else:
            title = f"Current full run · {state}"
            met = sum(run.get("checks", {}).values())
            total = len(run.get("checks", {}))
            summary = f"Seed {run['seed']} on `{run['accelerator']}`: {met} of {total} test checks met."
            if run["outcome"] == "error":
                summary = "The current full experiment encountered an execution error. See the recorded error below."
            elif run["outcome"] == "not_met":
                summary += " The experiment did not meet every criterion."
            recorded_hash = run.get("provenance", {}).get("code_sha256")
            if recorded_hash and recorded_hash != fingerprint(code):
                summary += (
                    " The experiment code has changed since this run; these measurements describe its earlier version."
                )
        group = header["evidence-group"]
        group_order = GROUP_ORDER.index(group) if group in GROUP_ORDER else len(GROUP_ORDER)
        published = dict(
            header,
            **{"evidence-status": state, "order": group_order * 10000 + int(identifier[1:]), "repo-actions": ["issue"]},
        )
        write(page, frontmatter(code, published))
        kind = "warning" if state in {"Failing", "Error"} else "note"
        write(
            output / f"{identifier}-status.md",
            f'::: {{.evidence-result .evidence-result-{kind} role="note"}}\n'
            f"**{identifier} · {title}**\n\n{summary}\n:::\n\n" + readout(header, run),
        )
        write(output / f"{identifier}-evidence.md", evidence(entry))
        write(
            output / f"{identifier}-script.md",
            "Evidence execution is optional and separate from building this site. "
            "Use a RelFlow build matching the recorded provenance; replace the wheel path below with that artifact. "
            "Run from the relflow/relflow repository's root:\n\n"
            f"```bash\nuv run --project docs --with /path/to/matching-relflow.whl "
            f"python evidence/experiments/run.py {identifier}\n```\n\n"
            "Or, from a prepared evidence environment, run the self-contained script directly:\n\n"
            f"```bash\nPYTHONPATH=evidence/experiments python evidence/{entry['script']}\n```\n\n"
            f"Add `--accelerator gpu` for CUDA or `--seed 42` for another seeded experiment. "
            "`--steps 2` checks execution with a short training budget; it is recorded as a smoke run.\n\n"
            f"[Download the complete evidence](../_generated/{identifier}.py).\n",
        )
        write(output / f"{identifier}.py", code)
    for page in docs.glob("*/*.py"):
        if page.parent != output and page not in pages:
            page.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="Remove generated evidence inputs after rendering.")
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if options.clean:
        clean(root)
    else:
        render(root)
