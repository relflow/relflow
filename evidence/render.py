"""Render recorded evidence and source listings without running experiments."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from evidence.experiments.source import fingerprint, frontmatter, registry


def clean(root: Path) -> None:
    """Remove temporary evidence inputs after Quarto has copied the rendered output."""
    docs = root / "docs/evidence"
    if not docs.exists():
        return
    entries = registry(root / "evidence", lane="all")
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


def cell(item: Any) -> str:
    """Display one recorded leaf literally, preserving nulls and empty values."""
    if item is None or isinstance(item, bool) or item == [] or item == {}:
        text = json.dumps(item)
    elif isinstance(item, float):
        text = f"{item:.6g}"
    else:
        text = str(item)
    return html.escape(text).replace("\n", "<br>")


def table(caption: str, columns: list[str], rows: list[list[Any]]) -> str:
    headings = "".join(f'<th scope="col">{cell(name)}</th>' for name in columns)
    body = "".join(
        '<tr><th scope="row">'
        + cell(row[0])
        + "</th>"
        + "".join(f"<td>{cell(item)}</td>" for item in row[1:])
        + "</tr>"
        for row in rows
    )
    return f"<table><caption>{cell(caption)}</caption><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table>"


def criteria(run: dict | None) -> str:
    """Keep the selected run's complete named criteria visible near the question."""
    if not run or not run.get("checks"):
        return '<p class="evidence-criteria">No behavioral checks have been recorded for this experiment.</p>\n'
    rows = "".join(
        f'<tr><th scope="row">{cell(name)}</th>'
        f'<td class="evidence-check-{"met" if met else "unmet"}">{"Met" if met else "Not met"}</td></tr>'
        for name, met in run["checks"].items()
    )
    return (
        '<div class="evidence-criteria"><table><caption>Predeclared behavioral checks</caption>'
        '<thead><tr><th scope="col">Criterion</th><th scope="col">Selected run</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>\n"
    )


def measurements(metrics: Any, path: tuple = ()) -> str:
    """Present nested measurements as named groups and complete, readable tables."""
    title = ".".join(str(part) for part in path) or "Measurements"
    if isinstance(metrics, dict):
        leaves = [[key, item] for key, item in metrics.items() if not isinstance(item, (dict, list)) or not item]
        groups = [(key, item) for key, item in metrics.items() if isinstance(item, (dict, list)) and item]
        sections = [table(title, ["Measurement", "Value"], leaves)] if leaves else []
    elif isinstance(metrics, list):
        if not metrics:
            return table(title, ["Measurement", "Value"], [["Value", metrics]])
        if metrics and all(
            isinstance(row, dict) and row and all(not isinstance(item, (dict, list)) for item in row.values())
            for row in metrics
        ):
            columns = list(dict.fromkeys(key for row in metrics for key in row))
            return table(
                title,
                ["Row", *columns],
                [
                    [index, *(row[key] if key in row else "Not recorded" for key in columns)]
                    for index, row in enumerate(metrics)
                ],
            )
        if not any(isinstance(item, (dict, list)) for item in metrics):
            return table(title, ["Index", "Value"], [[index, item] for index, item in enumerate(metrics)])
        groups = list(enumerate(metrics))
        sections = []
    else:
        return table(title, ["Measurement", "Value"], [["Value", metrics]])
    for key, item in groups:
        label = (
            f"{key} · {len(item)} {'values' if isinstance(item, list) else 'measurements'}"
            if isinstance(item, (dict, list))
            else str(key)
        )
        sections.append(
            '<details class="evidence-metric-group">'
            f"<summary>{cell(label)}</summary>\n{measurements(item, (*path, key))}\n</details>"
        )
    return "\n".join(sections) if sections else table(title, ["Measurement", "Value"], [["Value", metrics]])


def evidence(entry: dict[str, Any]) -> str:
    sections = []
    run = entry["run"]
    if run is not None:
        recorded = datetime.fromisoformat(run["started_at"])
        sections.append(
            "::: {.evidence-run-summary}\n\n"
            f"Seed **{run['seed']}**, `{run['accelerator']}`, recorded "
            f'<time datetime="{html.escape(run["started_at"], quote=True)}">'
            f"{recorded.strftime('%d %b %Y at %H:%M %Z')}</time>. "
            f"Outcome: **{status(entry)}**.\n\n:::"
        )
        if run.get("error"):
            sections.append(
                '<div class="evidence-error"><p><strong>Execution error</strong></p><pre><code>'
                + html.escape(run["error"])
                + "</code></pre></div>"
            )
        if run.get("metrics"):
            sections.append('<div class="evidence-metrics">\n' + measurements(run["metrics"]) + "\n</div>")
        else:
            sections.append("No measurements were recorded for this run.")
        origin = run.get("provenance", {})
        if origin:
            packages = origin.get("packages", {})
            rows = [
                ["relflow", packages.get("relflow")],
                ["Python", origin.get("python")],
                ["Torch", packages.get("torch")],
                ["Library revision", origin.get("relflow_revision")],
                ["Evidence revision", origin.get("evidence_revision")],
                ["Source fingerprint", origin.get("source_sha256")],
                ["Code fingerprint", origin.get("code_sha256")],
            ]
            sections.append(
                '<details class="evidence-provenance"><summary>Environment and provenance</summary>\n'
                + table(
                    "Recorded environment and source",
                    ["Property", "Recorded value"],
                    [row for row in rows if row[1] is not None],
                )
                + "\n</details>"
            )
        sections.append(
            '<details class="evidence-raw"><summary>Complete recorded JSON</summary>\n<pre><code>'
            + html.escape(json.dumps(run, indent=2, ensure_ascii=False, allow_nan=False))
            + "</code></pre>\n</details>"
        )
    else:
        sections.append(
            "No full experiment has been recorded. Measurements and provenance will appear here after a full run."
        )
    report = entry["report"]
    report_url = report.get("release") or report.get("github_run", {}).get("html_url")
    if report_url:
        sections.append(f"[Complete run report]({report_url}).")
    return "\n\n".join(sections) + "\n"


def presentation(source: str, identifier: str) -> str:
    """Arrange the authored details as tabs without changing executable source."""
    opening = '# <details class="evidence-details">\n# <summary>Experiment details and code</summary>\n'
    measured = f"# {{{{< evidence {identifier} evidence >}}}}"
    reproduction = f"# {{{{< evidence {identifier} script >}}}}"
    code = "# ### Complete experiment code"
    for marker in (opening, measured, reproduction, code, "# </details>"):
        if source.count(marker) != 1:
            raise ValueError(
                f"{identifier}: expected one standard experiment details section; missing or repeated {marker!r}"
            )
    overview, details = source.split(opening, 1)
    _, details = details.split(measured, 1)
    _, details = details.split(reproduction, 1)
    methods, implementation = details.split(code, 1)
    methods = methods.replace("# ### ", "# #### ")
    if not methods.strip("# \n"):
        methods = "#\n# The complete script defines the data, model, training, and evaluation.\n#\n"
    implementation = implementation.replace("# </details>", "# :::\n#\n# ::::\n#\n# </details>", 1)
    visible = f"# ## What this experiment checks\n#\n# {{{{< evidence {identifier} checks >}}}}\n#\n# ## Example"
    return (
        overview.replace("# ## Example", visible, 1)
        + opening
        + "#\n# :::: {.panel-tabset .evidence-tabs}\n#\n# ### Measurements\n#\n"
        + measured
        + "\n#\n# ### Method\n"
        + methods
        + "# ### Reproduce\n#\n"
        + reproduction
        + "\n#\n# ### Code\n#\n"
        + f"# [Download Python script](../_generated/{identifier}.py).\n#\n"
        + '# ::: {.evidence-source tabindex="0" role="region" aria-label="Complete experiment code"}\n'
        + implementation
    )


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


def render(root: Path, *, publication: Path | None = None) -> None:
    docs = root / "docs/evidence"
    entries = registry(root / "evidence", publication=publication, lane="all")
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
            **{
                "evidence-status": state,
                "subtitle": f'[{state}]{{.evidence-status data-evidence-status="{state}"}}',
                "evidence-lane": entry["lane"],
                "code-fold": False,
                "order": group_order * 10000 + int(identifier[1:]),
                "repo-actions": False,
            },
        )
        if entry["lane"] == "research":
            published["subtitle"] += " [Research]{.evidence-lane}"
        write(page, frontmatter(presentation(code, identifier), published))
        kind = "warning" if state in {"Failing", "Error"} else "note"
        write(
            output / f"{identifier}-status.md",
            f'::: {{.evidence-result .evidence-result-{kind} role="note"}}\n'
            f"**{identifier} · {title}**\n\n{summary}\n:::\n\n" + readout(header, run),
        )
        write(output / f"{identifier}-checks.md", criteria(run))
        write(output / f"{identifier}-evidence.md", evidence(entry))
        write(
            output / f"{identifier}-script.md",
            (
                "<p><strong>Research experiment.</strong> Run this explicitly with <code>--lane research</code>. Required resources:</p>\n<ul>"
                + "".join(f"<li>{html.escape(item)}</li>" for item in header["evidence-requires"])
                + "</ul>\n\n"
                if entry["lane"] == "research"
                else ""
            )
            + "Evidence execution is optional and separate from building this site. "
            "Use a relflow build matching the recorded provenance; replace the wheel path below with that artifact. "
            "Run from the relflow/relflow repository's root:\n\n"
            f"```bash\nuv run --project docs --with /path/to/matching-relflow.whl "
            f"python evidence/experiments/run.py {'--lane research ' if entry['lane'] == 'research' else ''}{identifier}\n```\n\n"
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
        selected = os.environ.get("RELFLOW_EVIDENCE_PUBLICATION")
        render(root, publication=Path(selected) if selected else None)
