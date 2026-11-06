"""Current measured results drive static evidence pages without model imports."""

from __future__ import annotations

import builtins
import importlib.util
import json
import os
import sys
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from evidence.experiments.source import metadata, registry

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evidence"))


class Tables(HTMLParser):
    """Read semantic table cells without depending on generated whitespace."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.tables = []
        self.tags = []
        self.table = None
        self.row = None
        self.cell = None
        self.caption = False
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self.tags.append((tag, attributes))
        if tag == "table":
            self.table = {"caption": "", "rows": []}
            self.tables.append(self.table)
        elif tag == "caption":
            self.caption = True
        elif tag == "tr":
            self.row = []
        elif tag in {"th", "td"}:
            self.cell = {"tag": tag, "attributes": attributes, "text": ""}
        elif tag == "br":
            if self.cell is not None:
                self.cell["text"] += "\n"
            elif self.caption and self.table is not None:
                self.table["caption"] += "\n"

    def handle_data(self, data):
        if self.caption and self.table is not None:
            self.table["caption"] += data
        if self.cell is not None:
            self.cell["text"] += data

    def handle_endtag(self, tag):
        if tag == "caption":
            self.caption = False
        elif tag in {"th", "td"} and self.cell is not None:
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table["rows"].append(self.row)
            self.row = None
        elif tag == "table":
            self.table = None

    def values(self):
        return [[cell["text"] for cell in row] for table in self.tables for row in table["rows"]]


def load(path: Path) -> ModuleType:
    """Load the standalone reporting tools without importing experiment scripts."""
    spec = importlib.util.spec_from_file_location(f"evidence_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(*, seed: int = 11, outcome: str = "met", mode: str = "full") -> dict:
    """Make a recorded experiment with enough provenance for its public report."""
    return {
        "started_at": "2026-09-14T12:00:00+00:00",
        "mode": mode,
        "seed": seed,
        "steps_override": 2 if mode == "smoke" else None,
        "accelerator": "cpu",
        "provenance": {
            "source_sha256": f"source-{seed}",
            "python": "3.12.0",
            "packages": {"torch": "2.8.0", "relflow": "0.4.1"},
            "relflow_revision": "tested-library",
            "evidence_revision": "tested-evidence",
        },
        "metrics": {"held_out_rmse": 0.125},
        "checks": {"Held-out RMSE is below 0.2": outcome == "met"},
        "outcome": outcome,
        "duration_seconds": 0.01,
    }


def set_result(identifier: str, recorded: dict, root: Path) -> None:
    path = root / "evidence/results.json"
    document = (
        json.loads(path.read_text())
        if path.exists()
        else {"schema": 1, "version": "test", "revision": "tested", "report": {}, "results": {}}
    )
    document["results"][identifier] = recorded
    path.write_text(json.dumps(document))


@pytest.fixture
def evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType, Path]:
    reporting = None
    renderer = load(ROOT / "evidence/render.py")
    scripts = tmp_path / "evidence/experiments/family"
    pages = tmp_path / "docs/evidence/family"
    scripts.mkdir(parents=True)
    pages.mkdir(parents=True)
    for identifier, slug in (("P001", "signal"), ("P002", "noise")):
        script = scripts / f"{slug}.py"
        metadata = {
            "evidence-group": "Family",
            "title": slug.title(),
            "categories": ["Family"],
            "description": slug,
            "execute": {"enabled": False, "eval": False},
        }
        header = (
            "---\n" + yaml.safe_dump(metadata) + "---\n\n# Static experiment\n\n## Example\n\nA static example.\n"
            "\n## Comparison\n\nA static control.\n\n## Result\n\nA measured limit.\n"
            '\n<details class="evidence-details">\n<summary>Experiment details and code</summary>\n\n'
            f"{{{{< evidence {identifier} evidence >}}}}\n\n### Run this experiment\n\n"
            f"{{{{< evidence {identifier} script >}}}}\n\n### Data and evaluation\n\n"
            "Independent held-out observations.\n\n### Complete experiment code\n"
        )
        script.write_text(
            "# %% [markdown]\n" + "\n".join("# " + line for line in header.splitlines()) + "\n\n# %%\n"
            '"""Literal Markdown example: ```python and ````.\n"""\n'
            "from pathlib import Path\n"
            f"Path({str(tmp_path / 'executed')!r}).write_text('executed')\n"
            "raise RuntimeError('a documentation build executed a evidence')\n"
            "from reporting import experiment, report\n"
            f"@experiment({identifier!r}, seed=7)\n"
            "def run(seed, steps, accelerator):\n    return {}, {'signal': True}\n"
            "if __name__ == '__main__':\n    report(run)\n"
            "\n# %% [markdown]\n# </details>\n"
        )
    return reporting, renderer, tmp_path


@pytest.mark.parametrize(
    ("outcome", "expected", "kind"),
    [
        ("met", "Passing", "note"),
        ("not_met", "Failing", "warning"),
        ("error", "Error", "warning"),
        (None, "Missing", "note"),
    ],
)
def test_statuses_match_page_badges_catalog_callouts_legend_and_colors(
    evidence: tuple[ModuleType, ModuleType, Path], outcome: str | None, expected: str, kind: str
) -> None:
    _, renderer, root = evidence
    if outcome is not None:
        set_result("P001", result(outcome=outcome), root)

    renderer.render(root)

    published = metadata((root / "docs/evidence/family/signal.py").read_text())
    assert published["evidence-status"] == expected
    badge = f'[{expected}]{{.evidence-status data-evidence-status="{expected}"}}'
    assert published["subtitle"] == badge
    assert published["repo-actions"] is False
    notice = (root / "docs/evidence/_generated/P001-status.md").read_text()
    title = f"Current full run · {expected}" if outcome is not None else expected
    assert f'::: {{.evidence-result .evidence-result-{kind} role="note"}}' in notice
    assert f"**P001 · {title}**" in notice
    legend = (ROOT / "docs/evidence.qmd").read_text()
    assert badge in legend
    styles = (ROOT / "docs/assets/stylesheets/evidence.css").read_text()
    assert f'\n.evidence-status[data-evidence-status="{expected}"] {{' in styles
    assert f'\nbody.quarto-dark .evidence-status[data-evidence-status="{expected}"] {{' in styles


def test_docs_refresh_from_yaml_and_embed_scripts_without_executing(
    evidence: tuple[ModuleType, ModuleType, Path],
) -> None:
    reporting, renderer, root = evidence
    generated = root / "docs/evidence/_generated"
    renderer.render(root)
    notice = (generated / "P001-status.md").read_text()
    assert "Missing" in notice and "No full experiment has been recorded." in notice
    assert "Historical summary" not in notice
    assert "Historical measurements" not in (generated / "P001-evidence.md").read_text()

    failed = result(outcome="not_met")
    failed["metrics"] = {"held_out_rmse": 0.875}
    set_result("P001", failed, root)
    renderer.render(root)
    assert metadata((root / "docs/evidence/family/signal.py").read_text())["evidence-status"] == "Failing"
    assert metadata((root / "docs/evidence/family/noise.py").read_text())["evidence-status"] == "Missing"
    assert "0 of 1 test checks met" in (generated / "P001-status.md").read_text()
    assert ["held_out_rmse", "0.875"] in Tables((generated / "P001-evidence.md").read_text()).values()

    set_result("P001", result(seed=29), root)
    renderer.render(root)
    assert "Passing" in (generated / "P001-status.md").read_text()
    text = (generated / "P001-evidence.md").read_text()
    assert "Seed **29**" in text and ["held_out_rmse", "0.125"] in Tables(text).values()
    assert ["relflow", "0.4.1"] in Tables(text).values()
    assert "tested-library" in text and "tested-evidence" in text
    assert ["held_out_rmse", "0.875"] not in Tables(text).values()
    assert ["Held-out RMSE is below 0.2", "Met"] not in Tables(text).values()
    criteria = (generated / "P001-checks.md").read_text()
    assert ["Held-out RMSE is below 0.2", "Met"] in Tables(criteria).values()
    assert "<details" not in criteria
    assert "Historical" not in text and "https://example.com/signal" not in text
    assert registry(root / "evidence")["P001"]["run"]["seed"] == 29
    script = (root / "evidence/experiments/family/signal.py").read_text()
    listing = (generated / "P001-script.md").read_text()
    assert renderer.fingerprint((root / "docs/evidence/family/signal.py").read_text()) == renderer.fingerprint(script)
    assert (generated / "P001.py").read_text() == script
    assert "Download the complete evidence" in listing
    assert "experiments/run.py P001" in listing
    assert "{python}" not in listing
    assert not (root / "executed").exists()


def test_render_preserves_unchanged_input_timestamps(evidence: tuple[ModuleType, ModuleType, Path]) -> None:
    _, renderer, root = evidence
    renderer.render(root)
    generated = root / "docs/evidence"
    files = {path for path in generated.rglob("*") if path.is_file()}
    timestamp = 1_700_000_000_000_000_000
    for path in files:
        os.utime(path, ns=(timestamp, timestamp))

    renderer.render(root)
    assert all(path.stat().st_mtime_ns == timestamp for path in files)

    script = root / "evidence/experiments/family/signal.py"
    script.write_text(script.read_text().replace("# title: Signal", "# title: Updated signal"))
    renderer.render(root)
    changed = {path.relative_to(generated).as_posix() for path in files if path.stat().st_mtime_ns != timestamp}
    assert changed == {"family/signal.py", "_generated/P001.py"}
    assert renderer.fingerprint((generated / "family/signal.py").read_text()) == renderer.fingerprint(
        script.read_text()
    )
    assert (generated / "_generated/P001.py").read_text() == script.read_text()


def test_details_tabs_keep_methods_code_and_downloads_without_executing(evidence):
    _, renderer, root = evidence
    script = root / "evidence/experiments/family/signal.py"
    original = script.read_bytes()
    renderer.render(root)

    page = (root / "docs/evidence/family/signal.py").read_text()
    headings = [page.index(f"# ### {name}\n") for name in ("Measurements", "Method", "Reproduce", "Code")]
    assert headings == sorted(headings)
    assert "# #### Data and evaluation" in page
    assert "Independent held-out observations." in page
    assert 'role="region" aria-label="Complete experiment code"' in page
    assert "../_generated/P001.py" in page
    assert metadata(page)["code-fold"] is False
    assert renderer.fingerprint(page) == renderer.fingerprint(original.decode())
    assert script.read_bytes() == original
    assert (root / "docs/evidence/_generated/P001.py").read_bytes() == original
    assert not (root / "executed").exists()


def test_run_details_fold_provenance_and_raw_data_without_hiding_errors(evidence):
    _, renderer, root = evidence
    recorded = result(outcome="error")
    recorded["error"] = '<script>alert("failure")</script>\n```'
    recorded["metrics"] = {"arm": {"score": 0.25, "values": list(range(12))}}
    set_result("P001", recorded, root)
    original = (root / "evidence/results.json").read_bytes()
    renderer.render(root)

    text = (root / "docs/evidence/_generated/P001-evidence.md").read_text()
    parsed = Tables(text)
    assert '<time datetime="2026-09-14T12:00:00+00:00">14 Sep 2026 at 12:00 UTC</time>' in text
    assert "Execution error" in text
    assert not any(tag == "script" for tag, _ in parsed.tags)
    assert ["Source fingerprint", "source-11"] in parsed.values()
    assert ["Library revision", "tested-library"] in parsed.values()
    assert ["Evidence revision", "tested-evidence"] in parsed.values()
    disclosures = [(attrs.get("class"), attrs) for tag, attrs in parsed.tags if tag == "details"]
    assert {name for name, _ in disclosures} >= {"evidence-provenance", "evidence-raw", "evidence-metric-group"}
    assert all("open" not in attrs for _, attrs in disclosures)
    assert (root / "evidence/results.json").read_bytes() == original


def test_research_pages_show_requirements_without_joining_the_default_execution_suite(evidence):
    _, renderer, root = evidence
    source = (root / "evidence/experiments/family/signal.py").read_text().replace("P001", "P003")
    header = metadata(source)
    header["evidence-requires"] = ['Offline encoder <snapshot> and the "text" extra']
    source = renderer.frontmatter(source, header)
    path = root / "evidence/research/family/optional.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)

    renderer.render(root)

    page = metadata((root / "docs/evidence/family/optional.py").read_text())
    assert page["evidence-status"] == "Missing"
    assert page["evidence-lane"] == "research"
    assert "[Research]{.evidence-lane}" in page["subtitle"]
    listing = (root / "docs/evidence/_generated/P003-script.md").read_text()
    assert "--lane research P003" in listing
    assert "&lt;snapshot&gt;" in listing
    assert "research/family/optional.py" in listing
    assert "P003" not in registry(root / "evidence")
    assert (root / "docs/evidence/_generated/P003.py").read_text() == source
    assert not (root / "executed").exists()


def test_docs_reject_enabled_experiments_before_staging(evidence: tuple[ModuleType, ModuleType, Path]) -> None:
    _, renderer, root = evidence
    script = root / "evidence/experiments/family/signal.py"
    script.write_text(script.read_text().replace("enabled: false", "enabled: true"))
    with pytest.raises(ValueError, match="requires execute.enabled: false"):
        renderer.render(root)
    assert not (root / "docs/evidence/family/signal.py").exists()
    assert not (root / "executed").exists()


def test_docs_cleanup_preserves_authored_sources_results_and_published_files(
    evidence: tuple[ModuleType, ModuleType, Path],
) -> None:
    _, renderer, root = evidence
    docs = root / "docs/evidence"
    catalog = root / "docs/evidence.qmd"
    catalog.write_text("# Evidences\n")
    notes = docs / "notes"
    notes.mkdir()
    (notes / "draft.md").write_text("Authored notes\n")
    (docs / "retired-family").mkdir()
    published = root / "docs/site/evidence/_generated"
    published.mkdir(parents=True)
    (published / "P001.py").write_text("Published evidence\n")
    source = root / "evidence/experiments/family/signal.py"
    before = source.read_bytes()
    script = (root / "evidence/experiments/family/signal.py").read_bytes()
    renderer.render(root)
    (docs / "_generated/P001-insights.md").write_text("Obsolete fragment\n")
    (docs / "family/signal.quarto_ipynb").write_text("Temporary notebook\n")
    (docs / "family/signal.quarto_ipynb_1").write_text("Temporary notebook cell\n")
    (docs / "family/signal.quarto_ipynb_2").write_text("Temporary notebook cell\n")

    renderer.clean(root)
    renderer.clean(root)

    assert {path.name for path in docs.iterdir()} == {"notes"}
    assert catalog.read_text() == "# Evidences\n"
    assert (notes / "draft.md").read_text() == "Authored notes\n"
    assert (published / "P001.py").read_text() == "Published evidence\n"
    assert source.read_bytes() == before
    assert (root / "evidence/experiments/family/signal.py").read_bytes() == script
    (notes / "draft.md").unlink()
    renderer.clean(root)
    renderer.clean(root)
    assert not docs.exists()


def test_docs_cleanup_rejects_paths_outside_generated_families(
    evidence: tuple[ModuleType, ModuleType, Path],
) -> None:
    _, renderer, root = evidence
    outside = root / "outside"
    outside.mkdir()
    family = root / "docs/evidence/family"
    family.rmdir()
    family.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Cannot clean evidence page outside"):
        renderer.clean(root)
    assert (root / "evidence/experiments/family/signal.py").is_file()


def test_changed_script_warns_without_rewriting_recorded_results(
    evidence: tuple[ModuleType, ModuleType, Path],
) -> None:
    reporting, renderer, root = evidence
    script = root / "evidence/experiments/family/signal.py"
    measured = result(seed=43)
    measured["provenance"]["code_sha256"] = renderer.fingerprint(script.read_text())
    set_result("P001", measured, root)
    before = (root / "evidence/results.json").read_bytes()
    generated = root / "docs/evidence/_generated"

    renderer.render(root)
    assert "code has changed" not in (generated / "P001-status.md").read_text()
    script.write_text(script.read_text().replace("# Static experiment", "# Clearer explanation"))
    renderer.render(root)
    assert "code has changed" not in (generated / "P001-status.md").read_text()
    script.write_text(script.read_text() + "\nexperiment_size = 2048\n")
    renderer.render(root)

    status = (generated / "P001-status.md").read_text()
    assert "code has changed" in status
    assert "earlier version" in status
    assert "Passing" in status
    evidence_text = (generated / "P001-evidence.md").read_text()
    assert "Seed **43**" in evidence_text
    assert ["held_out_rmse", "0.125"] in Tables(evidence_text).values()
    assert "experiment_size = 2048" in (generated / "P001.py").read_text()
    assert (root / "evidence/results.json").read_bytes() == before
    assert not (root / "executed").exists()


def test_readout_uses_explicit_paths_and_explains_scores(evidence: tuple[ModuleType, ModuleType, Path]) -> None:
    _, renderer, _ = evidence
    header = {
        "evidence-id": "P001",
        "evidence-readout": [
            {"label": "Prediction error", "metric": ["arm", "nrmse"], "format": "error"},
            {"label": "Correct answers", "metric": ["accuracy"], "format": "percent"},
            {"label": "Matching IDs", "metric": ["auc"], "format": "auc"},
            {"label": "Last count", "metric": ["counts", -1], "format": "number"},
        ],
    }
    run = result()
    run["metrics"] = {"arm": {"nrmse": 0.025}, "accuracy": 0.75, "auc": 0.9998, "counts": [2, 5]}
    text = renderer.readout(header, run)
    assert "| Prediction error | 2.5% of baseline error |" in text
    assert "| Correct answers | 75.0% |" in text
    assert "| Matching IDs | 0.9998 |" in text
    assert "| Last count | 5 |" in text
    assert "100% matches that guess; lower is better" in text
    assert "not percentages of correct answers" in text
    assert renderer.readout(header, None) == ""
    run["metrics"] = {"auc": float("nan")}
    text = renderer.readout(header, run)
    assert "| Prediction error | Not recorded in this run |" in text
    assert "| Matching IDs | Not available |" in text


@pytest.mark.parametrize("measured", [True, "0.125", {"nrmse": 0.125}])
def test_readout_rejects_non_numerical_selections(
    evidence: tuple[ModuleType, ModuleType, Path], measured: object
) -> None:
    _, renderer, _ = evidence
    header = {"evidence-id": "P001", "evidence-readout": [{"label": "Error", "metric": ["score"], "format": "error"}]}
    with pytest.raises(ValueError, match="P001: readout 'Error' requires a numerical metric"):
        renderer.readout(header, {"metrics": {"score": measured}})


def test_docs_reject_smoke_results(evidence):
    _, renderer, root = evidence
    set_result("P001", result(mode="smoke"), root)
    with pytest.raises(ValueError, match="only full runs may be published"):
        renderer.render(root)


def test_static_evidence_publication_imports_no_model_runtime(evidence, monkeypatch):
    _, _, root = evidence
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"relflow", "torch", "lightning"}, name
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    renderer = load(ROOT / "evidence/render.py")
    set_result("P001", result(), root)
    renderer.render(root)
    assert (root / "docs/evidence/_generated/P001-status.md").is_file()
    assert not (root / "executed").exists()
    listing = (root / "docs/evidence/_generated/P001-script.md").read_text()
    assert "separate from building this site" in listing
    assert "matching-relflow.whl" in listing
    assert "relflow/relflow repository" in listing


def test_criteria_are_visible_once_and_keep_named_failed_checks(evidence):
    _, renderer, root = evidence
    measured = result(outcome="not_met")
    measured["checks"] = {"Retains the signal": True, "Control remains distinguishable": False}
    set_result("P001", measured, root)
    original = (root / "evidence/results.json").read_bytes()

    renderer.render(root)

    generated = root / "docs/evidence/_generated"
    status = (generated / "P001-status.md").read_text()
    checks = (generated / "P001-checks.md").read_text()
    details = (generated / "P001-evidence.md").read_text()
    table = Tables(checks)
    assert any("evidence-criteria" in attrs.get("class", "").split() for _, attrs in table.tags)
    assert ["Retains the signal", "Met"] in table.values()
    assert ["Control remains distinguishable", "Not met"] in table.values()
    assert "<details" not in checks
    for name in measured["checks"]:
        assert checks.count(name) == 1
        assert name not in status
        assert not any(row and row[0] == name for row in Tables(details).values())
    page = (root / "docs/evidence/family/signal.py").read_text()
    assert page.index("# ## What this experiment checks") < page.index("# ## Example")
    assert page.count("{{< evidence P001 checks >}}") == 1
    assert renderer.fingerprint(page) == renderer.fingerprint(
        (root / "evidence/experiments/family/signal.py").read_text()
    )
    assert (root / "evidence/results.json").read_bytes() == original


def test_criteria_escape_check_names_and_have_accessible_table_headers(evidence):
    _, renderer, _ = evidence
    name = '<script>alert("check")</script> | a\nb'
    recorded = {"checks": {name: False}}
    original = deepcopy(recorded)
    text = renderer.criteria(recorded)
    parsed = Tables(text)
    assert [name, "Not met"] in parsed.values()
    assert not any(tag == "script" for tag, _ in parsed.tags)
    assert "<details" not in text
    assert len(parsed.tables) == 1
    table = parsed.tables[0]
    assert table["caption"].strip()
    assert all(cell["tag"] == "th" and cell["attributes"].get("scope") == "col" for cell in table["rows"][0])
    assert table["rows"][1][0]["attributes"].get("scope") == "row"
    assert recorded == original
    for missing in (None, {"checks": {}}):
        unavailable = renderer.criteria(missing)
        assert "No behavioral checks have been recorded" in unavailable
        assert Tables(unavailable).tables == []


def test_nested_measurements_keep_structure_all_list_values_and_empty_states(evidence):
    _, renderer, _ = evidence
    recorded = {
        "zero": 0,
        "off": False,
        "missing": None,
        "empty_items": [],
        "empty_group": {},
        "arm": {"nrmse": 0.125, "nested": {"counts": list(range(12))}},
        "trials": [{"seed": 17, "score": 0.25}, {"seed": 117, "score": 0.5}],
    }
    original = deepcopy(recorded)
    text = renderer.measurements(recorded)
    parsed = Tables(text)
    assert (
        sum(
            tag == "details" and "evidence-metric-group" in attrs.get("class", "").split() for tag, attrs in parsed.tags
        )
        >= 3
    )
    rows = parsed.values()
    assert ["zero", "0"] in rows
    assert ["off", "false"] in rows
    assert ["missing", "null"] in rows
    assert ["empty_items", "[]"] in rows
    assert ["empty_group", "{}"] in rows
    assert ["nrmse", "0.125"] in rows
    assert ["Row", "seed", "score"] in rows
    assert ["0", "17", "0.25"] in rows and ["1", "117", "0.5"] in rows
    cells = [cell["text"] for table in parsed.tables for row in table["rows"][1:] for cell in row[1:]]
    assert all(str(number) in cells for number in range(12))
    assert "final 8" not in text
    assert recorded == original
    for table in parsed.tables:
        assert table["caption"].strip()
        assert all(cell["tag"] == "th" and cell["attributes"].get("scope") == "col" for cell in table["rows"][0])
        assert all(row[0]["attributes"].get("scope") == "row" for row in table["rows"][1:])


def test_nested_measurement_labels_and_values_remain_literal_html_text(evidence):
    _, renderer, _ = evidence
    group = '</summary><script>alert("group")</script>'
    name = '<img src=x onerror="bad()"> | name\nnext'
    measured = '<a href="javascript:bad()">literal</a> | value\nnext'
    recorded = {group: {name: measured}}
    text = renderer.measurements(recorded)
    parsed = Tables(text)
    assert [name, measured] in parsed.values()
    assert not any(tag in {"script", "img", "a"} for tag, _ in parsed.tags)
    assert "&lt;/summary&gt;" in text and "&lt;script&gt;" in text
    assert recorded == {group: {name: measured}}


def test_record_tables_distinguish_absent_values_from_null_zero_and_false(evidence):
    _, renderer, _ = evidence
    records = [{"seed": 17, "score": 0}, {"seed": 18, "passed": False}, {"seed": 19, "score": None}]
    parsed = Tables(renderer.measurements(records, ("trials",)))
    assert parsed.values() == [
        ["Row", "seed", "score", "passed"],
        ["0", "17", "0", "Not recorded"],
        ["1", "18", "Not recorded", "false"],
        ["2", "19", "null", "Not recorded"],
    ]
