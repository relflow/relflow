"""Current measured results drive static evidence pages without model imports."""

from __future__ import annotations

import builtins
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from evidence.experiments.source import frontmatter, metadata, registry

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evidence"))


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
    entry = registry(root / "evidence")[identifier]
    script = root / "evidence" / entry["script"]
    script.write_text(frontmatter(script.read_text(), dict(entry["header"], **{"evidence-result": recorded})))


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
            "evidence-id": identifier,
            "evidence-group": "Family",
            "title": slug.title(),
            "categories": ["Family"],
            "description": slug,
            "execute": {"enabled": False, "eval": False},
        }
        header = "---\n" + yaml.safe_dump(metadata) + "---\n\n# Static experiment\n"
        script.write_text(
            "# %% [markdown]\n" + "\n".join("# " + line for line in header.splitlines()) + "\n\n# %%\n"
            '"""Literal Markdown example: ```python and ````.\n"""\n'
            "from pathlib import Path\n"
            f"Path({str(tmp_path / 'executed')!r}).write_text('executed')\n"
            "raise RuntimeError('a documentation build executed a evidence')\n"
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
def test_statuses_match_catalog_callouts_legend_and_colors(
    evidence: tuple[ModuleType, ModuleType, Path], outcome: str | None, expected: str, kind: str
) -> None:
    _, renderer, root = evidence
    if outcome is not None:
        set_result("P001", result(outcome=outcome), root)

    renderer.render(root)

    published = metadata((root / "docs/evidence/family/signal.py").read_text())
    assert published["evidence-status"] == expected
    assert published["repo-actions"] == ["issue"]
    notice = (root / "docs/evidence/_generated/P001-status.md").read_text()
    title = f"Current full run · {expected}" if outcome is not None else expected
    assert f'::: {{.evidence-result .evidence-result-{kind} role="note"}}' in notice
    assert f"**P001 · {title}**" in notice
    legend = (ROOT / "docs/evidence.qmd").read_text()
    assert f'[{expected}]{{.evidence-status data-evidence-status="{expected}"}}' in legend
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
    assert "| held_out_rmse | 0.875 |" in (generated / "P001-evidence.md").read_text()

    set_result("P001", result(seed=29), root)
    renderer.render(root)
    assert "Passing" in (generated / "P001-status.md").read_text()
    text = (generated / "P001-evidence.md").read_text()
    assert "Seed **29**" in text and "| held_out_rmse | 0.125 |" in text
    assert "RelFlow 0.4.1" in text and "tested-library" in text and "tested-evidence" in text
    assert "| held_out_rmse | 0.875 |" not in text
    assert "<details>\n<summary>Behavioral checks</summary>\n\n" in text
    assert "| Held-out RMSE is below 0.2 | Met |\n\n</details>" in text
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
    source = root / "evidence/experiments/family/signal.py"
    script = root / "evidence/experiments/family/signal.py"
    measured = result(seed=43)
    measured["provenance"]["code_sha256"] = renderer.fingerprint(script.read_text())
    set_result("P001", measured, root)
    before = metadata(source.read_text())["evidence-result"]
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
    assert "| held_out_rmse | 0.125 |" in evidence_text
    assert "experiment_size = 2048" in (generated / "P001.py").read_text()
    assert metadata(source.read_text())["evidence-result"] == before
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
        assert name.split(".")[0] not in {"relflow", "torch", "lightning", "numpy"}, name
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
