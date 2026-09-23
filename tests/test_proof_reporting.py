"""Evidence survives repeated runs and drives static proof documentation."""

from __future__ import annotations

import importlib.util
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Barrier
from types import ModuleType

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proofs"))


def load(path: Path) -> ModuleType:
    """Load the standalone reporting tools without importing experiment scripts."""
    spec = importlib.util.spec_from_file_location(f"proof_{path.stem}", path)
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
            "packages": {"torch": "2.8.0"},
        },
        "metrics": {"held_out_rmse": 0.125},
        "checks": {"Held-out RMSE is below 0.2": outcome == "met"},
        "outcome": outcome,
        "duration_seconds": 0.01,
    }


@pytest.fixture
def evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType, Path]:
    reporting = load(ROOT / "proofs/reporting.py")
    renderer = load(ROOT / "proofs/render.py")
    scripts = tmp_path / "proofs"
    pages = tmp_path / "docs/proofs/family"
    scripts.mkdir()
    pages.mkdir(parents=True)
    entries = {}
    for identifier, slug, state in (("P001", "signal", "Passing"), ("P002", "noise", "Limited")):
        script = scripts / f"{slug}.py"
        metadata = {
            "proof-id": identifier,
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
            "raise RuntimeError('a documentation build executed a proof')\n"
        )
        page = pages / f"{slug}.py"
        entries[identifier] = {
            "script": script.relative_to(tmp_path).as_posix(),
            "page": page.relative_to(tmp_path).as_posix(),
            "historical": {
                "status": state,
                "summary": f"Historical summary of {slug}.",
                "insights": f"The insight for {slug}.",
                "evidence": f"Historical measurements of {slug}.",
                "sources": [f"https://example.com/{slug}"],
            },
            "runs": [],
        }
    source = scripts / "results.yaml"
    source.write_text(yaml.safe_dump({"schema": 1, "proofs": entries}, sort_keys=False))
    monkeypatch.setattr(reporting, "RESULTS", source)
    monkeypatch.setattr(reporting, "provenance", lambda script: deepcopy(result()["provenance"]))
    monkeypatch.setattr(torch, "set_num_threads", lambda threads: None)
    return reporting, renderer, tmp_path


@pytest.mark.parametrize(
    ("historical", "outcome", "expected", "kind"),
    [
        ("Passing", "met", "Passing", "note"),
        ("Limited", "met", "Limited", "warning"),
        ("Partial", "met", "Partial", "note"),
        ("Passing", "not_met", "Failing", "warning"),
        ("Passing", "error", "Error", "warning"),
        ("Passing", None, "Missing", "note"),
    ],
)
def test_statuses_match_catalog_callouts_legend_and_colors(
    evidence: tuple[ModuleType, ModuleType, Path], historical: str, outcome: str | None, expected: str, kind: str
) -> None:
    _, renderer, root = evidence
    source = root / "proofs/results.yaml"
    document = yaml.safe_load(source.read_text())
    entry = document["proofs"]["P001"]
    entry["historical"]["status"] = historical
    entry["runs"] = [result(outcome=outcome)] if outcome is not None else []
    source.write_text(yaml.safe_dump(document))

    renderer.render(root)

    catalog = yaml.safe_load((root / "docs/proofs/catalog.yaml").read_text())
    assert next(row for row in catalog if row["id"] == "P001")["proof-status"] == expected
    notice = (root / "docs/proofs/_generated/P001-status.md").read_text()
    title = f"Latest full run · {expected}" if outcome is not None else expected
    assert f'::: {{.callout-{kind} title="P001 · {title}"}}' in notice
    legend = (ROOT / "docs/proofs.qmd").read_text()
    assert f'[{expected}]{{.proof-status data-proof-status="{expected}"}}' in legend
    styles = (ROOT / "docs/assets/stylesheets/proofs.css").read_text()
    assert f'\n.proof-status[data-proof-status="{expected}"] {{' in styles
    assert f'\nbody.quarto-dark .proof-status[data-proof-status="{expected}"] {{' in styles


def test_concurrent_appends_preserve_history_and_other_proofs(evidence: tuple[ModuleType, ModuleType, Path]) -> None:
    reporting, _, root = evidence
    output = root / "collected/results.yaml"
    historical = result(seed=1)
    reporting.save("P001", historical, output)
    before = yaml.safe_load(output.read_text())
    barrier = Barrier(8)

    def append(index: int) -> None:
        barrier.wait(timeout=10)
        reporting.save("P001" if index % 2 else "P002", result(seed=index + 10), output)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append, range(8)))

    after = yaml.safe_load(output.read_text())
    assert after["schema"] == 1
    for identifier in ("P001", "P002"):
        original = before["proofs"][identifier]
        recorded = after["proofs"][identifier]
        assert {key: value for key, value in recorded.items() if key != "runs"} == {
            key: value for key, value in original.items() if key != "runs"
        }
    assert after["proofs"]["P001"]["runs"][0] == historical
    assert sorted(run["seed"] for run in after["proofs"]["P001"]["runs"]) == [1, 11, 13, 15, 17]
    assert sorted(run["seed"] for run in after["proofs"]["P002"]["runs"]) == [10, 12, 14, 16]
    assert yaml.safe_load(reporting.RESULTS.read_text())["proofs"]["P001"]["runs"] == []


def test_interrupted_replacement_keeps_existing_evidence(
    evidence: tuple[ModuleType, ModuleType, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    reporting, _, root = evidence
    output = root / "collected/results.yaml"
    reporting.save("P001", result(seed=1), output)
    before = output.read_bytes()

    def interrupt(source: Path, destination: Path) -> None:
        raise OSError("simulated interrupted publication")

    monkeypatch.setattr(Path, "replace", interrupt)
    with pytest.raises(OSError, match="interrupted publication"):
        reporting.save("P001", result(seed=2), output)
    assert output.read_bytes() == before
    assert {path.name for path in output.parent.iterdir()} == {"results.yaml", "results.yaml.lock"}


def test_cli_records_actual_metrics_failed_checks_and_options(
    evidence: tuple[ModuleType, ModuleType, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    reporting, _, root = evidence
    output = root / "runs.yaml"
    calls = []

    def experiment(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
        calls.append((seed, steps, accelerator))
        return {"rmse": np.float64(0.375), "predictions": np.array([1.0, 2.0])}, {
            "Control remains at chance": np.bool_(True),
            "RMSE is below 0.2": np.bool_(False),
        }

    monkeypatch.setattr(
        sys, "argv", ["proof.py", "--seed", "23", "--accelerator", "auto", "--threads", "2", "--output", str(output)]
    )
    reporting.report("P001", experiment, seed=11)
    recorded = yaml.safe_load(output.read_text())["proofs"]["P001"]["runs"][-1]
    assert calls == [(23, None, "auto")]
    assert recorded["mode"] == "full"
    assert recorded["seed"] == 23
    assert recorded["threads"] == 2
    assert recorded["accelerator"] == "auto"
    assert recorded["metrics"] == {"rmse": 0.375, "predictions": [1.0, 2.0]}
    assert recorded["checks"] == {"Control remains at chance": True, "RMSE is below 0.2": False}
    assert recorded["outcome"] == "not_met"
    assert recorded["duration_seconds"] >= 0
    assert recorded["provenance"]["source_sha256"] == "source-11"
    assert "Recorded P001" in capsys.readouterr().out


def test_cli_records_execution_errors_before_exiting(
    evidence: tuple[ModuleType, ModuleType, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    reporting, renderer, root = evidence
    output = root / "proofs/results.yaml"
    reporting.save("P001", result(), output)

    def experiment(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
        raise RuntimeError("synthetic generation failed")

    monkeypatch.setattr(sys, "argv", ["proof.py", "--output", str(output)])
    with pytest.raises(SystemExit) as caught:
        reporting.report("P001", experiment, seed=19)
    assert caught.value.code == 1
    entry = yaml.safe_load(output.read_text())["proofs"]["P001"]
    assert len(entry["runs"]) == 2
    assert entry["runs"][-1]["outcome"] == "error"
    assert entry["runs"][-1]["seed"] == 19
    assert "RuntimeError: synthetic generation failed" in entry["runs"][-1]["error"]
    renderer.render(root)
    assert "Error" in (root / "docs/proofs/_generated/P001-status.md").read_text()
    assert "synthetic generation failed" in (root / "docs/proofs/_generated/P001-evidence.md").read_text()


def test_smoke_runs_do_not_replace_full_evidence_or_status(
    evidence: tuple[ModuleType, ModuleType, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    reporting, renderer, root = evidence
    output = root / "proofs/results.yaml"
    full = result(seed=7, outcome="not_met")
    reporting.save("P001", full, output)

    def experiment(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
        assert (seed, steps, accelerator) == (11, 2, "cpu")
        return {"smoke_only_measurement": 999}, {"Execution completed": True}

    monkeypatch.setattr(sys, "argv", ["proof.py", "--steps", "2", "--output", str(output)])
    reporting.report("P001", experiment, seed=11)
    entry = yaml.safe_load(output.read_text())["proofs"]["P001"]
    assert entry["runs"][-1]["mode"] == "smoke"
    assert entry["runs"][-1]["steps_override"] == 2
    assert renderer.latest(entry) == full
    renderer.render(root)
    generated = root / "docs/proofs/_generated"
    assert "Failing" in (generated / "P001-status.md").read_text()
    text = (generated / "P001-evidence.md").read_text()
    assert "Seed **7**" in text
    assert "| held_out_rmse | 0.125 |" in text
    assert "smoke_only_measurement" not in text
    assert "2-step cap" in text
    assert "does not establish learning capability or change the proof status" in text


def test_docs_refresh_from_yaml_and_embed_scripts_without_executing(
    evidence: tuple[ModuleType, ModuleType, Path],
) -> None:
    reporting, renderer, root = evidence
    source = root / "proofs/results.yaml"
    generated = root / "docs/proofs/_generated"
    renderer.render(root)
    notice = (generated / "P001-status.md").read_text()
    assert "Missing" in notice and "No full experiment has been recorded." in notice
    assert "Historical summary" not in notice
    assert "Historical measurements" not in (generated / "P001-evidence.md").read_text()

    failed = result(outcome="not_met")
    failed["metrics"] = {"held_out_rmse": 0.875}
    reporting.save("P001", failed, source)
    renderer.render(root)
    catalog = yaml.safe_load((root / "docs/proofs/catalog.yaml").read_text())
    assert {row["id"]: row["proof-status"] for row in catalog} == {
        "P001": "Failing",
        "P002": "Missing",
    }
    assert "0 of 1 behavioral checks met" in (generated / "P001-status.md").read_text()
    assert "| held_out_rmse | 0.875 |" in (generated / "P001-evidence.md").read_text()

    reporting.save("P001", result(seed=29), source)
    renderer.render(root)
    assert "Passing" in (generated / "P001-status.md").read_text()
    text = (generated / "P001-evidence.md").read_text()
    assert "Seed **29**" in text and "| held_out_rmse | 0.125 |" in text
    assert "| held_out_rmse | 0.875 |" not in text
    assert "<details>\n<summary>Behavioral checks</summary>\n\n" in text
    assert "| Held-out RMSE is below 0.2 | Met |\n\n</details>" in text
    assert "Historical" not in text and "https://example.com/signal" not in text
    assert (generated / "results.yaml").read_bytes() == source.read_bytes()
    assert len(yaml.safe_load((generated / "results.yaml").read_text())["proofs"]["P001"]["runs"]) == 2
    script = (root / "proofs/signal.py").read_text()
    listing = (generated / "P001-script.md").read_text()
    assert (root / "docs/proofs/family/signal.py").read_text() == script
    assert (generated / "P001.py").read_text() == script
    assert "Download the complete proof" in listing
    assert "proofs/run.py P001" in listing
    assert "{python}" not in listing
    assert not (root / "executed").exists()


def test_render_preserves_unchanged_input_timestamps(evidence: tuple[ModuleType, ModuleType, Path]) -> None:
    _, renderer, root = evidence
    renderer.render(root)
    generated = root / "docs/proofs"
    files = {path for path in generated.rglob("*") if path.is_file()}
    timestamp = 1_700_000_000_000_000_000
    for path in files:
        os.utime(path, ns=(timestamp, timestamp))

    renderer.render(root)
    assert all(path.stat().st_mtime_ns == timestamp for path in files)

    script = root / "proofs/signal.py"
    script.write_text(script.read_text().replace("# title: Signal", "# title: Updated signal"))
    renderer.render(root)
    changed = {path.relative_to(generated).as_posix() for path in files if path.stat().st_mtime_ns != timestamp}
    assert changed == {"family/signal.py", "_generated/P001.py", "catalog.yaml"}
    assert (generated / "family/signal.py").read_text() == script.read_text()
    assert (generated / "_generated/P001.py").read_text() == script.read_text()


def test_docs_reject_enabled_experiments_before_staging(evidence: tuple[ModuleType, ModuleType, Path]) -> None:
    _, renderer, root = evidence
    script = root / "proofs/signal.py"
    script.write_text(script.read_text().replace("enabled: false", "enabled: true"))
    with pytest.raises(ValueError, match="requires execute.enabled: false"):
        renderer.render(root)
    assert not (root / "docs/proofs/family/signal.py").exists()
    assert not (root / "executed").exists()


def test_docs_cleanup_preserves_authored_sources_results_and_published_files(
    evidence: tuple[ModuleType, ModuleType, Path],
) -> None:
    _, renderer, root = evidence
    docs = root / "docs/proofs"
    catalog = root / "docs/proofs.qmd"
    catalog.write_text("# Proofs\n")
    notes = docs / "notes"
    notes.mkdir()
    (notes / "draft.md").write_text("Authored notes\n")
    (docs / "retired-family").mkdir()
    published = root / "docs/site/proofs/_generated"
    published.mkdir(parents=True)
    (published / "results.yaml").write_text("Published evidence\n")
    source = root / "proofs/results.yaml"
    before = source.read_bytes()
    script = (root / "proofs/signal.py").read_bytes()
    renderer.render(root)
    (docs / "_generated/P001-insights.md").write_text("Obsolete fragment\n")

    renderer.clean(root)
    renderer.clean(root)

    assert {path.name for path in docs.iterdir()} == {"notes"}
    assert catalog.read_text() == "# Proofs\n"
    assert (notes / "draft.md").read_text() == "Authored notes\n"
    assert (published / "results.yaml").read_text() == "Published evidence\n"
    assert source.read_bytes() == before
    assert (root / "proofs/signal.py").read_bytes() == script
    (notes / "draft.md").unlink()
    renderer.clean(root)
    renderer.clean(root)
    assert not docs.exists()


def test_docs_cleanup_rejects_paths_outside_generated_families(
    evidence: tuple[ModuleType, ModuleType, Path],
) -> None:
    _, renderer, root = evidence
    source = root / "proofs/results.yaml"
    document = yaml.safe_load(source.read_text())
    document["proofs"]["P001"]["page"] = "proofs/signal.py"
    source.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="Cannot clean proof page outside"):
        renderer.clean(root)
    assert (root / "proofs/signal.py").is_file()


def test_provenance_failure_is_recorded_and_docs_remain_renderable(
    evidence: tuple[ModuleType, ModuleType, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    reporting, renderer, root = evidence
    source = root / "proofs/results.yaml"
    reporting.save("P001", result(seed=7), source)
    calls = []

    def unavailable(script: Path) -> dict:
        raise OSError("cannot read installed package source")

    def experiment(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
        calls.append(seed)
        return {"unexpected": 1}, {"unexpected": True}

    monkeypatch.setattr(reporting, "provenance", unavailable)
    monkeypatch.setattr(sys, "argv", ["proof.py", "--output", str(source)])
    with pytest.raises(SystemExit) as caught:
        reporting.report("P001", experiment, seed=31)
    assert caught.value.code == 1
    assert calls == []
    runs = yaml.safe_load(source.read_text())["proofs"]["P001"]["runs"]
    assert len(runs) == 2 and runs[0]["seed"] == 7
    assert runs[-1]["outcome"] == "error"
    assert runs[-1]["seed"] == 31
    assert not runs[-1].get("provenance")
    assert "OSError: cannot read installed package source" in runs[-1]["error"]
    assert "metrics" not in runs[-1]

    renderer.render(root)
    generated = root / "docs/proofs/_generated"
    assert "Error" in (generated / "P001-status.md").read_text()
    assert "cannot read installed package source" in (generated / "P001-evidence.md").read_text()
    catalog = yaml.safe_load((root / "docs/proofs/catalog.yaml").read_text())
    assert next(row for row in catalog if row["id"] == "P001")["proof-status"] == "Error"


def test_changed_script_warns_without_rewriting_recorded_results(
    evidence: tuple[ModuleType, ModuleType, Path],
) -> None:
    reporting, renderer, root = evidence
    source = root / "proofs/results.yaml"
    script = root / "proofs/signal.py"
    measured = result(seed=43)
    measured["provenance"]["code_sha256"] = renderer.fingerprint(script.read_text())
    reporting.save("P001", measured, source)
    before = source.read_bytes()
    generated = root / "docs/proofs/_generated"

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
    assert source.read_bytes() == before
    assert not (root / "executed").exists()
