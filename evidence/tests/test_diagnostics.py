"""Evidence owns observations and isolated plotting; docs only consume static data."""

from pathlib import Path

import yaml

from evidence import runtime
from evidence.diagnostics.capture import FUNCTIONS

ROOT = Path(__file__).resolve().parents[2]


def test_every_evidence_run_retains_observations_and_renders_without_write_permissions():
    workflow = yaml.safe_load((ROOT / ".github/workflows/evidence-hosted.yml").read_text())
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    jobs = workflow["jobs"]
    assert jobs["plots"]["needs"] == ["prepare", "collect"]
    assert jobs["plots"]["if"].startswith("always()")
    assert any("evidence-observations-shard-" in str(step) for step in jobs["shard"]["steps"])
    assert any("evidence-plots-attempt-" in str(step) for step in jobs["plots"]["steps"])
    docs = yaml.safe_load((ROOT / ".github/workflows/docs.yml").read_text())
    assert set(docs["jobs"]) == {"build"}
    assert not (ROOT / ".github/workflows/diagnostics.yml").exists()


def test_diagnostic_capture_observes_readouts_without_monitoring_generators_or_model_runtime():
    assert {"run", "predict", "prediction", "evaluate", "measure", "score"} <= FUNCTIONS
    assert not {"records", "forward", "training_step", "backward", "optimizer_step"} & FUNCTIONS


def test_plotting_runs_selected_code_only_in_a_bounded_offline_container(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.subprocess, "run", lambda command, **options: calls.append((command, options)))
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *args, **options: "sha256:" + "a" * 64 + "\n")
    result = runtime.render_plots(tmp_path / "reports", tmp_path / "observations", tmp_path / "output")
    command, options = calls[-1]
    assert command[:3] == ["docker", "run", "--rm"]
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command and "no-new-privileges" in command
    assert "--memory" in command and command[command.index("--memory") + 1] == "4g"
    assert "TMPDIR=/work" in command and any("dst=/work" in value for value in command)
    assert "/reports/source" not in command
    assert "evidence.diagnostics.build" in command
    assert options["timeout"] <= 1800
    assert result["image_id"] == "sha256:" + "a" * 64


def test_privileged_publication_uses_only_a_frozen_verifier():
    workflow = yaml.safe_load((ROOT / ".github/workflows/publish-evidence.yml").read_text())
    triggers = workflow.get("on", workflow.get(True))
    assert triggers["workflow_run"] == {"workflows": ["CI"], "types": ["completed"]}
    job = workflow["jobs"]["preview"]
    assert "head_repository.full_name == github.repository" in job["if"]
    checkout = next(step["with"] for step in job["steps"] if step.get("uses") == "actions/checkout@v4")
    assert len(checkout["ref"]) == 40 and checkout["persist-credentials"] is False
    assert checkout["repository"] == "relflow/relflow"
    assert not any("pip install" in step.get("run", "") and "executor/" in step["run"] for step in job["steps"])
