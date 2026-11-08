"""Diagnostic compute remains explicit and outside the required learning gate."""

from pathlib import Path

import yaml

from evidence.diagnostics.capture import FUNCTIONS
from evidence.diagnostics.host import policy

ROOT = Path(__file__).resolve().parents[2]


def test_visualization_compute_has_no_automatic_trigger_or_publication_permission():
    workflow = yaml.safe_load((ROOT / ".github/workflows/diagnostics.yml").read_text())
    assert set(workflow.get("on", workflow.get(True))) == {"workflow_call"}
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {"prepare", "capture", "collect"}
    assert workflow["jobs"]["capture"]["strategy"]["fail-fast"] is False
    assert workflow["jobs"]["capture"]["strategy"]["matrix"]["shard"] == list(range(12))


def test_diagnostic_capture_observes_readouts_without_monitoring_generators_or_model_runtime():
    assert {"run", "predict", "prediction", "evaluate", "measure", "score"} <= FUNCTIONS
    assert not {"records", "forward", "training_step", "backward", "optimizer_step"} & FUNCTIONS


def test_documentation_dispatch_explicitly_opts_into_proof_compute():
    workflow = yaml.safe_load((ROOT / ".github/workflows/docs.yml").read_text())
    triggers = workflow.get("on", workflow.get(True))
    assert triggers["workflow_dispatch"]["inputs"]["diagnostic_plots"]["default"] is False
    job = workflow["jobs"]["diagnostics"]
    assert "workflow_dispatch" in job["if"] and "inputs.diagnostic_plots" in job["if"]
    assert "needs" not in workflow["jobs"]["build"]


def test_preflight_and_worker_policy_supplies_identity_and_writable_cache_without_host_network():
    profile = policy()
    assert profile[profile.index("--network") + 1] == "none"
    assert "--read-only" in profile
    assert "USER=evidence" in profile and "LOGNAME=evidence" in profile
    assert "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor" in profile
    assert "mode=1777" in profile[profile.index("--tmpfs") + 1]
