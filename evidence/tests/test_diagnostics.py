"""Diagnostic compute remains explicit and outside the required learning gate."""

from pathlib import Path

import yaml

from evidence.diagnostics.capture import FUNCTIONS
from evidence.diagnostics.host import policy, schedule

ROOT = Path(__file__).resolve().parents[2]


def test_visualization_compute_has_no_automatic_trigger_or_publication_permission():
    workflow = yaml.safe_load((ROOT / ".github/workflows/diagnostics.yml").read_text())
    assert set(workflow.get("on", workflow.get(True))) == {"workflow_call"}
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {"prepare", "capture", "collect"}
    assert workflow["jobs"]["capture"]["strategy"]["fail-fast"] is False
    assert "needs.prepare.outputs.shards" in workflow["jobs"]["capture"]["strategy"]["matrix"]["shard"]


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


def test_full_diagnostic_schedule_keeps_every_whole_proof_and_uses_twelve_shards():
    identifiers = [f"P{index:03d}" for index in range(1, 97)]
    plan = schedule(identifiers, {})
    assert len(plan) == 12
    assert sorted(key for shard in plan for key in shard["ids"]) == identifiers
    assert all(len(shard["ids"]) == 8 for shard in plan)


def test_explicit_subset_has_no_empty_compute_shards():
    plan = schedule(["P025", "P085"], {})
    assert len(plan) == 2
    assert sorted(key for shard in plan for key in shard["ids"]) == ["P025", "P085"]
