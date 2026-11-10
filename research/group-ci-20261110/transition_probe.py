"""Smoke the migration planner and isolated baseline execution on the lab."""

import json
from pathlib import Path

from evidence_ci import execute, plan, runtime_inputs, script_paths
from experiments.source import registry
from migration import BASELINE
from runtime import isolate

root = Path("/work/input/evidence")
out = Path("/work/output")
manifest = plan(
    run_id="local-transition-smoke",
    candidate={"revision": "c" * 40},
    baseline={"revision": BASELINE},
    evidence_sha="uncommitted-working-tree",
    executor_sha="f" * 40,
    root=root,
    identifiers=["P108"],
    steps=32,
)
assert manifest["policy"] == "cluster-group-transition-v1"
task = next(task for task in manifest["tasks"] if task["role"] == "baseline")
assert task["id"] == "P018" and task["script"] == "migrations/cluster_to_group/P018.py"
view = isolate(root, out / "view", task, runtime_inputs(root), script_paths(root, registry(root, measurements=False)))
assert not (view / "migrations/cluster_to_group/P019.py").exists()
record = execute(task, root=view, observations=out / "observations")
(out / "record.json").write_text(json.dumps(record, indent=2) + "\n")
if record["error"] or not record["result"] or record["result"]["outcome"] == "error":
    raise RuntimeError(f"Baseline smoke execution failed: {record['error']}")
print("Isolated baseline", task["id"], record["result"]["mode"], record["result"]["outcome"])
