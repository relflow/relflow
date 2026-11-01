"""The privileged lab workflow accepts suite data without trusting its host code."""

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/evidence-lab.yml"
STEPS = yaml.safe_load(WORKFLOW.read_text())["jobs"]["run"]["steps"]


def test_suite_revision_cannot_replace_the_trusted_host_executor():
    checkouts = {step["with"]["path"]: step["with"] for step in STEPS if step.get("uses") == "actions/checkout@v4"}
    executor, suite = checkouts["executor"], checkouts["suite"]
    assert executor["repository"] == suite["repository"] == "relflow/relflow"
    assert re.fullmatch(r"[0-9a-f]{40}", executor["ref"])
    assert suite["ref"] == "${{ inputs.suite_ref }}"
    assert executor["persist-credentials"] is suite["persist-credentials"] is False


@pytest.mark.parametrize(
    "reference,accepted",
    [
        ("a" * 40, True),
        ("main", False),
        ("a" * 39, False),
        ("A" * 40, False),
        ("a" * 40 + "\n", False),
        ("$(echo injected)", False),
    ],
)
def test_suite_reference_guard_requires_an_exact_commit(reference, accepted):
    validation = next(step["run"] for step in STEPS if step.get("name") == "Validate the immutable suite reference")
    result = subprocess.run(["bash", "-c", validation], env=dict(os.environ, SUITE_REF=reference), check=False)
    assert (result.returncode == 0) == accepted
