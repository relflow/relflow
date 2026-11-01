"""Public typing failures must stay visible to editor and pre-commit users."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_invalid_public_inputs_are_rejected(tmp_path):
    source = (ROOT / "tests/typing/rejected.py.txt").read_text()
    snippet = tmp_path / "rejected.py"
    snippet.write_text(source)
    expected_lines = {index for index, line in enumerate(source.splitlines(), 1) if "# rejected" in line}
    assert expected_lines

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyrefly",
            "check",
            "--config",
            str(ROOT / "pyproject.toml"),
            "--python-interpreter-path",
            sys.executable,
            "--output-format",
            "json",
            str(snippet),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    errors = json.loads(result.stdout)["errors"]
    assert all(Path(error["path"]).resolve() == snippet for error in errors), errors
    assert {error["line"] for error in errors} == expected_lines, result.stdout
    assert all(error["severity"] == "error" for error in errors), errors


def test_runtime_constructor_retains_custom_scheduler_support():
    import relflow as rf

    class CustomScheduler:
        def __init__(self, optimizer):
            self.optimizer = optimizer

        def state_dict(self):
            return {}

        def load_state_dict(self, state_dict):
            pass

    model = rf.Model.xs(d_model=16, value=rf.Number(mask=True))
    configured = rf.Model(
        model.schema,
        optimizer=rf.adamw(1e-3),
        scheduler=lambda module, optimizer: CustomScheduler(optimizer),
    )
    training = configured.configure_optimizers()
    assert isinstance(training["lr_scheduler"], CustomScheduler)
    assert training["lr_scheduler"].optimizer is training["optimizer"]
