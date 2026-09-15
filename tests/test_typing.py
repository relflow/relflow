"""Public typing failures must stay visible to editor and pre-commit users."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    from relflow.helpers.optimizers import adamw

    class CustomScheduler:
        def __init__(self, optimizer):
            self.optimizer = optimizer

        def state_dict(self):
            return {}

        def load_state_dict(self, state_dict):
            pass

    model = rf.Model(d_model=16, n_layers=1, n_heads=4, value=rf.Number(mask=True))
    optimizer = adamw(1e-3)(model)
    scheduler = CustomScheduler(optimizer)
    configured = rf.Model(model.schema, optimizer=optimizer, scheduler=scheduler)
    assert configured.scheduler is scheduler
    assert configured.configure_optimizers() == {"optimizer": optimizer, "lr_scheduler": scheduler}
