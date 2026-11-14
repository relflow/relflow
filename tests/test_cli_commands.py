"""Typed parsing, generated help, and completion without executing model jobs."""

import json

import pytest
from typer.testing import CliRunner

import relflow as rf
import relflow.cli.commands as commands
from relflow.cli import app, main
from relflow.cli.config import Run


@pytest.mark.parametrize("command", [None, "inspect", "check", "train", "predict", "serve"])
def test_typer_generates_help_for_every_command(command):
    result = CliRunner().invoke(app, [command, "--help"] if command else ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output
    if command:
        assert "--json" in result.output
        assert "--config" in result.output
    else:
        assert all(name in result.output for name in ("inspect", "check", "train", "predict", "serve"))
        assert "--show-completion" in result.output


def test_version_and_short_help_keep_standard_exit_codes(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"relflow {rf.__version__}"
    assert main(["--version", "train"]) == 0
    assert capsys.readouterr().out.strip() == f"relflow {rf.__version__}"
    assert main(["-h"]) == 0
    assert "Usage:" in capsys.readouterr().out


def test_no_arguments_show_help_without_starting_a_workflow():
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 2
    assert "Usage:" in result.output


def test_repeatable_options_keep_their_values_and_order(monkeypatch):
    captured = []

    def register(run, imports):
        captured.extend(imports)

    monkeypatch.setattr(Run, "register", register)
    monkeypatch.setattr(commands, "inspect_model", lambda path: {"status": "success"})
    result = CliRunner().invoke(
        app,
        [
            "inspect",
            "model.rf",
            "--import",
            "first",
            "--import",
            "second",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured == ["first", "second"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["train", "model.yaml", "--output", "new-run", "--batch-size", "0"],
        ["train", "model.yaml", "--output", "new-run", "--max-epochs", "bad"],
        ["predict", "model.rf", "--output", "new-run", "--format", "invalid"],
        ["check", "model.yaml", "--stage", "invalid"],
        ["serve", "model.rf", "--port", "65536"],
        ["predict", "model.rf"],
    ],
)
def test_invalid_cli_types_choices_ranges_and_required_options_fail_before_execution(monkeypatch, arguments):
    def forbidden(*args, **kwargs):
        pytest.fail("an invalid command started a model workflow")

    monkeypatch.setattr(commands, "read", forbidden)
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 2, result.output
    assert "Error" in result.output


def test_unknown_commands_get_a_native_suggestion():
    result = CliRunner().invoke(app, ["predcit"])
    assert result.exit_code == 2
    assert "predict" in result.output
    assert "Did you mean" in result.output


def test_bash_completion_can_be_generated_without_installing_it(monkeypatch):
    monkeypatch.setattr("shellingham.detect_shell", lambda: ("bash", "/bin/bash"))
    result = CliRunner().invoke(app, ["--show-completion"])
    assert result.exit_code == 0, result.output
    assert "relflow" in result.output
    assert "_RELFLOW_COMPLETE" in result.output


def test_native_completion_suggests_commands_without_starting_a_workflow(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("shell completion started a model workflow")

    monkeypatch.setattr(commands, "read", forbidden)
    result = CliRunner().invoke(
        app,
        [],
        env={
            "_RELFLOW_COMPLETE": "complete_bash",
            "COMP_WORDS": "relflow pr",
            "COMP_CWORD": "1",
        },
    )
    assert result.exit_code == 0, result.output
    assert "predict" in result.output.splitlines()
    assert "Usage:" not in result.output


@pytest.mark.parametrize(
    "arguments",
    [
        ["train", "model.yaml", "--output", "unused", "--batch-size", "0", "--json"],
        ["inspect", "missing.rf", "--unknown-option", "--json"],
    ],
)
def test_native_usage_errors_keep_json_stdout_separate_from_rich_help(arguments, capsys, monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert main(arguments) == 2
    captured = capsys.readouterr()
    assert "\x1b[" not in captured.out
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out)["error"]["type"] == "UsageError"
    assert "Error" in captured.err


def test_workflow_exit_codes_are_preserved_by_the_typer_entry_point(monkeypatch, capsys):
    def fail(path):
        raise ValueError("invalid checkpoint")

    monkeypatch.setattr(commands, "inspect_model", fail)
    assert main(["inspect", "missing.rf", "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "failed"
    assert report["error"] == {"type": "ValueError", "message": "invalid checkpoint"}
