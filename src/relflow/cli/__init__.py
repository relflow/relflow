"""Typer entry point for relflow's model and file workflows."""

import json
import sys

from relflow.cli.commands import app

__all__ = ["app", "main"]


def main(argv: list[str] | None = None) -> int:
    """Preserve programmatic exit codes while Typer owns parsing and help."""
    arguments = sys.argv[1:] if argv is None else argv
    invocation: dict[str, bool] = {}
    try:
        app(args=arguments, prog_name="relflow", obj=invocation)
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else 1
        if code == 2 and not invocation.get("executed") and "--json" in arguments:
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "command": "cli",
                        "status": "failed",
                        "error": {
                            "type": "UsageError",
                            "message": "Invalid command-line arguments; see stderr for usage details.",
                        },
                    }
                )
            )
        return code
    return 0
