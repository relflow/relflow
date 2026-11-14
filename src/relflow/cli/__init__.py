"""Typer entry point for relflow's model and file workflows."""

import json
import sys

from click import ClickException, UsageError
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from relflow.cli.commands import app

__all__ = ["app", "main"]


def main(argv: list[str] | None = None) -> int:
    """Return an exit code; Typer dispatches commands and owns argument parsing."""
    arguments = sys.argv[1:] if argv is None else argv
    try:
        return app(args=arguments, prog_name="relflow", standalone_mode=False) or 0
    except ClickException as error:
        console = Console(stderr=True, markup=False, highlight=False)
        if isinstance(error, UsageError) and error.ctx is not None:
            console.print(Text(error.ctx.get_usage()))
        console.print(Panel(Text(error.format_message()), title="Error", border_style="red"))
        # A usage error can precede parsing --json; stop at the literal-argument delimiter.
        options = arguments[: arguments.index("--")] if "--" in arguments else arguments
        if "--json" in options:
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "command": "cli",
                        "status": "failed",
                        "error": {
                            "type": "UsageError",
                            "message": error.format_message(),
                        },
                    }
                )
            )
        return error.exit_code
