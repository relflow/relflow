# Contributing

Start with [AGENTS.md](AGENTS.md) for the package model, public API conventions,
code style, and ownership boundaries. Keep changes focused and test the public
contract and the invariants affected by the change.

## Development Setup

Use Python 3.12 or newer and install the development environment with uv:

```bash
uv sync
uv run pre-commit install
```

Run the smallest relevant test subset while developing. Before publishing a
branch, run the complete checks configured in `.pre-commit-config.yaml`:

```bash
uv run ruff format --check
uv run ruff check
uv run pyrefly check
uv run pytest
```

The commit hooks apply Ruff formatting and lint fixes, then run Pyrefly and the
full test suite. Review any files changed by the hooks before committing again.

Distributed tests launch every rank on the same machine. On macOS, the test
suite defaults `GLOO_SOCKET_IFNAME` to the loopback interface `lo0`. An explicitly
configured `GLOO_SOCKET_IFNAME` takes precedence.

## Type Checking And Editors

Pyrefly checks `src/relflow` and the static examples in `tests/typing` using
`pyproject.toml`. Development dependencies include Arrow stubs and the optional
Text, serving, and DuckDB dependencies so those APIs are checked too.

Install the Pyrefly editor extension and select `.venv/bin/python` as the
workspace interpreter. For VS Code, these settings use the same checker as the
commit hook:

```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "pyrefly.lspPath": ".venv/bin/pyrefly",
  "python.pyrefly.diagnosticMode": "workspace"
}
```

On Windows, use `.venv/Scripts/python.exe` and `.venv/Scripts/pyrefly.exe`.
Other editors can launch `uv run pyrefly lsp` through their language-server
client.

Add `assert_type` examples in `tests/typing` for public typing contracts.
`tests/test_typing.py` checks that invalid inputs produce diagnostics. Keep
annotations compatible with the minimum Python version in `requires-python`;
use `typing_extensions` when a feature needs a backport. For changes involving
newer typing syntax, also check the supported static targets:

```bash
uv run pyrefly check --python-version 3.14
uv run pyrefly check --python-version 3.15
```

Static checks do not replace runtime tests on the target interpreter. Avoid
project-wide suppressions; a local suppression should explain the upstream
stub gap or framework contract.

## Tests And Documentation

Place runtime and architectural tests beside the existing tests for the
subsystem. For extension-boundary changes, include a minimal third-party
extension that exercises registration, validation, encoding, and schema
round-tripping. The [agent guide](AGENTS.md) describes the contracts to preserve.

User documentation, executable learning proofs, recorded results, and branding
assets live in the separate [relflow/docs repository](https://github.com/relflow/docs).
Coordinate public API changes there and follow that repository's README and
agent guide for writing, previewing, checking, and publishing. Library checks
do not build the site or run learning experiments.
