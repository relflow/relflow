# Contributing

Start with [AGENTS.md](AGENTS.md) for the package model, public API conventions,
code style, and ownership boundaries. Keep changes focused and test the public
contract and the invariants affected by the change.

## Development Setup

Use Python 3.11 or newer and install the development environment with uv:

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
fast test suite. Review any files changed by the hooks before committing again.

Distributed tests launch every rank on the same machine. On macOS, the test
suite defaults `GLOO_SOCKET_IFNAME` to the loopback interface `lo0`. An explicitly
configured `GLOO_SOCKET_IFNAME` takes precedence.

## Coverage

Coverage is opt-in and measures package lines and branches, including pytest
workers, Python subprocesses, and Torch multiprocessing workers:

```bash
uv run pytest -n 2 --cov --cov-report=term-missing --cov-report=html --cov-report=xml --cov-report=json
```

Open `htmlcov/index.html` for the source report. `coverage.xml` supports coverage
tools and editors; `coverage.json` includes line and branch totals. Generated
reports are ignored by Git. Normal tests and commit hooks run without tracing.
There is no coverage threshold until the hosted baseline is established.

README status badges follow `main`. After a successful same-repository push
runs the Tests workflow, the Coverage badge workflow publishes a Shields JSON
endpoint at `badges:coverage.json` and a `summary.json` containing line and branch
totals, the tested commit, and the source run URL. The badge displays line
coverage and links to the test workflow's complete reports. These files live on
the separate `badges` branch; source branches keep generated reports ignored.
Pull requests and release validation cannot publish badge updates. Old runs
cannot replace the current main commit's results, and test jobs stay read-only.
The first live coverage badge appears after this workflow reaches `main` and
that branch's tests pass. No coverage-service account or additional secret is
required.

## Compatibility CI

The Tests workflow runs the complete suite on Linux with Python 3.11, 3.12,
3.13, and 3.14, plus macOS ARM and Windows with Python 3.12. It installs a wheel
with all extras from the lockfile and verifies that imports come from site-packages.
Linux/Python 3.12 also runs formatting, lint, typing, and coverage checks. Each
job uploads its environment and JUnit results; the coverage job adds HTML, XML,
and JSON reports and a workflow summary.

Every environment loads a checkpoint trained from the current checkout on Linux
and Python 3.12. The check compares schema, learned state, prediction structure
and values, then resaves the model and resumes one optimizer step. This verifies
checkpoint portability across operating systems and Python versions.

RelFlow 0.3.5 adds per-head query/key normalization and learned query scales to
attention. Attention checkpoints from earlier releases lack this state and are
intentionally rejected by strict loading. Use their original package version to
load them, or retrain with 0.3.5; the portability check does not promise loading
an older attention architecture.

To reproduce the artifact exchange across two installed environments:

```bash
python scripts/checkpoint.py create --output checkpoint-baseline
python scripts/checkpoint.py check --input checkpoint-baseline --output checkpoint-check
```

Output directories must not already exist. Keep checkpoint artifacts outside
the source tree when running locally.

The `Required tests` job fails if the checkpoint producer or any supported
environment fails or is skipped. Use that check in branch protection.
Publication runs the same workflow for the release commit and waits for
success before building or publishing.

Python 3.15 also runs on every push and pull request as an advisory preview.
It uses official nightly CPU PyTorch wheels while stable wheels are unavailable,
then resolves the other dependencies from PyPI. Torch and DuckDB require binary
wheels; other dependencies may build from source, and prereleases are allowed
when needed. It saves installation diagnostics and resolved package versions,
then runs the same checkpoint and test checks. This preview does not gate releases.

Dependency compatibility runs nightly or manually. Its minimum profile resolves
the lowest compatible direct dependencies on Python 3.11; its latest profile
resolves fresh dependencies on Python 3.14. A separate Python 3.15 advisory probe
checks stable dependency availability, without the preview's nightly Torch index.
These jobs resolve in disposable checkouts and save their resulting lockfiles.
The minimum profile targets CPU compatibility; accelerator jobs use the locked
dependency set.

Real accelerator jobs run nightly or manually after a runner is configured.
Set GitHub repository variables `CUDA_RUNNER` and/or `MPS_RUNNER` to JSON arrays
of labels for the corresponding self-hosted runner, for example:

```json
["self-hosted", "linux", "x64", "relflow-cuda"]
```

An unset variable leaves that hardware job disabled. Configured jobs require
the selected accelerator to be available before running, so skipped tests do
not substitute for GPU execution. They use a separate Actions workspace and
do not change an existing lab checkout or environment.

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

User documentation and branding assets live in [docs/](docs/README.md).
Canonical learning experiments, runtime tests, and recorded results live in
[evidence/](evidence/README.md). Coordinate those changes with public API changes
in the same branch. Follow the [documentation agent guide](docs/AGENTS.md) for
writing and diagrams, and the [evidence protocol](evidence/SPEC.md) for learning
experiments. Static documentation tests live in `docs/tests/`; package runtime
and architectural tests remain in `tests/`.

The documentation environment is isolated from the package environment. Run
these commands from the repository root:

```bash
uv sync --project docs --locked
make check
make check-docs
make dev
```

`make check` validates static examples, links, and evidence reporting;
`make check-docs` also renders and validates the full site. `make dev` previews
it at `http://127.0.0.1:4200`. The build uses pinned Quarto and never imports
RelFlow or executes models. Library checks do not build the site or run learning
experiments; explicit evidence execution requires the intended RelFlow wheel.
