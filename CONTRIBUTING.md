# Contributing

Start with [AGENTS.md](AGENTS.md) for the package model, public API conventions,
code style, and ownership boundaries. Keep changes focused and test the public
contract and the invariants affected by the change.

## Licensing and contribution rights

Public relflow releases beginning with 0.5.0 use `AGPL-3.0-only`, as set out in
[LICENSE](LICENSE) and [NOTICE](NOTICE). Project-authored documentation, examples,
schema machinery, caches, and preset definitions share that license unless a
file states otherwise. Preserve third-party licenses and attribution.

Additional licensing permissions and changes away from AGPL require written
approval from both Grantham Taylor and Reid Taylor. A pull request or a DCO
sign-off alone does not grant proprietary relicensing rights. Before merging
outside contributions, both maintainers must retain a separate written grant
from every relevant rights holder authorizing inclusion under AGPL and under
additional terms they jointly approve. This includes contributions copied from
forks and employer-owned work. The contribution instructions do not replace that
agreement or establish a transfer of copyright.

Check source licensing before a build creates any distributions:

```bash
python scripts/release.py licensing
```

The required CI source check precedes package and documentation builds. The
package workflow repeats it before building; archive validation then checks
0.5.0 and later wheels and source distributions for AGPL metadata, the unchanged
license text, and the attribution notice. Preserve already-published artifacts
and their original licensing and provenance.

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
uv run --script scripts/tree.py
uv run ruff check
uv run pyrefly check
uv run pytest
```

The commit hooks check repository structure, apply Ruff formatting and lint fixes, then run Pyrefly and the
fast test suite. Review any files changed by the hooks before committing again.

## Repository structure

`[tool.relflow.tree]` in `pyproject.toml` defines the allowed root files, durable
tool entrypoints, source directories, file types, and size limits. New source
files fit an existing ownership rule; a new repository responsibility requires
an explicit policy change alongside its implementation. Do not add broad
exceptions for scratch files or generated output.

- `src/relflow/` owns package Python code; `tests/` owns package checks.
- `docs/` owns static publishing, authored pages, examples, and curated assets.
- `evidence/experiments/` owns every Python learning experiment; each script
  declares its dependency environment and classification tags; every experiment
  belongs to the same required CI catalog. Each file owns its review, diagnostic
  components, highlights, and scheduling estimates. Adding or removing an experiment
  must require no edit to a central per-ID table.
- `native/` owns Rust sources, Python bindings, and their tests.
- `scripts/` contains explicitly admitted repository tools. One-off investigations
  and generated reports belong outside the tracked tree.

Run `make check-tree` to check tracked and nonignored untracked files. Commit
hooks and the required CI structure job use `--tracked`; ignored artifacts still
fail if force-added. The structure job must pass before package and docs builds
start. Archives, checkpoints, caches, rendered sites, dependency directories,
and large measurement artifacts stay in ignored local output or CI/release assets.

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

The installed-wheel import job uses required dependencies only, with no development
or package extras. It verifies training, checkpoint reload, and prediction without
Matplotlib, reproduces the upstream incomplete-namespace failure, then installs
Matplotlib 3.10.7 and checks modeling plus TorchMetrics plotting. See
[issue 67](https://github.com/relflow/relflow/issues/67) for the diagnosis and
environment remedy.

The Tests workflow runs the complete suite on Linux with Python 3.12, 3.13,
and 3.14, plus macOS ARM and Windows with Python 3.12. It installs a wheel
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
the lowest compatible direct dependencies on Python 3.12; its latest profile
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
writing and interactive components, and the [evidence protocol](evidence/SPEC.md)
for learning experiments. Starlight tests live in `docs/starlight/tests/`, static
publisher tests in `docs/tests/`, and Evidence protocol tests in `evidence/tests/`.
Package runtime and architectural tests remain in `tests/`.

The documentation environment is isolated from the package environment. Run
these commands from the repository root:

```bash
make prepare
make check
make check-docs
make dev
```

Use Node 24 and Python 3.12 or newer. `make check` validates formatting, types,
component contracts, and static publisher/Evidence tooling; `make check-docs`
builds Starlight and checks links, canonical URLs, and page/bundle budgets.
`make dev` previews the site at
`http://127.0.0.1:4300`. Install Chromium with
`cd docs/starlight && npx playwright install chromium`, then run
`make check-browser` from the repository root for browser checks.
Site builds never import relflow or execute models. Library checks do not build the site or run learning
experiments; explicit evidence execution requires the intended RelFlow wheel.

Read the Docs builds hosted versions and pull request previews using the root
[.readthedocs.yaml](.readthedocs.yaml). GitHub documentation CI retains source
and rendered-site validation with a downloadable artifact. Keep documentation
changes in the same PR as their API changes, then review the Read the Docs
preview before merging. See [docs/README.md](docs/README.md) for initial account
setup, version activation, Gold-backed ad-free hosting, and domain cutover.
