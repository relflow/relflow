# RelFlow evidence

This directory owns the 79 controlled synthetic experiments, their protocol,
lab runner, runtime tests, and current publication results.
The package, website, and evidence share this repository. Stable P identifiers remain
unchanged. Each script's frontmatter records its ID, group, selected full run,
metrics, checks, seed, environment, source provenance, and retained report link.

Run from the repository root:

```sh
make check
uv run --project docs python evidence/experiments/run.py --list
uv run --project docs --with /path/to/matching-relflow.whl python evidence/experiments/run.py P001 --steps 2
uv run --project docs --with /path/to/matching-relflow.whl pytest -c docs/pyproject.toml evidence/tests/runtime
```

Supply an actual wheel. Use the isolated documentation project so the root
package environment cannot substitute the editable SDK for that wheel.
A two-step run checks execution only. Each execution
writes a new JSON file under evidence/runs/local; --output-dir selects another
directory. Execution never edits the reviewed publication selection.

## Required lab execution

The library's workflow builds exact baseline and merge-candidate wheels and
calls .github/workflows/evidence-lab.yml at a reviewed full commit SHA, passing
a separate reviewed suite SHA. The helper pins the trusted executor. GitHub
joins Tailscale using short-lived OIDC and dispatches one experiment per Docker
container on lab. Containers have bounded CPU/memory, no network, a read-only
root, and no host home or Docker socket. The full catalog runs for both wheels
with the original seeds and budgets. Every baseline-passing check must pass on
the candidate; baseline misses remain reported. CI queues lab work across PRs
and cancels superseded attempts within each PR. Errors, missing records, invalid
provenance, smoke execution, and partial coverage fail CI / required.

The combined CI gate also requires the library test matrix and static
documentation validation. The suite normally comes from the candidate commit
in this repository. A new suite can be validated explicitly with
`gh workflow run ci.yml --repo relflow/relflow --ref main -f evidence_ref=<full-suite-commit>`.
This tests the selected suite without changing the trusted executor or
Tailscale trust.

See [lab.md](lab.md) for trust and repository variables. No AWS account,
Modal setup, stored Tailscale secret, or self-hosted GitHub runner is required.

## Reports and publication

Routine CI reports are Actions artifacts in relflow/relflow, retained for 90
days; input wheels and requirements are retained for seven days. Before expiry,
promote a selected complete run to an immutable Release in relflow/relflow:

```sh
gh workflow run publish-evidence.yml --repo relflow/relflow \
  -f tag=<unique-release-tag> -f library_run_id=<library-actions-run-id>
```

The workflow verifies completeness, provenance, wheel digests, and the runtime
lock; preserves reports and inputs with checksums; and generates a validated
publication patch for experiment frontmatter. Download evidence-publication-update,
review the patch on a branch at its recorded repository base, and open a normal PR.
The site's normal build reads the committed records and canonical sources.
There are no snapshot bundles, importers/exporters, or old Proofs redirects.

Existing report links in relflow/docs retain their original location. Raw
report releases retain execution provenance. Updating source or output
readers does not establish a fresh learning result. The checkout keeps the
current selection without archived result snapshots or history readers.
